import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny
from tqdm import tqdm

def adjust_gamma(img, gamma=0.4):
    npim = np.array(img)
    npim_gamma = 255.0 * (npim / 255.0) ** gamma
    return Image.fromarray(np.uint8(npim_gamma))

atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",200)
atkbold_smol = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",40)

image_size = (480, 360)
screen_size = (1100, 825)
screen_is_monochrome = True

demo_mode = True
keep_fp32_on_cpu = False

def mask_image(timestamp):
    is_two_line = False # our images are 4:3 instead of 1:1, so we have space for all on one line
    linesep = "\n" if is_two_line else ""
    mask_text = timestamp.strftime(f"%-I{linesep}%p").upper() if timestamp.minute == 0 else timestamp.strftime(f"%-I{linesep}%M")
    time_img = Image.new("L", image_size, (0,))
    draw = ImageDraw.Draw(time_img)
    draw.multiline_text(
        xy=(0,0),
        text=mask_text,
        fill=(255,),
        font=atkbold,
        align="center",
        spacing=-10,
    )
    cropped = time_img.crop(time_img.getbbox())
    resized = ImageOps.expand(ImageOps.pad(cropped, (max(*cropped.size), max(*cropped.size))), -10).resize(image_size, resample=Image.Resampling.LANCZOS)
    enhanced_image = ImageEnhance.Contrast(resized).enhance(9000)
    return enhanced_image


def qint8(m, per_channel=True, inplace=False):
    qconfig_spec = {torch.nn.Linear: (torch.quantization.per_channel_dynamic_qconfig if per_channel else torch.quantization.default_dynamic_qconfig)}
    return torch.quantization.quantize_dynamic(m, qconfig_spec=qconfig_spec, dtype=torch.qint8, inplace=inplace)


preferred_dtype = torch.float32
preferred_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.quantized.engine = "qnnpack"

if demo_mode:
    preferred_device = "cpu"
    preferred_dtype = torch.float32

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=preferred_dtype,
).to(preferred_device)
if preferred_device == "cpu" and not keep_fp32_on_cpu:
    qint8(controlnet, inplace=True)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    controlnet=controlnet,
    #vae=AutoencoderTiny.from_pretrained("madebyollin/taesd"),
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

pipe.vae.set_default_attn_processor()
pipe.unet.set_default_attn_processor()

if preferred_device == "cpu" and not keep_fp32_on_cpu:
    qint8(pipe.unet, inplace=True)
    qint8(pipe.text_encoder, inplace=True)
    qint8(pipe.vae, inplace=True)


current_denoising_steps = 3
current_latency = 0
rounding_minutes = 15
target_filename = "/tmp/beauty.png"
mask_image(timestamp=datetime.now()).save(target_filename)

cali1 = "desert landscape with tall mountains and cactus and boulders at sunrise with the sun on the horizon"
cali2 = "stony river in a sunny redwood forest with salmon and deer and bears and mushrooms"
cali3 = "beach, tall cliffs, sun at the horizon, albatross eating fish, no one on the beach, boulders and tide pools in the shallow water"
cali4 = "nighttime photo of a desert landscape with the milky way in the sky and boulders on a shallow lake bed surrounded by tall mountains"

prompts = [
    cali1,
    cali2,
    cali3,
    cali4,
]
conditioning_scales = {
    cali1: 0.8,
    cali2: 0.76,
    cali3: 0.79,
    cali4: 0.72,
}
negative_prompt = "low quality, ugly, wrong"

four_color_image = Image.new("P", (1,1))
four_color_image.putpalette([0,0,0,64,64,64,128,128,128,192,192,192,255,255,255])

if demo_mode:
    iteration_range = range(4 * 4 * 24)
else:
    iteration_range = range(86400 * 365 * 80)

for iteration in tqdm(iteration_range):
    if demo_mode:
        synthetic_time = datetime.fromtimestamp(iteration * 15 * 60 // 4 + 8 * 3600 - 2 * 15 * 60 // 4 + 1)
        target_filename = f"face-render-{iteration:06}.png"
        current_latency = 0
        pre_render_time = synthetic_time
    else:
        pre_render_time = datetime.now()

    target_time_plus_midpoint = pre_render_time + timedelta(seconds=(current_latency + rounding_minutes * 60 / 2))
    rounded_target_time = target_time_plus_midpoint - timedelta(minutes=target_time_plus_midpoint.minute - target_time_plus_midpoint.minute // rounding_minutes * rounding_minutes)
    current_mask_image = mask_image(timestamp=rounded_target_time)
    print(f"current_latency: {current_latency}, pre_render_time: {pre_render_time}, rounded_target_time: {rounded_target_time}, current_denoising_steps: {current_denoising_steps}\n")

    image = pipe(
        prompt=prompts[iteration % len(prompts)],
        negative_prompt=negative_prompt,
        image=current_mask_image,
        num_inference_steps=min(max(current_denoising_steps, 1),16),
        guidance_scale=7.0,
        controlnet_conditioning_scale=conditioning_scales[prompts[iteration % len(prompts)]],
        #control_guidance_start=0,
        #control_guidance_end=1,
        #cross_attention_kwargs={"scale": 1},
        generator=torch.manual_seed(int(pre_render_time.timestamp())),
        height=image_size[1],
        width=image_size[0],
    ).images[0]
    image = adjust_gamma(image, gamma=0.5)
    image = ImageEnhance.Sharpness(image).enhance(5)
    image = image.resize(screen_size)
    if demo_mode:
        draw = ImageDraw.Draw(image)
        draw.text((60, screen_size[1]-60), f"leebutterman.com", fill=(255,255,255), font=atkbold_smol)
    image.save(target_filename)

    post_render_time = datetime.now()
    current_latency = post_render_time.timestamp() - pre_render_time.timestamp()
