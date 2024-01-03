import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny

def adjust_gamma(img, gamma=0.4):
    npim = np.array(img)
    npim_gamma = 255.0 * (npim / 255.0) ** gamma
    return Image.fromarray(np.uint8(npim_gamma))

ptsans = ImageFont.truetype("pt-sans-narrow-regular.ttf",50)
atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",50)
atkreg = ImageFont.truetype("Atkinson-Hyperlegible-Regular-102.otf",50)

def mask_image(timestamp):
    mask_text = timestamp.strftime("%-I%M").upper()
    mask_size = (512,512)
    time_img = Image.new("L", mask_size, (0,))
    draw = ImageDraw.Draw(time_img)
    draw.text((0,0), mask_text, (255,), font=atkbold)
    cropped = time_img.crop(time_img.getbbox())
    resized = ImageOps.expand(ImageOps.pad(cropped, (max(*cropped.size), max(*cropped.size))), 5).resize(mask_size, resample=Image.Resampling.LANCZOS)
    enhanced_image = ImageEnhance.Contrast(resized).enhance(9000)
    return enhanced_image


def qint8(m, per_channel=True, inplace=False):
    qconfig_spec = {torch.nn.Linear: (torch.quantization.per_channel_dynamic_qconfig if per_channel else torch.quantization.default_dynamic_qconfig)}
    return torch.quantization.quantize_dynamic(m, qconfig_spec=qconfig_spec, dtype=torch.qint8, inplace=inplace)


preferred_dtype = torch.float32
preferred_device = "cpu"
torch.backends.quantized.engine = "qnnpack"
unet_preferred_dtype = torch.qint8

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=preferred_dtype,
).to(preferred_device)
qint8(controlnet, inplace=True)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    controlnet=controlnet,
    vae=AutoencoderTiny.from_pretrained("madebyollin/taesd"),
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

pipe.unet.set_default_attn_processor()

qint8(pipe.unet, inplace=True)
qint8(pipe.text_encoder, inplace=True)

print("Quantized.\n")

current_denoising_steps = 2
target_latency = 900
current_latency = 0
rounding_minutes = 15
target_filename = "/tmp/beauty.png"
mask_image(timestamp=datetime.now()).save(target_filename)

cali1 = "desert landscape with tall mountains and cactus and boulders at sunrise with the sun on the horizon"
cali2 = "stony river in a sunny redwood forest with salmon and deer and bears and mushrooms"
cali3 = "cliffs at the beach, sun at the horizon, piping plovers, dolphins in the distance jumping out of the water"

prompts = [
    cali1, cali2, cali3,
    "still life with fruit and flowers",
    cali1, cali2, cali3,
    "bowl of lettuces and root vegetables",
]
negative_prompt = "low quality, ugly, wrong"

for iteration in range(86400 * 365 * 80):
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
        controlnet_conditioning_scale=0.5,
        #control_guidance_start=0,
        #control_guidance_end=1,
        #cross_attention_kwargs={"scale": 1},
        generator=torch.manual_seed(int(pre_render_time.timestamp())),
        height=512,
        width=512,
    ).images[0]
    adjust_gamma(image, gamma=0.5).convert(mode="L").convert(mode="RGB").convert(mode="P").save(target_filename)

    post_render_time = datetime.now()
    current_latency = post_render_time.timestamp() - pre_render_time.timestamp()
    # current_denoising_steps += 1 if current_latency < target_latency else -1 # 2 is fine lol

