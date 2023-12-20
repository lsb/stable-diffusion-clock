import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from datetime import datetime, timedelta
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

ptsans = ImageFont.truetype("pt-sans-narrow-regular.ttf",50)
atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",50)
atkreg = ImageFont.truetype("Atkinson-Hyperlegible-Regular-102.otf",50)

def mask_image(timestamp):
    mask_text = timestamp.strftime("%a\n%-I%p").upper()
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
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

pipe.enable_attention_slicing()

qint8(pipe.unet, inplace=True)

print("Quantized.\n")

current_denoising_steps = 20
target_latency = 3000
current_latency = 0
half_an_hour = 3600 / 2
target_filename = "/tmp/beauty.png"
mask_image(timestamp=datetime.now()).save(target_filename)

prompt = "watercolor of a leafy pedestrian mall at golden hour with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing"
negative_prompt = "low quality, ugly, wrong"

prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt=prompt, device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=negative_prompt)

for iteration in range(86400 * 365 * 80):
    pre_render_time = datetime.now()
    rounded_target_time = pre_render_time + timedelta(seconds=current_latency+half_an_hour)
    current_mask_image = mask_image(timestamp=rounded_target_time)
    print(f"current_latency: {current_latency}, pre_render_time: {pre_render_time}, rounded_target_time: {rounded_target_time}, current_denoising_steps: {current_denoising_steps}\n")

    image = pipe(
        #prompt="detail of a new hieronymous bosch painting, high quality",
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        #prompt="corgis running in the park with trees, golden hour",
        #prompt="still life with fruit and flowers",
        #prompt="cute puppies in the park",
        image=current_mask_image,
        num_inference_steps=(current_denoising_steps if current_latency > 0 else 10),
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.95,
        #control_guidance_start=0,
        #control_guidance_end=1,
        #cross_attention_kwargs={"scale": 1},
        generator=torch.manual_seed(int(pre_render_time.timestamp())),
        height=512,
        width=512,
    ).images[0]
    image.save(target_filename)

    post_render_time = datetime.now()
    current_latency = post_render_time.timestamp() - pre_render_time.timestamp()
    current_denoising_steps += 1 if current_latency < target_latency else -1

