import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from datetime import datetime, timedelta

#epoch_seconds = int(datetime.now().timestamp())
#current_time = datetime.now(beautiful_downtown_oakland_california).strftime("%d%b").upper().strip('0')
ptsans = ImageFont.truetype("pt-sans-narrow-regular.ttf",50)
atkbold = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",50)
atkreg = ImageFont.truetype("Atkinson-Hyperlegible-Regular-102.otf",50)
mask_text = "OH\nNO"
mask_size = (512,512)
time_img = Image.new("L", mask_size, (0,))
draw = ImageDraw.Draw(time_img)
draw.text((0,0), mask_text, (255,), font=atkbold)
cropped = time_img.crop(time_img.getbbox())
resized = ImageOps.expand(ImageOps.pad(cropped, (max(*cropped.size), max(*cropped.size))), 5).resize(mask_size, resample=Image.Resampling.LANCZOS)
mask_image = ImageEnhance.Contrast(resized).enhance(9000)
mask_image

preferred_dtype = torch.float32
preferred_device = "cpu"
torch.backends.quantized.engine = "qnnpack"
unet_preferred_dtype = torch.qint8

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=preferred_dtype,
).to(preferred_device)
torch.quantization.quantize_dynamic(controlnet, dtype=torch.qint8, inplace=True)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=preferred_dtype,
    safety_checker=None,
).to(preferred_device)

pipe.enable_attention_slicing()

torch.quantization.quantize_dynamic(
    pipe.unet,
    dtype=torch.qint8,
    inplace=True,
    qconfig_spec={torch.nn.Linear: torch.quantization.per_channel_dynamic_qconfig}
)

pipe.unet = torch.compile(pipe.unet)
pipe.vae = torch.compile(pipe.vae)
pipe.text_encoder = torch.compile(pipe.text_encoder)
print("Finished compiling and quantizing.\n")

for iteration in range(86400 * 365 * 80):
    print(datetime.now())
    image = pipe(
        #prompt="detail of a new hieronymous bosch painting, high quality",
        #prompt="watercolor of a leafy pedestrian mall at golden hour with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing",
        #prompt="corgis running in the park with trees, golden hour",
        #prompt="still life with fruit and flowers",
        prompt="puppies in the park",
        negative_prompt="low quality, bad quality, sketches, wrong",
        image=mask_image,
        num_inference_steps=20,
        guidance_scale=7.0,
        controlnet_conditioning_scale=0.9,
        #control_guidance_start=0,
        #control_guidance_end=1,
        #cross_attention_kwargs={"scale": 1},
        generator=torch.manual_seed(int(datetime.now().timestamp())),
        height=512,
        width=512,
    ).images[0]
    image.save("puppies-in-the-park-oh-no.jpg")

