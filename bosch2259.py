import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
import random
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)
import time

preferred_dtype = torch.float16
preferred_device = "cuda"
BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=preferred_dtype).to(preferred_device)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=preferred_dtype).to(preferred_device)
main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=preferred_dtype,
).to(preferred_device)
image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension)/2
    top = (height - new_dimension)/2
    right = (width + new_dimension)/2
    bottom = (height + new_dimension)/2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


def common_upscale(samples, width, height, upscale_method, crop=False):
        if crop == "center":
            old_width = samples.shape[3]
            old_height = samples.shape[2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples[:,:,y:old_height-y,x:old_width-x]
        else:
            s = samples

        return torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)


def upscale(samples, upscale_method, scale_by):
        #s = samples.copy()
        width = round(samples["images"].shape[3] * scale_by)
        height = round(samples["images"].shape[2] * scale_by)
        s = common_upscale(samples["images"], width, height, upscale_method, "disabled")
        return (s)


def inference(
    control_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 8.0,
    controlnet_conditioning_scale: float = 1,
    control_guidance_start: float = 1,    
    control_guidance_end: float = 1,
    upscaler_strength: float = 0.5,
    seed: int = -1,
    sampler = "DPM++ Karras SDE",
):
    start_time = time.time()
    start_time_struct = time.localtime(start_time)
    start_time_formatted = time.strftime("%H:%M:%S", start_time_struct)
    print(f"Inference started at {start_time_formatted}")
    
    # Generate the initial image
    #init_image = init_pipe(prompt).images[0]

    # Rest of your existing code
    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = SAMPLER_MAP[sampler](main_pipe.scheduler.config)
    my_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    generator = torch.Generator(device=preferred_device).manual_seed(my_seed)
    
    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent"
    )
    upscaled_latents = upscale(out, "nearest-exact", 2)
    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,        
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=12,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )
    end_time = time.time()
    end_time_struct = time.localtime(end_time)
    end_time_formatted = time.strftime("%H:%M:%S", end_time_struct)
    print(f"Inference ended at {end_time_formatted}, taking {end_time-start_time}s")
    return out_image["images"][0]

from datetime import datetime
import pytz

beautiful_downtown_oakland_california = pytz.timezone("America/Los_Angeles")
prompts = {
  "bosch": "detail of a new hieronymus bosch painting",
  "train": "beautiful detail of a train map",
  "urban": "watercolor of a leafy pedestrian mall at golden hour with multiracial genderqueer joggers and bicyclists and wheelchair users talking and laughing",
}


atkinson = ImageFont.truetype("Atkinson-Hyperlegible-Regular-102.otf",100)
sigfont = ImageFont.truetype("Atkinson-Hyperlegible-Bold-102.otf",36)
for i in range(1000 * 1000 * 1000):
    epoch_seconds = int(datetime.now().timestamp())
    current_time = datetime.now(beautiful_downtown_oakland_california).strftime("%H\n%M")
    size = (512,512)
    time_img = Image.new("L", size, (0,))
    draw = ImageDraw.Draw(time_img)
    draw.text((0,0), current_time, (255,), font=atkinson)
    cropped = time_img.crop(time_img.getbbox())
    resized = ImageOps.expand(ImageOps.pad(cropped, (max(*cropped.size), max(*cropped.size))), 20).resize(size, resample=Image.Resampling.LANCZOS)
    enhanced = ImageEnhance.Contrast(resized).enhance(9000)

    for p in prompts:
        img = inference(
            control_image=enhanced,
            prompt=prompts[p],
            negative_prompt="low quality, ugly, wrong",
            controlnet_conditioning_scale=0.8,
            control_guidance_start=0,
            control_guidance_end=1,
            seed=epoch_seconds,
        )
        draw = ImageDraw.Draw(img)
        draw.text((25,975), "Lee Butterman", (255,255,255), font=sigfont)
        img.save(f"time.jpg", quality=80)
        img.save(f"time-{p}.jpg", quality=80)
        img.save(f"archive/{p}-{epoch_seconds}.jpg", quality=80)
