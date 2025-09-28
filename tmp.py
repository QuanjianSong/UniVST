import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLPipeline.from_pretrained("/data/lxy/sqj/base_models/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16)
pipe.to("cuda")
my_unet = pipe.unet
breakpoint()
input_image = load_image("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]