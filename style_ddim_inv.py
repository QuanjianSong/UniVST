import argparse
import logging
import inspect
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version

from transformers import CLIPTextModel, CLIPTokenizer
from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline
from video_diffusion.util import save_videos_grid, ddim_inversion, seed_everything
from einops import rearrange
import os
from PIL import Image


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def main(
    pretrained_model_path: str,
    style_path: str,
    output_dir: str,
    gradient_accumulation_steps: int = 1,
    num_frames: int = 16,
    time_steps: int = 50,
    mixed_precision: Optional[str] = "fp16",
    seed: Optional[int] = 33,
):  
    *_, config = inspect.getargvalues(inspect.currentframe())
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
        seed_everything(seed)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('stabilityai/stable-video-diffusion-img2vid', subfolder="vae")
    model_config = {}
    unet = UNetPseudo3DConditionModel.from_2d_model(os.path.join(pretrained_model_path, "unet"), model_config=model_config)
    # Set grad false
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(time_steps)
    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)
    # load style
    style_image = Image.open(style_path).convert("RGB").resize((512, 512))
    # transforms
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    style_tensor = transform(style_image)
    pixel_values = style_tensor.repeat(num_frames, 1, 1, 1).to(weight_dtype).to("cuda")
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = latents * 0.18215
    # make dir
    name = style_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, 'style', name, 'inversion')
    reconstruction_path = os.path.join(output_dir, 'style', name, 'reconstruction')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    # ----------------------------------content style inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", 
                                    is_opt=True, inversion_path=inversion_path)[-1].to(weight_dtype)
    # ---------------------------------content style construction------------------------------
    print(f"reconstruction:")
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(seed)
    sample = pipe("", generator=generator, latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "style.gif"))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--style_path", type=str, default="example/style/style1.png")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
