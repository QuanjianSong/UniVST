import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Optional

import torch
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version

from transformers import CLIPTextModel, CLIPTokenizer
from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.data.dataset import UniVSTDataset
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline
from video_diffusion.util import save_videos_grid, ddim_inversion, seed_everything, load_video_frames
from einops import rearrange
import os


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def main(
    pretrained_model_path: str,
    content_path: str,
    output_dir: str,
    gradient_accumulation_steps: int = 1,
    mixed_precision: Optional[str] = "fp16",
    up_ft_indices: int = 2,
    num_frames: int = 16,
    time_steps: int = 16,
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
    # load video data
    if content_path.endswith(".mp4"):
        train_dataset = UniVSTDataset(content_path, prompt="")
        # Preprocessing the dataset
        train_dataset.prompt_ids = tokenizer(
            train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1
        )
        # Prepare everything with our `accelerator`.
        train_dataloader = accelerator.prepare(train_dataloader)
        # load data
        batch = next(iter(train_dataloader))
        pixel_values = batch["pixel_values"].to(weight_dtype).to("cuda")
        video_length = pixel_values.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
    else:
        pixel_values = load_video_frames(content_path, num_frames).to(weight_dtype).to("cuda")
        video_length = pixel_values.shape[0]
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = latents * 0.18215
    # make dir
    name = content_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, 'content', name, 'inversion')
    reconstruction_path = os.path.join(output_dir, 'content', name, 'reconstruction')
    ft_path = os.path.join(output_dir, 'features', name)
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    # ----------------------------------content video inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", 
                                    is_opt=True, inversion_path=inversion_path,
                                    up_ft_indices=[up_ft_indices], ft_path=ft_path)[-1].to(weight_dtype)
    # ---------------------------------content video construction------------------------------
    print(f"reconstruction:")
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(seed)
    sample = pipe("", generator=generator, latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "video.gif"))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--up_ft_indices", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--content_path", type=str, default="example/content/libby")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
