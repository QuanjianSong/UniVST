import argparse
import logging
import inspect
import os
from typing import Optional

import torch
import torch.nn.functional as F
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
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline
from video_diffusion.util import save_folder, save_videos_grid, ddim_inversion, load_ddim_latents_at_t
import os
from video_diffusion.pnp_utils import register_spatial_attention_pnp


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def init_pnp(pipe):
    register_spatial_attention_pnp(pipe)

def main(
    pretrained_model_path: str,
    inv_path: str,
    style_path: str,
    mask_path: str,
    output_dir: str,
    gradient_accumulation_steps: int = 1,
    mixed_precision: Optional[str] = "fp16",
    seed: Optional[int] = 33,
    **kwargs,
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
    # Load model
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('stabilityai/stable-video-diffusion-img2vid', subfolder="vae")
    model_config = {}
    unet = UNetPseudo3DConditionModel.from_2d_model(os.path.join(pretrained_model_path, "unet"), model_config=model_config)
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)
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
    ddim_inv_scheduler.set_timesteps(50)
    # load data
    inv_latents_at_t = load_ddim_latents_at_t(
            981, ddim_latents_path=inv_path
        ).to(accelerator.device, dtype=weight_dtype)
    style_latents_at_t = load_ddim_latents_at_t(
            981, ddim_latents_path=style_path
        ).to(accelerator.device, dtype=weight_dtype)
    # -----------------------------------------------------------------------------------------------
    # Init latent-shift
    inv_latents_at_t = adain(inv_latents_at_t, style_latents_at_t)
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward
    init_pnp(pipe)
    # breakpoint()
    video = pipe.video_style_transfer("", latents=inv_latents_at_t, video_length=16, height=512, width=512, num_inference_steps=50, guidance_scale=1.0,
                                inv_path=inv_path, style_path=style_path, mask_path=mask_path).images
    video = video.permute(0, 4, 1, 2, 3).contiguous()

    output_dir = os.path.join(output_dir, 'edit')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_folder(video, output_dir)

def adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[0, 3, 4], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 3, 4], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 3, 4], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 3, 4], keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    # -------------------------------------------------------------------------------------------
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean
    return output.to(sty_feat.dtype)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="configs/cat.yaml")
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--inv_path", type=str, default="output/content/libby/inversion")
    parser.add_argument("--mask_path", type=str, default="output/mask/libby")
    parser.add_argument("--style_path", type=str, default="output/style/style1/inversion")
    parser.add_argument("--output_dir", type=str, default="output/")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
