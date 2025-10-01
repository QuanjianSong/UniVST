import argparse
import os
from typing import Optional

import torch
import torch.utils.checkpoint

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.video_diffusion_sd.models.unet_3d_condition import UNetPseudo3DConditionModel
from backbones.video_diffusion_sd.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline
from backbones.video_diffusion_sd.pnp_utils import register_spatial_attention_pnp, latent_adain

from src.util import save_folder, save_videos_grid, load_ddim_latents_at_t, seed_everything


def main(
    pretrained_model_path: str,
    content_inv_path: str,
    style_inv_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    time_steps: int = 50,
    seed: Optional[int] = 33,
    **kwargs,
):  
    if seed is not None:
        seed_everything(seed)
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('stabilityai/stable-video-diffusion-img2vid', subfolder="vae").requires_grad_(False)
    unet = UNetPseudo3DConditionModel.from_2d_model(os.path.join(pretrained_model_path, "unet")).requires_grad_(False)
    # set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()
    # custom pipe
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )

    
    content_inv_noise = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=content_inv_path
        ).to(weight_dtype).cuda()
    style_inv_noise = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=style_inv_path
        ).to(weight_dtype).cuda()
    # -----------------------------------------------------------------------------------------------
    # Init latent-shift
    inv_latents_at_t = latent_adain(content_inv_noise, style_inv_noise)
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward
    register_spatial_attention_pnp(pipe)
    # video_style_transfer
    sample = pipe.video_style_transfer("", latents=inv_latents_at_t,
                                       num_inference_steps=time_steps,
                                       content_inv_path=content_inv_path, style_inv_path=style_inv_path, mask_path=mask_path).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    # save
    output_path = os.path.join(output_path, 'sd', f'{content_inv_path.split("/")[-2]}_{style_inv_path.split("/")[-2]}')
    os.makedirs(output_path, exist_ok=True)
    save_folder(sample, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    # parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--content_inv_path", type=str, default="results/contents-inv/sd/mallard-fly/inversion")
    parser.add_argument("--style_inv_path", type=str, default="results/styles-inv/sd/00033/inversion")
    parser.add_argument("--mask_path", type=str, default="results/masks/sd/mallard-fly")
    parser.add_argument("--output_path", type=str, default="results/stylizations")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
