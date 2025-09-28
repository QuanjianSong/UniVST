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


def init_pnp(pipe):
    register_spatial_attention_pnp(pipe)


def main(
    pretrained_model_path: str,
    inv_path: str,
    style_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = 33,
    **kwargs,
):  
    if seed is not None:
        seed_everything(seed)
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('/data/lxy/sqj/base_models/stable-video-diffusion-img2vid', subfolder="vae").requires_grad_(False)
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

    
    inv_latents_at_t = load_ddim_latents_at_t(
            981, ddim_latents_path=inv_path
        ).to(weight_dtype).cuda()
    style_latents_at_t = load_ddim_latents_at_t(
            981, ddim_latents_path=style_path
        ).to(weight_dtype).cuda()
    output_path = os.path.join(output_path, 'edit')
    os.makedirs(output_path, exist_ok=True)
    # -----------------------------------------------------------------------------------------------
    # Init latent-shift
    inv_latents_at_t = latent_adain(inv_latents_at_t, style_latents_at_t)
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward
    init_pnp(pipe)
    # video_style_transfer
    sample = pipe.video_style_transfer("", latents=inv_latents_at_t, video_length=16, height=512, width=512, num_inference_steps=50, guidance_scale=1.0,
                                inv_path=inv_path, style_path=style_path, mask_path=mask_path).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    # save
    save_folder(sample, output_path)
    # save_videos_grid(sample, os.path.join(output_dir, "style.mp4"), fps=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-2-1-base")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-1.5")
    parser.add_argument("--inv_path", type=str, default="output/content/libby/inversion")
    parser.add_argument("--style_path", type=str, default="output/style/style1/inversion")
    parser.add_argument("--mask_path", type=str, default="output/mask/libby")
    parser.add_argument("--output_path", type=str, default="output/")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
