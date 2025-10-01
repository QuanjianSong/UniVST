import os

import argparse
from typing import Optional
from omegaconf import OmegaConf

import torch

from diffusers import DDIMScheduler
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.animatediff.models.unet import UNet3DConditionModel
from backbones.animatediff.pipelines.pipeline_animation import AnimationPipeline
from backbones.animatediff.utils.util import load_weights
from backbones.animatediff.pnp_utils import register_spatial_attention_pnp

from src.util import save_folder, save_videos_grid, load_ddim_latents_at_t, seed_everything


def main(
    pretrained_model_path: str,
    motion_module_path: str,
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
    vae = AutoencoderKLTemporalDecoder.from_pretrained('/data/lxy/sqj/base_models/stable-video-diffusion-img2vid', subfolder="vae").requires_grad_(False)
    inference_config = OmegaConf.load("backbones/animatediff/animatediff-v2.yaml")
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).requires_grad_(False)
    # set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()
    # custom pipe
    pipe = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    )
    pipe = load_weights(
        pipe,
        motion_module_path=motion_module_path,
        device="cuda",
        dtype=weight_dtype
    )

    # load data
    content_inv_noises = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=content_inv_path
        ).to(weight_dtype).cuda()
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward
    register_spatial_attention_pnp(pipe)
    # video_style_transfer
    sample = pipe.video_style_transfer("", latents=content_inv_noises,
                                    num_inference_steps=time_steps,
                                    content_inv_path=content_inv_path, style_inv_path=style_inv_path, mask_path=mask_path).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    # save
    output_path = os.path.join(output_path, 'animatediff', f'{content_inv_path.split("/")[-2]}_{style_inv_path.split("/")[-2]}')
    os.makedirs(output_path, exist_ok=True)
    save_folder(sample, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-1.5")
    parser.add_argument("--motion_module_path", type=str, default="ckpt/mm_sd_v15_v2.ckpt")
    parser.add_argument("--content_inv_path", type=str, default="output/content/libby/inversion")
    parser.add_argument("--style_inv_path", type=str, default="output/style/style1/inversion")
    parser.add_argument("--mask_path", type=str, default="output/mask/libby")
    parser.add_argument("--output_path", type=str, default="results/stylized")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
