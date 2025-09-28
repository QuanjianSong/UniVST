import os

import argparse
from typing import Optional
from omegaconf import OmegaConf

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.animatediff.models.unet import UNet3DConditionModel
from backbones.animatediff.pipelines.pipeline_animation import AnimationPipeline
from backbones.animatediff.utils.util import load_weights

from src.animatediff.tools.util_animatediff import content_inversion_reconstruction
from src.util import seed_everything

import decord
decord.bridge.set_bridge('torch')


def main(
    pretrained_model_path: str,
    content_path: str,
    output_path: str,
    num_frames: int = 16,
    height: int = 512,
    width: int = 512,
    time_steps: int = 50,
    weight_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = 33,
    is_opt: bool = False,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('/data/lxy/sqj/base_models/stable-video-diffusion-img2vid', subfolder="vae").requires_grad_(False)
    inference_config = OmegaConf.load("/data/lxy/sqj/code/UniVST/backbones/animatediff/animatediff-v2.yaml")
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
        motion_module_path         = "/data/lxy/sqj/code/UniVST/ckpt/mm_sd_v15_v2.ckpt",
        device = "cuda",
        dtype = weight_dtype
    )
    # inversion scheduler
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(time_steps)
    # go!
    # breakpoint()
    content_inversion_reconstruction(pipe, ddim_inv_scheduler, content_path, output_path,
                                    num_frames, height, width, time_steps, weight_dtype,
                                    is_opt=is_opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-1.5")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument('--is_opt', action='store_true', help='use Easy-Inv')
    parser.add_argument("--content_path", type=str, default="examples/content/libby")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
