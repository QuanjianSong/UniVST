import argparse
import os
from typing import Optional

import torch
from diffusers import DDIMScheduler
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPTextModel, CLIPTokenizer

from backbones.video_diffusion_sd.models.unet_3d_condition import UNetPseudo3DConditionModel
from backbones.video_diffusion_sd.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline

from inversion_tools.ddim_inversion import style_inversion_reconstruction
from src.util import seed_everything


def main(
    pretrained_model_path: str,
    style_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    num_frames: int = 16,
    height: int = 512,
    width: int = 512,
    time_steps: int = 50,
    #
    is_opt: bool = False,
    seed: Optional[int] = 33,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    # use 3d vae for more stable results
    vae = AutoencoderKLTemporalDecoder.from_pretrained('/data/lxy/sqj/base_models/stable-video-diffusion-img2vid', subfolder="vae").requires_grad_(False)
    unet = UNetPseudo3DConditionModel.from_2d_model(os.path.join(pretrained_model_path, "unet")).requires_grad_(False)
    # set device
    text_encoder = text_encoder.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    unet = unet.to(weight_dtype).cuda()
    # custom pipe
    pipe = SpatioTemporalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    # inversion scheduler
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(time_steps)
    # make dir
    output_path = os.path.join(output_path, 'sd', style_path.split('/')[-1].split('.')[0])
    inversion_path = os.path.join(output_path, 'inversion')
    reconstruction_path = os.path.join(output_path, 'reconstruction')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    # go!
    with torch.no_grad():
        style_inversion_reconstruction(pipe, ddim_inv_scheduler, style_path, inversion_path, reconstruction_path,
                                    num_frames, height, width, time_steps, weight_dtype,
                                    is_opt=is_opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-1.5")
    # parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-2-1-base")
    parser.add_argument("--style_path", type=str, default="examples/style/style1.png")
    parser.add_argument("--output_path", type=str, default="results/style-inv")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--time_steps", type=int, default=50)
    #
    parser.add_argument('--is_opt', action='store_true', help='use Easy-Inv')
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
