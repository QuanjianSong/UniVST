import argparse
import os
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from backbones.video_diffusion_sd3.models.transformer_3D_model import CustomSD3Transformer2DModel
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from inversion_tools.flow_inversion import content_inversion_reconstruction
from backbones.video_diffusion_sd3.pipelines.custom_pipeline import CustomStableDiffusion3Pipeline
from backbones.video_diffusion_sd3.pnp_utils import CrossFrameProcessor
from src.util import seed_everything

import decord
decord.bridge.set_bridge('torch')
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行化


def main(
    pretrained_model_path: str,
    content_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    num_frames: int = 16,
    height: int = 1024,
    width: int = 1024,
    time_steps: int = 50,
    #
    ft_indices: int = None,
    ft_timesteps: int = None,
    is_rf_solver: bool = False,
    seed: Optional[int] = 42,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    tokenizer      = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    tokenizer_2    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
    tokenizer_3    = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_3")
    #
    text_encoder   = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2").requires_grad_(False)
    text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_3").requires_grad_(False)
    #
    vae         = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").requires_grad_(False)
    scheduler   = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    transformer = CustomSD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer").requires_grad_(False)
    # Set cross-frame attention to support video
    new_procs = {}
    for idx, (name, processor) in enumerate(transformer.attn_processors.items()):
        if idx >= 0:
            if 'attn' in name:
                new_procs[name] = CrossFrameProcessor()
            else:
                new_procs[name] = processor
        else:
            new_procs[name] = processor
    transformer.set_attn_processor(new_procs)

    #
    text_encoder = text_encoder.to(weight_dtype).cuda()
    text_encoder_2 = text_encoder_2.to(weight_dtype).cuda()
    text_encoder_3 = text_encoder_3.to(weight_dtype).cuda()
    vae = vae.to(weight_dtype).cuda()
    transformer = transformer.to(weight_dtype).cuda()

    # custom pipe
    pipe = CustomStableDiffusion3Pipeline(
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3,
        vae=vae, transformer=transformer,
        scheduler=scheduler,
    )
    # make dir
    output_path = os.path.join(output_path, 'sd3', content_path.split('/')[-1])
    inversion_path = os.path.join(output_path, 'inversion')
    reconstruction_path = os.path.join(output_path, 'reconstruction')
    ft_path = os.path.join(output_path, 'features')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    # go!
    with torch.no_grad():
        content_inversion_reconstruction(pipe, content_path, inversion_path, reconstruction_path, 
                                        num_frames, height, width, time_steps, weight_dtype,
                                        ft_indices=[ft_indices], ft_timesteps=[ft_timesteps], ft_path=ft_path,
                                        is_rf_solver=is_rf_solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--content_path", type=str, default="examples/contents/mallard-fly")
    parser.add_argument("--output_path", type=str, default="results/contents-inv")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.bfloat16)
    #
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=50)
    #
    parser.add_argument("--ft_indices", type=int, default=20)
    parser.add_argument("--ft_timesteps", type=int, default=5)
    parser.add_argument('--is_rf_solver', action='store_true', help='use rf-solver')
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
