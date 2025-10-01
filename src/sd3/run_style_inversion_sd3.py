from typing import Optional
import argparse
import os

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

from inversion_tools.flow_inversion import style_inversion_reconstruction
from backbones.video_diffusion_sd3.pipelines.custom_pipeline import CustomStableDiffusion3Pipeline
from backbones.video_diffusion_sd3.pnp_utils import CrossFrameProcessor
from src.util import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    pretrained_model_path: str,
    style_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
    #
    num_frames: int = 16,
    height: int = 1024,
    width: int = 1024, 
    time_steps: int = 50,
    #
    is_rf_solver: bool = False,
    seed: Optional[int] = 42,
    **kwargs,
):
    if seed is not None:
        seed_everything(seed)
    tokenizer      = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    tokenizer_2    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
    tokenizer_3    = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_3")
    # --- Text Encoders（T5 + 两套 CLIP）---
    text_encoder   = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder").requires_grad_(False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2").requires_grad_(False)
    text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_3").requires_grad_(False)
    #
    vae         = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").requires_grad_(False)
    scheduler   = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    transformer = CustomSD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer").requires_grad_(False)
    # add cross-frame attention to support video
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
    
    # set device
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
    output_path = os.path.join(output_path, 'sd3', style_path.split('/')[-1].split('.')[0])
    inversion_path = os.path.join(output_path, 'inversion')
    reconstruction_path = os.path.join(output_path, 'reconstruction')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    # go!
    with torch.no_grad():
        style_inversion_reconstruction(pipe, style_path, inversion_path, reconstruction_path,
                                    num_frames, height, width, time_steps, weight_dtype,
                                    is_rf_solver=is_rf_solver)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3.5-medium")
    parser.add_argument("--style_path", type=str, default="examples/style/style1.png")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=50)
    #
    parser.add_argument('--is_rf_solver', action='store_true', help='use rf-solver')
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
