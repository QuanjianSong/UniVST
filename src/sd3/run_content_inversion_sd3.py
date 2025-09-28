import argparse
import os
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models.transformers import SD3Transformer2DModel
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from util import seed_everything
from sd.tools.util_sd import content_inversion_reconstruction
from base_models.sd3.custom_pipeline import CustomStableDiffusion3Pipeline


import decord
decord.bridge.set_bridge('torch')
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行化


def main(
    pretrained_model_path: str,
    content_path: str,
    output_path: str,
    up_ft_indices: int = 2,
    num_frames: int = 16,
    height: int = 512,
    width: int = 512,
    time_steps: int = 50,
    weight_dtype: torch.dtype = torch.float16,
    seed: Optional[int] = 42,
    is_rf_solver: bool = False,
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
    transformer = SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer").requires_grad_(False)
    # Set cross-frame attention to support video
    from base_models.sd3.pnp_utils import CrossFrameProcessor
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

    # breakpoint()
    # custom pipe
    pipe = CustomStableDiffusion3Pipeline(
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
        text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3,
        vae=vae, transformer=transformer,
        scheduler=scheduler,
    )

    # go!
    # breakpoint()
    with torch.no_grad():
        content_inversion_reconstruction(content_path, output_path, up_ft_indices, num_frames,
                                            height, width, time_steps, weight_dtype, pipe,
                                            is_rf_solver=is_rf_solver)
        

    # with torch.no_grad():
    #     for content_sub_path in sorted(os.listdir(content_path)):
    #         if ".DS_Store" in content_sub_path:
    #             continue
    #         content_path_final = os.path.join(content_path, content_sub_path)
    #         content_inversion_reconstruction(content_path_final, output_path, up_ft_indices, num_frames,
    #                                         height, width, time_steps, weight_dtype, pipe,
    #                                         is_rf_solver=is_rf_solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3.5-medium")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.bfloat16)
    parser.add_argument("--up_ft_indices", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument('--is_rf_solver', action='store_true', help='use rf-solver')
    parser.add_argument("--content_path", type=str, default="examples/content/libby")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
