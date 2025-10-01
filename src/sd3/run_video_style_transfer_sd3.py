import argparse
from typing import Optional
import argparse
import os

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
from backbones.video_diffusion_sd3.pnp_utils import register_spatial_attention_pnp, latent_adain
from backbones.video_diffusion_sd3.pipelines.custom_pipeline import CustomStableDiffusion3Pipeline
from backbones.video_diffusion_sd3.pnp_utils import CrossFrameProcessor

from util import load_ddim_latents_at_t
from util import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    # go
    content_inv_noises = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=content_inv_path
        ).to(weight_dtype).cuda()
    content_inv_latents = load_ddim_latents_at_t(
            0, ddim_latents_path=content_inv_path
        ).to(weight_dtype).cuda()
    style_inv_noises = load_ddim_latents_at_t(
            time_steps, ddim_latents_path=style_inv_path
        ).to(weight_dtype).cuda()
    # -----------------------------------------------------------------------------------------------
    # Init latent-shift, [f, c, h, w]
    content_inv_noises = latent_adain(content_inv_noises, style_inv_noises)
    # -----------------------------------------------------------------------------------------------
    # Init Pnp, modify attention forward to support AdaIN-Guided Attention-shift
    register_spatial_attention_pnp(pipe)
    # video_style_transfer
    samples = pipe.video_style_transfer("", latents=content_inv_noises, img_latents=content_inv_latents,
                                num_inference_steps=time_steps,
                                content_inv_path=content_inv_path, style_inv_path=style_inv_path, mask_path=mask_path,
                                eta_base=0.85, eta_trend='constant', start_step=25, end_step=39,).images
    # save
    output_path = os.path.join(output_path, 'sd3', f'{content_inv_path.split('/')[-2]}_{style_inv_path.split('/')[-2]}')
    os.makedirs(output_path, exist_ok=True)
    for idx, sample in enumerate(samples):
        sample.save(os.path.join(output_path, f"%05d.png" % (idx * 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    # parser.add_argument("--pretrained_model_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--content_inv_path", type=str, default="results/contents-inv/sd3/mallard-fly/inversion/content-inv/sd3-rf-solver/davis2016/blackswan/inversion")
    parser.add_argument("--style_inv_path", type=str, default="results/styles-inv/sd3/00033/inversion")
    parser.add_argument("--mask_path", type=str, default="results/masks/sd3/mallard-fly")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    #
    parser.add_argument("--time_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
