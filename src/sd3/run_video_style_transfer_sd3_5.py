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
from diffusers.utils import export_to_video
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,              # 有的权重用 CLIPTextModelWithProjection；见下方注释
    CLIPTokenizer,              # 对 CLIP
)
from base_models.sd3.pnp_utils import register_spatial_attention_pnp, latent_adain
from base_models.sd3.custom_pipeline import CustomStableDiffusion3Pipeline

from util import save_folder, save_videos_grid, load_ddim_latents_at_t
from util import seed_everything
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行化


def init_pnp(pipe):
    register_spatial_attention_pnp(pipe)


def main(
    pretrained_model_path: str,
    inv_path: str,
    style_path: str,
    mask_path: str,
    output_path: str,
    weight_dtype: torch.dtype = torch.float16,
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
    transformer = SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer").requires_grad_(False)
    # add cross-frame attention to support video
    from base_models.sd3.pnp_utils import CrossFrameProcessor
    new_procs = {}
    # breakpoint()
    # Set cross-frame attention to support video
    for idx, (name, processor) in enumerate(transformer.attn_processors.items()):
        if idx >= 0:
        # if True:
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
    # breakpoint()



    # go
    # inv_latents_at_t = load_ddim_latents_at_t(
    #         50, ddim_latents_path=inv_path
    #     ).to(weight_dtype).cuda()
    # content_latents = load_ddim_latents_at_t(
    #         0, ddim_latents_path=inv_path
    #     ).to(weight_dtype).cuda()
    # style_latents_at_t = load_ddim_latents_at_t(
    #         50, ddim_latents_path=style_path
    #     ).to(weight_dtype).cuda()
    # output_path = os.path.join(output_path, 'edit')
    # os.makedirs(output_path, exist_ok=True)
    # # -----------------------------------------------------------------------------------------------
    # # breakpoint()
    # # Init latent-shift, [f, c, h, w]
    # inv_latents_at_t = latent_adain(inv_latents_at_t, style_latents_at_t)
    # # -----------------------------------------------------------------------------------------------
    # # Init Pnp, modify attention forward to support AdaIN-Guided Attention-shift
    # init_pnp(pipe)
    # # video_style_transfer
    # samples = pipe.video_style_transfer("", latents=inv_latents_at_t, img_latents=content_latents,
    #                             num_inference_steps=50, guidance_scale=1.0,
    #                             inv_path=inv_path, style_path=style_path, mask_path=mask_path,
    #                             eta_base=0.85, eta_trend='constant', start_step=25, end_step=39,).images
    # # breakpoint()
    # # save
    # export_to_video(samples, os.path.join(output_path, 'edit.mp4'), fps=8,)
    # for idx, sample in enumerate(samples):
    #             sample.save(os.path.join(output_path, f"%05d.png" % (idx * 1)))


    # for loop
    final_output = os.path.join(output_path, inv_path.split('/')[-1] + '_' + style_path.split('/')[-1])
    for i_index, sub_style_path in enumerate(sorted(os.listdir(style_path))):
        for j_index, sub_inv_path in enumerate(sorted(os.listdir(inv_path))):
            # breakpoint()
            final_inv_path = os.path.join(inv_path, sub_inv_path, 'inversion')
            final_style_path = os.path.join(style_path, sub_style_path, 'inversion')
            final_mask = os.path.join(mask_path, sub_inv_path)
            save_out = os.path.join(final_output, sub_inv_path + '_' + sub_style_path)
            os.makedirs(save_out, exist_ok=True)

            # load data
            inv_latents_at_t = load_ddim_latents_at_t(
                    50, ddim_latents_path=final_inv_path
                ).to(weight_dtype).cuda()
            content_latents = load_ddim_latents_at_t(
                    0, ddim_latents_path=final_inv_path
                ).to(weight_dtype).cuda()
            style_latents_at_t = load_ddim_latents_at_t(
                    50, ddim_latents_path=final_style_path
                ).to(weight_dtype).cuda()
            # -----------------------------------------------------------------------------------------------
            # Init latent-shift
            inv_latents_at_t = latent_adain(inv_latents_at_t, style_latents_at_t)
            # -----------------------------------------------------------------------------------------------
            # Init Pnp, modify attention forward
            init_pnp(pipe)
            # video_style_transfer
            samples = pipe.video_style_transfer("", latents=inv_latents_at_t, img_latents=content_latents,
                                num_inference_steps=50, guidance_scale=1.0,
                                inv_path=final_inv_path, style_path=final_style_path, mask_path=final_mask,
                                eta_base=0.85, eta_trend='constant', start_step=25, end_step=39,).images
            # 0.8 for sd3.5， 0.75 for sd3.0
            # save
            for idx, sample in enumerate(samples):
                sample.save(os.path.join(save_out, f"%05d.png" % (idx * 1)))
            # save_videos_grid(sample, os.path.join(output_dir, "style.mp4"), fps=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--pretrained_model_path", type=str, default="/data/lxy/sqj/base_models/stable-diffusion-3.5-medium")
    parser.add_argument("--inv_path", type=str, default="outputs/content-inv/sd3-rf-solver/davis2016/blackswan/inversion")
    parser.add_argument("--style_path", type=str, default="outputs/style-inv/sd3-rf-solver/laion/00041/inversion")
    parser.add_argument("--mask_path", type=str, default="/data/lxy/sqj/datasets/my_davis2016/mask/blackswan")
    parser.add_argument("--output_path", type=str, default="output/")
    args = parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
