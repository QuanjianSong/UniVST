import os
from einops import rearrange
from PIL import Image
import numpy as np
from typing import Union
from tqdm import tqdm

import torch
from torchvision import transforms
from src.util import save_videos_grid, load_video_frames

import decord
decord.bridge.set_bridge('torch')


def content_inversion_reconstruction(pipe, ddim_inv_scheduler, content_path, inversion_path, reconstruction_path,
                                    num_frames, height, width, time_steps, weight_dtype,
                                    ft_indices=None, ft_timesteps=None, ft_path=None,
                                    is_opt=True):
    if content_path.endswith(".mp4"):
        vr = decord.VideoReader(content_path, width=width, height=height)
        sample_index = list(range(0, len(vr), 1))[: num_frames]
        video = vr.get_batch(sample_index)
        pixel_values = (video / 127.5 - 1.0).unsqueeze(0)
        pixel_values = rearrange(pixel_values, 'b f h w c -> (b f) c h w').to(weight_dtype).cuda()
    else:
        pixel_values = load_video_frames(content_path, num_frames, image_size=(width, height)).to(weight_dtype).cuda()
    # vae
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = latents * pipe.vae.config.scaling_factor
    # ----------------------------------content video inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", inversion_path=inversion_path,
                                    ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path,
                                    is_opt=is_opt,)[-1].to(weight_dtype)
    # ---------------------------------content video construction------------------------------
    print(f"reconstruction:")
    sample = pipe.reconstruction("", latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "content_video.mp4"))


def style_inversion_reconstruction(pipe, ddim_inv_scheduler, style_path, inversion_path, reconstruction_path,
                                num_frames, height, width, time_steps, weight_dtype,
                                is_opt=True):
    style_image = Image.open(style_path).convert("RGB").resize((width, height))
    style_tensor = transforms.ToTensor()(style_image)
    style_tensor = 2.0 * style_tensor - 1.0
    pixel_values = style_tensor.repeat(num_frames, 1, 1, 1).to(weight_dtype).cuda()
    # vae
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=num_frames)
    latents = latents * pipe.vae.config.scaling_factor
    # ----------------------------------content style inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", inversion_path=inversion_path,
                                    is_opt=is_opt,)[-1].to(weight_dtype)
    # ---------------------------------content style construction------------------------------
    print(f"reconstruction:")
    sample = pipe.reconstruction("", latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "style_video.mp4"), fps=8)


# *********************************************************************************************


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent,
                   num_inv_steps, prompt="", inversion_path=None,
                   ft_indices=None, ft_timesteps=None, ft_path=None,
                   is_opt=False,):
    if is_opt:
        ddim_latents = ddim_loop_plus(pipeline, ddim_scheduler, video_latent,
                                    num_inv_steps, prompt, inversion_path,
                                    ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path)
    else:
        ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent,
                                num_inv_steps, prompt, inversion_path,
                                ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path)
    return ddim_latents


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent,
            num_inv_steps, prompt, inversion_path,
            ft_indices=None, ft_timesteps=None, ft_path=None
    ):
    context = init_prompt(pipeline, prompt)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    if inversion_path is not None:
        torch.save(
                    latent.detach().clone(),
                    os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
                )
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(pipeline, latent, t, cond_embeddings, 
                                            ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        # save latent
        if inversion_path is not None:
            torch.save(
                latent.detach().clone(),
                os.path.join(inversion_path, f"ddim_latents_{i + 1}.pt"),
            )
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_loop_plus(pipeline, ddim_scheduler, latent,
                num_inv_steps, prompt, inversion_path,
                ft_indices=None, ft_timesteps=None, ft_path=None
    ):
    context = init_prompt(pipeline, prompt)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    if inversion_path is not None:
        torch.save(
                    latent.detach().clone(),
                    os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
                )
    or_latent_idx = 0.5
    inject_steps = 0.05
    inject_len = 0.2
    num_inference_steps=50
    num_fix_itr = 0
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(pipeline, latent, t, cond_embeddings,
                                            ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path)
        noise_pred = noise_pred.requires_grad_(True)

        last_noise = noise_pred
        if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
            print("add!")
            if i > 0:
                latent = or_latent_idx * latent + (1 - or_latent_idx) * last_latent
        for fix_itr in range(num_fix_itr):
            if fix_itr == 0:
                print("fix!")
            if fix_itr > 0:
                latents_tmp = next_step((noise_pred + last_noise) / 2, t, latent, ddim_scheduler)
            else:
                latents_tmp = next_step(noise_pred, t, latent, ddim_scheduler)
            last_noise = noise_pred
            noise_pred = get_noise_pred_single(latents_tmp, t, cond_embeddings, pipeline.unet)

        last_latent = latent
        latent = next_step(noise_pred, t, latent, ddim_scheduler)

        # save latent
        if inversion_path is not None:
            torch.save(
                        latent.detach().clone(),
                        os.path.join(inversion_path, f"ddim_latents_{i + 1}.pt"),
                    )
        all_latent.append(latent)
        continue
    return all_latent


@torch.no_grad()
def init_prompt(pipeline, prompt):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t

    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_epsilon = model_output

    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    return next_sample


def get_noise_pred_single(pipeline, latents, t, context,
                        ft_indices=None, ft_timesteps=None, ft_path=None):
    noise_pred = pipeline.unet(latents, t, encoder_hidden_states=context,
                            ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path
    )["sample"]
    return noise_pred

