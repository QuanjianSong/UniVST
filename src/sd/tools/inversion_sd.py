import os
import numpy as np
from typing import Union
import torch
from tqdm import tqdm

@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent,
                   num_inv_steps, prompt="", inversion_path=None,
                   up_ft_indices=None, ft_path=None,
                   is_opt=False,):
    if is_opt:
        ddim_latents = ddim_loop_plus(pipeline, ddim_scheduler, video_latent,
                                    num_inv_steps, prompt, inversion_path,
                                    up_ft_indices=up_ft_indices, ft_path=ft_path)
    else:
        ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent,
                                num_inv_steps, prompt, inversion_path,
                                up_ft_indices=up_ft_indices, ft_path=ft_path)
    return ddim_latents


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent,
            num_inv_steps, prompt, inversion_path,
            up_ft_indices=None, ft_path=None):
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
                                            up_ft_indices=up_ft_indices, ft_path=ft_path)
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
                up_ft_indices=None, ft_path=None):
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
                                            up_ft_indices=up_ft_indices, ft_path=ft_path)
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
                        up_ft_indices=None, ft_path=None):
    noise_pred = pipeline.unet(latents, t, encoder_hidden_states=context, up_ft_indices=up_ft_indices, ft_path=ft_path)["sample"]
    return noise_pred
