import os

from einops import rearrange
from PIL import Image

from torchvision import transforms
import torch

from diffusers.utils import export_to_video
from src.util import save_videos_grid, load_video_frames

import decord
decord.bridge.set_bridge('torch')


def content_inversion_reconstruction(pipe, content_path, inversion_path, reconstruction_path,
                                    num_frames, height, width, time_steps, weight_dtype,
                                    ft_indices, ft_timesteps, ft_path,
                                    is_rf_solver=False):
    if content_path.endswith(".mp4"):
        vr = decord.VideoReader(content_path, width=width, height=height)
        sample_index = list(range(0, len(vr), 1))[: num_frames]
        video = vr.get_batch(sample_index)
        pixel_values = (video / 127.5 - 1.0).unsqueeze(0)
        pixel_values = rearrange(pixel_values, 'b f h w c -> (b f) c h w').to(weight_dtype).cuda()
    else:
        pixel_values = load_video_frames(content_path, num_frames, image_size=(width, height)).to(weight_dtype).cuda()
    # vae
    img_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    img_latents = (img_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # ----------------------------------content inversion--------------------------------
    print(f"inversion:")
    if is_rf_solver: # rf-solver
        inv_latent = rf_solver(
            pipe, 
            img_latents,
            prompt = "",
            num_inference_steps=time_steps,
            inversion_path=inversion_path,
            #
            ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path,
        )
    else: # rf-inversion
        inv_latent = rf_inversion(
            pipe,
            img_latents,
            prompt = "",
            DTYPE=weight_dtype,
            gamma=0.0,
            num_inference_steps=time_steps,
            inversion_path=inversion_path,
            #
            ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path,
        )
    # ---------------------------------content construction------------------------------
    print(f"reconstruction:")
    images = pipe.reconstruction(
        prompt="",
        img_latents=img_latents,
        inversed_latents=inv_latent,
        eta_base=0.85,
        eta_trend='constant',
        start_step=25,
        end_step=39,
        guidance_scale=1.0,
        DTYPE=weight_dtype,
        num_inference_steps=time_steps,
    )
    export_to_video(images, os.path.join(reconstruction_path, 'content_video.mp4'), fps=8,)


def style_inversion_reconstruction(pipe, style_path, inversion_path, reconstruction_path,
                                num_frames, height, width, time_steps, weight_dtype,
                                is_rf_solver=False):
    style_image = Image.open(style_path).convert("RGB").resize((width, height))
    style_tensor = transforms.ToTensor()(style_image)
    style_tensor = 2.0 * style_tensor - 1.0
    pixel_values = style_tensor.repeat(num_frames, 1, 1, 1).to(weight_dtype).cuda()
    # vae
    img_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    img_latents = (img_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # ----------------------------------style inversion--------------------------------
    print(f"inversion:")
    if is_rf_solver: # rf-solver
        inv_latent = rf_solver(
            pipe, 
            img_latents,
            prompt = "",
            num_inference_steps=time_steps,
            inversion_path=inversion_path,
        )
    else: # rf-inversion
        inv_latent = rf_inversion(
            pipe,
            img_latents,
            prompt = "",
            DTYPE=weight_dtype,
            gamma=0.0,
            num_inference_steps=time_steps,
            inversion_path=inversion_path,
        )
    # ---------------------------------style construction------------------------------
    print(f"reconstruction:")
    images = pipe.reconstruction(
        prompt="",
        img_latents=img_latents,
        inversed_latents=inv_latent,
        eta_base=0.85,
        eta_trend='constant',
        start_step=25,
        end_step=39,
        guidance_scale=1.0,
        DTYPE=weight_dtype,
        num_inference_steps=time_steps,
    )
    export_to_video(images, os.path.join(reconstruction_path, 'style_video.mp4'), fps=8,)


# *********************************************************************************************


@torch.no_grad()
def rf_inversion(
    pipeline, 
    image_latents,
    prompt = "",
    gamma=0.5,
    num_inference_steps=50,
    inversion_path=None,
    ft_indices=None, ft_timesteps=None, ft_path=None,
):
    # Getting null-text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt( # null text
        prompt=prompt, 
        prompt_2=prompt,
        prompt_3=prompt,
    )
    # set timestep
    pipeline.scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
    timesteps = pipeline.scheduler.sigmas
    timesteps = torch.flip(timesteps, dims=[0])
    # save inversion result
    if inversion_path is not None:
        torch.save(
            image_latents.detach().clone(),
            os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
        )
    # generate gaussain noise with seed
    target_noise = torch.randn_like(image_latents)

    # # Image inversion with interpolated velocity field.  t goes from 0.0 to 1.0
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((image_latents.shape[0],), t_curr * 1000, dtype=image_latents.dtype, device=image_latents.device)

            # Null-text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=image_latents,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                idx=idx,
                ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path,
                return_dict=False,
            )[0]

            # Target noise velocity
            target_noise_velocity = (target_noise - image_latents) / (1.0 - t_curr)
            # interpolated velocity
            interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * pred_velocity   
            # one step Euler, similar to pipeline.scheduler.step but in the forward to noise instead of denosing
            image_latents = image_latents + (t_prev - t_curr) * interpolated_velocity

            # save
            if inversion_path is not None:
                torch.save(
                    image_latents.detach().clone(),
                    os.path.join(inversion_path, f"ddim_latents_{idx + 1}.pt"),
                )

            progress_bar.update()
            
    return image_latents


def rf_solver(
    pipeline, 
    image_latents,
    prompt = "",
    num_inference_steps=50,
    inversion_path=None,
    ft_indices=None, ft_timesteps=None, ft_path=None,
):
    # Getting null-text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
    )
    
    # set timestep
    pipeline.scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
    timesteps = pipeline.scheduler.sigmas
    timesteps = torch.flip(timesteps, dims=[0])

    # save inversion result
    if inversion_path is not None:
        torch.save(
            image_latents.detach().clone(),
            os.path.join(inversion_path, f"ddim_latents_{0}.pt"),
        )

    # 7. Denoising loop
    with pipeline.progress_bar(total=len(timesteps)-1) as progress_bar:
        for idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            t_vec = torch.full((image_latents.shape[0],), 1000 * t_curr, dtype=image_latents.dtype, device=image_latents.device)

            pred = pipeline.transformer(
                hidden_states=image_latents,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                idx=idx,
                ft_indices=ft_indices, ft_timesteps=ft_timesteps, ft_path=ft_path,
                return_dict=False,
            )[0]
            # get the conditional vector field
            img_mid = image_latents + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((image_latents.shape[0],), 1000 * (t_curr + (t_prev - t_curr) / 2), dtype=image_latents.dtype, device=image_latents.device)
            pred_mid = pipeline.transformer(
                hidden_states=img_mid,
                timestep=t_vec_mid,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                idx=idx,
                return_dict=False,
            )[0]
            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)

            # compute the previous noisy sample x_t -> x_t-1
            image_latents = image_latents + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

            # save
            if inversion_path is not None:
                torch.save(
                    image_latents.detach().clone(),
                    os.path.join(inversion_path, f"ddim_latents_{idx + 1}.pt"),
                )

            progress_bar.update()

    return image_latents
