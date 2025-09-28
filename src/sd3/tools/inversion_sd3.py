import os

from typing import Union, Optional, List
import torch
import inspect
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def generate_eta_values(
    timesteps, 
    start_step, 
    end_step, 
    eta, 
    eta_trend,
):
    assert start_step < end_step and start_step >= 0 and end_step <= len(timesteps), "Invalid start_step and end_step"
    # timesteps are monotonically decreasing, from 1.0 to 0.0
    eta_values = [0.0] * len(timesteps)
    
    if eta_trend == 'constant':
        for i in range(start_step, end_step):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[start_step] - timesteps[i]) / total_time
    elif eta_trend == 'linear_decrease':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[i] - timesteps[end_step - 1]) / total_time
    else:
        raise NotImplementedError(f"Unsupported eta_trend: {eta_trend}")
    
    return eta_values


@torch.no_grad()
def reconstruction(
    pipeline, 
    img_latents,
    inversed_latents,            # can be none if not using inversed latents
    eta_base,                    # base eta value
    eta_trend,                   # constant, linear_increase, linear_decrease
    start_step,                  # 0-based indexing, closed interval
    end_step,                    # 0-based indexing, open interval
    guidance_scale=1.0,
    prompt='',
    DTYPE=torch.bfloat16,
    num_inference_steps=50,
):

    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, pipeline.device)

    # Getting text embedning
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipeline.encode_prompt(
        prompt=prompt, 
        prompt_2=prompt,
        prompt_3=prompt
    )

    latents = inversed_latents
    target_img = img_latents.clone().to(torch.float32)

    # get the eta values for each steps in 
    eta_values = generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

    # handle guidance scale if need
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # Editing text velocity
            pred_velocity = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance scale
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = pred_velocity.chunk(2)
                pred_velocity = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Prevents precision issues
            latents = latents.to(torch.float32)
            pred_velocity = pred_velocity.to(torch.float32)

            # Target image velocity
            t_curr = t / pipeline.scheduler.config.num_train_timesteps
            target_velocity = -(target_img - latents) / t_curr

            # interpolated velocity
            eta = eta_values[i]
            interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

            # denosing
            latents = pipeline.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]
            
            latents = latents.to(DTYPE)
            progress_bar.update()
    
    imgs = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")

    return imgs




@torch.no_grad()
def rf_inversion(
    pipeline, 
    image_latents,
    prompt = "",
    DTYPE=torch.float16,
    gamma=0.5,
    num_inference_steps=50,
    inversion_path=None,
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
            # breakpoint()
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            t_vec = torch.full((image_latents.shape[0],), 1000 * t_curr, dtype=image_latents.dtype, device=image_latents.device)

            pred = pipeline.transformer(
                hidden_states=image_latents,
                timestep=t_vec,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            # breakpoint()
            # get the conditional vector field
            img_mid = image_latents + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((image_latents.shape[0],), 1000 * (t_curr + (t_prev - t_curr) / 2), dtype=image_latents.dtype, device=image_latents.device)
            pred_mid = pipeline.transformer(
                hidden_states=img_mid,
                timestep=t_vec_mid,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
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