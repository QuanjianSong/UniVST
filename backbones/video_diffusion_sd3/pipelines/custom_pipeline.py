from typing import Any, Callable, Dict, List, Optional, Union
import inspect

import torch
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from backbones.video_diffusion_sd3.pnp_utils import latent_adain

from src.util import load_mask, load_ddim_latents_at_t


class CustomStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def generate_eta_values(
        self,
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
        self,
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

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device)

        # Getting text embedning
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.encode_prompt(
            prompt=prompt, 
            prompt_2=prompt,
            prompt_3=prompt
        )

        latents = inversed_latents
        target_img = img_latents.clone().to(torch.float32)
        # get the eta values for each steps in 
        eta_values = self.generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)
        # handle guidance scale if need
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                # Editing text velocity
                pred_velocity = self.transformer(
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
                t_curr = t / self.scheduler.config.num_train_timesteps
                target_velocity = -(target_img - latents) / t_curr
                # interpolated velocity
                eta = eta_values[i]
                interpolate_velocity = pred_velocity + eta * (target_velocity - pred_velocity)

                # denosing
                latents = self.scheduler.step(interpolate_velocity, t, latents, return_dict=False)[0]
                latents = latents.to(DTYPE)
                progress_bar.update()
        
        imgs = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        imgs = self.vae.decode(imgs)[0]
        imgs = self.image_processor.postprocess(imgs, output_type="pil")

        return imgs

    @torch.no_grad()
    def video_style_transfer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
        #
        content_inv_path=None,
        style_inv_path=None,
        mask_path=None,
        # additional
        eta_base=0.95,
        eta_trend='constant',
        start_step=10,
        end_step=20,
        img_latents=None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device



        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=False,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        frame_nums = latents.shape[0]
        prompt_embeds_all = torch.cat([prompt_embeds.repeat(frame_nums, 1, 1), prompt_embeds.repeat(frame_nums, 1, 1), prompt_embeds.repeat(frame_nums, 1, 1)])
        pooled_prompt_embeds_all = torch.cat([pooled_prompt_embeds.repeat(frame_nums, 1), pooled_prompt_embeds.repeat(frame_nums, 1), pooled_prompt_embeds.repeat(frame_nums, 1)])


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu



        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        

        target_img = img_latents.clone()
        # get the eta values for each steps in 
        eta_values = self.generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)


        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # ---------------------------------add code------------------------------------
                content_inv_latents_at_t = load_ddim_latents_at_t(50 - i, content_inv_path).to(latents.dtype).to(self.device)
                style_inv_latents_at_t = load_ddim_latents_at_t(50 - i, style_inv_path).to(latents.dtype).to(self.device)
                # localized latent blending
                if mask_path and i <= 0.9 * num_inference_steps:
                    mask = load_mask(mask_path)
                    resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                    mode='bilinear', align_corners=False)
                    resized_mask = resized_mask.permute(1, 0, 2, 3).contiguous()
                    latents = (1 - resized_mask) * latents + resized_mask * content_inv_latents_at_t
                #
                if i >= 0.8 * num_inference_steps and i <= 0.9 * num_inference_steps:
                    if mask_path:
                        mask = load_mask(mask_path)
                        resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                        mode='bilinear', align_corners=False)
                        resized_mask = resized_mask.permute(1, 0, 2, 3).contiguous()
                    else:
                        resized_mask = 0.0
                    latents = (1.0 - resized_mask) * latent_adain(latents, style_inv_latents_at_t) + resized_mask * ddim_inv_latents_at_t 
                # ------------------------------------------------------------------------------------------------------------------
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([content_inv_latents_at_t, style_inv_latents_at_t, latents])
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_all,
                    pooled_projections=pooled_prompt_embeds_all,
                    return_dict=False,
                    joint_attention_kwargs={'idx': i}
                )[0]

                # perform guidance
                _noise_pred_content_inv, _noise_pred_style_inv, noise_pred_editing = noise_pred.chunk(3)


                # ---------------------------- modify ----------------------------
                t_curr = t / self.scheduler.config.num_train_timesteps
                target_pred = -(target_img - latents) / t_curr
                # interpolated velocity
                eta = eta_values[i]
                noise_pred_editing = noise_pred_editing + eta * (target_pred - noise_pred_editing)
                # ----------------------------------------------------------------


                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred_editing, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


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