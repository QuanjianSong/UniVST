from diffusers import (
    StableDiffusion3Pipeline
)
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import inspect
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from base_models.sd3.pnp_utils import latent_adain
from util import load_mask, load_ddim_latents_at_t
import torch.nn.functional as F


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
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
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
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        # additional
        eta_base=0.95,
        eta_trend='constant',
        start_step=10,
        end_step=20,
        img_latents=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*):
                A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                with a scale of `1`.
            skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
            skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

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

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
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
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

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

        # breakpoint()

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)


        target_img = img_latents.clone()
        # get the eta values for each steps in 
        eta_values = self.generate_eta_values(timesteps, start_step, end_step, eta_base, eta_trend)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # breakpoint()
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # ---------------------------- modify ----------------------------
                t_curr = t / self.scheduler.config.num_train_timesteps
                target_pred = -(target_img - latents) / t_curr
                # interpolated velocity
                eta = eta_values[i]
                noise_pred = noise_pred + eta * (target_pred - noise_pred)
                # ----------------------------------------------------------------

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

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

    @torch.no_grad()
    def video_style_transfer(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
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
        inv_path=None,
        style_path=None,
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



        # breakpoint()
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
                # register time
                # register_time(self, i)
                # load inversion latent, [BF, c, h, w]
                ddim_inv_latents_at_t = load_ddim_latents_at_t(50 - i, inv_path).to(latents.dtype).to(self.device)
                style_inv_latents_at_t = load_ddim_latents_at_t(50 - i, style_path).to(latents.dtype).to(self.device)
                # localized
                if i <= 0.9 * num_inference_steps:
                    mask = load_mask(mask_path)
                    resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                    mode='bilinear', align_corners=False)
                    resized_mask = resized_mask.permute(1, 0, 2, 3).contiguous()
                    ori_latents = load_ddim_latents_at_t(
                    50 - i, ddim_latents_path=inv_path
                    ).to(latents.dtype).to(self.device)
                    latents = (1 - resized_mask) * latents + resized_mask * ori_latents
                #
                if i >= 0.8 * num_inference_steps and i <= 0.9 * num_inference_steps:
                    mask = load_mask(mask_path)
                    # resized_mask shape: [1, 4, 16, 64, 64]
                    resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                    mode='bilinear', align_corners=False)
                    resized_mask = resized_mask.permute(1, 0, 2, 3).contiguous()
                    latents = (1 - resized_mask) * latent_adain(latents, style_inv_latents_at_t) + resized_mask * ori_latents 
                # ------------------------------------------------------------------------------------------------------------------
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([ddim_inv_latents_at_t, style_inv_latents_at_t, latents])
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # breakpoint()
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_all,
                    pooled_projections=pooled_prompt_embeds_all,
                    return_dict=False,
                    joint_attention_kwargs={'idx': i}
                )[0]

                # perform guidance
                _noise_pred_ddim_inv, _noise_pred_style_inv, noise_pred_editing = noise_pred.chunk(3)


                # ---------------------------- modify ----------------------------
                t_curr = t / self.scheduler.config.num_train_timesteps
                # breakpoint()
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