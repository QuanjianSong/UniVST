# code mostly taken from https://github.com/huggingface/diffusers
import inspect
from typing import Callable, List, Optional, Union
import os, sys
import numpy as np
from PIL import Image
import math
import PIL
from torchvision import transforms
import random
from diffusers.utils.torch_utils import is_compiled_module

import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
# from diffusers.pipeline_utils import DiffusionPipeline
try:
    from diffusers.pipeline_utils import DiffusionPipeline
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from video_diffusion.pnp_utils import register_time
from ..models.unet_3d_condition import UNetPseudo3DConditionModel

from video_diffusion.cal_optica_flow import get_warp

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SpatioTemporalStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNetPseudo3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_before_train_loop(self, params_to_optimize=None):
        # Set xformers in train.py
        
        # self.disable_xformers_memory_efficient_attention()

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        if params_to_optimize is not None:
            params_to_optimize.requires_grad = True
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _encode_image(
        self,
        image,
        device,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        noise_level,
        generator,
        image_embeds,
        return_image_embeds=False
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_videos_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_videos_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(image).image_embeds

        if return_image_embeds:
            return image_embeds

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds
    
    def decode_latents(self, latents, num_frames=16, decode_chunk_size=16):
        latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        latents = latents.flatten(0, 1)
        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = (frames / 2 + 0.5).clamp(0, 1)

        frames = rearrange(frames, "(b f) c h w -> b f h w c", f=16)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.cpu().float().numpy()

        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        clip_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            clip_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(
                    device
                )
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        video_length: int = 8,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        up_ft_indices=None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)
        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        # [1, 4, 8, 64, 64]
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                # [2, 4, 8, 64, 64]
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings, up_ft_indices=up_ft_indices,
                ).sample.to(dtype=latents_dtype)
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1 [1, 4, 8, 64, 64]
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 8. Post-processing
        image = self.decode_latents(latents)
        # 9. Run safety checker
        has_nsfw_concept = None
        # 10. Convert to PIL or tensor
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.from_numpy(image)
        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    @torch.no_grad()
    def video_style_transfer(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        inv_path=None,
        style_path=None,
        mask_path=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)
        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # Encode input prompt
        if do_classifier_free_guidance:
            negative_prompt_embeds, prompt_embeds = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        else:
            prompt_embeds = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        # 3.1 Encode ddim inversion prompt
        if do_classifier_free_guidance:
            _, ddim_inv_prompt_embeds = self._encode_prompt(
                "",
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt=None,
            )
        else:
            ddim_inv_prompt_embeds = self._encode_prompt(
                "",
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt=None,
            )
        if do_classifier_free_guidance:
            # prompt_embeds_all = torch.cat([negative_prompt_embeds.unsqueeze(0), prompt_embeds.unsqueeze(0)])
            prompt_embeds_all = torch.cat([ddim_inv_prompt_embeds.unsqueeze(0), ddim_inv_prompt_embeds.unsqueeze(0), negative_prompt_embeds.unsqueeze(0), prompt_embeds.unsqueeze(0)])
        else:
            prompt_embeds_all = torch.cat([ddim_inv_prompt_embeds, ddim_inv_prompt_embeds, prompt_embeds])
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeds_all.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # ---------------------------------add code------------------------------------
                ddim_inv_latents_at_t = load_ddim_latents_at_t(t, inv_path).to(latents.dtype).to(self.device)
                style_inv_latents_at_t = load_ddim_latents_at_t(t, style_path).to(latents.dtype).to(self.device)
                #
                if t >= 100:
                    mask = load_mask(mask_path)
                    # resized_mask shape: [1, 4, 16, 64, 64]
                    resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                    mode='bilinear', align_corners=False)
                    resized_mask = resized_mask[None, :]
                    ori_latents = load_ddim_latents_at_t(
                    t, ddim_latents_path=inv_path
                    ).to(latents.dtype).to(self.device)
                    latents = (1 - resized_mask) * latents + resized_mask * ori_latents 
                # ------------------------------------------------------------------------------------------------------------------
                if t >= 100 and t <= 200:
                    mask = load_mask(mask_path)
                    # resized_mask shape: [1, 4, 16, 64, 64]
                    resized_mask = F.interpolate(mask.to(latents.device).to(latents.dtype), size=(latents.shape[-2], latents.shape[-1]), 
                                                    mode='bilinear', align_corners=False)
                    resized_mask = resized_mask[None, :]
                    latents = (1 - resized_mask) * self.adain(latents, style_inv_latents_at_t) + resized_mask * ori_latents 

                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([ddim_inv_latents_at_t, style_inv_latents_at_t, latents, latents])
                else:
                    latent_model_input = torch.cat([ddim_inv_latents_at_t, style_inv_latents_at_t, latents])
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # register time
                register_time(self, t.item())
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_all).sample.to(dtype=latents_dtype)
                # perform guidance
                if do_classifier_free_guidance:
                    _noise_pred_ddim_inv, _noise_pred_style_inv, noise_pred_negative, noise_pred_editing  = noise_pred.chunk(4)
                    noise_pred = noise_pred_negative + guidance_scale * (noise_pred_editing - noise_pred_negative)
                else:
                    _noise_pred_ddim_inv, _noise_pred_style_inv, noise_pred_editing = noise_pred.chunk(3)
                    noise_pred = noise_pred_editing
                # -------------------------------Sliding window smoothing--------------------------
                smoother = 'pixel'
                if i >= 20 and i < 25 and smoother is not None:
                    # cal Z_0
                    pred_original_sample = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                    if smoother == 'pixel':
                        # decode it to pixel
                        estimated_frames = self.get_images_from_latents(pred_original_sample)
                        # copy data
                        ori_estimated_frames = estimated_frames.copy()
                        # ----------------------------------------------------------------------------------------------------
                        r = 2
                        cnt_nums = 1
                        for iters in range(cnt_nums):
                            # store temp result
                            estimated_frames_tmp = np.zeros_like(estimated_frames).astype(np.float32)
                            # for each key frame, sliding window
                            for key_index in range(0, 16):
                                key_frame = estimated_frames[:, :, key_index, :, :][0].transpose(1, 2, 0).copy()
                                weight = 0
                                # consider the window of key frame
                                for bias in range(-r, r+1):
                                    now_index = key_index + bias
                                    if now_index >= 0 and now_index < 16:
                                        # choose from update estimated_frames
                                        now_frame = estimated_frames[:,:, now_index, :, :][0].transpose(1, 2, 0).copy()
                                        if bias == 0:
                                            estimated_frames_tmp[:, :, key_index, :, :] = estimated_frames_tmp[:,:, key_index, :, :] + now_frame.transpose(2, 0, 1).astype(np.float32)
                                        else:
                                            warp_result = get_warp(key_frame, now_frame, key_frame, now_frame).transpose(2, 0, 1)
                                            estimated_frames_tmp[:,:, key_index, :, :] = estimated_frames_tmp[:,:, key_index, :, :] + warp_result.astype(np.float32)     
                                        weight += 1
                                # estimated_frames_tmp[:,:, key_index, :, :] = estimated_frames_tmp[:,:, key_index, :, :] / weight
                                estimated_frames[:,:, key_index, :, :] = estimated_frames_tmp[:,:, key_index, :, :] / weight      
                        # sliding window end and get final result
                        estimated_frames = estimated_frames.astype(np.uint8)
                        # apply mask
                        estimated_frames = ori_estimated_frames * mask[None, :].numpy() + (1 - mask[None, :].numpy()) * estimated_frames
                        # encoder it to latent, [-1, 1]
                        pred_original_sample = self.get_latent_image(estimated_frames)
                    else:
                        print('error')
                        return
                    # adjust noise_pred
                    noise_pred = self.return_to_timestep(t, latents, pred_original_sample, self.scheduler)
                # ----------------------------------------------------------------------------------
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 8. Post-processing
        image = self.decode_latents(latents)
        # 9. Run safety checker
        has_nsfw_concept = None
        # 10. Convert to PIL or tensor
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.from_numpy(image)
        if not return_dict:
            return (image, has_nsfw_concept)
        torch.cuda.empty_cache()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def return_to_timestep(
        self,
        timestep: int,
        sample: torch.FloatTensor,
        sample_stablized: torch.FloatTensor,
        schedules,
    ):
        alpha_prod_t = schedules.alphas_cumprod[timestep]
        noise_pred = (sample - alpha_prod_t ** (0.5) * sample_stablized) / ((1 - alpha_prod_t) ** (0.5))
        return noise_pred

    def get_images_from_latents(
        self,
        latents,
        decode_chunk_size=16,
    ):
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        frames = (frames * 255).round().astype("uint8")
        frames = rearrange(frames, "(b f) c h w -> b c f h w", f=16)

        return frames

    def get_latent_image(
        self,
        image: Image.Image,
    ):
        image = rearrange(image, "b c f h w -> (b f) c h w")
        # transforms to [-1, 1]
        image = (image / 127.5) - 1.0
        image = torch.from_numpy(image).to(device=self.device, dtype=self.vae.dtype) 
        # breakpoint()
        latents = self.vae.encode(image).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=16)
        latents = 0.18215 * latents
        
        return latents

    def adain(self, cnt_feat, sty_feat, ad=True):
        beta = 1.0
        # breakpoint()
        cnt_mean = cnt_feat.mean(dim=[0, 3, 4], keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 3, 4], keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 3, 4], keepdim=True)
        sty_std = sty_feat.std(dim=[0, 3, 4], keepdim=True)
        output_mean = beta * sty_mean + (1 - beta) * cnt_mean
        output_std = beta * sty_std + (1 - beta) * cnt_std
        output = cnt_feat
        # -------------------------------------------------------------------------------------------
        if ad:
            output = F.instance_norm(output) * output_std + output_mean
        return output.to(sty_feat.dtype)

    @staticmethod
    def numpy_to_pil(images):
        # (1, 16, 512, 512, 3)
        pil_images = []
        is_video = (len(images.shape)==5)
        if is_video:
            for sequence in images:
                pil_images.append(DiffusionPipeline.numpy_to_pil(sequence))
        else:
            pil_images.append(DiffusionPipeline.numpy_to_pil(images))
        return pil_images

    def print_pipeline(self, logger):
        print('Overview function of pipeline: ')
        print(self.__class__)

        print(self)
        
        expected_modules, optional_parameters = self._get_signature_keys(self)        
        components_details = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }
        import json
        logger.info(str(components_details))
        
        print(f"python version {sys.version}")
        print(f"torch version {torch.__version__}")
        print(f"validate gpu status:")
        print( torch.tensor(1.0).cuda()*2)
        os.system("nvcc --version")

        import diffusers
        print(diffusers.__version__)
        print(diffusers.__file__)

        try:
            import bitsandbytes
            print(bitsandbytes.__file__)
        except:
            print("fail to import bitsandbytes")

def load_ddim_latents_at_t(t, ddim_latents_path, is_x0=False):
    if is_x0:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_x0_{t}.pt")
    else:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path)
    logger.debug(f"Loaded ddim_latents_at_t from {ddim_latents_at_t_path}")
    return ddim_latents_at_t

def load_mask(mask_path='', n_frames=16):
    # image_files = [os.path.join(mask_path, file) for file in os.listdir(mask_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = [f"{mask_path}/%05d.png" % (i * 1) for i in range(n_frames)]
    image_files = sorted(image_files)
    # breakpoint()
    images = [np.array(Image.open(image)) * 255 for image in image_files]
    # images = [np.array(Image.open(image)) for image in image_files]
    # breakpoint()
    image_tensor = np.stack(images)
    image_tensor_torch = torch.from_numpy(image_tensor).unsqueeze(0)
    image_tensor_torch = image_tensor_torch.clip(0, 1)
    return image_tensor_torch

def load_image(
    image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.
    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    if convert_method is not None:
        image = convert_method(image)
    else:
        image = image.convert("RGB")
    return image

def load_video_frames(frames_path, n_frames, image_size=(512, 512)):
    # Load paths
    to_tensor = transforms.ToTensor()
    paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
    frames = []
    for p in paths:
        img = load_image(p)
        # check!
        if img.size != image_size:
            logger.error(f"Frame size {f.size} does not match config.image_size {image_size}")
            raise ValueError(f"Frame size {f.size} does not match config.image_size {image_size}")
        # transforms tensor
        np_img = np.array(img)
        normalized_img = (np_img / 127.5) - 1.0
        tensor_img = torch.from_numpy(normalized_img).permute(2, 0, 1).float()
        frames.append(tensor_img)
    video_tensor = torch.stack(frames)
    return video_tensor
