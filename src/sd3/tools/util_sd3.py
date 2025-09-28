import os

from einops import rearrange
from PIL import Image
from torchvision import transforms

from sd.tools.inversion_sd import rf_inversion, rf_solver
from util import save_videos_grid, load_video_frames

import decord
decord.bridge.set_bridge('torch')
from diffusers.utils import export_to_video
from sd.tools.inversion_sd import reconstruction


def content_inversion_reconstruction(pipe, content_path, output_dir,
                                    num_frames, height, width, time_steps, weight_dtype,
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
    # breakpoint()
    img_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    img_latents = (img_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # breakpoint()
    # make dir
    name = content_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, name, 'inversion')
    reconstruction_path = os.path.join(output_dir, name, 'reconstruction')
    ft_path = os.path.join(output_dir, name, 'features')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    # ----------------------------------content inversion--------------------------------
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
    # ---------------------------------content construction------------------------------
    print(f"reconstruction:")
    images = reconstruction(
        pipe, 
        img_latents=img_latents,
        inversed_latents=inv_latent,
        eta_base=0.95,
        eta_trend='constant',
        start_step=10,
        end_step=20,
        guidance_scale=1.0,
        prompt="",
        DTYPE=weight_dtype,
        num_inference_steps=50,
    )
    export_to_video(images, os.path.join(reconstruction_path, 'content_video.mp4'), fps=8,)


def style_inversion_reconstruction(pipe, style_path, output_dir,
                                num_frames, height, width, time_steps, weight_dtype,
                                is_rf_solver=False):
    style_image = Image.open(style_path).convert("RGB").resize((width, height))
    style_tensor = transforms.ToTensor()(style_image)
    style_tensor = 2.0 * style_tensor - 1.0
    pixel_values = style_tensor.repeat(num_frames, 1, 1, 1).to(weight_dtype).cuda()
    # vae
    img_latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    img_latents = (img_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    # make dir
    name = style_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, name, 'inversion')
    reconstruction_path = os.path.join(output_dir, name, 'reconstruction')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
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
    images = reconstruction(
        pipe, 
        img_latents=img_latents,
        inversed_latents=inv_latent,
        eta_base=0.85,
        eta_trend='constant',
        start_step=25,
        end_step=39,
        guidance_scale=1.0,
        prompt="",
        DTYPE=weight_dtype,
        num_inference_steps=50,
    )
    export_to_video(images, os.path.join(reconstruction_path, 'style_video.mp4'), fps=8,)
