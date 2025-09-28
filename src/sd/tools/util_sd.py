import os
from einops import rearrange
from PIL import Image
from torchvision import transforms

from src.sd.tools.inversion_sd import ddim_inversion
from src.util import save_videos_grid, load_video_frames

import decord
decord.bridge.set_bridge('torch')


def content_inversion_reconstruction(pipe, ddim_inv_scheduler, content_path, output_dir, 
                                    num_frames, height, width, time_steps, weight_dtype,
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
    # breakpoint()
    # latents = latents * pipe.vae.config.scaling_factor
    latents = latents * 0.18215
    # make dir
    name = content_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, name, 'inversion')
    reconstruction_path = os.path.join(output_dir, name, 'reconstruction')
    ft_path = os.path.join(output_dir, name, 'features')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    os.makedirs(ft_path, exist_ok=True)
    # ----------------------------------content video inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", inversion_path=inversion_path,
                                    up_ft_indices=[2], ft_path=ft_path,
                                    is_opt=is_opt,)[-1].to(weight_dtype)
    # ---------------------------------content video construction------------------------------
    print(f"reconstruction:")
    sample = pipe("", latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "content_video.mp4"))


def style_inversion_reconstruction(pipe, ddim_inv_scheduler, style_path, output_dir,
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
    # make dir
    name = style_path.split('/')[-1].split('.')[0]
    inversion_path = os.path.join(output_dir, name, 'inversion')
    reconstruction_path = os.path.join(output_dir, name, 'reconstruction')
    os.makedirs(inversion_path, exist_ok=True)
    os.makedirs(reconstruction_path, exist_ok=True)
    # ----------------------------------content style inversion--------------------------------
    print(f"inversion:")
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, video_latent=latents,
                                    num_inv_steps=time_steps, prompt="", 
                                    is_opt=is_opt, inversion_path=inversion_path)[-1].to(weight_dtype)
    # ---------------------------------content style construction------------------------------
    print(f"reconstruction:")
    sample = pipe("", latents=ddim_inv_latent, video_length=num_frames, guidance_scale=1.0).images
    # breakpoint()
    sample = sample.permute(0, 4, 1, 2, 3).contiguous()
    save_videos_grid(sample, os.path.join(reconstruction_path, "style_video.mp4"), fps=8)
