import os
import imageio
import numpy as np
from typing import Union, Sequence, Callable, Tuple
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange

import abc
import copy
from datetime import datetime
import logging
import PIL
import random
logger = logging.getLogger(__name__)

def save_folder(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        imageio.imsave(os.path.join(path, f"%05d.png" % (i * 1)), x)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []

    for i, x in enumerate(videos):
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def save_images_as_gif(
    images: Sequence[Image.Image],
    save_path: str,
    loop=0,
    duration=100,
    optimize=False,
) -> None:
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )

def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()

def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.png"), np.array(init_image))

def save_gif_mp4_folder_type(images, save_path, save_gif=True):
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    breakpoint()
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    if save_gif: save_images_as_gif(images, save_path)
    save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
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

    # SD1.5
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_epsilon = model_output

    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    return next_sample

def get_noise_pred_single(latents, t, context, unet, up_ft_indices=None, ft_path=None):
    noise_pred = unet(latents, t, encoder_hidden_states=context, up_ft_indices=up_ft_indices, ft_path=ft_path)["sample"]
    return noise_pred

@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, inversion_path,
                up_ft_indices=None, ft_path=None):
    context = init_prompt(prompt, pipeline)
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
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet,
                                            up_ft_indices=up_ft_indices, ft_path=ft_path)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        # save latent
        if inversion_path is not None:
            torch.save(
                        latent.detach().clone(),
                        os.path.join(inversion_path, f"ddim_latents_{t}.pt"),
                    )
        all_latent.append(latent)
    return all_latent

@torch.no_grad()
def ddim_loop_plus(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, inversion_path,
                    up_ft_indices=None, ft_path=None):
    context = init_prompt(prompt, pipeline)
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
    no_inject = 0
    num_inference_steps=50
    num_fix_itr = 0
    inject_times = 0
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet, 
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
                        os.path.join(inversion_path, f"ddim_latents_{t}.pt"),
                    )
        all_latent.append(latent)
        continue
    return all_latent

@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", is_opt=False, inversion_path=None,
                    up_ft_indices=None, ft_path=None):
    if is_opt:
        ddim_latents = ddim_loop_plus(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, 
                                        inversion_path, up_ft_indices=up_ft_indices, ft_path=ft_path)
    else:
        ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt,
                                        inversion_path, up_ft_indices=up_ft_indices, ft_path=ft_path)
    return ddim_latents

def load_ddim_latents_at_t(t, ddim_latents_path):
    ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path)
    logger.debug(f"Loaded ddim_latents_at_t from {ddim_latents_at_t_path}")
    return ddim_latents_at_t

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else -1
    to_tensor = transforms.ToTensor()
    paths = [f"{frames_path}/%05d.png" % (i * 1) for i in range(n_frames)]
    # paths = [os.path.join(frames_path, item) for item in sorted(os.listdir(frames_path), key=extract_number)]
    frames = []
    for p in paths:
        img = load_image(p)
        # check!
        if img.size != image_size:
            img = img.resize(image_size)
            logger.error(f"Frame size {f.size} does not match config.image_size {image_size}")
            raise ValueError(f"Frame size {f.size} does not match config.image_size {image_size}")
        # transforms to tensor
        np_img = np.array(img)
        # transforms to [-1, 1]
        normalized_img = (np_img / 127.5) - 1.0
        tensor_img = torch.from_numpy(normalized_img).permute(2, 0, 1).float()
        frames.append(tensor_img)
    video_tensor = torch.stack(frames)
    return video_tensor
