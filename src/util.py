import os
import imageio
import numpy as np
from typing import Union, Sequence, Callable
import requests

import torch
import torchvision
from einops import rearrange

import PIL
import random
from PIL import Image


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_folder(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")

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


def load_video_frames(frames_path, n_frames, image_size=(512, 512)):
    # Load paths
    paths = [f"{frames_path}/%05d.png" % (i * 1) for i in range(n_frames)]
    # paths = [os.path.join(frames_path, item) for item in sorted(os.listdir(frames_path), key=extract_number)]
    frames = []
    for p in paths:
        img = load_image(p, image_size=image_size)
        # check!
        if img.size != image_size:
            img = img.resize(image_size)
            raise ValueError(f"Frame size does not match config.image_size")
        # transforms to tensor
        np_img = np.array(img)
        # transforms to [-1, 1]
        normalized_img = (np_img / 127.5) - 1.0
        tensor_img = torch.from_numpy(normalized_img).permute(2, 0, 1).float()
        frames.append(tensor_img)
    video_tensor = torch.stack(frames)
    return video_tensor


def load_image(
    image: Union[str, PIL.Image.Image], convert_method: Callable[[PIL.Image.Image], PIL.Image.Image] = None,
    image_size=None,
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
            image = PIL.Image.open(image).resize(image_size)
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


def load_ddim_latents_at_t(t, ddim_latents_path, is_x0=False):
    if is_x0:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_x0_{t}.pt")
    else:
        ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path, weights_only=True)
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
