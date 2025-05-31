import os

import numpy as np
from PIL import Image
from einops import rearrange
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .transform import short_size_scale, random_crop, center_crop, offset_crop
from ..common.image_util import IMAGE_EXTENSION
import decord
decord.bridge.set_bridge('torch')

class UniVSTDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 16,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
            **kwargs,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        # breakpoint()
        video = rearrange(video, "f h w c -> f c h w")
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }
        return example
