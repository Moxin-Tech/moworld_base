import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data

from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.data_transform import TemporalRandomCrop
from mindspeed_mm.data.data_utils.utils import DecordInit


class TrainVideoDataset(data.Dataset):
    video_exts = ["avi", "mp4", "webm"]

    def __init__(
        self,
        video_folder,
        num_frames,
        resolution=64,
        sample_rate=1,
        dynamic_sample=True,
        transform_pipeline=None
    ):

        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.v_decoder = DecordInit()
        self.video_folder = video_folder
        self.dynamic_sample = dynamic_sample
        self.transform = get_transforms(
            is_video=True,
            train_pipeline=transform_pipeline
        )
        print("Building datasets...")
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        samples += sum(
            [
                glob(osp.join(self.video_folder, "**", f"*.{ext}"), recursive=True)
                for ext in self.video_exts
            ],
            [],
        )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        try:
            video = self.decord_read(video_path)
            video = self.transform(video)  # T C H W -> T C H W
            video = video.transpose(0, 1)  # T C H W -> C T H W
            return dict(video=video, label="")
        except Exception as e:
            print(f"Error with {e}, {video_path}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def decord_read(self, path):
        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)
        # Sampling video frames
        if self.dynamic_sample:
            sample_rate = random.randint(1, self.sample_rate)
        else:
            sample_rate = self.sample_rate
        size = self.num_frames * sample_rate
        temporal_sample = TemporalRandomCrop(size)
        start_frame_ind, end_frame_ind = temporal_sample(total_frames)
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
        )

        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2).contiguous()
        return video_data
