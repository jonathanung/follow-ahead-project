import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from ..config import DataConfig
from .augmentation import augment_sequence
from .labels import generate_target


class TrajectoryDataset(Dataset):
    """Human3.6M walking trajectory dataset with augmentation.

    Loads raw 3D position data, extracts 2D hip joint coordinates,
    downsamples to target frequency, windows into sequences, applies
    augmentation, generates soft labels, and normalizes input to be
    relative to the last observed point.

    Args:
        config: DataConfig with data parameters.
        split: "train" or "test".
    """

    def __init__(self, config: DataConfig = DataConfig(), split: str = "train"):
        self.config = config

        samples, targets = self._load_and_process()

        # Normalize: make trajectories relative to last input point
        last_points = samples[:, config.input_length - 1 : config.input_length, :]
        samples = samples - np.broadcast_to(last_points, samples.shape)

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(config.seed)
        idx = rng.permutation(len(samples))
        samples = samples[idx]
        targets = targets[idx]

        # Split
        split_idx = int(config.train_ratio * len(samples))
        if split == "train":
            self.samples = torch.from_numpy(samples[:split_idx].astype(np.float32))
            self.targets = torch.from_numpy(targets[:split_idx].astype(np.float32))
        else:
            self.samples = torch.from_numpy(samples[split_idx:].astype(np.float32))
            self.targets = torch.from_numpy(targets[split_idx:].astype(np.float32))

    def _load_and_process(self) -> Tuple[np.ndarray, np.ndarray]:
        raw_data = np.load(self.config.data_path, allow_pickle=True)[
            "positions_3d"
        ].item()

        freq_ratio = self.config.source_freq // self.config.target_freq
        samples = []
        targets = []

        for subject in self.config.subjects:
            data_3d = raw_data[subject][self.config.motion]
            data_2d = data_3d[:, 0, :2]  # Hip joint, 2D

            for phase_offset in range(freq_ratio):
                indices = np.arange(phase_offset, data_2d.shape[0], freq_ratio)
                downsampled = data_2d[indices]

                for i in range(downsampled.shape[0] - self.config.seq_length):
                    window = downsampled[i : i + self.config.seq_length]

                    for aug_seq in augment_sequence(window):
                        samples.append(aug_seq[: self.config.input_length])
                        targets.append(
                            generate_target(
                                aug_seq, self.config.input_length, self.config.tanh_power
                            )
                        )

        return np.array(samples), np.array(targets)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.targets[idx]
