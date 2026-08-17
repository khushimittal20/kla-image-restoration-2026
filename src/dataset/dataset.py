from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KLARestorationDataset(Dataset):
    """
    Dataset for the KLA image-restoration challenge.

    Expected structure:
        data/
        └── train/
            ├── NoisyLR/
            │   ├── 000000.npy
            │   └── ...
            └── GT/
                ├── 000000.npy
                └── ...

    NoisyLR: degraded low-resolution image
    GT: clean high-resolution ground-truth image
    """

    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)

        self.noisy_files = sorted(self.noisy_dir.glob("*.npy"))
        self.gt_files = sorted(self.gt_dir.glob("*.npy"))

        if len(self.noisy_files) == 0:
            raise RuntimeError(f"No .npy files found in {self.noisy_dir}")

        if len(self.noisy_files) != len(self.gt_files):
            raise RuntimeError(
                f"Number of NoisyLR files ({len(self.noisy_files)}) "
                f"does not match GT files ({len(self.gt_files)})"
            )

        # Verify that every NoisyLR file has a matching GT file.
        for noisy_path, gt_path in zip(self.noisy_files, self.gt_files):
            if noisy_path.stem != gt_path.stem:
                raise RuntimeError(
                    f"Filename mismatch: {noisy_path.name} vs {gt_path.name}"
                )

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index):
        noisy = np.load(self.noisy_files[index]).astype(np.float32)
        gt = np.load(self.gt_files[index]).astype(np.float32)

        # Add channel dimension:
        # (H, W) -> (1, H, W)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt