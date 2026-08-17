import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class DetailSR(nn.Module):
    """
    Experiment 3: High-Frequency Detail CNN

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]

    Uses:
        - bicubic reconstruction baseline
        - residual feature extraction
        - explicit high-frequency branch
        - learned residual correction
    """

    def __init__(self, channels=64, num_blocks=6):
        super().__init__()

        # Main feature branch
        self.head = nn.Conv2d(1, channels, 3, padding=1)

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Detail branch
        self.detail = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fuse main + detail features
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            channels,
            1,
            3,
            padding=1,
        )

    def forward(self, x):

        # Bicubic ×2 baseline
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        # Main restoration branch
        main = self.head(base)
        main = self.body(main)

        # High-frequency input
        blur = F.avg_pool2d(
            base,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        high_freq = base - blur

        detail = self.detail(high_freq)

        # Fuse both branches
        features = torch.cat(
            [main, detail],
            dim=1,
        )

        features = self.fusion(features)

        # Predict residual correction
        residual = self.tail(features)

        output = base + residual

        return output