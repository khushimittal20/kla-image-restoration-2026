import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class BaselineSR(nn.Module):
    """
    Baseline:
        NoisyLR
          ↓
        Bicubic 2× upsampling
          ↓
        CNN restoration
          ↓
        Restored HR

    Input:  [B, 1, 128, 128]
    Output: [B, 1, 256, 256]
    """

    def __init__(self, num_blocks=4, channels=64):
        super().__init__()

        self.head = nn.Conv2d(1, channels, 3, padding=1)

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        # 2× spatial upsampling
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        # CNN restoration
        residual = self.head(x)
        residual = self.body(residual)
        residual = self.tail(residual)

        # Residual learning:
        # model predicts a correction to the bicubic image
        output = x + residual

        return output