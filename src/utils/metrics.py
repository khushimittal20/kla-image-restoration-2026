import math

import torch
import torch.nn.functional as F


def calculate_psnr(prediction, target, max_value=1.0):
    """
    Calculate PSNR assuming target images are in [0, 1].
    """
    prediction = prediction.clamp(0.0, 1.0)

    mse = F.mse_loss(prediction, target)

    if mse.item() == 0:
        return float("inf")

    return 10 * math.log10((max_value ** 2) / mse.item())


def calculate_ssim(prediction, target):
    """
    Placeholder for SSIM.

    We'll add the actual SSIM implementation/library
    after the basic training loop is verified.
    """
    raise NotImplementedError("SSIM will be added next.")