from pathlib import Path
import time

import torch
import torch.nn.functional as F
import lpips
from torch.utils.data import DataLoader, random_split
from skimage.metrics import structural_similarity

from src.dataset.dataset import KLARestorationDataset
from src.models.detail_sr import DetailSR

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/train")
WEIGHTS = Path("weights/detail_sr.pth")

BATCH_SIZE = 8
VAL_RATIO = 0.10
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Metrics
# ============================================================

def calculate_psnr(prediction, target):
    prediction = prediction.clamp(0.0, 1.0)

    mse = F.mse_loss(prediction, target).item()

    if mse == 0:
        return float("inf")

    return 10 * torch.log10(
        torch.tensor(1.0 / mse)
    ).item()


def calculate_ssim(prediction, target):
    prediction = prediction.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    return structural_similarity(
        target,
        prediction,
        data_range=1.0,
    )


# ============================================================
# Dataset
# ============================================================

dataset = KLARestorationDataset(
    noisy_dir=DATA_DIR / "NoisyLR",
    gt_dir=DATA_DIR / "GT",
)

val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size

generator = torch.Generator().manual_seed(RANDOM_SEED)

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Models
# ============================================================

model = DetailSR().to(DEVICE)
model.load_state_dict(
    torch.load(
        WEIGHTS,
        map_location=DEVICE,
    )
)

model.eval()


# LPIPS expects 3-channel images with values in [-1, 1]
lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
lpips_model.eval()


# ============================================================
# Evaluation
# ============================================================

model_psnr = 0.0
model_ssim = 0.0
model_lpips = 0.0

bicubic_psnr = 0.0
bicubic_ssim = 0.0
bicubic_lpips = 0.0

total_model_time = 0.0
total_images = 0


with torch.no_grad():

    for noisy, gt in val_loader:

        noisy = noisy.to(DEVICE)
        gt = gt.to(DEVICE)

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = F.interpolate(
            noisy,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        bicubic = bicubic.clamp(0.0, 1.0)

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        start = time.perf_counter()

        prediction = model(noisy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        total_model_time += time.perf_counter() - start

        prediction = prediction.clamp(0.0, 1.0)

        # ----------------------------------------------------
        # Convert grayscale → 3 channel for LPIPS
        # ----------------------------------------------------

        prediction_3 = prediction.repeat(1, 3, 1, 1)
        bicubic_3 = bicubic.repeat(1, 3, 1, 1)
        gt_3 = gt.repeat(1, 3, 1, 1)

        # LPIPS expects [-1, 1]
        prediction_3 = prediction_3 * 2 - 1
        bicubic_3 = bicubic_3 * 2 - 1
        gt_3 = gt_3 * 2 - 1

        model_lpips_batch = lpips_model(
            prediction_3,
            gt_3,
        ).mean().item()

        bicubic_lpips_batch = lpips_model(
            bicubic_3,
            gt_3,
        ).mean().item()

        # ----------------------------------------------------
        # Per-image metrics
        # ----------------------------------------------------

        for i in range(noisy.size(0)):

            model_psnr += calculate_psnr(
                prediction[i:i + 1],
                gt[i:i + 1],
            )

            model_ssim += calculate_ssim(
                prediction[i],
                gt[i],
            )

            bicubic_psnr += calculate_psnr(
                bicubic[i:i + 1],
                gt[i:i + 1],
            )

            bicubic_ssim += calculate_ssim(
                bicubic[i],
                gt[i],
            )

        model_lpips += model_lpips_batch * noisy.size(0)
        bicubic_lpips += bicubic_lpips_batch * noisy.size(0)

        total_images += noisy.size(0)


# ============================================================
# Final results
# ============================================================

model_psnr /= total_images
model_ssim /= total_images
model_lpips /= total_images

bicubic_psnr /= total_images
bicubic_ssim /= total_images
bicubic_lpips /= total_images

avg_time_ms = (
    total_model_time / total_images
) * 1000


print("\n================ RESULTS ================\n")

print("BICUBIC")
print(f"PSNR  : {bicubic_psnr:.4f} dB")
print(f"SSIM  : {bicubic_ssim:.4f}")
print(f"LPIPS : {bicubic_lpips:.4f}")

print("\nDETAIL-AWARE CNN")
print(f"PSNR  : {model_psnr:.4f} dB")
print(f"SSIM  : {model_ssim:.4f}")
print(f"LPIPS : {model_lpips:.4f}")

print("\nPERFORMANCE")
print(f"Inference time: {avg_time_ms:.2f} ms/image")

print("\n==========================================")