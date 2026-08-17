# KLA Image Restoration 2026

## AI-Based Restoration of Degraded Images for Semiconductor Inspection

A deep-learning based image restoration system designed to recover clean, high-resolution grayscale semiconductor inspection images from noisy and low-resolution inputs.

The model performs **2× super-resolution and image restoration simultaneously**, learning from paired degraded and ground-truth images.

## Problem

Semiconductor inspection images need to preserve fine structures and edges because small defects can be difficult to detect when images contain noise or have insufficient spatial resolution.

The input images in this challenge are degraded through combinations of:

* Speckle noise
* Gaussian noise / image degradation
* Spatial resolution reduction

The goal is to transform a degraded low-resolution image into a clean image at the original resolution.

**Input:** 128 × 128 grayscale degraded image

**Output:** 256 × 256 grayscale restored image

## Approach

We experimented with progressively stronger restoration approaches:

1. **Bicubic interpolation** — non-learning baseline
2. **Baseline CNN** — residual CNN restoration
3. **Detail-aware CNN** — added edge/detail-aware loss
4. **DetailSR** — final model with deeper restoration and explicit detail processing/fusion

The final submission uses **DetailSR**.

## Model Architecture

The final model is implemented in PyTorch.

```text
128×128 Degraded Input
        │
        ▼
  2× Bicubic Upsampling
        │
        ▼
  Feature Extraction
        │
        ▼
  Residual CNN Blocks
        │
        ├──────────────► Detail Branch
        │                    │
        │                    ▼
        │              Detail Features
        │                    │
        └──────────────► Feature Fusion
                             │
                             ▼
                  Restored 256×256 Image
```

The model uses residual learning to refine the upsampled image while dedicated feature processing helps preserve fine structures.

**Model parameters:** 592,641

## Loss Function

The training objective combines pixel-level reconstruction with detail preservation.

The detail-aware loss is implemented in:

```text
src/losses/detail_loss.py
```

The loss encourages the restored image to:

* Match ground-truth pixel values
* Preserve important edges and fine structures
* Avoid excessive smoothing during restoration

## Dataset

The training data consists of paired grayscale NumPy images:

```text
data/
└── train/
    ├── NoisyLR/
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    │
    └── GT/
        ├── 000000.npy
        ├── 000001.npy
        └── ...
```

Each degraded image has a corresponding ground-truth image with the same filename.

For our training setup:

* Total training samples: **3200**
* Training split: **2880**
* Validation split: **320**
* Input shape: **1 × 128 × 128**
* Ground-truth shape: **1 × 256 × 256**

The degraded input can contain values outside `[0, 1]` because of the noise degradation. Model outputs are clipped to `[0, 1]` before saving.

## Training

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the final DetailSR model:

```bash
python train_detail_sr.py
```

The trained weights are saved to:

```text
weights/detail_sr.pth
```

## Inference

The inference script accepts:

* Input directory
* Output directory
* Model weights

Example:

```bash
python inference.py --input_dir data/test/NoisyLR --output_dir outputs/final_test --weights weights/detail_sr.pth
```

The script processes every `.npy` image in the input directory and saves restored images using the original filenames.

Example:

```text
Input:
data/test/NoisyLR/000000.npy

Output:
outputs/final_test/000000.npy
```

## Evaluation

For validation experiments, we compared the final model against bicubic interpolation using:

* PSNR
* SSIM
* LPIPS

### Validation Results

| Method   |         PSNR ↑ |     SSIM ↑ |    LPIPS ↓ |
| -------- | -------------: | ---------: | ---------: |
| Bicubic  |     23.0831 dB |     0.5469 |     0.4351 |
| DetailSR | **27.2095 dB** | **0.7188** | **0.3223** |

### Improvement over Bicubic

* PSNR: **+4.1264 dB**
* SSIM: **+0.1719**
* LPIPS: **−0.1128**

Higher PSNR and SSIM indicate better reconstruction quality, while lower LPIPS indicates greater perceptual similarity to the ground truth.

## Inference Performance

The final model was tested in our CPU-only development environment.

```text
Inference time: 357.63 ms/image
Device: CPU
```

Competition benchmarking may differ depending on the evaluation hardware.

## Repository Structure

```text
kla-image-restoration-2026/
│
├── inference.py
├── train_detail_sr.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── dataset/
│   │   └── dataset.py
│   │
│   ├── losses/
│   │   ├── detail_loss.py
│   │   └── losses.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   └── detail_sr.py
│   │
│   └── utils/
│       └── metrics.py
│
└── weights/
    └── detail_sr.pth
```

## Requirements

The project uses:

* Python
* PyTorch
* NumPy
* OpenCV
* scikit-image
* LPIPS
* Matplotlib

Install all required packages using:

```bash
pip install -r requirements.txt
```

## Reproducibility

The repository contains:

* Final model architecture
* Training script
* Dataset loader
* Loss implementation
* Evaluation utilities
* Inference script
* Trained model weights
* Dependency specification

The inference pipeline can be run by providing a directory containing `.npy` input images and an output directory.

## Final Model

**Model:** DetailSR
**Task:** Grayscale image restoration + 2× super-resolution
**Input:** 128 × 128
**Output:** 256 × 256
**Parameters:** 592,641
**Framework:** PyTorch

Final trained weights:

```text
weights/detail_sr.pth
```

## Final Status

The final DetailSR model has been trained and evaluated on a held-out validation split.

The model was also used to generate restored outputs for all **400 provided test images**, producing 256 × 256 `.npy` outputs with the original filenames.
