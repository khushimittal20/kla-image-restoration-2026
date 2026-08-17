from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.dataset import KLARestorationDataset
from src.models.detail_sr import DetailSR
from src.losses.losses import L1Loss
from src.utils.metrics import calculate_psnr


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/train")
WEIGHTS_DIR = Path("weights")

BATCH_SIZE = 8
VAL_RATIO = 0.10
RANDOM_SEED = 42

LEARNING_RATE = 1e-4
EPOCHS = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Model
# ============================================================

model = DetailSR().to(DEVICE)

criterion = L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Training
# ============================================================

print(f"Device: {DEVICE}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters())}"
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (noisy, gt) in enumerate(train_loader):

        noisy = noisy.to(DEVICE)
        gt = gt.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(noisy)

        loss = criterion(
            prediction,
            gt,
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    average_loss = (
        running_loss / len(train_loader)
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    validation_psnr = 0.0
    validation_images = 0

    with torch.no_grad():

        for noisy, gt in val_loader:

            noisy = noisy.to(DEVICE)
            gt = gt.to(DEVICE)

            prediction = model(noisy)

            for i in range(noisy.size(0)):

                validation_psnr += calculate_psnr(
                    prediction[i:i + 1],
                    gt[i:i + 1],
                )

            validation_images += noisy.size(0)

    validation_psnr /= validation_images

    print(
        f"\nEpoch {epoch + 1} complete"
        f"\nTraining Loss: {average_loss:.6f}"
        f"\nValidation PSNR: "
        f"{validation_psnr:.4f} dB\n"
    )


# ============================================================
# Save checkpoint
# ============================================================

WEIGHTS_DIR.mkdir(exist_ok=True)

checkpoint_path = (
    WEIGHTS_DIR / "detail_sr.pth"
)

torch.save(
    model.state_dict(),
    checkpoint_path,
)

print(
    f"Model saved to: {checkpoint_path}"
)