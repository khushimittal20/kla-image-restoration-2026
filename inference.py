from pathlib import Path
import argparse

import numpy as np
import torch

from src.models.detail_sr import DetailSR

def load_model(weights_path, device):
    model = DetailSR().to(device)

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def restore_image(model, image, device):
    """
    Restore a single grayscale 128x128 numpy image
    into a 256x256 restored image.
    """

    tensor = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().cpu().numpy()

    # Ground truth is in [0, 1].
    # The degraded input may go outside this range.
    output = np.clip(output, 0.0, 1.0)

    return output.astype(np.float32)


def main():

    parser = argparse.ArgumentParser(
        description="KLA semiconductor image restoration inference"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input .npy images",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored .npy images will be saved",
    )

    parser.add_argument(
        "--weights",
        default="weights/detail_sr.pth",
        help="Path to trained model weights",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Weights: {weights_path}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        weights_path,
        device,
    )

    # --------------------------------------------------------
    # Find input images
    # --------------------------------------------------------

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:
        raise RuntimeError(
            f"No .npy files found in {input_dir}"
        )

    print(f"Found {len(input_files)} input images")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for index, input_file in enumerate(input_files):

        image = np.load(input_file)

        restored = restore_image(
            model,
            image,
            device,
        )

        output_file = output_dir / input_file.name

        np.save(
            output_file,
            restored,
        )

        if index % 50 == 0:
            print(
                f"Processed {index + 1}/{len(input_files)}"
            )

    print("\nInference complete.")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()