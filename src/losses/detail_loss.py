import torch
import torch.nn.functional as F


def sobel_edges(x):
    """Compute horizontal and vertical Sobel gradients."""

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=x.dtype,
        device=x.device
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    return gx, gy


def detail_loss(prediction, target):
    """
    Detail-aware restoration loss.

    Combines:
      1. L1 pixel loss
      2. Sobel edge loss
    """

    pixel_loss = F.l1_loss(prediction, target)

    pred_gx, pred_gy = sobel_edges(prediction)
    target_gx, target_gy = sobel_edges(target)

    edge_loss = (
        F.l1_loss(pred_gx, target_gx)
        + F.l1_loss(pred_gy, target_gy)
    )

    total_loss = pixel_loss + 0.1 * edge_loss

    return total_loss, pixel_loss, edge_loss