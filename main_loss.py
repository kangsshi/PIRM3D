# PIRM3D: physics-inspired 3D radio map construction.
"""Measured MSE + slice SI-MSE + signed finite differences in physical units."""
import torch
import torch.nn as nn


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


class PhysicalStructureLoss(nn.Module):
    def __init__(self, lambda_structure=0.1, gradient_axes="xyz", spacing_m=(1, 1, 1)):
        super().__init__()
        self.weight, self.axes, self.spacing = float(lambda_structure), gradient_axes, spacing_m

    def forward(self, logits, measured, y2, known, occupied):
        pred = torch.sigmoid(logits)
        blind = ~known & ~occupied
        mse = masked_mean((pred - measured).square(), known)
        count = blind.sum(dim=(-2, -1))
        diff = pred - y2
        mean = (diff * blind).sum(dim=(-2, -1)) / count.clamp_min(1)
        variance = ((diff - mean[..., None, None]).square() * blind).sum(dim=(-2, -1)) / count.clamp_min(1)
        si = masked_mean(variance, count > 0)
        gradients = {}
        for name, dimension, spacing in (("x", 3, self.spacing[2]), ("y", 2, self.spacing[1]),
                                          ("z", 1, self.spacing[0])):
            if name not in self.axes:
                continue
            left, right = [slice(None)] * 4, [slice(None)] * 4
            left[dimension], right[dimension] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            dp = (pred[right] - pred[left]) / spacing
            dt = (y2[right] - y2[left]) / spacing
            gradients[name] = masked_mean((dp - dt).square(), blind[right] & blind[left])
        gradient = sum(gradients.values())
        total = mse + self.weight * (si + gradient)
        return total, {"mse": mse, "si_mse": si, "gradient": gradient,
                       **{f"grad_{k}": v for k, v in gradients.items()}}
