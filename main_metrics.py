# PIRM3D: physics-inspired 3D radio map construction.
"""Consistent per-sample macro metrics for validation and testing."""
import math
import numpy as np
from scipy.ndimage import uniform_filter


def local_ssim_map(first, second, window=7):
    """Uniform-window SSIM map, K1=.01, K2=.03, range=1, sample covariance.

    Same local statistics as skimage's non-Gaussian structural_similarity;
    callers explicitly choose how to average the map over valid air pixels.
    Reference: https://scikit-image.org/docs/stable/api/skimage.metrics.html
    """
    if first.shape != second.shape or first.ndim != 2 or window < 3 or window % 2 == 0:
        raise ValueError("SSIM needs matching 2D planes and an odd window >=3")
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    mean_a = uniform_filter(a, size=window, mode="reflect")
    mean_b = uniform_filter(b, size=window, mode="reflect")
    correction = window**2 / (window**2 - 1)
    var_a = correction * (uniform_filter(a*a, size=window, mode="reflect") - mean_a**2)
    var_b = correction * (uniform_filter(b*b, size=window, mode="reflect") - mean_b**2)
    cov = correction * (uniform_filter(a*b, size=window, mode="reflect") - mean_a*mean_b)
    luminance = (2 * mean_a * mean_b + .01**2) / (mean_a**2 + mean_b**2 + .01**2)
    structure = (2 * cov + .03**2) / (var_a + var_b + .03**2)
    return luminance * structure


class Metrics:
    def __init__(self):
        self.rows = []

    def update(self, pred, gt, valid, names=None):
        for index in range(pred.shape[0]):
            mask = valid[index]
            n = int(mask.sum())
            if not n:
                continue
            error = (pred[index][mask].double() - gt[index][mask].double()).square().sum().item()
            energy = gt[index][mask].double().square().sum().item()
            mse = error / n
            self.rows.append({"filename": names[index] if names else str(len(self.rows)),
                              "mse": mse, "rmse": math.sqrt(mse), "nmse": error / max(energy, 1e-12),
                              "psnr": -10 * math.log10(max(mse, 1e-12)), "valid_voxels": n})

    def compute(self):
        if not self.rows:
            raise ValueError("No valid evaluation samples")
        return {key: float(np.mean([r[key] for r in self.rows])) for key in ("mse", "rmse", "nmse", "psnr")}
