import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from dataset_3d import UrbanRadio3D_Dataset
from model_3d import ResUNet3D

os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# Utility Functions
# ==========================================
def plot_3d_volume(ax, volume, title, cmap='jet', threshold=0.05, drop_rate=0.90):
    """ Plot a sparse 3D volume for visualization
    Args:
        ax: Matplotlib 3D axis
        volume: 3D numpy array of shape (20, 256, 256)
        title: Plot title
        cmap: Colormap
        threshold: Minimum value to plot
        drop_rate: Fraction of points to drop for performance
    Returns:
        Scatter plot object
    """
    z, y, x = np.where(volume > threshold)
    values = volume[z, y, x]
    if len(x) > 0:
        keep_indices = np.random.rand(len(x)) > drop_rate
        x, y, z, values = x[keep_indices], y[keep_indices], z[keep_indices], values[keep_indices]
        scatter = ax.scatter(x, y, z, c=values, cmap=cmap, s=2, alpha=0.7, vmin=0, vmax=1)
    else:
        scatter = ax.scatter([0], [0], [0], c=[0], cmap=cmap, s=0.1, alpha=0.1, vmin=0, vmax=1)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 19)
    ax.set_box_aspect((1, 1, 0.6))
    return scatter


def atomic_save(checkpoint_data, save_path):
    """ Save checkpoint atomically to avoid corruption """
    temp_path = save_path + ".tmp"
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, save_path)


# ==========================================
# Metric Tracker (GPU-Accelerated, OOM-Safe)
# ==========================================
class MetricTracker:
    """
    GPU-based scalar aggregator to avoid system RAM overflow
    All heavy computations are done on GPU, only Python floats are returned
    """

    def __init__(self):
        self.sq_err = 0.0
        self.gt_sq = 0.0
        self.pixels = 0

    def update(self, pred, gt, mask):
        """ Update tracker with new batch
        Args:
            pred: Predicted 3D radio map (torch.Tensor)
            gt: Ground truth 3D radio map (torch.Tensor)
            mask: Boolean mask of valid voxels (torch.Tensor)
        """
        valid_count = mask.sum().item()
        if valid_count > 0:
            sq_err_val = torch.sum((pred[mask] - gt[mask]) ** 2).item()
            gt_sq_val = torch.sum(gt[mask] ** 2).item()

            self.sq_err += sq_err_val
            self.gt_sq += gt_sq_val
            self.pixels += valid_count

    def compute(self):
        """ Compute final metrics
        Returns:
            tuple: (rmse, nmse, num_pixels)
        """
        if self.pixels == 0:
            return 0.0, 0.0, 0
        mse = self.sq_err / self.pixels
        rmse = np.sqrt(mse)
        nmse = self.sq_err / (self.gt_sq + 1e-8)
        return rmse, nmse, self.pixels


# ==========================================
# Main Test Function
# ==========================================
def main():
    # Configuration
    DATA_ROOT = '/root/autodl-tmp/3DRadioMap'
    SAVE_DIR = os.path.join(DATA_ROOT, 'Checkpoints_PIRM3D')
    LOG_PATH = os.path.join(SAVE_DIR, 'test_results.log')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Simple logging setup (consistent with train.py)
    def log(message):
        print(message)
        with open(LOG_PATH, 'a') as f:
            f.write(message + '\n')

    log("=" * 50)
    log("[START] Testing PIRM3D: Physics-Inspired 3D Radio Map Construction")
    log("=" * 50)

    # Load weights
    best_weight = os.path.join(SAVE_DIR, 'best_model.pth')
    latest_weight = os.path.join(SAVE_DIR, 'latest_model.pth')

    if os.path.exists(best_weight):
        WEIGHT_PATH = best_weight
        log("-> Testing with best_model.pth")
    elif os.path.exists(latest_weight):
        WEIGHT_PATH = latest_weight
        log("-> Testing with latest_model.pth")
    else:
        log("-> Error: No weights found!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Test Dataset
    log("-> Loading Test dataset...")
    test_dataset = UrbanRadio3D_Dataset(DATA_ROOT, split='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Preload global UAV Mask to GPU
    uav_mask_global = torch.from_numpy(test_dataset.uav_mask).bool().to(device)

    # Initialize Model and Load Weights
    model = ResUNet3D(in_channels=5, out_channels=1, base_dim=16).to(device)
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        log(f"-> Weights loaded from Epoch {checkpoint.get('epoch', 'Unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Initialize Metric Trackers
    tracker_full = MetricTracker()
    tracker_known = MetricTracker()
    tracker_unknown = MetricTracker()
    tracker_uav = MetricTracker()

    ssim_sum = 0.0
    total_samples = 0
    vis_data = None

    # Inference Loop
    log("-> Starting pixel-level inference...")
    with torch.inference_mode():
        for step, batch in enumerate(test_loader):
            # Move tensors to GPU and detach
            x = batch['x'].to(device).detach()
            gt = batch['gt'].to(device).detach()

            bldg_mask = (x[:, 1, :, :, :] == 1.0)

            # Expand global UAV mask to batch size
            batch_uav_mask = uav_mask_global.unsqueeze(0).expand(x.size(0), -1, -1, -1)

            # Generate ground H1 mask
            batch_h1_mask = torch.zeros_like(bldg_mask, dtype=torch.bool)
            batch_h1_mask[:, 0, :, :] = True

            # Build mask logic
            known_mask = (batch_h1_mask | batch_uav_mask) & (~bldg_mask)
            unknown_mask = (~known_mask) & (~bldg_mask)
            full_air_mask = ~bldg_mask
            pure_uav_mask = batch_uav_mask & (~bldg_mask)

            # Forward pass
            logits = model(x).squeeze(1).detach()
            pred = torch.sigmoid(logits)
            pred[bldg_mask] = 0.0

            # Update trackers (only scalar values are transferred back to CPU)
            tracker_full.update(pred, gt, full_air_mask)
            tracker_known.update(pred, gt, known_mask)
            tracker_unknown.update(pred, gt, unknown_mask)
            tracker_uav.update(pred, gt, pure_uav_mask)

            # Compute SSIM (must be done on CPU, clean up immediately)
            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            for b in range(x.size(0)):
                for z in range(20):
                    # Handle edge cases where image is flat
                    if gt_np[b, z].var() < 1e-5 and pred_np[b, z].var() < 1e-5:
                        s = 1.0
                    else:
                        s = ssim(gt_np[b, z], pred_np[b, z], data_range=1.0)
                    ssim_sum += s

            total_samples += x.size(0)

            # Save first sample for visualization
            if step == 0:
                vis_data = {
                    'hint': x[0, 3].cpu().numpy(),
                    'pred': pred_np[0],
                    'gt': gt_np[0]
                }

            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                current_full_rmse, _, _ = tracker_full.compute()
                print(f"  Step {step + 1}/{len(test_loader)} | Full RMSE: {current_full_rmse:.4f}")

            # Clean up large temporary tensors to prevent OOM
            del x, gt, logits, pred, bldg_mask, batch_uav_mask, batch_h1_mask
            del known_mask, unknown_mask, full_air_mask, pure_uav_mask
            del pred_np, gt_np

    # Compute final metrics
    rmse_full, nmse_full, px_full = tracker_full.compute()
    rmse_known, nmse_known, px_known = tracker_known.compute()
    rmse_unknown, nmse_unknown, px_unknown = tracker_unknown.compute()
    rmse_uav, nmse_uav, px_uav = tracker_uav.compute()

    ssim_avg = ssim_sum / (total_samples * 20)
    psnr_full = 10 * np.log10(1.0 / (rmse_full ** 2 + 1e-8))

    # Log final results
    log("=" * 60)
    log(f"3D Radio Map Evaluation Completed (Samples: {total_samples})")

    log("-" * 60)
    log("Case 1: Metrics on 100% 3D Volume (Exclude Buildings)")
    log(f"   - Valid Pixels : {px_full:,}")
    log(f"   - RMSE         : {rmse_full:.5f}")
    log(f"   - NMSE         : {nmse_full:.5f}")
    log(f"   - SSIM         : {ssim_avg:.5f}")
    log(f"   - PSNR         : {psnr_full:.2f} dB")

    log("-" * 60)
    log("Case 2: All Known Samples (Ground h1 + 20% UAV)")
    log(f"   - Valid Pixels : {px_known:,}")
    log(f"   - RMSE         : {rmse_known:.5f}")
    log(f"   - NMSE         : {nmse_known:.5f}")

    log("-" * 60)
    log("Case 3: Unknown Blind Regions (~80%)")
    log(f"   - Valid Pixels : {px_unknown:,}")
    log(f"   - RMSE         : {rmse_unknown:.5f}")
    log(f"   - NMSE         : {nmse_unknown:.5f}")

    log("-" * 60)
    log("Case 4: 20% UAV Trajectory Only")
    log(f"   - Valid Pixels : {px_uav:,}")
    log(f"   - RMSE         : {rmse_uav:.5f}")
    log(f"   - NMSE         : {nmse_uav:.5f}")
    log("=" * 60)

    # Generate and save visualization
    if vis_data is not None:
        log("-> Rendering 3D comparison plot...")
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_volume(ax1, vis_data['hint'], 'Input: 5% Sparse Hint', drop_rate=0.0)
        ax2 = fig.add_subplot(132, projection='3d')
        plot_3d_volume(ax2, vis_data['pred'], 'Prediction: PIRM3D', drop_rate=0.90)
        ax3 = fig.add_subplot(133, projection='3d')
        sc = plot_3d_volume(ax3, vis_data['gt'], 'Ground Truth (100%)', drop_rate=0.90)
        plt.colorbar(sc, ax=[ax1, ax2, ax3], shrink=0.5, label='Signal Pathloss')
        plot_path = os.path.join(DATA_ROOT, 'test_inference_preview.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        log(f"-> Plot saved as {plot_path}")


if __name__ == '__main__':
    main()