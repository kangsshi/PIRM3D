import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from dataset_3d import UrbanRadio3D_Dataset
from model_3d import ResUNet3D

os.environ["OMP_NUM_THREADS"] = "1"


# ==========================================
# Core Loss Function: Physics-Informed Loss
# ==========================================
class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for 3D Radio Map Construction
    1. Known Region: MSE with ground truth
    2. Unknown Region:
       - Slice-wise Shift-Invariant MSE (SI-MSE)
       - Spatial Gradient Matching Loss
    """

    def __init__(self, lambda_structure=0.1):
        super().__init__()
        self.lambda_structure = lambda_structure

    def forward(self, logits, gt, y2, loss_mask, bldg_mask):
        pred = torch.sigmoid(logits)
        unknown_mask = (~loss_mask) & (~bldg_mask)

        # --- 1. MSE Loss on Known Regions ---
        if loss_mask.sum() > 0:
            mse_loss = F.mse_loss(pred[loss_mask], gt[loss_mask])
        else:
            mse_loss = torch.tensor(0.0, device=logits.device)

        # --- 2. Shift-Invariant MSE (SI-MSE) on Unknown Regions ---
        diff = pred - y2
        diff_masked = diff * unknown_mask

        count_per_slice = unknown_mask.sum(dim=(2, 3))
        valid_slices = count_per_slice > 50

        if valid_slices.sum() > 0:
            mean_per_slice = diff_masked.sum(dim=(2, 3)) / (count_per_slice + 1e-8)
            diff_centered = (diff - mean_per_slice.unsqueeze(-1).unsqueeze(-1)) * unknown_mask
            var_per_slice = (diff_centered ** 2).sum(dim=(2, 3)) / (count_per_slice + 1e-8)
            si_mse_loss = var_per_slice[valid_slices].mean()
        else:
            si_mse_loss = torch.tensor(0.0, device=logits.device)

        # --- 3. Spatial Gradient Matching Loss ---
        grad_p_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_t_x = torch.abs(y2[:, :, :, 1:] - y2[:, :, :, :-1])

        grad_p_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_t_y = torch.abs(y2[:, :, 1:, :] - y2[:, :, :-1, :])

        mask_x = unknown_mask[:, :, :, 1:] & unknown_mask[:, :, :, :-1]
        mask_y = unknown_mask[:, :, 1:, :] & unknown_mask[:, :, :-1, :]

        grad_loss_x = F.mse_loss(grad_p_x[mask_x], grad_t_x[mask_x]) if mask_x.sum() > 0 else 0.0
        grad_loss_y = F.mse_loss(grad_p_y[mask_y], grad_t_y[mask_y]) if mask_y.sum() > 0 else 0.0

        grad_loss = grad_loss_x + grad_loss_y

        # --- Total Loss ---
        structure_loss = si_mse_loss + grad_loss
        total_loss = mse_loss + self.lambda_structure * structure_loss

        return total_loss, mse_loss, si_mse_loss, grad_loss


def atomic_save(checkpoint_data, save_path):
    """ Save checkpoint atomically to avoid corruption """
    temp_path = save_path + ".tmp"
    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, save_path)


def main():
    # Configuration
    DATA_ROOT = '/root/autodl-tmp/3DRadioMap'
    SAVE_DIR = os.path.join(DATA_ROOT, 'Checkpoints_PIRM3D')
    os.makedirs(SAVE_DIR, exist_ok=True)

    LOG_PATH = os.path.join(SAVE_DIR, 'train.log')
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 2e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simple logging setup
    def log(message):
        print(message)
        with open(LOG_PATH, 'a') as f:
            f.write(message + '\n')

    log("=" * 50)
    log("[START] Training PIRM3D: Physics-Inspired 3D Radio Map Construction")
    log("=" * 50)

    # Load Datasets
    log("-> Loading Datasets...")
    train_dataset = UrbanRadio3D_Dataset(DATA_ROOT, split='train')
    test_dataset = UrbanRadio3D_Dataset(DATA_ROOT, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # Initialize Model, Loss, Optimizer
    model = ResUNet3D(in_channels=5, out_channels=1, base_dim=16).to(device)
    criterion = PhysicsInformedLoss(lambda_structure=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Checkpoint Resume
    start_epoch = 1
    best_test_rmse = float('inf')
    latest_ckpt_path = os.path.join(SAVE_DIR, "latest_model.pth")

    if os.path.exists(latest_ckpt_path):
        log(f"-> Found checkpoint at {latest_ckpt_path}, resuming...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_rmse = checkpoint.get('test_rmse', float('inf'))
        log(f"-> Resumed successfully. Best Test RMSE: {best_test_rmse:.5f}.")

    # Training Loop
    log("-> Starting Training Loop...")
    for epoch in range(start_epoch, EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        print(f"\nTrain Epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            gt = batch['gt'].to(device)
            y2 = batch['y2'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            bldg_mask = (x[:, 1, :, :, :] > 0.5)

            # Forward pass
            logits = model(x).squeeze(1)
            loss, mse_err, simse_err, grad_err = criterion(logits, gt, y2, loss_mask, bldg_mask)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{len(train_loader)} | "
                      f"Tot: {loss.item():.3f} | "
                      f"MSE: {mse_err.item():.4f} | "
                      f"SI: {simse_err.item():.4f} | "
                      f"Grad: {grad_err.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # Test
        model.eval()
        test_sq_err = 0.0
        test_pixels = 0

        print(f"\nTest Epoch {epoch}")
        with torch.no_grad():
            for v_step, v_batch in enumerate(test_loader):
                vx = v_batch['x'].to(device)
                vgt = v_batch['gt'].to(device)
                vbldg = (vx[:, 1, :, :, :] == 1.0)

                vair = ~vbldg
                vlogits = model(vx).squeeze(1)
                vpred = torch.sigmoid(vlogits)
                vpred[vbldg] = 0.0

                if vair.sum() > 0:
                    test_sq_err += torch.sum((vpred[vair] - vgt[vair]) ** 2).item()
                    test_pixels += vair.sum().item()

        test_rmse = np.sqrt(test_sq_err / max(test_pixels, 1))

        # Logging
        log_message = (f"Epoch [{epoch}/{EPOCHS}] | "
                       f"Train Loss: {avg_train_loss:.4f} | "
                       f"Test RMSE: {test_rmse:.5f}")
        log(log_message)

        # Save Checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_rmse': test_rmse
        }

        atomic_save(checkpoint_data, latest_ckpt_path)

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            atomic_save(checkpoint_data, os.path.join(SAVE_DIR, "best_model.pth"))
            log(f"*** New Best Model Saved (Test RMSE: {best_test_rmse:.5f}) ***")


if __name__ == '__main__':
    main()