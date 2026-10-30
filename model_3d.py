"""Anisotropic 3D Res-UNet backbone; outputs logits. Architecture unchanged."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm3d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResUNet3D(nn.Module):

    def __init__(self, in_channels=5, out_channels=1, base_dim=16):
        """
        in_channels: 5 [Tx, Bldg, Depth, SparseHint, Mask]
        out_channels: 1 [Predicted 3D Pathloss]
        """
        super().__init__()
        self.inc = ResBlock3D(in_channels, base_dim)
        self.down1 = nn.Sequential(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), ResBlock3D(base_dim, base_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), ResBlock3D(base_dim * 2, base_dim * 4))
        self.down3 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)), ResBlock3D(base_dim * 4, base_dim * 8))
        self.down4 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)), ResBlock3D(base_dim * 8, base_dim * 16))
        self.up1 = nn.ConvTranspose3d(base_dim * 16, base_dim * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_up1 = ResBlock3D(base_dim * 16, base_dim * 8)
        self.up2 = nn.ConvTranspose3d(base_dim * 8, base_dim * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_up2 = ResBlock3D(base_dim * 8, base_dim * 4)
        self.up3 = nn.ConvTranspose3d(base_dim * 4, base_dim * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up3 = ResBlock3D(base_dim * 4, base_dim * 2)
        self.up4 = nn.ConvTranspose3d(base_dim * 2, base_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up4 = ResBlock3D(base_dim * 2, base_dim)
        self.outc = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5)
        u1 = self._pad_if_needed(u1, x4)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = self._pad_if_needed(u2, x3)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv_up2(u2)
        u3 = self.up3(u2)
        u3 = self._pad_if_needed(u3, x2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv_up3(u3)
        u4 = self.up4(u3)
        u4 = self._pad_if_needed(u4, x1)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv_up4(u4)
        out = self.outc(u4)
        return out

    def _pad_if_needed(self, upsampled, target):
        diffZ = target.size()[2] - upsampled.size()[2]
        diffY = target.size()[3] - upsampled.size()[3]
        diffX = target.size()[4] - upsampled.size()[4]
        if diffZ > 0 or diffY > 0 or diffX > 0:
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        return upsampled
