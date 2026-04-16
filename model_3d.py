import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['OMP_NUM_THREADS'] = '1'


class ResBlock3D(nn.Module):
    """ 3D Residual Building Block
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

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
    """ Anisotropic 3D Res-UNet for 3D Radio Map Construction
    Designed for flat 3D volumetric data (Z=20 << X,Y=256), preserves altitude resolution in shallow layers
    and applies isotropic downsampling in deep layers.

    Args:
        in_channels (int): Number of input channels, default 5 [Tx, Bldg, Depth, SparseHint, Mask]
        out_channels (int): Number of output channels, default 1 [Predicted 3D Pathloss]
        base_dim (int): Base channel dimension of the network, default 16
    """

    def __init__(self, in_channels=5, out_channels=1, base_dim=16):
        super().__init__()

        # --- Encoder Branch ---
        self.inc = ResBlock3D(in_channels, base_dim)

        # Shallow layers: only downsample on (X,Y) plane to preserve full altitude (Z) resolution
        self.down1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ResBlock3D(base_dim, base_dim * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            ResBlock3D(base_dim * 2, base_dim * 4)
        )

        # Deep layers: isotropic 3D downsampling after (X,Y) dimension is sufficiently reduced
        self.down3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ResBlock3D(base_dim * 4, base_dim * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ResBlock3D(base_dim * 8, base_dim * 16)
        )

        # --- Decoder Branch ---
        # Symmetric upsampling strictly mirroring the downsampling strategy
        self.up1 = nn.ConvTranspose3d(base_dim * 16, base_dim * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_up1 = ResBlock3D(base_dim * 16, base_dim * 8)

        self.up2 = nn.ConvTranspose3d(base_dim * 8, base_dim * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_up2 = ResBlock3D(base_dim * 8, base_dim * 4)

        self.up3 = nn.ConvTranspose3d(base_dim * 4, base_dim * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up3 = ResBlock3D(base_dim * 4, base_dim * 2)

        self.up4 = nn.ConvTranspose3d(base_dim * 2, base_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_up4 = ResBlock3D(base_dim * 2, base_dim)

        # --- Output Head ---
        # Linear logits output without Sigmoid, to avoid gradient vanishing in deep shadow and LoS regions
        self.outc = nn.Conv3d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward pass with skip connections
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
        """ Pad upsampled feature map to match the spatial size of the target skip connection feature map """
        diffZ = target.size()[2] - upsampled.size()[2]
        diffY = target.size()[3] - upsampled.size()[3]
        diffX = target.size()[4] - upsampled.size()[4]
        if diffZ > 0 or diffY > 0 or diffX > 0:
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2,
                                          diffZ // 2, diffZ - diffZ // 2])
        return upsampled


if __name__ == "__main__":
    print("Initializing Anisotropic 3D Res-UNet Model...")
    model = ResUNet3D(in_channels=5, out_channels=1, base_dim=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    print("Running forward pass test...")
    dummy_input = torch.randn(2, 5, 20, 256, 256).to(device)

    with torch.no_grad():
        out = model(dummy_input)

    print("Forward pass completed successfully.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape}")