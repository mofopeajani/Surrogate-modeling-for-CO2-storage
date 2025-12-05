## 3D Residual U-Net Architecture for CO2 surrogate modelling

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# -------------------------------------------------------------
#  Basic residual block (3D)
# -------------------------------------------------------------
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

        # Kaiming initialization
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if isinstance(self.skip, nn.Conv3d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity='relu')

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)
        return out



# -------------------------------------------------------------
#  Downsample block (encoder path)
# -------------------------------------------------------------
class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.res = ResBlock3D(out_ch, out_ch)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


# -------------------------------------------------------------
#  Upsample block (decoder path)
# -------------------------------------------------------------
class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # interpolate + 1×1×1 conv to reduce channels
        self.conv1x1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.res = ResBlock3D(out_ch * 2, out_ch)  # after concat with skip
        nn.init.kaiming_normal_(self.conv1x1.weight, nonlinearity='relu')
        if self.conv1x1.bias is not None:
            nn.init.zeros_(self.conv1x1.bias)

    def forward(self, x, skip):
        # Trilinear upsample
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = self.conv1x1(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


# -------------------------------------------------------------
#  Full 3D Residual U-Net
# -------------------------------------------------------------
class ResUNet3D(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, out_ch=2):
        super().__init__()

        # Encoder
        self.enc1 = ResBlock3D(in_ch, base_ch)          # 11 → 32
        self.down1 = DownBlock3D(base_ch, base_ch * 2)  # 32 → 64
        self.down2 = DownBlock3D(base_ch * 2, base_ch * 4)  # 64 → 128
        self.down3 = DownBlock3D(base_ch * 4, base_ch * 8)  # 128 → 256
        self.down4 = DownBlock3D(base_ch * 8, base_ch * 12) # 256 → 384 (bottleneck)

        # Decoder
        self.up1 = UpBlock3D(base_ch * 12, base_ch * 8)  # 384 → 256
        self.up2 = UpBlock3D(base_ch * 8,  base_ch * 4)  # 256 → 128
        self.up3 = UpBlock3D(base_ch * 4,  base_ch * 2)  # 128 → 64
        self.up4 = UpBlock3D(base_ch * 2,  base_ch)      # 64  → 32

        # Output heads
        self.head_p = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, 1, kernel_size=1)
        )
        self.head_s = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Weight initialization
        for m in [self.head_p[0], self.head_p[2], self.head_s[0], self.head_s[2]]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # 32
        e2 = self.down1(e1)      # 64
        e3 = self.down2(e2)      # 128
        e4 = self.down3(e3)      # 256
        b  = self.down4(e4)      # 384 bottleneck

        # Decoder
        d1 = self.up1(b, e4)     # 256
        d2 = self.up2(d1, e3)    # 128
        d3 = self.up3(d2, e2)    # 64
        d4 = self.up4(d3, e1)    # 32

        # Heads
        p = self.head_p(d4)
        s = self.head_s(d4)
        out = torch.cat([p, s], dim=1)  # (B,2,35,35,11)
        return out

C_in = 4   # number of input channels (set 7 if you include time)
C_out = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the network
SEED = 42
torch.manual_seed(SEED)
model = ResUNet3D(in_ch=C_in, base_ch=32, out_ch=C_out).to(device)