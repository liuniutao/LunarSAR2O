# LunarS2OUNet.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.conv_blocks import ResidualConvBlock



# ========== 网络结构 ==========

class LGAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(LGAM, self).__init__()
        
        # ---- 全局 SE 分支 ----
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc_global = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        
        # ---- 局部细节分支 ----
        self.local_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel, bias=False),  # depthwise conv
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)  # pointwise conv
        )
        self.fc_local = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # === 全局分支 ===
        y_global = self.global_pool(x).view(b, c)  
        y_global = self.fc_global(y_global).view(b, c, 1, 1)
        
        # === 局部分支 ===
        y_local = self.local_conv(x)
        y_local = self.fc_local(y_local)  
        
        # === 融合并激活 ===
        y = self.sigmoid(y_global + y_local)
        
        return x * y

class LunarS2OUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = ResidualConvBlock(input_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)

        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            LGAM(base_channels * 4)
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec3 = ResidualConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = ResidualConvBlock(base_channels * 2 + base_channels, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 编码阶段
        e1 = self.enc1(x)                                # [B, 64, 256, 256]
        e2 = self.enc2(F.avg_pool2d(e1, 2))              # [B, 128, 128, 128]
        e3 = self.enc3(F.avg_pool2d(e2, 2))              # [B, 256, 64, 64]
        m = self.middle(F.avg_pool2d(e3, 2))             # [B, 256, 32, 32]

        # 解码阶段
        d3 = self.up3(m)                                 # [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))       # [B, 384, 64, 64] -> 128

        d2 = self.up2(d3)                                # [B, 64, 128, 128]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))       # [B, 192, 128, 128] -> 64

        d1 = self.up1(d2)                                # [B, 32, 256, 256]
        out = self.final(d1)                             # [B, 3, 256, 256]
        return out
    



