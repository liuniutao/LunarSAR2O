# terrain_unet_ablation.py
# 包含10个不同结构的 TerrainUNet 网络用于结构消融实验

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.conv_blocks import ResidualConvBlock
from models.blocks.attention import SEBlock, MultiScaleAttention, CBAMBlock, GlobalContextBlock


# 普通Conv替代残差块
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# 上采样替代：插值 + Conv
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ========== 网络结构 ==========

# net1: 原始结构
class TerrainUNet_Full(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = ResidualConvBlock(input_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)

        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            SEBlock(base_channels * 4)
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


# net2: 移除注意力 SEBlock
class TerrainUNet_NoSE(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_channels = kwargs.get('base_channels', 64)
        self.middle = ResidualConvBlock(base_channels * 4, base_channels * 4)


# net3: 移除 skip connection
class TerrainUNet_NoSkip(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_channels = kwargs.get('base_channels', 64)
        self.dec3 = ResidualConvBlock(base_channels * 2, base_channels * 2)  # 256+128=384 -> 128
        self.dec2 = ResidualConvBlock(base_channels * 1, base_channels)  # 128+64=192 -> 64
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.middle(F.avg_pool2d(e3, 2))

        d3 = self.dec3(self.up3(m))
        d2 = self.dec2(self.up2(d3))
        d1 = self.up1(d2)
        return self.final(d1)


# net4: 无 middle block
class TerrainUNet_NoMiddle(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.middle = nn.Identity()


# net5: 所有残差块改为普通Conv
class TerrainUNet_BasicConv(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = BasicConvBlock(input_channels, base_channels)
        self.enc2 = BasicConvBlock(base_channels, base_channels * 2)
        self.enc3 = BasicConvBlock(base_channels * 2, base_channels * 4)
        self.middle = nn.Sequential(
            BasicConvBlock(base_channels * 4, base_channels * 4),            # 256 -> 256
            SEBlock(base_channels * 4),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec3 = BasicConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec2 = BasicConvBlock(base_channels * 2 + base_channels, base_channels)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.middle(F.avg_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.up1(d2)
        return self.final(d1)


# net6: 上采样改为 interpolate+Conv
class TerrainUNet_InterpUp(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_channels = kwargs.get('base_channels', 64)
        self.up3 = UpsampleBlock(base_channels * 4,base_channels * 2)
        self.up2 = UpsampleBlock(base_channels * 2, base_channels)
        self.up1 = UpsampleBlock(base_channels, base_channels // 2)


# net7: base_channels=32

class TerrainUNet_Channel32(TerrainUNet_Full):
    def __init__(self, **kwargs):
        kwargs['base_channels'] = 32
        super().__init__(**kwargs)


# net8: base_channels=128
class TerrainUNet_Channel128(TerrainUNet_Full):
    def __init__(self, **kwargs):
        kwargs['base_channels'] = 128
        super().__init__(**kwargs)


# net9: 输出Sigmoid
class TerrainUNet_Sigmoid(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_channels = kwargs.get('base_channels', 64)
        output_channels = kwargs.get('output_channels', 3)
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, kernel_size=3, padding=1),  # 32 -> 3
            nn.Sigmoid()
        )


# net10: 不归一化输出
class TerrainUNet_LinearOut(TerrainUNet_Full):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_channels = kwargs.get('base_channels', 64)
        output_channels = kwargs.get('output_channels', 3)
        self.final = nn.Conv2d(base_channels // 2, output_channels, kernel_size=3, padding=1) # 32 -> 3
        # 输出大于0
        self.final = nn.Sequential(
            self.final,
            nn.ReLU()
        )

# net11: Sigmoid+ch128
class TerrainUNet_SigmoidCh128(TerrainUNet_Full):
    def __init__(self, **kwargs):
        kwargs['base_channels'] = 128
        base_channels = kwargs.get('base_channels', 64)
        output_channels = kwargs.get('output_channels', 3)
        super().__init__(**kwargs)
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, kernel_size=3, padding=1),  # 32 -> 3
            nn.Sigmoid()
        )


'''# net12: 原始结构
class TerrainUNet_Full_SE(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = ResidualConvBlock(input_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)

        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            SE_LocalDetail(base_channels * 4)
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

# net13: 原始结构+CBAM替换SE
class TerrainUNet_Full_CBAM(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = ResidualConvBlock(input_channels, base_channels)
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)

        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 4, base_channels * 4),
            CBAM(base_channels * 4)
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
    
   ''' 
# net14: deep
class TerrainUNet_Full_Deep(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        # 编码部分
        self.enc1 = ResidualConvBlock(input_channels, base_channels)                # [B, 64, 256, 256]
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)             # [B, 128, 128, 128]
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)         # [B, 256, 64, 64]
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)         # [B, 512, 32, 32]

        # 中间瓶颈
        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 8, base_channels * 8),
            SEBlock(base_channels * 8)
        )

        # 解码部分
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)   # [B, 256, 64, 64]
        self.dec4 = ResidualConvBlock(base_channels * 12, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)   # [B, 128, 128, 128]
        self.dec3 = ResidualConvBlock(base_channels * 6, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)       # [B, 64, 256, 256]
        self.dec2 = ResidualConvBlock(base_channels * 3, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)      # [B, 32, 512, 512]
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)                           # [B, 64, 256, 256]
        e2 = self.enc2(F.avg_pool2d(e1, 2))         # [B, 128, 128, 128]
        e3 = self.enc3(F.avg_pool2d(e2, 2))         # [B, 256, 64, 64]
        e4 = self.enc4(F.avg_pool2d(e3, 2))         # [B, 512, 32, 32]
        m = self.middle(F.avg_pool2d(e4, 2))        # [B, 512, 16, 16]

        # 解码
        d4 = self.up4(m)                            # [B, 256, 32, 32]
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # [B, 768, 32, 32] -> 256

        d3 = self.up3(d4)                           # [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # [B, 384, 64, 64] -> 128

        d2 = self.up2(d3)                           # [B, 64, 128, 128]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 192, 128, 128] -> 64

        d1 = self.up1(d2)                           # [B, 32, 256, 256]
        out = self.final(d1)                        # [B, 3, 256, 256]
        return out
    

class TerrainUNet_Full_Deep(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        
        # 编码部分 - 5层
        self.enc1 = ResidualConvBlock(input_channels, base_channels)                # [B, 64, 256, 256]
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)             # [B, 128, 128, 128]
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)         # [B, 256, 64, 64]
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)         # [B, 512, 32, 32]
        self.enc5 = ResidualConvBlock(base_channels * 8, base_channels * 16)        # [B, 1024, 16, 16] - 新增第5层

        # 🔥 在第三层添加注意力机制
        self.attn3 = SEBlock(base_channels * 4)  # 或者使用其他注意力机制

        # 中间瓶颈
        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 16, base_channels * 16),              # [B, 1024, 8, 8]
            SEBlock(base_channels * 16)
        )

        # 解码部分 - 5层
        self.up5 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)  # [B, 512, 16, 16] - 新增上采样
        self.dec5 = ResidualConvBlock(base_channels * 16, base_channels * 8)        # [B, 1024, 16, 16] -> 512

        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)   # [B, 256, 32, 32]
        self.dec4 = ResidualConvBlock(base_channels * 8, base_channels * 4)         # [B, 512, 32, 32] -> 256

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)   # [B, 128, 64, 64]
        self.dec3 = ResidualConvBlock(base_channels * 4, base_channels * 2)         # [B, 256, 64, 64] -> 128

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)       # [B, 64, 128, 128]
        self.dec2 = ResidualConvBlock(base_channels * 2, base_channels)             # [B, 128, 128, 128] -> 64

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)      # [B, 32, 256, 256]
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)                           # [B, 64, 256, 256]
        e2 = self.enc2(F.avg_pool2d(e1, 2))         # [B, 128, 128, 128]
        e3 = self.enc3(F.avg_pool2d(e2, 2))         # [B, 256, 64, 64]
        
        # 🔥 在第三层应用注意力
        e3_attn = self.attn3(e3)                    # [B, 256, 64, 64]
        
        e4 = self.enc4(F.avg_pool2d(e3_attn, 2))    # [B, 512, 32, 32]
        e5 = self.enc5(F.avg_pool2d(e4, 2))         # [B, 1024, 16, 16] - 新增第5层
        
        # 中间瓶颈
        m = self.middle(F.avg_pool2d(e5, 2))        # [B, 1024, 8, 8]

        # 解码
        d5 = self.up5(m)                            # [B, 512, 16, 16]
        d5 = self.dec5(torch.cat([d5, e5], dim=1))  # [B, 1024, 16, 16] -> 512

        d4 = self.up4(d5)                           # [B, 256, 32, 32]
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # [B, 512, 32, 32] -> 256

        d3 = self.up3(d4)                           # [B, 128, 64, 64]
        d3 = self.dec3(torch.cat([d3, e3_attn], dim=1))  # 🔥 使用带注意力的特征 [B, 256, 64, 64] -> 128

        d2 = self.up2(d3)                           # [B, 64, 128, 128]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 128, 128, 128] -> 64

        d1 = self.up1(d2)                           # [B, 32, 256, 256]
        out = self.final(d1)                        # [B, 3, 256, 256]
        
        return out
    
class MultiScaleAttentionUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=32):
        super().__init__()
        
        # 编码部分 - 5层
        self.enc1 = ResidualConvBlock(input_channels, base_channels)                # [B, 64, 256, 256]
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)             # [B, 128, 128, 128]
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)         # [B, 256, 64, 64]
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)         # [B, 512, 32, 32]
        self.enc5 = ResidualConvBlock(base_channels * 8, base_channels * 16)        # [B, 1024, 16, 16]

        # 🔥 多尺度注意力策略
        self.detail_attn1 = CBAMBlock(base_channels)
        self.detail_attn2 = CBAMBlock(base_channels * 2)
        self.balance_attn3 = MultiScaleAttention(base_channels * 4)
        self.global_attn4 = SEBlock(base_channels * 8)
        self.global_attn5 = GlobalContextBlock(base_channels * 16)

        # 中间瓶颈
        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 16, base_channels * 16),
            GlobalContextBlock(base_channels * 16)
        )

        # 🔥 修复：解码器输入通道数要与拼接后的通道数匹配
        # 解码部分
        self.up5 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)  # [1024->512]
        self.dec5 = ResidualConvBlock(base_channels * 8 + base_channels * 16, base_channels * 8)  # 🔥 512+1024=1536->512
        self.dec5_attn = CBAMBlock(base_channels * 8)

        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)   # [512->256]
        self.dec4 = ResidualConvBlock(base_channels * 4 + base_channels * 8, base_channels * 4)   # 🔥 256+512=768->256
        self.dec4_attn = CBAMBlock(base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)   # [256->128]
        self.dec3 = ResidualConvBlock(base_channels * 2 + base_channels * 4, base_channels * 2)   # 🔥 128+256=384->128
        self.dec3_attn = MultiScaleAttention(base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)       # [128->64]
        self.dec2 = ResidualConvBlock(base_channels + base_channels * 2, base_channels)          # 🔥 64+128=192->64
        self.dec2_attn = CBAMBlock(base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)      # [64->32]
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码 + 注意力
        e1 = self.enc1(x)                                    # [B, 64, 256, 256]
        e1_attn = self.detail_attn1(e1)                      
        
        e2 = self.enc2(F.avg_pool2d(e1_attn, 2))             # [B, 128, 128, 128]
        e2_attn = self.detail_attn2(e2)                      
        
        e3 = self.enc3(F.avg_pool2d(e2_attn, 2))             # [B, 256, 64, 64]
        e3_attn = self.balance_attn3(e3)                     
        
        e4 = self.enc4(F.avg_pool2d(e3_attn, 2))             # [B, 512, 32, 32]
        e4_attn = self.global_attn4(e4)                      
        
        e5 = self.enc5(F.avg_pool2d(e4_attn, 2))             # [B, 1024, 16, 16]
        e5_attn = self.global_attn5(e5)                      
        
        # 中间瓶颈
        m = self.middle(F.avg_pool2d(e5_attn, 2))            # [B, 1024, 8, 8]

        # 解码 + 注意力
        d5 = self.up5(m)                                     # [B, 512, 16, 16]
        d5_cat = torch.cat([d5, e5_attn], dim=1)             # [B, 512+1024=1536, 16, 16]
        d5_conv = self.dec5(d5_cat)                          # [B, 512, 16, 16]
        d5_attn = self.dec5_attn(d5_conv)                    
        
        d4 = self.up4(d5_attn)                               # [B, 256, 32, 32]
        d4_cat = torch.cat([d4, e4_attn], dim=1)             # [B, 256+512=768, 32, 32]
        d4_conv = self.dec4(d4_cat)                          # [B, 256, 32, 32]
        d4_attn = self.dec4_attn(d4_conv)
        
        d3 = self.up3(d4_attn)                               # [B, 128, 64, 64]
        d3_cat = torch.cat([d3, e3_attn], dim=1)             # [B, 128+256=384, 64, 64]
        d3_conv = self.dec3(d3_cat)                          # [B, 128, 64, 64]
        d3_attn = self.dec3_attn(d3_conv)
        
        d2 = self.up2(d3_attn)                               # [B, 64, 128, 128]
        d2_cat = torch.cat([d2, e2_attn], dim=1)             # [B, 64+128=192, 128, 128]
        d2_conv = self.dec2(d2_cat)                          # [B, 64, 128, 128]
        d2_attn = self.dec2_attn(d2_conv)
        
        d1 = self.up1(d2_attn)                               # [B, 32, 256, 256]
        out = self.final(d1)                                 # [B, 3, 256, 256]
        
        return out
    
class EnhancedCBAMBlock(nn.Module):
    """增强的CBAM注意力，特别关注小目标"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x

class DetailEnhancementBranch(nn.Module):
    """高频细节增强分支"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, 3, padding=1)
        self.conv2 = nn.Conv2d(channels//2, channels//2, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 提取高频细节
        detail = self.conv1(x)
        detail = F.relu(detail)
        detail = self.conv2(detail)
        
        # 细节重要性权重
        attention_weights = self.attention(detail)
        enhanced_detail = detail * attention_weights
        
        return enhanced_detail

class UpsampleBlock1(nn.Module):
    """改进的上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x
class EnhancedMultiScaleAttentionUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, base_channels=64):
        super().__init__()
        
        # 增强浅层特征提取
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 编码部分
        self.enc1 = ResidualConvBlock(base_channels, base_channels)                # [B, 64, 256, 256]
        self.enc2 = ResidualConvBlock(base_channels, base_channels * 2)             # [B, 128, 128, 128]
        self.enc3 = ResidualConvBlock(base_channels * 2, base_channels * 4)         # [B, 256, 64, 64]
        self.enc4 = ResidualConvBlock(base_channels * 4, base_channels * 8)         # [B, 512, 32, 32]
        self.enc5 = ResidualConvBlock(base_channels * 8, base_channels * 16)        # [B, 1024, 16, 16]

        # 注意力机制
        self.detail_attn1 = EnhancedCBAMBlock(base_channels, reduction=2)
        self.detail_attn2 = EnhancedCBAMBlock(base_channels * 2, reduction=4)
        self.balance_attn3 = MultiScaleAttention(base_channels * 4)
        self.global_attn4 = SEBlock(base_channels * 8)
        self.global_attn5 = GlobalContextBlock(base_channels * 16)

        # 中间瓶颈
        self.middle = nn.Sequential(
            ResidualConvBlock(base_channels * 16, base_channels * 16),
            GlobalContextBlock(base_channels * 16)
        )
        
        # 细节分支
        self.detail_branch = DetailEnhancementBranch(base_channels)
        
        # 🔥 修复：正确设置解码器通道数
        # 解码部分
        self.up5 = UpsampleBlock1(base_channels * 16, base_channels * 8)  # 1024->512
        # d5_cat通道数: 512(up5输出) + 1024(e5_attn) = 1536
        self.dec5 = ResidualConvBlock(1536, base_channels * 8)  # 1536->512
        self.dec5_attn = EnhancedCBAMBlock(base_channels * 8)

        self.up4 = UpsampleBlock1(base_channels * 8, base_channels * 4)   # 512->256
        # d4_cat通道数: 256(up4输出) + 512(e4_attn) = 768
        self.dec4 = ResidualConvBlock(768, base_channels * 4)   # 768->256
        self.dec4_attn = EnhancedCBAMBlock(base_channels * 4)

        self.up3 = UpsampleBlock1(base_channels * 4, base_channels * 2)   # 256->128
        # d3_cat通道数: 128(up3输出) + 256(e3_attn) = 384
        self.dec3 = ResidualConvBlock(384, base_channels * 2)   # 384->128
        self.dec3_attn = MultiScaleAttention(base_channels * 2)

        self.up2 = UpsampleBlock1(base_channels * 2, base_channels)       # 128->64
        # d2_cat通道数: 64(up2输出) + 128(e2_attn) = 192
        self.dec2 = ResidualConvBlock(192, base_channels)       # 192->64
        self.dec2_attn = EnhancedCBAMBlock(base_channels)

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels + base_channels//2, base_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 浅层特征提取
        shallow_feat = self.shallow_conv(x)
        
        # 编码 + 注意力
        e1 = self.enc1(shallow_feat)
        e1_attn = self.detail_attn1(e1)
        
        e2 = self.enc2(F.avg_pool2d(e1_attn, 2))
        e2_attn = self.detail_attn2(e2)
        
        e3 = self.enc3(F.avg_pool2d(e2_attn, 2))
        e3_attn = self.balance_attn3(e3)
        
        e4 = self.enc4(F.avg_pool2d(e3_attn, 2))
        e4_attn = self.global_attn4(e4)
        
        e5 = self.enc5(F.avg_pool2d(e4_attn, 2))
        e5_attn = self.global_attn5(e5)
        
        # 中间瓶颈
        m = self.middle(F.avg_pool2d(e5_attn, 2))

        # 细节分支
        detail_feat = self.detail_branch(e1_attn)  # [B, 32, 256, 256]

        # 解码 + 注意力 + 跳跃连接
        d5 = self.up5(m)                                     # [B, 512, 16, 16]
        d5_cat = torch.cat([d5, e5_attn], dim=1)             # [B, 1536, 16, 16]
        d5_conv = self.dec5(d5_cat)                          # [B, 512, 16, 16]
        d5_attn = self.dec5_attn(d5_conv)
        
        d4 = self.up4(d5_attn)                               # [B, 256, 32, 32]
        d4_cat = torch.cat([d4, e4_attn], dim=1)             # [B, 768, 32, 32]
        d4_conv = self.dec4(d4_cat)                          # [B, 256, 32, 32]
        d4_attn = self.dec4_attn(d4_conv)
        
        d3 = self.up3(d4_attn)                               # [B, 128, 64, 64]
        d3_cat = torch.cat([d3, e3_attn], dim=1)             # [B, 384, 64, 64]
        d3_conv = self.dec3(d3_cat)                          # [B, 128, 64, 64]
        d3_attn = self.dec3_attn(d3_conv)
        
        d2 = self.up2(d3_attn)                               # [B, 64, 128, 128]
        d2_cat = torch.cat([d2, e2_attn], dim=1)             # [B, 192, 128, 128]
        d2_conv = self.dec2(d2_cat)                          # [B, 64, 128, 128]
        d2_attn = self.dec2_attn(d2_conv)
        
        # 上采样到原始分辨率并融合细节特征
        d1 = F.interpolate(d2_attn, scale_factor=2, mode='bilinear', align_corners=False)  # [B, 64, 256, 256]
        
        # 融合细节特征
        d1_enhanced = torch.cat([d1, detail_feat], dim=1)    # [B, 96, 256, 256]
        
        out = self.final_conv(d1_enhanced)                   # [B, 3, 256, 256]
        return out