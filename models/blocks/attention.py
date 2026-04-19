
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    """多尺度注意力 - 同时关注局部细节和全局上下文"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 局部注意力（小核） - 关注陨石坑细节
        self.local_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 全局注意力（大核+全局池化） - 关注地形变化
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 关注重要空间位置
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.weights = nn.Parameter(torch.ones(3))  # 可学习的权重

    def forward(self, x):
        # 局部注意力
        local_weight = self.local_attn(x)
        
        # 全局注意力
        global_weight = self.global_attn(x)
        global_weight = F.interpolate(global_weight, size=x.shape[2:], mode='nearest')
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attn(spatial_input)
        
        # 加权融合
        weights = F.softmax(self.weights, dim=0)
        combined_attn = (weights[0] * local_weight + 
                        weights[1] * global_weight + 
                        weights[2] * spatial_weight)
        
        return x * combined_attn

class CBAMBlock(nn.Module):
    """CBAM注意力 - 适合细节感知"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力 - 使用大核捕捉局部模式
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 7x7卷积核，适合陨石坑大小
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

class GlobalContextBlock(nn.Module):
    """全局上下文注意力 - 适合地形感知"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.context_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.LayerNorm([channels // 4, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        context_weight = self.context_transform(x)
        context_weight = F.interpolate(context_weight, size=x.shape[2:], mode='nearest')
        return x * context_weight