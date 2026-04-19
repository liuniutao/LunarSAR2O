# models/blocks/conv_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """
    多尺度残差卷积块，带 dilation 可调，用于特征提取 + 地形保持
    """
    # dilation作用
    # dilation的作用是增加卷积核的感受野，使模型能够捕捉更大范围的上下文信息。
    # 通过在卷积操作中引入空洞（dilation），可以在保持特征图分辨率的同时，扩大感受野。
    # 这对于处理具有不同尺度特征的图像非常有用，尤其是在地形保持任务中，可以更好地捕捉地形的细节和结构。
    # dilation的大小
    # dilation的大小直接影响感受野的大小。较大的dilation值可以捕捉更大范围的上下文信息，但也会增加计算复杂度。
    # 在地形保持任务中，通常使用dilation=1或2是合适的，这样可以在保持较高分辨率的同时，提取足够的上下文信息。
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv_block(x)
        skip = self.skip_connection(x)
        return F.relu(out + skip)
