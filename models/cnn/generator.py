import torch
import torch.nn as nn
import torch.nn.functional as F

# generator.py - 增加残差连接和实例归一化
class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, num_filters=64, use_attention=True):
        super().__init__()
        # 下采样
        self.down1 = self.down_block(input_channels, num_filters, normalization=False)
        self.down2 = self.down_block(num_filters, num_filters * 2)  # num_filters的作用是什么？
        # num_filters的作用是什么？
        # num_filters用于控制每一层的卷积核数量，影响特征提取能力
        # num_filters * 2用于增加特征图的深度，增强模型的表达能力
        # num_filters 大小有什么影响
        # num_filters的大小直接影响模型的容量和计算复杂度。较大的num_filters可以捕捉更多的特征，但也会增加计算量和内存消耗。
        # 在256*256的地图图像上，通常使用64或128作为num_filters的初始值是合适的，这样可以在保持较低计算成本的同时，提取足够的特征。
        # 所以这个网络的num_filters通常设置为64或128，这样可以在256x256的图像上有效地提取特征，同时保持计算效率。

        self.down3 = self.down_block(num_filters * 2, num_filters * 4)
        self.down4 = self.down_block(num_filters * 4, num_filters * 8)
        
        # 瓶颈层 - 增加残差块
        self.bottleneck = nn.Sequential(
            ResnetBlock(num_filters * 8),
            ResnetBlock(num_filters * 8)
        )
        # 这个瓶颈层的作用是什么？
        # 瓶颈层的作用是提取更深层次的特征，并通过残差连接增强梯度流动，防止梯度消失。
        # 通过残差连接，瓶颈层可以更好地捕捉复杂的特征，同时保持模型的稳定性和训练效率。
        # 是如何连接的？
        # 瓶颈层通过残差连接将输入特征与经过卷积处理的特征相加，形成一个跳跃连接。
        # 这种连接方式允许模型在学习过程中保留输入特征，同时学习更深层次的特征表示。
        # 此处的瓶颈层使用了两个残差块，每个块包含两个卷积层和一个跳跃连接。每个块包含两个卷积层和一个跳跃连接是怎么体现的？
        # 每个残差块包含两个卷积层和一个跳跃连接，具体实现如下：
        # 1. 第一个卷积层对输入特征进行卷积操作，提取初始特征。
        # 2. 第二个卷积层对第一个卷积层的输出进行卷积操作，进一步提取特征。
        # 3. 最后，将输入特征与第二个卷积层的输出相加，形成跳跃连接。

        # 上采样
        self.up4 = self.up_block(num_filters * 16, num_filters * 4)
        self.up3 = self.up_block(num_filters * 8, num_filters * 2)
        self.up2 = self.up_block(num_filters * 4, num_filters)
        self.up1 = self.up_block(num_filters * 2, num_filters)
        
        # 输出层 - 确保使用 tanh 激活
        self.output = nn.Sequential(
            nn.Conv2d(num_filters, output_channels, kernel_size=1),
            nn.Tanh()  # 明确添加 tanh 激活确保输出范围在 [-1, 1]
        )
        
    def down_block(self, in_channels, out_channels, normalization=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):    # 输入x的大小是 [B, 1, 256, 256]
        # 下采样路径
        d1 = self.down1(x)   # d1 的大小是 [B, 64, 128, 128]，因为num_filters=64。那256，256的图像经过卷积和池化后，大小变为128, 128
        # 256，256的图像经过卷积和池化后，大小变为128, 128，是如何计算的？
        # 计算方式是：输入大小除以2（因为stride=2），padding=1，kernel_size=4
        # 所以输出大小是 (输入大小 - kernel_size + 2 * padding) / stride + 1
        # 例如：256 - 4 + 2 * 1 = 254，254 / 2 + 1 = 128；
        # kernel_size 决定了卷积操作的感受野大小，较大的kernel_size可以捕捉更大范围的特征，但也会增加计算量.
        # 在256×256的地图图像上，通常使用kernel_size=4是合适的，这样可以在保持较低计算成本的同时，提取足够的特征。
        # stride 决定了特征图的下采样倍数，较大的stride会使特征图尺寸减小得更快
        d2 = self.down2(d1)  # d2 的大小是 [B, 128, 64, 64]
        d3 = self.down3(d2)  # d3 的大小是 [B, 256, 32, 32]
        d4 = self.down4(d3)  # d4 的大小是 [B, 512, 16, 16]
        # Batch size B 是指一次处理的图像数量，通常在训练时设置为一个较小的值（如16或32），以便在内存允许的范围内进行批处理。
        # 在推理时，可以将多个图像合并为一个批次进行处理，以提高效率。
        # 瓶颈层
        bottleneck = self.bottleneck(d4) #  # bottleneck 的大小是 [B, 512, 16, 16]
        #
        # 上采样路径
        u4 = self.up4(torch.cat([bottleneck, d4], dim=1)) # u4 的大小是 [B, 256, 32, 32]
        # 为什么要使用torch.cat([bottleneck, d4], dim=1)？
        # 使用torch.cat([bottleneck, d4], dim=1)是为了将瓶颈层的特征与上采样路径的特征进行连接，形成跳跃连接。
        # 这种连接方式允许模型在上采样过程中保留更深层次的特征，同时结合更高分辨率的特征。
        # 是如何concat的？
        # torch.cat([bottleneck, d4], dim=1)将瓶颈层的特征和上采样路径的特征在通道维度上进行连接。
        # 这意味着将两个特征图的通道数相加，形成一个新的特征图。
        # 例如，如果bottleneck的大小是[B, 512, 16, 16]，d4的大小是[B, 512, 16, 16]，那么连接后的特征图大小将是[B, 1024, 16, 16]。
        # 那为什么u4 的大小是 [B, 256, 32, 32]？
        # u4的大小是[B, 256, 32, 32]是因为上采样操作将特征图的空间尺寸从16x16扩大到32x32，同时通道数从1024减少到256。
        u3 = self.up3(torch.cat([u4, d3], dim=1))
        u2 = self.up2(torch.cat([u3, d2], dim=1))
        u1 = self.up1(torch.cat([u2, d1], dim=1))
        
        return self.output(u1)  # 直接返回 output 的结果（已包含 tanh）
        # 这个网络的输入输出，也就是网络结构是 U-Net
        # 输入x的大小是 [B, 1, 256, 256]，输出的大小是 [B, 3, 256, 256]
        
# 新增残差块
class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


# discriminator.py - 使用谱归一化稳定训练
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, n_layers=3):
        super().__init__()
        layers = []
        
        # 第一层不使用归一化
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 中间层
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(num_filters * mult_prev, num_filters * mult, 
                         kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.InstanceNorm2d(num_filters * mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 最后一层
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers.append(nn.utils.spectral_norm(
            nn.Conv2d(num_filters * mult_prev, num_filters * mult, 
                     kernel_size=4, stride=1, padding=1)
        ))
        layers.append(nn.InstanceNorm2d(num_filters * mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 输出层
        layers.append(nn.Conv2d(num_filters * mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class SelfAttentionBlock(nn.Module):
    """自注意力机制模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, height, width = x.size()
        
        # 计算查询、键、值
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        value = self.value(x).view(batch, -1, height * width)
        
        # 注意力图
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        
        # 加权和
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, height, width)
        
        return self.gamma * out + x