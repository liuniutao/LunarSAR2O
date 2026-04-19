import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """PatchGAN判别器 - 用于CNN训练"""
    
    def __init__(self, input_channels, num_filters=64, n_layers=3, use_sigmoid=True):
        super().__init__()
        layers = []
        
        # 初始层
        layers.append(nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 中间层
        for i in range(1, n_layers):
            in_ch = num_filters * min(2**(i-1), 8)
            out_ch = num_filters * min(2**i, 8)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        # 保存所有中间层用于特征提取
        self.feature_layers = nn.ModuleList()
        # 倒数第二层
        in_ch = num_filters * min(2**(n_layers-1), 8)
        out_ch = num_filters * min(2**n_layers, 8)
        layers.extend([
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 输出层
        layers.append(nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        """提取中间层特征"""
        features = []
        for layer in self.feature_layers:
            x = layer(x)
            features.append(x)
        return features