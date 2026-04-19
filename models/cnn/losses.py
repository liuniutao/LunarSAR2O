import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import calculate_psnr, calculate_ssim
class GANLoss(nn.Module):
    """GAN损失函数，支持多种模式"""
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
def total_variation_loss(x):
    """TV loss: encourage spatial smoothness, suppress noise/blur"""
    # 作用是鼓励空间平滑性，抑制噪声/模糊
    # 原理是计算图像梯度差异，鼓励相邻像素值相似
    # 计算水平和垂直方向的梯度差异
    # 会不会使图像过于平滑，导致模糊？
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss

def histogram_loss(gen_img, target_img, bins=256):
    """Simple histogram matching loss"""
    gen = (gen_img * 0.5 + 0.5).clamp(0, 1)
    tgt = (target_img * 0.5 + 0.5).clamp(0, 1)
    loss = 0.0
    for c in range(gen.shape[1]):
        gen_hist = torch.histc(gen[:, c], bins=bins, min=0.0, max=1.0)
        tgt_hist = torch.histc(tgt[:, c], bins=bins, min=0.0, max=1.0)
        gen_hist = gen_hist / (gen_hist.sum() + 1e-6)
        tgt_hist = tgt_hist / (tgt_hist.sum() + 1e-6)
        loss += torch.mean((gen_hist - tgt_hist) ** 2)
    return loss


class PerceptualLoss(nn.Module):
    """感知损失 - 基于VGG特征，增加颜色约束"""
    
    def __init__(self, layer_weights={'relu1_2': 0.5, 'relu2_2': 0.5, 'relu3_3': 1.0}):
        super().__init__()
        from torchvision.models import vgg16
        self.vgg = VGG16Features().eval()
        self.layer_weights = layer_weights
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 降低内容损失的权重
        self.content_loss = ContentLoss()
    
    def forward(self, pred, target):
        # 归一化到VGG输入范围
        pred_features = self.vgg((pred + 1) * 0.5)
        target_features = self.vgg((target + 1) * 0.5)
        
        loss = 0
        for layer in self.layer_weights:
            # 使用 L1 损失减少过平滑
            loss += F.l1_loss(pred_features[layer], target_features[layer]) * self.layer_weights[layer]
        
        # 降低内容损失的权重
        content_loss = self.content_loss(pred, target)
        return loss + content_loss * 0.1  # 降低内容损失权重
class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # 计算结构相似性损失
        ssim_loss = 1 - calculate_ssim(pred, target, size_average=True)
        
        # 计算梯度差异损失
        pred_grad = self.image_gradients(pred)
        target_grad = self.image_gradients(target)
        grad_loss = F.l1_loss(pred_grad, target_grad)
        
        return ssim_loss + grad_loss * 0.2  # 降低梯度损失权重
    
    def image_gradients(self, img):
        # 使用 Sobel 算子计算更准确的梯度
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        
        # 分别计算各通道的梯度
        grad_x = F.conv2d(img, sobel_x.repeat(img.size(1), 1, 1, 1), padding=1, groups=img.size(1))
        grad_y = F.conv2d(img, sobel_y.repeat(img.size(1), 1, 1, 1), padding=1, groups=img.size(1))
        
        return torch.cat([grad_x, grad_y], dim=1)
    
class VGG16Features(nn.Module):
    """提取VGG16中间层特征"""
    
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        for x in range(4):  # relu1_2
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  # relu2_2
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):  # relu3_3
            self.slice3.add_module(str(x), vgg[x])
    
    def forward(self, x):
        h = self.slice1(x)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        return {'relu1_2': relu1_2, 'relu2_2': relu2_2, 'relu3_3': relu3_3}