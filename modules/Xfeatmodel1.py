import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):
    """
    Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
            # print(f"Shape after normalization: {x.shape}")
            
            return self.layer(x)
    	
        

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        """
        channels: 输入特征图的通道数
        reduction_ratio: 压缩比例，用于减少全连接层的参数量
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),  # 压缩通道
            nn.GELU(),
            nn.Linear(channels // reduction_ratio, channels),  # 恢复通道
            nn.Sigmoid()  # 输出权重，范围在 [0, 1]
        )
        
		

    def forward(self, x):
        """
        x: 输入特征图，形状为 (B, C, H, W)
        返回：加权后的特征图，形状与输入相同
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class XFeatModel(nn.Module):
    """
    Implementation of architecture described in
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(12)

        # CNN Backbone & Heads
        self.skip1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(12, 24, 1, stride=1, padding=0)
        )

        self.block1 = nn.Sequential(
            BasicLayer(12, 16, stride=1),
            BasicLayer(16, 16, stride=1),
            BasicLayer(16, 24, stride=2),
            BasicLayer(24, 24, stride=1),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )

        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer(64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 64, 1, padding=0),
        )

        self.block_fusion = nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d(64, 64, 1, padding=0)
        )

        # 添加通道注意力机制
        self.channel_attention = ChannelAttention(channels=64, reduction_ratio=8)

    def forward(self, x):
        """
        input:
            x -> torch.Tensor(B, C, H, W) grayscale or rgb images
        return:
            feats -> torch.Tensor(B, 64, H/8, W/8) dense local features
        """
        # 预处理：将一维信号转换为二维图像
        if x.dim() == 3:  # (B, 1, seq_len)
            x = x.unsqueeze(-1)  # (B, 1, seq_len, 1)

        # 归一化
        with torch.no_grad():
            x = self.norm(x)
            # print(f"Shape after normalization: {x.shape}")


        # 主干网络
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        # 金字塔融合
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion(x3 + x4 + x5)

        # 应用通道注意力机制
        feats = self.channel_attention(feats)

        return feats