import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block with stride."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pw.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class FaceResidualBlock(nn.Module):
    """Residual unit for face feature extractor."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        res = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        return x + self.bn2(res)


class ForestFaceNet(nn.Module):
    """
    Custom 128-D Metric Learning Face Representation Network.
    Input: (B, 3, 112, 112) Aligned Face Crop.
    Output: (B, 128) L2-Normalized Unit Vector on the 128-D Hypersphere.
    Zero Pretrained Weights: Initialized from scratch via Kaiming Normal.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Input conv
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(32)
        self.input_act = nn.PReLU(32)

        # Stage 1: (32 -> 64, 56x56)
        self.stage1_down = DepthwiseSeparableConv(32, 64, stride=2)
        self.stage1_res = nn.Sequential(*[FaceResidualBlock(64) for _ in range(3)])

        # Stage 2: (64 -> 128, 28x28)
        self.stage2_down = DepthwiseSeparableConv(64, 128, stride=2)
        self.stage2_res = nn.Sequential(*[FaceResidualBlock(128) for _ in range(4)])

        # Stage 3: (128 -> 256, 14x14)
        self.stage3_down = DepthwiseSeparableConv(128, 256, stride=2)
        self.stage3_res = nn.Sequential(*[FaceResidualBlock(256) for _ in range(6)])

        # Stage 4: (256 -> 512, 7x7)
        self.stage4_down = DepthwiseSeparableConv(256, 512, stride=2)
        self.stage4_res = nn.Sequential(*[FaceResidualBlock(512) for _ in range(2)])

        # Global Pooling & 128-D Bottleneck
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

        self._init_fc_weights()

    def _init_fc_weights(self):
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        # x: (B, 3, 112, 112)
        x = self.input_act(self.input_bn(self.input_conv(x)))

        x = self.stage1_res(self.stage1_down(x))  # (B, 64, 56, 56)
        x = self.stage2_res(self.stage2_down(x))  # (B, 128, 28, 28)
        x = self.stage3_res(self.stage3_down(x))  # (B, 256, 14, 14)
        x = self.stage4_res(self.stage4_down(x))  # (B, 512, 7, 7)

        x = self.gap(x)                           # (B, 512, 1, 1)
        x = torch.flatten(x, 1)                   # (B, 512)

        x = self.bn_fc(self.fc(x))                # (B, 128)

        # L2 Unit Normalization: f(x) = x / ||x||_2
        embedding = F.normalize(x, p=2, dim=1)
        return embedding
