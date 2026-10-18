import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU with Kaiming Normal weight initialization."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Standard ResBlock with bottleneck connections."""
    def __init__(self, channels):
        super().__init__()
        self.cv1 = ConvBlock(channels, channels // 2, k=1, s=1, p=0)
        self.cv2 = ConvBlock(channels // 2, channels, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class ForestPersonNet(nn.Module):
    """
    Custom Human Detector for Wildland & SAR Environments.
    Input: (B, 3, 640, 640) Image Tensor.
    Zero Pretrained Weights: Initialized from scratch via Kaiming Normal.
    """
    def __init__(self, num_classes=1, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_channels_per_head = num_anchors * (5 + num_classes)  # [tx, ty, tw, th, conf, prob_person]

        # Stage 1: 640 -> 320
        self.stage1 = ConvBlock(3, 32, k=3, s=2, p=1)

        # Stage 2: 320 -> 160
        self.stage2_res = nn.Sequential(*[ResidualBlock(32) for _ in range(2)])
        self.stage2_down = ConvBlock(32, 64, k=3, s=2, p=1)

        # Stage 3: 160 -> 80 (P3)
        self.stage3_res = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.stage3_down = ConvBlock(64, 128, k=3, s=2, p=1)

        # Stage 4: 80 -> 40 (P4)
        self.stage4_res = nn.Sequential(*[ResidualBlock(128) for _ in range(6)])
        self.stage4_down = ConvBlock(128, 256, k=3, s=2, p=1)

        # Stage 5: 40 -> 20 (P5)
        self.stage5_res = nn.Sequential(*[ResidualBlock(256) for _ in range(4)])
        self.stage5_down = ConvBlock(256, 512, k=3, s=2, p=1)

        # FPN / PAN Neck
        # Top-down pathway: P5 -> P4 -> P3
        self.lateral_p5 = ConvBlock(512, 256, k=1, s=1, p=0)
        self.lateral_p4 = ConvBlock(256, 128, k=1, s=1, p=0)

        self.smooth_p4 = ConvBlock(256, 256, k=3, s=1, p=1)
        self.smooth_p3 = ConvBlock(128, 128, k=3, s=1, p=1)

        # Multi-Scale Detection Heads
        # Head P3: 80x80 (Distant / Occluded persons)
        self.head_p3 = nn.Conv2d(128, self.out_channels_per_head, kernel_size=1)
        # Head P4: 40x40 (Medium-range persons)
        self.head_p4 = nn.Conv2d(256, self.out_channels_per_head, kernel_size=1)
        # Head P5: 20x20 (Close / Large persons)
        self.head_p5 = nn.Conv2d(512, self.out_channels_per_head, kernel_size=1)

        self._init_head_weights()

    def _init_head_weights(self):
        for m in [self.head_p3, self.head_p4, self.head_p5]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Backbone Forward Pass
        x = self.stage1(x)               # (B, 32, 320, 320)
        x = self.stage2_res(x)
        x = self.stage2_down(x)          # (B, 64, 160, 160)

        x = self.stage3_res(x)
        p3 = self.stage3_down(x)         # (B, 128, 80, 80)

        x = self.stage4_res(p3)
        p4 = self.stage4_down(x)         # (B, 256, 40, 40)

        x = self.stage5_res(p4)
        p5 = self.stage5_down(x)         # (B, 512, 20, 20)

        # Neck Aggregation
        p5_lat = self.lateral_p5(p5)     # (B, 256, 20, 20)
        p5_upsample = F.interpolate(p5_lat, size=p4.shape[2:], mode="nearest")
        fused_p4 = self.smooth_p4(p4 + p5_upsample)  # (B, 256, 40, 40)

        p4_lat = self.lateral_p4(fused_p4) # (B, 128, 40, 40)
        p4_upsample = F.interpolate(p4_lat, size=p3.shape[2:], mode="nearest")
        fused_p3 = self.smooth_p3(p3 + p4_upsample)  # (B, 128, 80, 80)

        # Multi-scale Head Outputs
        out_p3 = self.head_p3(fused_p3)  # (B, 18, 80, 80)
        out_p4 = self.head_p4(fused_p4)  # (B, 18, 40, 40)
        out_p5 = self.head_p5(p5)        # (B, 18, 20, 20)

        return {
            "p3": out_p3,  # 80x80
            "p4": out_p4,  # 40x40
            "p5": out_p5   # 20x20
        }
