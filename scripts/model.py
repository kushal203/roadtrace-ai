"""
RoadTrace AI - Lightweight U-Net for Road Segmentation
Architecture: U-Net with MobileNetV2 encoder (pretrained on ImageNet)
Output: 3-class segmentation (background, direct drivable, alternative drivable)
Target resolution: 256x256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling + skip connection + double conv."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches from encoder
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RoadSegmentationNet(nn.Module):
    """
    Lightweight U-Net with MobileNetV2 encoder for road segmentation.
    
    Input: RGB image [B, 3, 256, 256]
    Output: Class logits [B, 3, 256, 256]
      - Class 0: Background (non-drivable)
      - Class 1: Direct drivable area
      - Class 2: Alternative drivable area
    """
    
    NUM_CLASSES = 3

    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load MobileNetV2 as encoder
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights).features
        
        # Extract encoder stages (feature maps at different scales)
        # MobileNetV2 features indices and their output channels:
        # [0]:   stride 2, 32ch   (conv + bn + relu)
        # [1]:   stride 1, 16ch   (inverted residual)
        # [2-3]: stride 2, 24ch
        # [4-6]: stride 2, 32ch
        # [7-13]: stride 2, 96ch
        # [14-17]: stride 2, 320ch -> 1280ch
        self.enc1 = backbone[0:2]    # 32 -> 16ch, stride 2
        self.enc2 = backbone[2:4]    # 24ch, stride 4
        self.enc3 = backbone[4:7]    # 32ch, stride 8
        self.enc4 = backbone[7:14]   # 96ch, stride 16
        self.enc5 = backbone[14:18]  # 320ch, stride 32
        
        # Lightweight decoder
        self.up4 = UpBlock(320, 96, 128)
        self.up3 = UpBlock(128, 32, 64)
        self.up2 = UpBlock(64, 24, 32)
        self.up1 = UpBlock(32, 16, 16)
        
        # Final upsampling to original resolution (256x256)
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, self.NUM_CLASSES, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)    # [B, 16, 128, 128]
        e2 = self.enc2(e1)   # [B, 24, 64, 64]
        e3 = self.enc3(e2)   # [B, 32, 32, 32]
        e4 = self.enc4(e3)   # [B, 96, 16, 16]
        e5 = self.enc5(e4)   # [B, 320, 8, 8]
        
        # Decoder with skip connections
        d4 = self.up4(e5, e4)  # [B, 128, 16, 16]
        d3 = self.up3(d4, e3)  # [B, 64, 32, 32]
        d2 = self.up2(d3, e2)  # [B, 32, 64, 64]
        d1 = self.up1(d2, e1)  # [B, 16, 128, 128]
        
        # Final upsample to 256x256
        out = self.final_up(d1)    # [B, 16, 256, 256]
        out = self.final_conv(out)  # [B, 3, 256, 256]
        
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def create_model(pretrained: bool = True, device: str = 'cuda') -> RoadSegmentationNet:
    """Create and return the road segmentation model."""
    model = RoadSegmentationNet(pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Smoke test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_model(pretrained=True, device=device)
    x = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
