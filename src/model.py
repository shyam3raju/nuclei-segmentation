"""
U-Net architecture implementation for semantic segmentation.

U-Net is the gold standard for biomedical image segmentation because:
1. Skip connections preserve fine spatial details lost during downsampling
2. Encoder-decoder structure captures both context (what) and localization (where)
3. Works well with small datasets (common in medical imaging)

Architecture overview:
- Encoder (left side): Series of conv layers with downsampling - captures "what"
- Decoder (right side): Series of conv layers with upsampling - captures "where"
- Skip connections: Copy feature maps from encoder to decoder at same resolution
  This helps recover fine-grained spatial information lost during downsampling

Original paper: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Basic building block: (Conv -> BatchNorm -> ReLU) x 2
    
    Why this pattern:
    - Two convs capture more complex patterns than one
    - BatchNorm stabilizes training and allows higher learning rates
    - ReLU introduces non-linearity (without it, stacked convs = one conv)
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    
    Why MaxPool:
    - Reduces spatial dimensions by 2x (increases receptive field)
    - Maintains important features while discarding less important details
    - More efficient than strided convolutions for this purpose
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Concat with skip connection -> DoubleConv
    
    Why skip connections:
    - Encoder has high-res features but limited context
    - Decoder has context but limited resolution
    - Skip connections give decoder both context AND fine-grained spatial info
    This is the key innovation that makes U-Net work so well!
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Upsample to double the spatial dimensions
        # We use bilinear interpolation + conv instead of transposed conv
        # to avoid checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # After upsampling and concatenating with skip connection:
        # We receive in_channels from below and out_channels from skip
        # Total input to conv: in_channels + out_channels
        self.conv = DoubleConv(in_channels + out_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder (lower resolution)
            x2: Feature map from encoder (higher resolution) - skip connection
        """
        # Upsample x1
        x1 = self.up(x1)
        
        # Handle size mismatch if input wasn't exactly divisible by 16
        # This can happen with non-standard input sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for binary semantic segmentation.
    
    Architecture (for 256x256 input):
        Input (3, 256, 256)
        ↓
        Encoder:
        - inc: Conv block (3→64 channels, 256x256)
        - down1: MaxPool + Conv (64→128, 128x128)
        - down2: MaxPool + Conv (128→256, 64x64)
        - down3: MaxPool + Conv (256→512, 32x32)
        - down4: MaxPool + Conv (512→1024, 16x16) ← bottleneck
        ↓
        Decoder:
        - up1: Upsample + Concat + Conv (1024→512, 32x32)
        - up2: Upsample + Concat + Conv (512→256, 64x64)
        - up3: Upsample + Concat + Conv (256→128, 128x128)
        - up4: Upsample + Concat + Conv (128→64, 256x256)
        ↓
        Output: Conv 1x1 (64→1 channel, 256x256) + Sigmoid
    
    Total parameters: ~31M (most are in the bottleneck and decoder)
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        """
        Args:
            in_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            out_channels (int): Number of output channels (1 for binary segmentation)
        """
        super().__init__()
        
        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, 64)
        
        # Encoder (downsampling path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Final 1x1 convolution to produce output
        # Why 1x1 conv: Projects 64 channels to 1 output channel (our mask)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x (torch.Tensor): Input image, shape (batch, 3, H, W)
        
        Returns:
            torch.Tensor: Output mask, shape (batch, 1, H, W)
                         Values in (0, 1) after sigmoid
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        logits = self.outc(x)
        
        return logits


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing U-Net model...")
    
    # Create model
    model = UNet(in_channels=3, out_channels=1)
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check if model can run on GPU
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        model = model.cuda()
        x = x.cuda()
        output = model(x)
        print("✓ Model successfully runs on GPU")
    else:
        print("\n✓ GPU not available, model runs on CPU")
    
    print("\nModel test passed! ✓")
