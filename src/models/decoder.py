"""
UNet decoder with attention gates and skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        """
        Initialize attention gate.

        Args:
            gate_channels: Number of channels in gating signal
            skip_channels: Number of channels in skip connection
            inter_channels: Number of intermediate channels
        """
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.gate_conv = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.skip_conv = nn.Conv2d(skip_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            gate: Gating signal from lower level
            skip: Skip connection from encoder

        Returns:
            Attention-weighted skip connection
        """
        # Resize gate to match skip dimensions
        gate_resized = F.interpolate(
            gate, size=skip.shape[2:], mode="bilinear", align_corners=False
        )

        # Compute attention coefficients
        gate_proj = self.gate_conv(gate_resized)
        skip_proj = self.skip_conv(skip)

        psi = self.relu(gate_proj + skip_proj)
        psi = self.psi(psi)
        attention_weights = self.sigmoid(psi)

        # Apply attention
        return skip * attention_weights


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            dropout: Dropout probability
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            skip_channels: Number of skip connection channels
            out_channels: Number of output channels
            use_attention: Whether to use attention gate
            dropout: Dropout probability
        """
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)

        if use_attention:
            self.attention = AttentionGate(in_channels // 2, skip_channels)
        else:
            self.attention = None

        conv_in_channels = (in_channels // 2) + skip_channels
        self.conv_block = ConvBlock(conv_in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor from lower level
            skip: Skip connection from encoder

        Returns:
            Upsampled and processed tensor
        """
        # Upsample
        x = self.upsample(x)

        # Apply attention to skip connection
        if self.attention is not None:
            skip = self.attention(x, skip)

        # Concatenate and process
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)

        return x


class DeepSupervision(nn.Module):
    """Deep supervision module for auxiliary losses."""

    def __init__(self, in_channels: int, num_classes: int):
        """
        Initialize deep supervision module.

        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
        """
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Forward pass with upsampling to target size.

        Args:
            x: Input feature map
            target_size: Target output size (H, W)

        Returns:
            Upsampled predictions
        """
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class UNetDecoder(nn.Module):
    """UNet decoder with attention gates and deep supervision."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = [1024, 512, 256, 128, 64],
        num_classes: int = 2,
        use_attention: bool = True,
        use_deep_supervision: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize UNet decoder.

        Args:
            encoder_channels: List of encoder channel dimensions (from deepest to shallowest)
            decoder_channels: List of decoder channel dimensions
            num_classes: Number of output classes
            use_attention: Whether to use attention gates
            use_deep_supervision: Whether to use deep supervision
            dropout: Dropout probability
        """
        super().__init__()

        self.use_deep_supervision = use_deep_supervision

        # Reverse encoder channels to match decoder order
        encoder_channels = encoder_channels[::-1]  # Deepest to shallowest

        # Center/bottleneck
        self.center = ConvBlock(
            encoder_channels[0], decoder_channels[0], dropout=dropout
        )

        # Decoder blocks
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            in_channels = decoder_channels[i]
            skip_channels = (
                encoder_channels[i + 1] if i + 1 < len(encoder_channels) else 0
            )
            out_channels = decoder_channels[i + 1]

            if skip_channels > 0:
                block = UpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_attention=use_attention,
                    dropout=dropout,
                )
            else:
                # No skip connection available
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
                    ConvBlock(in_channels // 2, out_channels, dropout=dropout),
                )

            self.blocks.append(block)

        # Final segmentation head
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, 1)

        # Deep supervision heads
        if use_deep_supervision:
            self.deep_supervision = nn.ModuleList(
                [
                    DeepSupervision(channels, num_classes)
                    for channels in decoder_channels[1:]  # Skip the deepest level
                ]
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of encoder features (from shallowest to deepest)

        Returns:
            Segmentation logits and optionally deep supervision outputs
        """
        # Reverse features to match decoder order (deepest to shallowest)
        features = features[::-1]

        # Center/bottleneck
        x = self.center(features[0])

        deep_outputs = []

        # Decoder blocks
        for i, block in enumerate(self.blocks):
            skip_idx = i + 1

            if skip_idx < len(features):
                # Use skip connection
                if isinstance(block, UpBlock):
                    x = block(x, features[skip_idx])
                else:
                    # Fallback for blocks without skip connections
                    x = block(x)
            else:
                # No skip connection available
                if isinstance(block, UpBlock):
                    # Create dummy skip connection with zeros
                    dummy_skip = torch.zeros_like(x[:, : x.size(1) // 2])
                    x = block(x, dummy_skip)
                else:
                    x = block(x)

            # Deep supervision
            if self.use_deep_supervision and i < len(self.deep_supervision):
                target_size = features[-1].shape[2:]  # Use input image size
                deep_output = self.deep_supervision[i](x, target_size)
                deep_outputs.append(deep_output)

        # Final segmentation
        final_output = self.final_conv(x)

        if self.use_deep_supervision:
            return final_output, deep_outputs
        else:
            return final_output


def test_decoder():
    """Test UNet decoder."""
    # Create dummy encoder features
    features = [
        torch.randn(2, 64, 128, 128),  # Shallowest
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 512, 16, 16),
        torch.randn(2, 1024, 8, 8),  # Deepest
    ]

    # Test decoder
    decoder = UNetDecoder(
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        use_attention=True,
        use_deep_supervision=True,
    )

    # Forward pass
    outputs = decoder(features)

    if isinstance(outputs, tuple):
        final_output, deep_outputs = outputs
        print(f"Final output shape: {final_output.shape}")
        print(f"Number of deep supervision outputs: {len(deep_outputs)}")
        for i, deep_out in enumerate(deep_outputs):
            print(f"Deep output {i} shape: {deep_out.shape}")
    else:
        print(f"Output shape: {outputs.shape}")

    # Test without deep supervision
    decoder_simple = UNetDecoder(
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        use_attention=False,
        use_deep_supervision=False,
    )

    simple_output = decoder_simple(features)
    print(f"Simple decoder output shape: {simple_output.shape}")

    return True


if __name__ == "__main__":
    test_decoder()
    print("Decoder tests passed!")
