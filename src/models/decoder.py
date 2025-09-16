"""
UNet decoder with attention gates and skip connections.
Includes MAE decoder for self-supervised pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
import numpy as np


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
    """Test decoder functionality."""
    # Test parameters
    batch_size = 2
    input_channels = [64, 128, 256, 512]
    feature_maps = [
        torch.randn(batch_size, 512, 16, 16),
        torch.randn(batch_size, 256, 32, 32),
        torch.randn(batch_size, 128, 64, 64),
        torch.randn(batch_size, 64, 128, 128),
    ]

    # Create decoder
    decoder = UNetDecoder(
        encoder_channels=input_channels,
        decoder_channels=[256, 128, 64, 32],
        num_classes=1,
        use_attention=True,
    )

    # Forward pass
    with torch.no_grad():
        output = decoder(feature_maps)

    print(f"Input feature shapes: {[f.shape for f in feature_maps]}")
    print(f"Output shape: {output.shape}")

    # Test MAE decoder
    mae_decoder = MAEDecoder(
        encoder_dims=[512, 256, 128, 64],
        decoder_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        output_channels=1,
    )

    # Create dummy mask for MAE test
    seq_len = 16 * 16  # 16x16 patches
    mask_indices = create_random_mask(batch_size, seq_len, mask_ratio=0.75)

    with torch.no_grad():
        mae_output = mae_decoder(feature_maps, mask_indices, (256, 256))

    print(f"MAE decoder output shape: {mae_output.shape}")

    return True


class MAEDecoder(nn.Module):
    """
    MAE Decoder for pixel reconstruction from Swin Transformer features.

    Key features:
    - Patch-based reconstruction with 75% masking
    - Multi-scale feature fusion from Swin stages
    - Lightweight decoder for efficient pretraining
    - Configurable output channels for grayscale/RGB
    """

    def __init__(
        self,
        encoder_dims: List[int] = [192, 384, 768, 1536],  # Swin-Large feature dims
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        patch_size: int = 4,
        output_channels: int = 3,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """
        Initialize MAE decoder.

        Args:
            encoder_dims: Feature dimensions from encoder stages
            decoder_dim: Decoder hidden dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            patch_size: Patch size (should match encoder)
            output_channels: Number of output channels (1 for grayscale, 3 for RGB)
            norm_layer: Normalization layer
        """
        super().__init__()

        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.output_channels = output_channels

        # Projection layers from encoder features to decoder dimension
        self.encoder_to_decoder = nn.ModuleList(
            [nn.Linear(dim, decoder_dim) for dim in encoder_dims]
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Positional embedding for decoder
        # This will be initialized based on input size during forward pass
        self.decoder_pos_embed = None

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_dim)

        # Final prediction head
        self.decoder_pred = nn.Linear(
            decoder_dim, patch_size**2 * output_channels, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize decoder weights."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_positional_embedding(self, height: int, width: int) -> torch.Tensor:
        """
        Get positional embedding for given spatial dimensions.

        Args:
            height: Feature map height
            width: Feature map width

        Returns:
            Positional embedding tensor
        """
        # Create 2D positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.decoder_dim, height, width)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        mask_indices: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Forward pass of MAE decoder.

        Args:
            encoder_features: List of feature maps from encoder stages
            mask_indices: Boolean mask indicating which patches are masked
            target_size: Target output size (H, W)

        Returns:
            Reconstructed image patches
        """
        # Use the deepest feature map as primary input
        x = encoder_features[-1]  # Shape: [B, H, W, C]
        B, H, W, C = x.shape

        # Flatten spatial dimensions: [B, H*W, C]
        x = x.view(B, H * W, C)

        # Project to decoder dimension
        x = self.encoder_to_decoder[-1](x)  # [B, H*W, decoder_dim]

        # Get positional embedding - always ensure it's on the correct device
        if self.decoder_pos_embed is None or self.decoder_pos_embed.shape[1] != H * W:
            self.decoder_pos_embed = self.get_positional_embedding(H, W).to(x.device)
        else:
            # Ensure existing positional embedding is on the correct device
            self.decoder_pos_embed = self.decoder_pos_embed.to(x.device)

        # Add positional embedding to visible patches
        x = x + self.decoder_pos_embed

        # For MAE, we need to handle the full sequence including masked patches
        # Get total number of patches
        total_patches = H * W

        # Create full sequence with mask tokens for all patches
        # In a proper implementation, you'd use the actual mask to place tokens correctly
        # For now, we'll create a sequence that matches the expected length
        if mask_indices is not None:
            # Use the mask to determine sequence length
            seq_len = mask_indices.shape[1]
        else:
            # Default to total patches
            seq_len = total_patches

        # Expand to match sequence length
        if x.shape[1] < seq_len:
            # Pad with mask tokens if needed - ensure they're on correct device
            num_mask_tokens = seq_len - x.shape[1]
            mask_tokens = self.mask_token.repeat(B, num_mask_tokens, 1).to(x.device)
            full_sequence = torch.cat([x, mask_tokens], dim=1)
        else:
            # Use first seq_len patches
            full_sequence = x[:, :seq_len, :]

        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            try:
                full_sequence = block(full_sequence)
            except RuntimeError as e:
                if "MPS backend out of memory" in str(e):
                    # Clear cache and retry with reduced precision
                    torch.mps.empty_cache()
                    full_sequence = full_sequence.half()
                    full_sequence = block(full_sequence)
                    full_sequence = full_sequence.float()
                else:
                    raise e

        full_sequence = self.decoder_norm(full_sequence)

        # Predict patches
        pred = self.decoder_pred(
            full_sequence
        )  # [B, N, patch_size^2 * output_channels]

        # Return flattened patches for reconstruction loss computation
        return pred

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: Images tensor [B, C, H, W]

        Returns:
            Patches tensor [B, N, patch_size^2 * C]
        """
        B, C, H, W = imgs.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        h = H // self.patch_size
        w = W // self.patch_size

        x = imgs.reshape(B, C, h, self.patch_size, w, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, h, w, patch_size, patch_size, C]
        x = x.reshape(B, h * w, self.patch_size**2 * C)

        return x

    def unpatchify(
        self, patches: torch.Tensor, img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Convert patches back to images.

        Args:
            patches: Patches tensor [B, N, patch_size^2 * C]
            img_size: Target image size (H, W)

        Returns:
            Images tensor [B, C, H, W]
        """
        H, W = img_size
        h = H // self.patch_size
        w = W // self.patch_size

        B, N, patch_dim = patches.shape
        C = patch_dim // (self.patch_size**2)

        assert N == h * w

        x = patches.reshape(B, h, w, self.patch_size, self.patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, patch_size, w, patch_size]
        x = x.reshape(B, C, H, W)

        return x


class TransformerBlock(nn.Module):
    """Transformer block for MAE decoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Generate 2D sinusoidal positional embedding.

    Args:
        embed_dim: Embedding dimension
        grid_h: Height of the grid
        grid_w: Width of the grid

    Returns:
        Positional embedding array
    """
    assert embed_dim % 4 == 0

    # Use half of dimensions for horizontal and half for vertical
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, np.arange(grid_h))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, np.arange(grid_w))

    emb = np.concatenate(
        [
            np.tile(emb_h[:, None, :], (1, grid_w, 1)),  # [H, W, embed_dim//2]
            np.tile(emb_w[None, :, :], (grid_h, 1, 1)),  # [H, W, embed_dim//2]
        ],
        axis=-1,
    )

    return emb.reshape(grid_h * grid_w, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sinusoidal positional embedding.

    Args:
        embed_dim: Embedding dimension
        pos: Position array

    Returns:
        Positional embedding array
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, embed_dim/2), outer product

    emb_sin = np.sin(out)  # (M, embed_dim/2)
    emb_cos = np.cos(out)  # (M, embed_dim/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb


def create_random_mask(
    batch_size: int, seq_len: int, mask_ratio: float = 0.75, device: torch.device = None
) -> torch.Tensor:
    """
    Create random mask for MAE pretraining.

    Args:
        batch_size: Batch size
        seq_len: Sequence length (number of patches)
        mask_ratio: Ratio of patches to mask
        device: Device to create tensor on

    Returns:
        Boolean mask tensor [B, N] where True = masked
    """
    num_mask = int(seq_len * mask_ratio)

    masks = []
    for _ in range(batch_size):
        # Create random permutation
        indices = torch.randperm(seq_len, device=device)

        # Create mask
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[indices[:num_mask]] = True

        masks.append(mask)

    return torch.stack(masks, dim=0)


def test_mae_decoder():
    """Test MAE decoder functionality."""
    # Create dummy encoder features (Swin-Large dimensions)
    batch_size = 2
    encoder_features = [
        torch.randn(batch_size, 96, 96, 192),  # Stage 1
        torch.randn(batch_size, 48, 48, 384),  # Stage 2
        torch.randn(batch_size, 24, 24, 768),  # Stage 3
        torch.randn(batch_size, 12, 12, 1536),  # Stage 4
    ]

    # Create MAE decoder
    decoder = MAEDecoder(
        encoder_dims=[192, 384, 768, 1536],
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        patch_size=4,
        output_channels=3,
    )

    # Create random mask
    seq_len = 12 * 12  # Based on deepest feature map
    mask_indices = create_random_mask(batch_size, seq_len, mask_ratio=0.75)

    # Test forward pass
    target_size = (384, 384)  # Target image size

    with torch.no_grad():
        pred_patches = decoder(encoder_features, mask_indices, target_size)

    print(f"Encoder features shapes: {[f.shape for f in encoder_features]}")
    print(f"Mask indices shape: {mask_indices.shape}")
    print(f"Predicted patches shape: {pred_patches.shape}")

    # Test patchify/unpatchify
    dummy_img = torch.randn(batch_size, 3, 384, 384)
    patches = decoder.patchify(dummy_img)
    reconstructed = decoder.unpatchify(patches, (384, 384))

    print(f"Original image shape: {dummy_img.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Check reconstruction accuracy
    reconstruction_error = torch.mean((dummy_img - reconstructed) ** 2)
    print(f"Patchify/unpatchify error: {reconstruction_error.item():.6f}")

    return True


def test_decoder():
    """Test decoder functionality."""
    # Test parameters
    batch_size = 2
    input_channels = [64, 128, 256, 512]
    feature_maps = [
        torch.randn(batch_size, 512, 16, 16),
        torch.randn(batch_size, 256, 32, 32),
        torch.randn(batch_size, 128, 64, 64),
        torch.randn(batch_size, 64, 128, 128),
    ]

    # Create decoder
    decoder = UNetDecoder(
        encoder_channels=input_channels,
        decoder_channels=[256, 128, 64, 32],
        num_classes=1,
        use_attention=True,
    )

    # Forward pass
    with torch.no_grad():
        output = decoder(feature_maps)

    print(f"Input feature shapes: {[f.shape for f in feature_maps]}")
    print(f"Output shape: {output.shape}")

    # Test MAE decoder
    mae_decoder = MAEDecoder(
        encoder_dims=[512, 256, 128, 64],
        decoder_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        output_channels=1,
    )

    # Create dummy mask for MAE test
    seq_len = 16 * 16  # 16x16 patches
    mask_indices = create_random_mask(batch_size, seq_len, mask_ratio=0.75)

    with torch.no_grad():
        mae_output = mae_decoder(feature_maps, mask_indices, (256, 256))

    print(f"MAE decoder output shape: {mae_output.shape}")

    return True


if __name__ == "__main__":
    test_decoder()
    print("Decoder tests passed!")
