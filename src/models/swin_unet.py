"""
Swin-UNet model combining Swin Transformer backbone with UNet decoder.
Includes MAE (Masked Autoencoder) capabilities for self-supervised pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
from .decoder import UNetDecoder, MAEDecoder, create_random_mask


class SwinUNet(nn.Module):
    """
    Swin-UNet model for semantic segmentation.

    Combines a Swin Transformer backbone from timm with a UNet decoder
    for binary defect segmentation on wafer images.
    """

    def __init__(
        self,
        backbone_name: str = "swin_large_patch4_window12_384",
        pretrained: bool = True,
        decoder_channels: List[int] = [1024, 512, 256, 128, 64],
        num_classes: int = 2,
        use_attention: bool = True,
        use_deep_supervision: bool = False,
        dropout: float = 0.1,
        aux_params: Optional[dict] = None,
        in_channels: int = 3,
    ):
        """
        Initialize Swin-UNet model.

        Args:
            backbone_name: Name of the Swin Transformer backbone from timm
            pretrained: Whether to use pretrained weights
            decoder_channels: List of decoder channel dimensions
            num_classes: Number of output classes
            use_attention: Whether to use attention gates in decoder
            use_deep_supervision: Whether to use deep supervision
            dropout: Dropout probability
            aux_params: Additional parameters for auxiliary losses
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_deep_supervision = use_deep_supervision
        self.in_channels = in_channels

        # Initialize backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(
                0,
                1,
                2,
                3,
            ),  # Get features from all stages (Swin has 4 stages)
        )

        # Modify first layer for single channel input if needed
        if in_channels != 3:
            # Get the original first layer
            first_layer = self.backbone.patch_embed.proj

            # Create new layer with same out_channels but different in_channels
            new_first_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias is not None,
            )

            # Initialize new layer weights
            if pretrained and in_channels == 1:
                # For single channel, average the RGB weights
                with torch.no_grad():
                    new_first_layer.weight.data = first_layer.weight.data.mean(
                        dim=1, keepdim=True
                    )
                    if (
                        new_first_layer.bias is not None
                        and first_layer.bias is not None
                    ):
                        new_first_layer.bias.data = first_layer.bias.data

            # Replace the first layer
            self.backbone.patch_embed.proj = new_first_layer

        # Determine input size from backbone name to match expected resolution
        if "224" in backbone_name:
            input_size = 224
        elif "384" in backbone_name:
            input_size = 384
        else:
            # Default to 224 for unknown backbones (most common)
            input_size = 224

        # Store backbone input size for forward pass
        self.backbone_input_size = input_size

        # Get feature information from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, input_size, input_size)
            features = self.backbone(dummy_input)
            # Swin transformer outputs are in NHWC format
            feature_channels = [f.shape[-1] for f in features]

        print(f"Backbone feature channels: {feature_channels}")

        # Initialize decoder
        self.decoder = UNetDecoder(
            encoder_channels=feature_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_attention=use_attention,
            use_deep_supervision=use_deep_supervision,
            dropout=dropout,
        )

        # Auxiliary classifier for deep supervision
        if aux_params is not None:
            self.aux_classifier = nn.Conv2d(
                feature_channels[-1], num_classes, 1  # Use deepest feature map
            )
        else:
            self.aux_classifier = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize decoder weights (backbone is already pretrained)."""
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if self.aux_classifier is not None:
            nn.init.kaiming_normal_(
                self.aux_classifier.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.aux_classifier.bias is not None:
                nn.init.constant_(self.aux_classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Segmentation logits and optionally auxiliary outputs
        """
        input_size = x.shape[2:]

        # Swin transformer requires specific input size based on the backbone variant
        # If input is different size, we need to resize
        backbone_input_size = (self.backbone_input_size, self.backbone_input_size)
        if input_size != backbone_input_size:
            x_backbone = torch.nn.functional.interpolate(
                x, size=backbone_input_size, mode="bilinear", align_corners=False
            )
        else:
            x_backbone = x

        # Extract features using backbone
        features = self.backbone(x_backbone)

        # Convert from NHWC to NCHW format for CNN decoder
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        # Decode features
        decoder_output = self.decoder(features)

        if self.use_deep_supervision and isinstance(decoder_output, tuple):
            main_output, deep_outputs = decoder_output
        else:
            main_output = decoder_output
            deep_outputs = []

        # Resize main output to original input size
        main_output = torch.nn.functional.interpolate(
            main_output, size=input_size, mode="bilinear", align_corners=False
        )

        # Prepare outputs
        outputs = [main_output]

        # Add deep supervision outputs
        if deep_outputs:
            for deep_out in deep_outputs:
                deep_out = torch.nn.functional.interpolate(
                    deep_out, size=input_size, mode="bilinear", align_corners=False
                )
                outputs.append(deep_out)

        # Add auxiliary output if available
        if self.aux_classifier is not None:
            aux_output = self.aux_classifier(features[-1])  # Use deepest features
            aux_output = torch.nn.functional.interpolate(
                aux_output, size=input_size, mode="bilinear", align_corners=False
            )
            outputs.append(aux_output)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract encoder features for analysis.

        Args:
            x: Input tensor

        Returns:
            List of feature maps from encoder
        """
        return self.backbone(x)

    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_model_size(self) -> dict:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        encoder_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params_mb": total_params * 4 / (1024 * 1024),  # Assume FP32
        }


class SwinUNetWithCAM(SwinUNet):
    """Swin-UNet with Class Activation Map (CAM) support."""

    def __init__(self, *args, **kwargs):
        """Initialize Swin-UNet with CAM support."""
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.activations = None

        # Register hooks for CAM
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks for CAM."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks on the last encoder layer
        last_layer = list(self.backbone.children())[-1]
        last_layer.register_forward_hook(forward_hook)
        last_layer.register_backward_hook(backward_hook)

    def get_cam(self, x: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        """
        Generate Class Activation Map.

        Args:
            x: Input tensor
            class_idx: Class index for CAM generation

        Returns:
            CAM tensor
        """
        # Forward pass
        output = self.forward(x)
        if isinstance(output, tuple):
            output = output[0]

        # Backward pass for gradients
        self.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward()

        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = torch.nn.functional.relu(cam)

            # Normalize CAM
            cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
            cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            # Resize to input size
            cam = torch.nn.functional.interpolate(
                cam, size=x.shape[2:], mode="bilinear", align_corners=False
            )

            return cam

        return None


def create_model(config: dict) -> SwinUNet:
    """
    Create Swin-UNet model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SwinUNet model
    """
    model_config = config["model"]

    # Handle different config formats
    backbone_name = model_config.get("backbone", "swin_large_patch4_window12_384")
    if "encoder" in model_config:
        # Map encoder names to backbone names
        encoder_map = {
            "swin_large": "swin_large_patch4_window12_384",
            "swin_base": "swin_base_patch4_window12_384",
            "swin_small": "swin_small_patch4_window7_224",
        }
        backbone_name = encoder_map.get(model_config["encoder"], backbone_name)

    # Get decoder channels from config or use defaults
    decoder_channels = model_config.get("decoder_channels", [1024, 512, 256, 128, 64])
    if "unet" in model_config and "features" in model_config["unet"]:
        # Convert from encoder features to decoder channels
        features = model_config["unet"]["features"]
        decoder_channels = features[1:] + [
            features[-1] // 2
        ]  # Skip first, add final layer

    model = SwinUNet(
        backbone_name=backbone_name,
        pretrained=True,
        decoder_channels=decoder_channels,
        num_classes=model_config.get("num_classes", 2),  # Binary segmentation
        use_attention=(
            model_config.get("unet", {}).get("use_attention", True)
            if "unet" in model_config
            else True
        ),
        use_deep_supervision=(
            model_config.get("unet", {}).get("use_deep_supervision", False)
            if "unet" in model_config
            else False
        ),
        dropout=model_config.get("dropout", 0.1),
        in_channels=model_config.get(
            "in_channels", 3
        ),  # Default to 3 for RGB, 1 for C-SAM
    )

    return model


def test_swin_unet():
    """Test Swin-UNet model."""
    # Test model creation
    model = SwinUNet(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,  # Don't download weights for testing
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        use_attention=True,
        use_deep_supervision=False,
        in_channels=3,
    )

    # Test forward pass with RGB input
    x = torch.randn(2, 3, 512, 512)
    output = model(x)

    print(f"RGB Input shape: {x.shape}")
    print(f"RGB Output shape: {output.shape}")

    # Test single-channel model
    model_1ch = SwinUNet(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,  # Don't download weights for testing
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        use_attention=True,
        use_deep_supervision=False,
        in_channels=1,
    )

    # Test forward pass with single channel input
    x_1ch = torch.randn(2, 1, 512, 512)
    output_1ch = model_1ch(x_1ch)

    print(f"Single-channel Input shape: {x_1ch.shape}")
    print(f"Single-channel Output shape: {output_1ch.shape}")

    # Test model info
    model_info = model.get_model_size()
    print(f"Model info: {model_info}")

    # Test with deep supervision
    model_ds = SwinUNet(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        use_attention=True,
        use_deep_supervision=True,
        in_channels=3,
    )

    output_ds = model_ds(x)
    if isinstance(output_ds, tuple):
        print(f"Deep supervision outputs: {len(output_ds)}")
        for i, out in enumerate(output_ds):
            print(f"Output {i} shape: {out.shape}")

    # Test CAM model
    cam_model = SwinUNetWithCAM(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2,
        in_channels=3,
    )

    cam = cam_model.get_cam(x, class_idx=1)
    if cam is not None:
        print(f"CAM shape: {cam.shape}")

    return True


class MAESwinUNet(SwinUNet):
    """
    Swin-UNet with Masked Autoencoder pretraining support.

    Key features:
    - Self-supervised pretraining via pixel reconstruction
    - 75% masking ratio for efficient learning
    - Seamless transition from pretraining to segmentation fine-tuning
    - Compatible with existing SwinUNet architecture
    """

    def __init__(
        self,
        backbone_name: str = "swin_large_patch4_window12_384",
        pretrained: bool = True,
        decoder_channels: List[int] = [1024, 512, 256, 128, 64],
        num_classes: int = 2,
        use_attention: bool = True,
        use_deep_supervision: bool = False,
        dropout: float = 0.1,
        aux_params: Optional[dict] = None,
        in_channels: int = 3,
        # MAE-specific parameters
        mae_decoder_dim: int = 512,
        mae_decoder_depth: int = 8,
        mae_decoder_heads: int = 16,
        mae_mask_ratio: float = 0.75,
        **kwargs,
    ):
        """
        Initialize MAE-enabled Swin-UNet.

        Args:
            All SwinUNet args plus:
            mae_decoder_dim: MAE decoder hidden dimension
            mae_decoder_depth: Number of MAE decoder transformer blocks
            mae_decoder_heads: Number of attention heads in MAE decoder
            mae_mask_ratio: Ratio of patches to mask during pretraining
        """
        # Initialize base SwinUNet
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            use_attention=use_attention,
            use_deep_supervision=use_deep_supervision,
            dropout=dropout,
            aux_params=aux_params,
            in_channels=in_channels,
            **kwargs,
        )

        self.mae_mask_ratio = mae_mask_ratio
        self.training_mode = "segmentation"  # "mae" or "segmentation"

        # Get encoder feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(
                1, in_channels, self.backbone_input_size, self.backbone_input_size
            )
            features = self.backbone(dummy_input)
            encoder_dims = [f.shape[-1] for f in features]  # NHWC format

        # Initialize MAE decoder
        self.mae_decoder = MAEDecoder(
            encoder_dims=encoder_dims,
            decoder_dim=mae_decoder_dim,
            decoder_depth=mae_decoder_depth,
            decoder_num_heads=mae_decoder_heads,
            patch_size=4,  # Match Swin patch size
            output_channels=in_channels,
        )

        print(f"Initialized MAE decoder with encoder dims: {encoder_dims}")

    def set_training_mode(self, mode: str):
        """
        Set training mode between MAE pretraining and segmentation fine-tuning.

        Args:
            mode: "mae" for pretraining, "segmentation" for fine-tuning
        """
        assert mode in ["mae", "segmentation"], f"Invalid mode: {mode}"
        self.training_mode = mode

        if mode == "mae":
            # Freeze segmentation decoder during MAE pretraining
            for param in self.decoder.parameters():
                param.requires_grad = False
            if self.aux_classifier is not None:
                for param in self.aux_classifier.parameters():
                    param.requires_grad = False

            # Enable MAE decoder
            for param in self.mae_decoder.parameters():
                param.requires_grad = True

        else:  # segmentation mode
            # Enable segmentation decoder
            for param in self.decoder.parameters():
                param.requires_grad = True
            if self.aux_classifier is not None:
                for param in self.aux_classifier.parameters():
                    param.requires_grad = True

            # Freeze MAE decoder during segmentation training
            for param in self.mae_decoder.parameters():
                param.requires_grad = False

    def forward_mae(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MAE pretraining.

        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Masking ratio (defaults to self.mae_mask_ratio)

        Returns:
            Dictionary with 'pred', 'target', 'mask', 'loss'
        """
        B, C, H, W = x.shape
        mask_ratio = mask_ratio or self.mae_mask_ratio

        # Ensure input size matches backbone requirements
        backbone_input_size = (self.backbone_input_size, self.backbone_input_size)
        if (H, W) != backbone_input_size:
            x_backbone = F.interpolate(
                x, size=backbone_input_size, mode="bilinear", align_corners=False
            )
        else:
            x_backbone = x

        # Create target patches from original image
        target_patches = self.mae_decoder.patchify(
            x_backbone
        )  # [B, N, patch_size^2 * C]

        # Create random mask
        seq_len = target_patches.shape[1]
        mask = create_random_mask(B, seq_len, mask_ratio, device=x.device)  # [B, N]

        # Apply mask to input before encoder
        # For Swin Transformer, we need to modify the patch embedding
        masked_input = self._apply_spatial_mask(x_backbone, mask)

        # Extract features with masked input
        features = self.backbone(masked_input)

        # MAE reconstruction
        pred_patches = self.mae_decoder(features, mask, target_size=backbone_input_size)

        # Compute reconstruction loss only on masked patches
        loss = self._compute_mae_loss(pred_patches, target_patches, mask)

        return {
            "pred": pred_patches,
            "target": target_patches,
            "mask": mask,
            "loss": loss,
            "masked_input": masked_input,
        }

    def _apply_spatial_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial masking to input image.

        Args:
            x: Input image [B, C, H, W]
            mask: Patch mask [B, N]

        Returns:
            Masked input image
        """
        B, C, H, W = x.shape
        patch_size = 4  # Swin patch size

        # Calculate patch grid
        h_patches = H // patch_size
        w_patches = W // patch_size

        # Reshape mask to spatial grid
        spatial_mask = mask.view(B, h_patches, w_patches)  # [B, h_patches, w_patches]

        # Expand mask to pixel level
        pixel_mask = spatial_mask.repeat_interleave(
            patch_size, dim=1
        ).repeat_interleave(patch_size, dim=2)
        pixel_mask = pixel_mask.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, H, W]

        # Apply mask (0 for masked patches, 1 for visible)
        masked_x = x * (~pixel_mask).float()

        return masked_x

    def _compute_mae_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MAE reconstruction loss.

        Args:
            pred: Predicted patches [B, N, patch_dim]
            target: Target patches [B, N, patch_dim]
            mask: Mask indicating which patches were masked [B, N]

        Returns:
            MSE loss on masked patches only
        """
        # Compute MSE loss
        loss = F.mse_loss(pred, target, reduction="none")  # [B, N, patch_dim]
        loss = loss.mean(dim=-1)  # [B, N] - average over patch dimensions

        # Only compute loss on masked patches
        loss = loss * mask.float()

        # Average over masked patches
        num_masked = mask.sum(dim=1).float()  # [B]
        loss = loss.sum(dim=1) / (num_masked + 1e-8)  # [B]

        return loss.mean()  # Average over batch

    def forward(
        self, x: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Forward pass - automatically switches between MAE and segmentation modes.

        Args:
            x: Input tensor
            mask_ratio: Masking ratio for MAE mode

        Returns:
            Segmentation output or MAE reconstruction results
        """
        if self.training_mode == "mae":
            return self.forward_mae(x, mask_ratio)
        else:
            # Standard segmentation forward pass
            return super().forward(x)

    def load_mae_pretrained_encoder(self, checkpoint_path: str):
        """
        Load MAE pretrained encoder weights.

        Args:
            checkpoint_path: Path to MAE checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract encoder weights
        encoder_state_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.startswith("backbone."):
                encoder_state_dict[key] = value

        # Load encoder weights
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            encoder_state_dict, strict=False
        )

        print(f"Loaded MAE pretrained encoder from {checkpoint_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")


def create_mae_model(config: dict) -> MAESwinUNet:
    """
    Create MAE-enabled Swin-UNet from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MAESwinUNet model
    """
    model_config = config["model"]

    # Get backbone name
    backbone_name = model_config.get("backbone", "swin_large_patch4_window12_384")
    if "encoder" in model_config:
        encoder_map = {
            "swin_large": "swin_large_patch4_window12_384",
            "swin_base": "swin_base_patch4_window12_384",
            "swin_small": "swin_small_patch4_window7_224",
        }
        backbone_name = encoder_map.get(model_config["encoder"], backbone_name)

    # MAE specific config
    mae_config = model_config.get("mae", {})

    model = MAESwinUNet(
        backbone_name=backbone_name,
        pretrained=True,
        decoder_channels=model_config.get(
            "decoder_channels", [1024, 512, 256, 128, 64]
        ),
        num_classes=model_config.get("num_classes", 2),
        use_attention=model_config.get("use_attention", True),
        use_deep_supervision=model_config.get("use_deep_supervision", False),
        dropout=model_config.get("dropout", 0.1),
        in_channels=model_config.get("in_channels", 3),
        # MAE parameters
        mae_decoder_dim=mae_config.get("decoder_dim", 512),
        mae_decoder_depth=mae_config.get("decoder_depth", 8),
        mae_decoder_heads=mae_config.get("decoder_heads", 16),
        mae_mask_ratio=mae_config.get("mask_ratio", 0.75),
    )

    return model


def test_mae_swin_unet():
    """Test MAE-enabled Swin-UNet."""
    # Create model
    model = MAESwinUNet(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,  # Don't download for testing
        in_channels=3,
        mae_decoder_dim=512,
        mae_decoder_depth=4,  # Smaller for testing
        mae_decoder_heads=8,
        mae_mask_ratio=0.75,
    )

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 384, 384)

    print(f"Input shape: {x.shape}")

    # Test MAE mode
    model.set_training_mode("mae")
    mae_output = model(x)

    print("MAE mode output keys:", mae_output.keys())
    print(f"MAE loss: {mae_output['loss'].item():.4f}")
    print(f"Pred shape: {mae_output['pred'].shape}")
    print(f"Target shape: {mae_output['target'].shape}")
    print(f"Mask shape: {mae_output['mask'].shape}")
    print(f"Masked ratio: {mae_output['mask'].float().mean().item():.3f}")

    # Test segmentation mode
    model.set_training_mode("segmentation")
    seg_output = model(x)

    print(f"Segmentation output shape: {seg_output.shape}")

    # Test encoder feature extraction
    features = model.get_encoder_features_for_patchcore(x)
    print(f"Encoder features shapes: {[f.shape for f in features]}")

    # Test model size
    model_info = model.get_model_size()
    print(f"Model info: {model_info}")

    return True


if __name__ == "__main__":
    test_swin_unet()
    test_mae_swin_unet()
    print("Swin-UNet tests passed!")
