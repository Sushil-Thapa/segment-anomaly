"""
Swin-UNet model combining Swin Transformer backbone with UNet decoder.
Includes MAE (Masked Autoencoder) and DINOv3 capabilities for self-supervised pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import copy
import logging
from .decoder import UNetDecoder, MAEDecoder, create_random_mask

logger = logging.getLogger(__name__)


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


class DINOv3SwinUNet(SwinUNet):
    """
    Swin-UNet with DINOv3 self-distillation pretraining support.

    Key features:
    - Self-distillation via student-teacher framework
    - Multi-crop strategy for robust feature learning
    - Momentum teacher with EMA updates
    - Can be initialized from MAE pretrained weights for sequential SSL
    - Seamless transition to segmentation fine-tuning
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
        # DINOv3-specific parameters
        dino_out_dim: int = 65536,
        dino_hidden_dim: int = 2048,
        dino_bottleneck_dim: int = 256,
        dino_teacher_temp: float = 0.04,
        dino_student_temp: float = 0.1,
        dino_momentum_teacher: float = 0.996,
        dino_center_momentum: float = 0.9,
        **kwargs,
    ):
        """
        Initialize DINOv3-enabled Swin-UNet.

        Args:
            All SwinUNet args plus:
            dino_out_dim: Output dimension for DINO head
            dino_hidden_dim: Hidden dimension in DINO projection head
            dino_bottleneck_dim: Bottleneck dimension in DINO head
            dino_teacher_temp: Teacher temperature for sharpening
            dino_student_temp: Student temperature for softmax
            dino_momentum_teacher: EMA momentum for teacher update
            dino_center_momentum: Momentum for centering operation
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

        self.dino_teacher_temp = dino_teacher_temp
        self.dino_student_temp = dino_student_temp
        self.dino_momentum_teacher = dino_momentum_teacher
        self.dino_center_momentum = dino_center_momentum
        self.training_mode = "segmentation"  # "dino" or "segmentation"

        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(
                1, in_channels, self.backbone_input_size, self.backbone_input_size
            )
            features = self.backbone(dummy_input)
            encoder_dim = features[-1].shape[-1]  # NHWC format, take channel dim

        # DINO projection head (student)
        self.dino_head = DINOHead(
            in_dim=encoder_dim,
            out_dim=dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            use_bn=True,
            nlayers=3,
        )

        # Teacher network (EMA of student) - create new instances instead of deepcopy
        # Deepcopy doesn't work well with weight_norm on MPS
        self.teacher_backbone = timm.create_model(
            backbone_name,
            pretrained=False,  # Don't reload pretrained, will copy weights manually
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
            features_only=True,  # Match student backbone format - returns list of features
            out_indices=(0, 1, 2, 3),  # Get all feature stages like student
        )
        self.teacher_head = DINOHead(
            in_dim=encoder_dim,
            out_dim=dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
            nlayers=3,
        )

        # Copy weights from student to teacher (unless MAE checkpoint will be loaded later)
        # This initialization will be overridden by load_mae_pretrained_encoder if called
        self.teacher_backbone.load_state_dict(self.backbone.state_dict(), strict=False)
        self.teacher_head.load_state_dict(self.dino_head.state_dict(), strict=False)

        # Disable gradients for teacher
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        # Center for teacher output (moving average)
        self.register_buffer("center", torch.zeros(1, dino_out_dim))

        print(
            f"Initialized DINOv3 with encoder dim: {encoder_dim}, output dim: {dino_out_dim}"
        )

    def set_training_mode(self, mode: str):
        """
        Set training mode between DINOv3 pretraining and segmentation fine-tuning.

        Args:
            mode: "dino" for pretraining, "segmentation" for fine-tuning
        """
        assert mode in ["dino", "segmentation"], f"Invalid mode: {mode}"
        self.training_mode = mode

        if mode == "dino":
            # Enable student encoder and DINO head
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.dino_head.parameters():
                param.requires_grad = True

            # Freeze segmentation decoder during DINO pretraining
            for param in self.decoder.parameters():
                param.requires_grad = False
            if self.aux_classifier is not None:
                for param in self.aux_classifier.parameters():
                    param.requires_grad = False

        else:  # segmentation mode
            # Enable segmentation decoder
            for param in self.decoder.parameters():
                param.requires_grad = True
            if self.aux_classifier is not None:
                for param in self.aux_classifier.parameters():
                    param.requires_grad = True

            # Freeze DINO head during segmentation training
            for param in self.dino_head.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher networks using exponential moving average."""
        for param_student, param_teacher in zip(
            self.backbone.parameters(), self.teacher_backbone.parameters()
        ):
            param_teacher.data.mul_(self.dino_momentum_teacher).add_(
                param_student.data, alpha=1 - self.dino_momentum_teacher
            )

        for param_student, param_teacher in zip(
            self.dino_head.parameters(), self.teacher_head.parameters()
        ):
            param_teacher.data.mul_(self.dino_momentum_teacher).add_(
                param_student.data, alpha=1 - self.dino_momentum_teacher
            )

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # Update center with momentum
        self.center = self.center * self.dino_center_momentum + batch_center * (
            1 - self.dino_center_momentum
        )

    def forward_dino(
        self, global_views: List[torch.Tensor], local_views: List[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for DINOv3 pretraining with multi-crop strategy.

        Args:
            global_views: List of global crop tensors (typically 2)
            local_views: List of local crop tensors (typically 4-8)

        Returns:
            Dictionary with 'student_output', 'teacher_output', 'loss'
        """
        # Concatenate all views
        if local_views is not None:
            all_views = global_views + local_views
        else:
            all_views = global_views

        n_global = len(global_views)

        # Process all views through student
        student_outputs = []
        for view in all_views:
            # Ensure correct input size
            if view.shape[2:] != (self.backbone_input_size, self.backbone_input_size):
                view = F.interpolate(
                    view,
                    size=(self.backbone_input_size, self.backbone_input_size),
                    mode="bilinear",
                    align_corners=False,
                )

            # Extract features
            features = self.backbone(view)

            # Global average pooling over spatial dimensions
            if len(features[-1].shape) == 4:  # NHWC format
                pooled = (
                    F.adaptive_avg_pool2d(
                        features[-1].permute(0, 3, 1, 2), 1  # NHWC -> NCHW
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                )
            elif len(features[-1].shape) == 3:  # [B, H*W, C]
                pooled = features[-1].mean(dim=1)
            else:  # Already [B, C]
                pooled = features[-1]

            # Project through DINO head
            output = self.dino_head(pooled)
            student_outputs.append(output)

        # Process only global views through teacher
        teacher_outputs = []
        with torch.no_grad():
            for idx, view in enumerate(global_views):
                logger.debug(f"teacher loop {idx}: view.shape={view.shape}")

                if view.shape[2:] != (
                    self.backbone_input_size,
                    self.backbone_input_size,
                ):
                    view = F.interpolate(
                        view,
                        size=(self.backbone_input_size, self.backbone_input_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                features = self.teacher_backbone(view)
                logger.debug(
                    f"teacher loop {idx}: features[-1].shape={features[-1].shape}"
                )

                # Swin returns [B, H, W, C] - always expect 4D
                feat = features[-1]
                if len(feat.shape) != 4:
                    raise RuntimeError(
                        f"Expected 4D features [B, H, W, C], got {feat.shape}"
                    )

                # Pool: [B, H, W, C] -> [B, C, H, W] -> [B, C]
                pooled = (
                    F.adaptive_avg_pool2d(feat.permute(0, 3, 1, 2), 1)  # NHWC -> NCHW
                    .squeeze(-1)
                    .squeeze(-1)
                )

                logger.debug(f"teacher loop {idx}: pooled.shape={pooled.shape}")

                output = self.teacher_head(pooled)
                logger.debug(f"teacher loop {idx}: output.shape={output.shape}")
                teacher_outputs.append(output)

        # Concatenate outputs
        logger.debug(
            f"forward_dino: len(student_outputs)={len(student_outputs)}, len(teacher_outputs)={len(teacher_outputs)}"
        )
        for i, out in enumerate(student_outputs):
            logger.debug(f"forward_dino: student_outputs[{i}].shape={out.shape}")
        for i, out in enumerate(teacher_outputs):
            logger.debug(f"forward_dino: teacher_outputs[{i}].shape={out.shape}")

        student_output = torch.cat(student_outputs, dim=0)
        teacher_output = torch.cat(teacher_outputs, dim=0)

        logger.debug(
            f"forward_dino: After concat: student={student_output.shape}, teacher={teacher_output.shape}"
        )

        # Compute loss
        loss = self._compute_dino_loss(
            student_output, teacher_output, n_global, len(all_views)
        )

        # Update center
        self.update_center(teacher_output)

        return {
            "student_output": student_output,
            "teacher_output": teacher_output,
            "loss": loss,
            "center": self.center.clone(),
        }

    def _compute_dino_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        n_global: int,
        n_total: int,
    ) -> torch.Tensor:
        """
        Compute DINO loss using cross-entropy between student and teacher.

        Args:
            student_output: Student predictions for all crops [n_total * B, out_dim]
            teacher_output: Teacher predictions for global crops [n_global * B, out_dim]
            n_global: Number of global crops
            n_total: Total number of crops (global + local)

        Returns:
            DINO loss value
        """
        # Infer batch size from concatenated outputs
        batch_size = teacher_output.shape[0] // n_global

        logger.debug(
            f"teacher_output.shape={teacher_output.shape}, student_output.shape={student_output.shape}"
        )
        logger.debug(f"n_global={n_global}, n_total={n_total}, batch_size={batch_size}")

        # Center and sharpen teacher output
        teacher_output = F.softmax(
            (teacher_output - self.center) / self.dino_teacher_temp, dim=-1
        )

        # Student output with temperature
        student_output = F.log_softmax(student_output / self.dino_student_temp, dim=-1)

        # Compute cross-entropy loss
        # Each global view from teacher is compared with all student views except itself
        total_loss = 0
        n_loss_terms = 0

        for t_idx in range(n_global):
            # Teacher crop t_idx spans rows [t_idx * batch_size : (t_idx + 1) * batch_size]
            t_start = t_idx * batch_size
            t_end = (t_idx + 1) * batch_size
            teacher_crop = teacher_output[t_start:t_end]  # [B, out_dim]

            for s_idx in range(n_total):
                if s_idx == t_idx:  # Skip comparing view with itself
                    continue

                # Student crop s_idx
                s_start = s_idx * batch_size
                s_end = (s_idx + 1) * batch_size
                student_crop = student_output[s_start:s_end]  # [B, out_dim]

                logger.debug(
                    f"t_idx={t_idx}, s_idx={s_idx}, teacher_crop.shape={teacher_crop.shape}, student_crop.shape={student_crop.shape}"
                )

                # Cross-entropy: -sum(teacher * log_student)
                loss = -torch.sum(teacher_crop * student_crop, dim=-1).mean()

                total_loss += loss
                n_loss_terms += 1

        return total_loss / n_loss_terms if n_loss_terms > 0 else total_loss

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        local_views: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Forward pass - automatically switches between DINOv3 and segmentation modes.

        Args:
            x: Input tensor or list of global views for DINO
            local_views: List of local views for DINO (optional)

        Returns:
            Segmentation output or DINO results
        """
        if self.training_mode == "dino":
            if isinstance(x, list):
                return self.forward_dino(x, local_views)
            else:
                # Single image, create pseudo multi-crop for testing
                return self.forward_dino([x, x])
        else:
            # Standard segmentation forward pass
            if isinstance(x, list):
                x = x[0]  # Take first view if list provided
            return super().forward(x)

    def load_mae_pretrained_encoder(self, checkpoint_path: str):
        """
        Load MAE pretrained encoder weights before DINOv3 training.

        Args:
            checkpoint_path: Path to MAE checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle both checkpoint formats: direct state_dict or wrapped in 'model_state_dict'
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Extract encoder weights and fix key naming (underscore -> dot)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # Remove 'backbone.' prefix
                new_key = key.replace("backbone.", "")
                # Fix naming: layers_0 -> layers.0, layers_1 -> layers.1, etc.
                new_key = new_key.replace("layers_", "layers.")
                encoder_state_dict[new_key] = value

        # Load into student encoder
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            encoder_state_dict, strict=False
        )

        # Load into teacher encoder (use same remapped state_dict)
        self.teacher_backbone.load_state_dict(encoder_state_dict, strict=False)

        print(f"Loaded MAE pretrained encoder from {checkpoint_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5


class DINOHead(nn.Module):
    """
    Projection head for DINOv3.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        use_bn: bool = True,
        norm_last_layer: bool = True,
    ):
        super().__init__()

        nlayers = max(nlayers, 1)

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def create_dino_model(config: dict, mae_checkpoint: str = None) -> DINOv3SwinUNet:
    """
    Create DINOv3-enabled Swin-UNet from configuration.

    Args:
        config: Configuration dictionary
        mae_checkpoint: Optional path to MAE checkpoint for sequential SSL

    Returns:
        DINOv3SwinUNet model
    """
    model_config = config["model"]

    # Get backbone name
    backbone_name = model_config.get("backbone", "swin_large_patch4_window12_384")
    if "encoder" in model_config:
        encoder_map = {
            "swin_large": "swin_large_patch4_window12_384",
            "swin_base": "swin_base_patch4_window12_384",
            "swin_small": "swin_small_patch4_window7_224",
            "swin_tiny": "swin_tiny_patch4_window7_224",
        }
        backbone_name = encoder_map.get(model_config["encoder"], backbone_name)

    # DINOv3 specific config
    dino_config = model_config.get("dino", {})

    model = DINOv3SwinUNet(
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
        # DINOv3 parameters
        dino_out_dim=dino_config.get("out_dim", 65536),
        dino_hidden_dim=dino_config.get("hidden_dim", 2048),
        dino_bottleneck_dim=dino_config.get("bottleneck_dim", 256),
        dino_teacher_temp=dino_config.get("teacher_temp", 0.04),
        dino_student_temp=dino_config.get("student_temp", 0.1),
        dino_momentum_teacher=dino_config.get("momentum_teacher", 0.996),
        dino_center_momentum=dino_config.get("center_momentum", 0.9),
    )

    # Load MAE weights if provided (for sequential SSL)
    if mae_checkpoint:
        model.load_mae_pretrained_encoder(mae_checkpoint)

    return model


def test_dino_swin_unet():
    """Test DINOv3-enabled Swin-UNet."""
    # Create model
    model = DINOv3SwinUNet(
        backbone_name="swin_large_patch4_window12_384",
        pretrained=False,
        in_channels=3,
        dino_out_dim=8192,  # Smaller for testing
        dino_hidden_dim=1024,
        dino_bottleneck_dim=256,
    )

    # Test input - multi-crop
    batch_size = 2
    global_view1 = torch.randn(batch_size, 3, 384, 384)
    global_view2 = torch.randn(batch_size, 3, 384, 384)
    local_view1 = torch.randn(batch_size, 3, 192, 192)
    local_view2 = torch.randn(batch_size, 3, 192, 192)

    print("Testing DINOv3 mode...")

    # Test DINO mode
    model.set_training_mode("dino")
    dino_output = model([global_view1, global_view2], [local_view1, local_view2])

    print("DINO mode output keys:", dino_output.keys())
    print(f"DINO loss: {dino_output['loss'].item():.4f}")
    print(f"Student output shape: {dino_output['student_output'].shape}")
    print(f"Teacher output shape: {dino_output['teacher_output'].shape}")
    print(f"Center shape: {dino_output['center'].shape}")

    # Test teacher update
    model.update_teacher()
    print("Teacher updated successfully")

    # Test segmentation mode
    model.set_training_mode("segmentation")
    seg_output = model(global_view1)

    print(f"Segmentation output shape: {seg_output.shape}")

    # Test model size
    model_info = model.get_model_size()
    print(f"Model info: {model_info}")

    return True


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
    test_dino_swin_unet()
    print("All Swin-UNet tests passed!")
