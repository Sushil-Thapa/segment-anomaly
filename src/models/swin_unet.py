"""
Swin-UNet model combining Swin Transformer backbone with UNet decoder.
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Tuple, Union
from .decoder import UNetDecoder


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


if __name__ == "__main__":
    test_swin_unet()
    print("Swin-UNet tests passed!")
