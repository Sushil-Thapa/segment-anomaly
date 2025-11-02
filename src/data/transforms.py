"""
Data augmentation transforms for wafer defect and SAM acoustic microscopy segmentation with industry-specific augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, List
import torch
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
import random


# ImageNet normalization constants (not recommended for wafer images)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default wafer-specific normalization (will be computed from actual data)
WAFER_MEAN = [0.5, 0.5, 0.5]  # Placeholder - compute from your data
WAFER_STD = [0.2, 0.2, 0.2]  # Placeholder - compute from your data


class RandomShadow(A.ImageOnlyTransform):
    """Add random shadow patterns mimicking wafer inspection artifacts."""

    def __init__(
        self,
        shadow_dimension_lower=0.6,
        shadow_dimension_upper=1,
        shadow_roi_lower=0,
        shadow_roi_upper=1,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.shadow_dimension_lower = shadow_dimension_lower
        self.shadow_dimension_upper = shadow_dimension_upper
        self.shadow_roi_lower = shadow_roi_lower
        self.shadow_roi_upper = shadow_roi_upper

    def apply(self, img, shadow_dimension=0.5, shadow_roi=0.5, **params):
        image_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # Generate random shadow pattern
        height, width = img.shape[:2]
        shadow_mask = np.zeros((height, width), dtype=np.uint8)

        # Create multiple shadow regions
        num_shadows = np.random.randint(1, 4)
        for _ in range(num_shadows):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = x1 + np.random.randint(width // 4, width // 2)
            y2 = y1 + np.random.randint(height // 4, height // 2)

            cv2.rectangle(shadow_mask, (x1, y1), (x2, y2), 255, -1)

        # Apply Gaussian blur to make shadows realistic
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)

        # Reduce brightness in shadow regions
        shadow_factor = shadow_dimension
        image_HLS[:, :, 1] = image_HLS[:, :, 1] * (
            1 - shadow_factor * (shadow_mask / 255.0)
        )

        return cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)

    def get_params(self):
        return {
            "shadow_dimension": np.random.uniform(
                self.shadow_dimension_lower, self.shadow_dimension_upper
            ),
            "shadow_roi": np.random.uniform(
                self.shadow_roi_lower, self.shadow_roi_upper
            ),
        }

    def get_transform_init_args_names(self):
        return (
            "shadow_dimension_lower",
            "shadow_dimension_upper",
            "shadow_roi_lower",
            "shadow_roi_upper",
        )


# =====================================
# SAM ACOUSTIC MICROSCOPY TRANSFORMS
# =====================================


class SpeckleNoise(A.ImageOnlyTransform):
    """Add speckle noise typical in acoustic microscopy imaging."""

    def __init__(self, intensity=(0.1, 0.3), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.intensity = intensity

    def apply(self, img, noise_intensity=0.2, **params):
        # Convert to float for processing
        image = img.astype(np.float32) / 255.0

        # Generate multiplicative speckle noise
        noise = np.random.normal(1.0, noise_intensity, image.shape)
        noisy_image = image * noise

        # Clip and convert back
        noisy_image = np.clip(noisy_image, 0, 1) * 255
        return noisy_image.astype(np.uint8)

    def get_params(self):
        return {
            "noise_intensity": np.random.uniform(self.intensity[0], self.intensity[1])
        }

    def get_transform_init_args_names(self):
        return ("intensity",)


class AcousticNoise(A.ImageOnlyTransform):
    """Add acoustic-specific noise patterns including coherent and incoherent noise."""

    def __init__(
        self,
        coherent_strength=(0.05, 0.15),
        incoherent_strength=(0.1, 0.25),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.coherent_strength = coherent_strength
        self.incoherent_strength = incoherent_strength

    def apply(self, img, coherent_str=0.1, incoherent_str=0.15, **params):
        image = img.astype(np.float32) / 255.0

        # Add coherent noise (structured patterns)
        if coherent_str > 0:
            h, w = image.shape[:2]
            # Create periodic patterns
            x = np.arange(w)
            y = np.arange(h)[:, np.newaxis]

            # Multiple frequency components
            freq1 = np.random.uniform(0.05, 0.2)
            freq2 = np.random.uniform(0.1, 0.3)

            coherent_noise = coherent_str * np.sin(2 * np.pi * freq1 * x) * np.sin(
                2 * np.pi * freq1 * y
            ) + coherent_str * 0.5 * np.sin(2 * np.pi * freq2 * x) * np.cos(
                2 * np.pi * freq2 * y
            )

            if len(image.shape) == 3:
                coherent_noise = coherent_noise[:, :, np.newaxis]

            image = image + coherent_noise

        # Add incoherent noise (random)
        if incoherent_str > 0:
            incoherent_noise = np.random.normal(0, incoherent_str, image.shape)
            image = image + incoherent_noise

        # Clip and convert back
        image = np.clip(image, 0, 1) * 255
        return image.astype(np.uint8)

    def get_params(self):
        return {
            "coherent_str": np.random.uniform(
                self.coherent_strength[0], self.coherent_strength[1]
            ),
            "incoherent_str": np.random.uniform(
                self.incoherent_strength[0], self.incoherent_strength[1]
            ),
        }

    def get_transform_init_args_names(self):
        return ("coherent_strength", "incoherent_strength")


class DefectCopyPaste(A.DualTransform):
    """Copy-paste defects from one image to another for data augmentation."""

    def __init__(
        self,
        defect_images=None,
        paste_prob=0.3,
        max_defects=3,
        size_range=(20, 80),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.defect_images = defect_images or []
        self.paste_prob = paste_prob
        self.max_defects = max_defects
        self.size_range = size_range

    def apply(self, img, defect_patches=None, paste_locations=None, **params):
        if not defect_patches or not paste_locations:
            return img

        result = img.copy()
        for patch, (x, y, w, h) in zip(defect_patches, paste_locations):
            # Ensure patch fits in image
            h_img, w_img = img.shape[:2]
            if x + w <= w_img and y + h <= h_img:
                # Blend patch into image
                alpha = 0.7 + 0.3 * np.random.random()
                result[y : y + h, x : x + w] = (
                    alpha * patch + (1 - alpha) * result[y : y + h, x : x + w]
                ).astype(np.uint8)

        return result

    def apply_to_mask(self, mask, defect_masks=None, paste_locations=None, **params):
        if not defect_masks or not paste_locations:
            return mask

        result = mask.copy()
        for mask_patch, (x, y, w, h) in zip(defect_masks, paste_locations):
            # Ensure patch fits in mask
            h_mask, w_mask = mask.shape[:2]
            if x + w <= w_mask and y + h <= h_mask:
                result[y : y + h, x : x + w] = np.maximum(
                    result[y : y + h, x : x + w], mask_patch
                )

        return result

    def get_params_dependent_on_targets(self, params):
        if not self.defect_images or np.random.random() > self.paste_prob:
            return {
                "defect_patches": None,
                "defect_masks": None,
                "paste_locations": None,
            }

        img = params["image"]
        h_img, w_img = img.shape[:2]

        num_defects = np.random.randint(1, self.max_defects + 1)
        defect_patches = []
        defect_masks = []
        paste_locations = []

        for _ in range(num_defects):
            # Select random defect image
            if self.defect_images:
                defect_img = random.choice(self.defect_images)

                # Extract random patch
                patch_size = np.random.randint(self.size_range[0], self.size_range[1])
                if (
                    defect_img.shape[0] >= patch_size
                    and defect_img.shape[1] >= patch_size
                ):
                    start_x = np.random.randint(0, defect_img.shape[1] - patch_size)
                    start_y = np.random.randint(0, defect_img.shape[0] - patch_size)

                    patch = defect_img[
                        start_y : start_y + patch_size, start_x : start_x + patch_size
                    ]

                    # Create corresponding mask patch (assume defects are non-zero)
                    if len(patch.shape) == 3:
                        mask_patch = (np.mean(patch, axis=2) > 50).astype(np.uint8)
                    else:
                        mask_patch = (patch > 50).astype(np.uint8)

                    # Find random paste location
                    paste_x = np.random.randint(0, max(1, w_img - patch_size))
                    paste_y = np.random.randint(0, max(1, h_img - patch_size))

                    defect_patches.append(patch)
                    defect_masks.append(mask_patch)
                    paste_locations.append((paste_x, paste_y, patch_size, patch_size))

        return {
            "defect_patches": defect_patches,
            "defect_masks": defect_masks,
            "paste_locations": paste_locations,
        }

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("paste_prob", "max_defects", "size_range")


class TileJitter(A.DualTransform):
    """Apply small random translations to simulate tile positioning uncertainty."""

    def __init__(self, max_jitter=32, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.max_jitter = max_jitter

    def apply(self, img, dx=0, dy=0, **params):
        h, w = img.shape[:2]

        # Create transformation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation with reflection padding
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return result

    def apply_to_mask(self, mask, dx=0, dy=0, **params):
        h, w = mask.shape[:2]

        # Create transformation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation with constant padding (background)
        result = cv2.warpAffine(
            mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return result

    def get_params(self):
        return {
            "dx": np.random.randint(-self.max_jitter, self.max_jitter + 1),
            "dy": np.random.randint(-self.max_jitter, self.max_jitter + 1),
        }

    def get_transform_init_args_names(self):
        return ("max_jitter",)


class GrayscaleToRGB(A.ImageOnlyTransform):
    """Convert grayscale images to RGB by replicating channels."""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        if len(img.shape) == 2:
            return np.stack([img, img, img], axis=2)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            return np.concatenate([img, img, img], axis=2)
        else:
            return img

    def get_transform_init_args_names(self):
        return ()


def get_sam_train_transform(
    tile_size: int = 512,
    use_dataset_normalization: bool = True,
    data_dir: Optional[str] = None,
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    defect_images: Optional[List[np.ndarray]] = None,
    grayscale_mode: bool = False,
) -> Callable:
    """
    Get SAM-specific training augmentation pipeline for acoustic microscopy.

    Args:
        tile_size: Size of input tiles
        use_dataset_normalization: Whether to use dataset-specific normalization
        data_dir: Path to data directory for computing statistics
        normalization_stats: Pre-computed (mean, std) tuple
        defect_images: List of defect images for copy-paste augmentation
        grayscale_mode: If True, keeps single channel; if False, converts to RGB

    Returns:
        Albumentations compose transform
    """
    # Determine normalization parameters
    if normalization_stats is not None:
        mean, std = normalization_stats
    elif use_dataset_normalization and data_dir:
        mean, std = compute_dataset_statistics(data_dir)
    else:
        # For grayscale SAM images, use single-channel normalization
        if grayscale_mode:
            mean, std = [0.5], [0.2]  # Single channel stats
        else:
            mean, std = [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]  # Replicated to 3 channels

    transforms = []

    # Channel conversion based on mode
    if not grayscale_mode:
        # Convert grayscale to RGB for pretrained models
        transforms.append(GrayscaleToRGB(always_apply=True))

    transforms.extend(
        [
            # SAM-specific positioning jitter (±32px as specified)
            TileJitter(max_jitter=32, p=0.6),
            # Standard geometric augmentations (reduced rotation for acoustic)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.05,  # Reduced for acoustic precision
                scale_limit=0.05,  # Reduced scale variation
                rotate_limit=15,  # Reduced rotation
                border_mode=cv2.BORDER_REFLECT,
                p=0.6,
            ),
            # SAM acoustic-specific noise
            SpeckleNoise(intensity=(0.1, 0.3), p=0.5),
            AcousticNoise(
                coherent_strength=(0.05, 0.15), incoherent_strength=(0.1, 0.25), p=0.4
            ),
            # Defect synthesis (if defect images provided)
            DefectCopyPaste(
                defect_images=defect_images,
                paste_prob=0.3,
                max_defects=2,
                size_range=(20, 60),
                p=0.3 if defect_images else 0.0,
            ),
            # Reduced geometric distortions for acoustic precision
            A.GridDistortion(
                num_steps=3, distort_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.2
            ),
            A.ElasticTransform(
                alpha=60,  # Reduced from wafer defaults
                sigma=60 * 0.05,
                alpha_affine=60 * 0.03,
                border_mode=cv2.BORDER_REFLECT,
                p=0.2,
            ),
            # Acoustic-appropriate intensity augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # Reduced for acoustic stability
                contrast_limit=0.25,
                p=0.6,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),  # Narrower range
            A.CLAHE(
                clip_limit=2.0,  # Conservative for acoustic
                tile_grid_size=(8, 8),
                p=0.4,
            ),
            # Minimal dropout for acoustic (defects are subtle)
            A.CoarseDropout(
                max_holes=6,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=5,
                min_width=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.2,
            ),
            # Final normalization (handle both grayscale and RGB)
            A.Normalize(
                mean=mean if grayscale_mode else (mean * 3 if len(mean) == 1 else mean),
                std=std if grayscale_mode else (std * 3 if len(std) == 1 else std),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms, additional_targets={"mask": "mask"})


def get_sam_val_transform(
    tile_size: int = 512,
    use_dataset_normalization: bool = True,
    data_dir: Optional[str] = None,
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    grayscale_mode: bool = False,
) -> Callable:
    """
    Get SAM-specific validation transform pipeline (minimal processing).

    Args:
        tile_size: Size of input tiles
        use_dataset_normalization: Whether to use dataset-specific normalization
        data_dir: Path to data directory for computing statistics
        normalization_stats: Pre-computed (mean, std) tuple
        grayscale_mode: If True, keeps single channel; if False, converts to RGB

    Returns:
        Albumentations compose transform
    """
    # Determine normalization parameters
    if normalization_stats is not None:
        mean, std = normalization_stats
    elif use_dataset_normalization and data_dir:
        mean, std = compute_dataset_statistics(data_dir)
    else:
        # For grayscale SAM images
        if grayscale_mode:
            mean, std = [0.5], [0.2]  # Single channel stats
        else:
            mean, std = [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]  # Replicated to 3 channels

    transforms = []

    # Channel conversion based on mode
    if not grayscale_mode:
        # Convert grayscale to RGB for pretrained models
        transforms.append(GrayscaleToRGB(always_apply=True))

    transforms.extend(
        [
            # Final normalization
            A.Normalize(
                mean=mean if grayscale_mode else (mean * 3 if len(mean) == 1 else mean),
                std=std if grayscale_mode else (std * 3 if len(std) == 1 else std),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms, additional_targets={"mask": "mask"})


# =====================================
# END SAM ACOUSTIC MICROSCOPY TRANSFORMS
# =====================================


def compute_dataset_statistics(
    data_dir: str, sample_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dataset-specific normalization statistics from actual wafer images.

    Args:
        data_dir: Path to training data directory
        sample_size: Number of samples to use for statistics

    Returns:
        Tuple of (mean, std) arrays
    """
    import os
    from PIL import Image

    image_dir = os.path.join(data_dir, "train", "images")
    if not os.path.exists(image_dir):
        print("Warning: Training images not found, using default normalization")
        return np.array(WAFER_MEAN), np.array(WAFER_STD)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(image_files) == 0:
        print("Warning: No images found, using default normalization")
        return np.array(WAFER_MEAN), np.array(WAFER_STD)

    # Sample random images
    sample_files = np.random.choice(
        image_files, min(sample_size, len(image_files)), replace=False
    )

    pixel_values = []
    for file in sample_files:
        try:
            img_path = os.path.join(image_dir, file)
            img = np.array(Image.open(img_path).convert("RGB"))
            # Randomly sample pixels to avoid memory issues
            sample_pixels = img.reshape(-1, 3)[::100]  # Sample every 100th pixel
            pixel_values.append(sample_pixels)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if not pixel_values:
        print("Warning: Failed to process images, using default normalization")
        return np.array(WAFER_MEAN), np.array(WAFER_STD)

    # Compute statistics
    all_pixels = np.concatenate(pixel_values, axis=0)
    mean = np.mean(all_pixels, axis=0) / 255.0
    std = np.std(all_pixels, axis=0) / 255.0

    print(f"Computed dataset statistics: mean={mean}, std={std}")
    return mean, std


def get_train_transform(
    tile_size: int = 512,
    use_dataset_normalization: bool = True,
    data_dir: Optional[str] = None,
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Callable:
    """
    Get training augmentation pipeline with wafer-specific augmentations.

    Args:
        tile_size: Size of input tiles
        use_dataset_normalization: Whether to use dataset-specific normalization
        data_dir: Path to data directory for computing statistics
        normalization_stats: Pre-computed (mean, std) tuple

    Returns:
        Albumentations compose transform
    """

    # Determine normalization parameters
    if normalization_stats is not None:
        mean, std = normalization_stats
    elif use_dataset_normalization and data_dir:
        mean, std = compute_dataset_statistics(data_dir)
    else:
        print("Using ImageNet normalization (not recommended for wafer images)")
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    return A.Compose(
        [
            # Geometric augmentations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-30, 30),
                mode=cv2.BORDER_REFLECT,
                p=0.7,
            ),
            # Wafer-specific texture augmentations
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,  # Increased for wafer surface variations
                border_mode=cv2.BORDER_REFLECT,
                p=0.3,
            ),
            A.OpticalDistortion(
                distort_limit=0.1,
                border_mode=cv2.BORDER_REFLECT,
                p=0.25,
            ),
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3,
            ),
            # Intensity augmentations for wafer inspection conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # Increased for varying illumination
                contrast_limit=0.3,
                p=0.7,
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),  # Wider range for industrial imaging
                p=0.4,  # Increased probability
            ),
            RandomShadow(  # Custom wafer shadow augmentation
                shadow_dimension_lower=0.4, shadow_dimension_upper=0.8, p=0.3
            ),
            A.CLAHE(
                clip_limit=3.0,  # Higher for industrial contrast
                tile_grid_size=(8, 8),
                p=0.5,
            ),
            # Color/channel augmentations
            A.HueSaturationValue(
                hue_shift_limit=15,  # Slightly increased
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.4,
            ),
            A.ChannelShuffle(p=0.1),  # Minimal channel mixing for industrial images
            # Noise and artifacts (common in wafer inspection)
            A.GaussNoise(variance_limit=(10, 60), p=0.3),  # Fixed parameter name
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 12),  # Fixed parameter structure
                hole_height_range=(8, 40),
                hole_width_range=(8, 40),
                fill_value=0,
                p=0.4,
            ),
            # Advanced texture variations
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.1),
            # Final normalization and tensor conversion
            A.Normalize(mean=mean.tolist(), std=std.tolist(), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transform(
    tile_size: int = 512,
    use_dataset_normalization: bool = True,
    data_dir: Optional[str] = None,
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Callable:
    """
    Get validation transform pipeline (no augmentations, dataset-specific normalization).

    Args:
        tile_size: Size of input tiles
        use_dataset_normalization: Whether to use dataset-specific normalization
        data_dir: Path to data directory for computing statistics
        normalization_stats: Pre-computed (mean, std) tuple

    Returns:
        Albumentations compose transform
    """
    # Determine normalization parameters
    if normalization_stats is not None:
        mean, std = normalization_stats
    elif use_dataset_normalization and data_dir:
        mean, std = compute_dataset_statistics(data_dir)
    else:
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    return A.Compose(
        [
            A.Normalize(mean=mean.tolist(), std=std.tolist(), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_inference_transform(
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Callable:
    """
    Get inference transform pipeline.

    Args:
        normalization_stats: Pre-computed (mean, std) tuple

    Returns:
        Albumentations compose transform
    """
    if normalization_stats is not None:
        mean, std = normalization_stats
    else:
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    return A.Compose(
        [
            A.Normalize(mean=mean.tolist(), std=std.tolist(), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(
    normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Dict[str, Callable]:
    """
    Get Test Time Augmentation transforms.

    Args:
        normalization_stats: Pre-computed (mean, std) tuple

    Returns:
        Dictionary of TTA transforms
    """
    if normalization_stats is not None:
        mean, std = normalization_stats
    else:
        mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    base_transform = [
        A.Normalize(mean=mean.tolist(), std=std.tolist(), max_pixel_value=255.0),
        ToTensorV2(),
    ]

    return {
        "original": A.Compose(base_transform),
        "hflip": A.Compose([A.HorizontalFlip(always_apply=True)] + base_transform),
        "vflip": A.Compose([A.VerticalFlip(always_apply=True)] + base_transform),
        "rot90": A.Compose(
            [A.Rotate(limit=(90, 90), always_apply=True)] + base_transform
        ),
        "rot180": A.Compose(
            [A.Rotate(limit=(180, 180), always_apply=True)] + base_transform
        ),
        "rot270": A.Compose(
            [A.Rotate(limit=(270, 270), always_apply=True)] + base_transform
        ),
    }


class WaferSpecificTransforms:
    """Factory class for wafer-specific transforms with computed normalization."""

    def __init__(self, data_dir: Optional[str] = None, sample_size: int = 1000):
        """
        Initialize with dataset statistics computation.

        Args:
            data_dir: Path to data directory
            sample_size: Number of samples for statistics computation
        """
        self.data_dir = data_dir
        self.sample_size = sample_size
        self._normalization_stats = None

        if data_dir:
            self._compute_statistics()

    def _compute_statistics(self):
        """Compute and cache normalization statistics."""
        if self.data_dir:
            self._normalization_stats = compute_dataset_statistics(
                self.data_dir, self.sample_size
            )
            print(
                f"Computed wafer dataset statistics: mean={self._normalization_stats[0]}, std={self._normalization_stats[1]}"
            )
        else:
            self._normalization_stats = (np.array(WAFER_MEAN), np.array(WAFER_STD))

    def get_train_transform(self, tile_size: int = 512) -> Callable:
        """Get training transform with dataset normalization."""
        return get_train_transform(
            tile_size=tile_size,
            use_dataset_normalization=True,
            normalization_stats=self._normalization_stats,
        )

    def get_val_transform(self, tile_size: int = 512) -> Callable:
        """Get validation transform with dataset normalization."""
        return get_val_transform(
            tile_size=tile_size,
            use_dataset_normalization=True,
            normalization_stats=self._normalization_stats,
        )

    def get_inference_transform(self) -> Callable:
        """Get inference transform with dataset normalization."""
        return get_inference_transform(normalization_stats=self._normalization_stats)

    def get_tta_transforms(self) -> Dict[str, Callable]:
        """Get TTA transforms with dataset normalization."""
        return get_tta_transforms(normalization_stats=self._normalization_stats)

    @property
    def normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization statistics."""
        return self._normalization_stats


class SAMSpecificTransforms:
    """Factory class for SAM acoustic microscopy transforms with computed normalization."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        sample_size: int = 1000,
        defect_images: Optional[List[np.ndarray]] = None,
        grayscale_mode: bool = True,
    ):
        """
        Initialize with dataset statistics computation and defect images.

        Args:
            data_dir: Path to data directory
            sample_size: Number of samples for statistics computation
            defect_images: List of defect images for copy-paste augmentation
            grayscale_mode: If True, keeps single channel; if False, converts to RGB
        """
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.defect_images = defect_images or []
        self.grayscale_mode = grayscale_mode
        self._normalization_stats = None

        if data_dir:
            self._compute_statistics()
        else:
            # Default SAM normalization
            if grayscale_mode:
                self._normalization_stats = (np.array([0.5]), np.array([0.2]))
            else:
                self._normalization_stats = (
                    np.array([0.5, 0.5, 0.5]),
                    np.array([0.2, 0.2, 0.2]),
                )

    def _compute_statistics(self):
        """Compute and cache normalization statistics for SAM data."""
        if self.data_dir:
            self._normalization_stats = compute_dataset_statistics(
                self.data_dir, self.sample_size
            )
            print(
                f"Computed SAM dataset statistics: mean={self._normalization_stats[0]}, std={self._normalization_stats[1]}"
            )
        else:
            if self.grayscale_mode:
                self._normalization_stats = (np.array([0.5]), np.array([0.2]))
            else:
                self._normalization_stats = (
                    np.array([0.5, 0.5, 0.5]),
                    np.array([0.2, 0.2, 0.2]),
                )

    def get_train_transform(self, tile_size: int = 512) -> Callable:
        """Get SAM training transform with dataset normalization."""
        return get_sam_train_transform(
            tile_size=tile_size,
            use_dataset_normalization=True,
            normalization_stats=self._normalization_stats,
            defect_images=self.defect_images,
            grayscale_mode=self.grayscale_mode,
        )

    def get_val_transform(self, tile_size: int = 512) -> Callable:
        """Get SAM validation transform with dataset normalization."""
        return get_sam_val_transform(
            tile_size=tile_size,
            use_dataset_normalization=True,
            normalization_stats=self._normalization_stats,
            grayscale_mode=self.grayscale_mode,
        )

    def add_defect_images(self, defect_images: List[np.ndarray]):
        """Add defect images for copy-paste augmentation."""
        self.defect_images.extend(defect_images)

    @property
    def normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization statistics."""
        return self._normalization_stats


def create_weighted_sampler_transforms(
    positive_ratio: float = 0.75,
) -> Dict[str, Callable]:
    """
    Create transforms for weighted sampling with different positive ratios.

    Args:
        positive_ratio: Target ratio of positive samples

    Returns:
        Dictionary of transform sets
    """
    # More aggressive augmentation for positive samples (to increase diversity)
    positive_augment = A.Compose(
        [
            A.RandomRotate90(p=0.8),
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT,
                p=0.8,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.RandomGamma(gamma_limit=(60, 140), p=0.6),
            A.GridDistortion(
                num_steps=5, distort_limit=0.2, border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
        ]
    )

    # Light augmentation for negative samples
    negative_augment = A.Compose(
        [
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.15),
            A.VerticalFlip(p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    )

    return {
        "positive_heavy": positive_augment,
        "negative_light": negative_augment,
        "target_ratio": positive_ratio,
    }


# Export key functions for backward compatibility
__all__ = [
    "get_train_transform",
    "get_val_transform",
    "get_train_transforms",  # Alias for backward compatibility
    "get_val_transforms",  # Alias for backward compatibility
    "get_inference_transform",
    "get_tta_transforms",
    "get_mae_transform",  # Added MAE transform
    "compute_dataset_statistics",
    "WaferSpecificTransforms",
    "SAMSpecificTransforms",
    "RandomShadow",
    "create_weighted_sampler_transforms",
    # SAM acoustic microscopy transforms
    "SpeckleNoise",
    "AcousticNoise",
    "DefectCopyPaste",
    "TileJitter",
    "GrayscaleToRGB",
    "get_sam_train_transform",
    "get_sam_val_transform",
    # Constants
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "WAFER_MEAN",
    "WAFER_STD",
]

# Backward compatibility aliases
get_train_transforms = get_train_transform
get_val_transforms = get_val_transform


def test_transforms():
    """Test transform functions."""
    import torch

    # Test basic transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Create dummy data
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

    # Test training transform
    train_result = train_transform(image=dummy_image, mask=dummy_mask)
    print(
        f"Train transform output shapes: image={train_result['image'].shape}, mask={train_result['mask'].shape}"
    )

    # Test validation transform
    val_result = val_transform(image=dummy_image, mask=dummy_mask)
    print(
        f"Val transform output shapes: image={val_result['image'].shape}, mask={val_result['mask'].shape}"
    )

    # Test TTA transforms
    tta_transforms = get_tta_transforms()
    for name, transform in tta_transforms.items():
        result = transform(image=dummy_image)
        print(f"TTA {name} output shape: {result['image'].shape}")

    # Test wafer-specific transforms factory
    wafer_transforms = WaferSpecificTransforms()
    train_transform = wafer_transforms.get_train_transform()
    result = train_transform(image=dummy_image, mask=dummy_mask)
    print(
        f"Wafer-specific transform output shapes: image={result['image'].shape}, mask={result['mask'].shape}"
    )

    print("All transform tests passed!")
    return True


def get_mae_transform(tile_size: int = 384, in_channels: int = 3) -> Callable:
    """
    Get transform for MAE pretraining.

    Args:
        tile_size: Target image size
        in_channels: Number of input channels

    Returns:
        Transform function
    """
    # Minimal augmentation for MAE - just normalization and light intensity jitter
    if in_channels == 1:
        # Grayscale transforms
        transforms = [
            A.Resize(tile_size, tile_size, interpolation=cv2.INTER_LINEAR),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485],  # Grayscale mean
                std=[0.229],  # Grayscale std
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    else:
        # RGB transforms
        transforms = [
            A.Resize(tile_size, tile_size, interpolation=cv2.INTER_LINEAR),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225],  # ImageNet stds
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


def get_dino_multicrop_transform(
    global_crop_size: int = 384,
    local_crop_size: int = 192,
    global_crop_scale: Tuple[float, float] = (0.4, 1.0),
    local_crop_scale: Tuple[float, float] = (0.05, 0.4),
    n_global_crops: int = 2,
    n_local_crops: int = 6,
    in_channels: int = 3,
) -> Callable:
    """
    Create multi-crop augmentation for DINOv3 pretraining.

    Args:
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        n_global_crops: Number of global crops to generate
        n_local_crops: Number of local crops to generate
        in_channels: Number of input channels

    Returns:
        Callable that takes an image and returns dict with global and local crops
    """

    # Global crop augmentation pipeline
    global_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(global_crop_size, global_crop_size),
                scale=global_crop_scale,
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1.0
                    ),
                    A.ToGray(p=1.0),
                ],
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406] if in_channels == 3 else [0.5],
                std=[0.229, 0.224, 0.225] if in_channels == 3 else [0.5],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    # Local crop augmentation pipeline (more aggressive)
    local_transform = A.Compose(
        [
            A.RandomResizedCrop(
                size=(local_crop_size, local_crop_size),
                scale=local_crop_scale,
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR,
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1.0
                    ),
                    A.ToGray(p=1.0),
                ],
                p=0.8,
            ),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
            A.Solarize(threshold=128, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406] if in_channels == 3 else [0.5],
                std=[0.229, 0.224, 0.225] if in_channels == 3 else [0.5],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    def transform_fn(image):
        """Apply multi-crop augmentation to image."""
        # Ensure image is numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Handle grayscale
        if in_channels == 1 and image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=-1)
        elif in_channels == 3 and image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Generate global crops
        global_crops = []
        for _ in range(n_global_crops):
            crop = global_transform(image=image)["image"]
            global_crops.append(crop)

        # Generate local crops
        local_crops = []
        for _ in range(n_local_crops):
            crop = local_transform(image=image)["image"]
            local_crops.append(crop)

        return {
            "global_views": global_crops,
            "local_views": local_crops,
            "image": global_crops[0],  # For compatibility
        }

    return transform_fn


if __name__ == "__main__":
    test_transforms()
