"""
Data augmentation transforms for wafer defect segmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Callable, Dict, Any


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(tile_size: int = 512) -> Callable:
    """
    Get training augmentation pipeline.
    
    Args:
        tile_size: Size of input tiles
        
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Geometric augmentations
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        A.CLAHE(
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            p=0.5
        ),
        
        # Heavy augmentations
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            border_mode=cv2.BORDER_REFLECT,
            p=0.3
        ),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            mask_fill_value=0,
            p=0.4
        ),
        
        # Additional augmentations
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            border_mode=cv2.BORDER_REFLECT,
            p=0.2
        ),
        A.OpticalDistortion(
            distort_limit=0.1,
            shift_limit=0.1,
            border_mode=cv2.BORDER_REFLECT,
            p=0.2
        ),
        
        # Color augmentations
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.2
        ),
        
        # Noise
        A.GaussNoise(
            var_limit=(10, 50),
            mean=0,
            p=0.2
        ),
        
        # Normalization and tensor conversion
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transform(tile_size: int = 512) -> Callable:
    """
    Get validation transform pipeline (no augmentations).
    
    Args:
        tile_size: Size of input tiles
        
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_test_transform() -> Callable:
    """
    Get test transform pipeline (same as validation).
    
    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


class MixUp:
    """MixUp augmentation for segmentation."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply MixUp to a batch.
        
        Args:
            batch: Batch dictionary with 'image' and 'mask' keys
            
        Returns:
            Mixed batch
        """
        images = batch['image']
        masks = batch['mask']
        
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Mix images and masks
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_masks = lam * masks + (1 - lam) * masks[indices]
        
        return {
            'image': mixed_images,
            'mask': mixed_masks,
            'lambda': lam,
            'indices': indices
        }


class CutMix:
    """CutMix augmentation for segmentation."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CutMix to a batch.
        
        Args:
            batch: Batch dictionary with 'image' and 'mask' keys
            
        Returns:
            CutMixed batch
        """
        images = batch['image']
        masks = batch['mask']
        
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Generate random bounding box
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_masks = masks.clone()
        
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        mixed_masks[:, :, bbx1:bbx2, bby1:bby2] = masks[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to actual area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return {
            'image': mixed_images,
            'mask': mixed_masks,
            'lambda': lam,
            'indices': indices
        }


def test_transforms():
    """Test transform pipelines."""
    import torch
    
    # Create dummy data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
    
    # Test training transforms
    train_transform = get_train_transform()
    augmented = train_transform(image=image, mask=mask)
    
    print(f"Original image shape: {image.shape}")
    print(f"Augmented image shape: {augmented['image'].shape}")
    print(f"Original mask shape: {mask.shape}")
    print(f"Augmented mask shape: {augmented['mask'].shape}")
    
    # Test validation transforms
    val_transform = get_val_transform()
    val_data = val_transform(image=image, mask=mask)
    
    print(f"Validation image shape: {val_data['image'].shape}")
    print(f"Validation mask shape: {val_data['mask'].shape}")
    
    return True


if __name__ == "__main__":
    test_transforms()
    print("Transform tests passed!")
