"""
Dataset implementation for wafer defect segmentation with tiling support.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List, Tuple, Dict, Optional, Callable, Union
import glob
from pathlib import Path
import pickle
from .tiling import TileGenerator
from .transforms import get_train_transform, get_val_transform


class WaferTileDataset(Dataset):
    """
    Dataset for wafer defect segmentation with tile-based loading and caching.
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 tile_size: int = 512,
                 stride: int = 256,
                 transform: Optional[Callable] = None,
                 oversample_ratio: float = 3.0,
                 cache_tiles: bool = True,
                 precompute_tiles: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing images and masks
            split: Dataset split ('train', 'val', 'test')
            tile_size: Size of tiles to extract
            stride: Stride for tile extraction
            transform: Transform pipeline to apply
            oversample_ratio: Ratio of positive to negative samples
            cache_tiles: Whether to cache extracted tiles
            precompute_tiles: Whether to precompute all tile indices
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tile_size = tile_size
        self.stride = stride
        self.transform = transform
        self.oversample_ratio = oversample_ratio
        self.cache_tiles = cache_tiles
        
        # Initialize tiler
        self.tiler = TileGenerator(tile_size=tile_size, stride=stride)
        
        # Load file paths
        self.image_paths = []
        self.mask_paths = []
        self._load_file_paths()
        
        # Precompute tile information
        self.tile_info = []  # List of (image_idx, tile_coords, has_defect)
        self.tile_cache = {}  # Cache for tiles
        
        if precompute_tiles:
            self._precompute_tiles()
        
        # Setup weighted sampling for training
        if split == 'train' and oversample_ratio > 1.0:
            self.sample_weights = self._compute_sample_weights()
        else:
            self.sample_weights = None
    
    def _load_file_paths(self):
        """Load image and mask file paths."""
        images_dir = self.data_root / 'images' / self.split
        masks_dir = self.data_root / 'masks' / self.split
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(str(images_dir / ext)))
        
        self.image_paths.sort()
        
        # Find corresponding mask files
        for img_path in self.image_paths:
            img_name = Path(img_path).stem
            mask_path = None
            
            # Try different mask extensions
            for ext in ['png', 'jpg', 'jpeg', 'tiff', 'tif']:
                potential_mask = masks_dir / f"{img_name}.{ext}"
                if potential_mask.exists():
                    mask_path = str(potential_mask)
                    break
            
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for image {img_path}")
            
            self.mask_paths.append(mask_path)
        
        print(f"Loaded {len(self.image_paths)} images for {self.split} split")
    
    def _precompute_tiles(self):
        """Precompute tile information for all images."""
        cache_path = self.data_root / f"tile_cache_{self.split}_{self.tile_size}_{self.stride}.pkl"
        
        if cache_path.exists():
            print(f"Loading precomputed tiles from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tile_info = pickle.load(f)
            return
        
        print("Precomputing tile information...")
        self.tile_info = []
        
        for img_idx, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            # Load image to get dimensions
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            height, width = image.shape[:2]
            
            # Get tile coordinates
            tile_coords = self.tiler.get_tile_indices(height, width)
            
            # Load mask to check for defects in each tile
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            for coords in tile_coords:
                start_y, end_y, start_x, end_x = coords
                tile_mask = mask[start_y:end_y, start_x:end_x]
                
                # Check if tile contains defects
                has_defect = np.any(tile_mask > 0)
                
                self.tile_info.append({
                    'image_idx': img_idx,
                    'coords': coords,
                    'has_defect': has_defect
                })
        
        # Save precomputed information
        with open(cache_path, 'wb') as f:
            pickle.dump(self.tile_info, f)
        
        print(f"Precomputed {len(self.tile_info)} tiles")
    
    def _compute_sample_weights(self) -> torch.Tensor:
        """Compute sample weights for weighted random sampling."""
        positive_count = sum(1 for tile in self.tile_info if tile['has_defect'])
        negative_count = len(self.tile_info) - positive_count
        
        if positive_count == 0:
            return torch.ones(len(self.tile_info))
        
        # Weight calculation: inversely proportional to class frequency
        pos_weight = 1.0 / positive_count
        neg_weight = 1.0 / negative_count
        
        # Apply oversampling ratio
        pos_weight *= self.oversample_ratio
        
        weights = []
        for tile in self.tile_info:
            if tile['has_defect']:
                weights.append(pos_weight)
            else:
                weights.append(neg_weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _load_tile(self, image_idx: int, coords: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Load a tile from an image and mask."""
        cache_key = (image_idx, coords)
        
        if self.cache_tiles and cache_key in self.tile_cache:
            return self.tile_cache[cache_key]
        
        # Load full image and mask
        img_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Extract tile
        image_tile = self.tiler.extract_tile(image, coords)
        mask_tile = self.tiler.extract_tile(mask, coords)
        
        # Ensure mask is binary
        mask_tile = (mask_tile > 0).astype(np.uint8)
        
        if self.cache_tiles:
            self.tile_cache[cache_key] = (image_tile, mask_tile)
        
        return image_tile, mask_tile
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.tile_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tile."""
        tile_info = self.tile_info[idx]
        image_idx = tile_info['image_idx']
        coords = tile_info['coords']
        
        # Load tile
        image_tile, mask_tile = self._load_tile(image_idx, coords)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image_tile, mask=mask_tile)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask']
        else:
            image_tensor = torch.from_numpy(image_tile.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_tile).long()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_idx': image_idx,
            'coords': coords
        }
    
    def get_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get weighted random sampler for training."""
        if self.sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        positive_count = sum(1 for tile in self.tile_info if tile['has_defect'])
        negative_count = len(self.tile_info) - positive_count
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'total': len(self.tile_info),
            'positive_ratio': positive_count / len(self.tile_info) if len(self.tile_info) > 0 else 0
        }


def create_dataloaders(config: dict, 
                      num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_train_transform(tile_size=config['data']['tile_size'])
    val_transform = get_val_transform(tile_size=config['data']['tile_size'])
    
    # Create datasets
    train_dataset = WaferTileDataset(
        data_root=config['data']['root'],
        split='train',
        tile_size=config['data']['tile_size'],
        stride=config['data']['stride'],
        transform=train_transform,
        oversample_ratio=config['data']['oversample_ratio']
    )
    
    val_dataset = WaferTileDataset(
        data_root=config['data']['root'],
        split='val',
        tile_size=config['data']['tile_size'],
        stride=config['data']['stride'],
        transform=val_transform,
        oversample_ratio=1.0  # No oversampling for validation
    )
    
    test_dataset = WaferTileDataset(
        data_root=config['data']['root'],
        split='test',
        tile_size=config['data']['tile_size'],
        stride=config['data']['stride'],
        transform=val_transform,
        oversample_ratio=1.0  # No oversampling for test
    )
    
    # Create samplers
    train_sampler = train_dataset.get_sampler()
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset functionality."""
    # Create dummy data structure
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (temp_path / 'images' / split).mkdir(parents=True)
            (temp_path / 'masks' / split).mkdir(parents=True)
            
            # Create dummy images and masks
            for i in range(3):
                # Create dummy image
                image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
                cv2.imwrite(str(temp_path / 'images' / split / f'image_{i}.png'), image)
                
                # Create dummy mask
                mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8) * 255
                cv2.imwrite(str(temp_path / 'masks' / split / f'image_{i}.png'), mask)
        
        # Test dataset
        dataset = WaferTileDataset(
            data_root=str(temp_path),
            split='train',
            tile_size=512,
            stride=256,
            precompute_tiles=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # Test sample loading
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        
        return True


if __name__ == "__main__":
    test_dataset()
    print("Dataset tests passed!")
