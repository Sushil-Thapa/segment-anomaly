"""
Tiling utilities for large image segmentation with deterministic tile extraction and stitching.
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Optional
import math


class TileGenerator:
    """Generate tiles from large images with overlap and padding."""
    
    def __init__(self, tile_size: int = 512, stride: int = 256, padding_mode: str = 'reflect'):
        """
        Initialize tile generator.
        
        Args:
            tile_size: Size of each tile (square)
            stride: Stride between tiles (controls overlap)
            padding_mode: Padding mode for edge tiles ('reflect', 'constant')
        """
        self.tile_size = tile_size
        self.stride = stride
        self.padding_mode = padding_mode
        
    def get_tile_indices(self, image_height: int, image_width: int) -> List[Tuple[int, int, int, int]]:
        """
        Get all tile indices for an image.
        
        Args:
            image_height: Height of the input image
            image_width: Width of the input image
            
        Returns:
            List of tuples (start_y, end_y, start_x, end_x) for each tile
        """
        tiles = []
        
        # Calculate number of tiles needed
        n_tiles_y = math.ceil((image_height - self.tile_size) / self.stride) + 1
        n_tiles_x = math.ceil((image_width - self.tile_size) / self.stride) + 1
        
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                start_y = i * self.stride
                start_x = j * self.stride
                
                # Ensure we don't go beyond image boundaries
                end_y = min(start_y + self.tile_size, image_height)
                end_x = min(start_x + self.tile_size, image_width)
                
                # Adjust start coordinates if tile would be smaller than tile_size
                if end_y - start_y < self.tile_size:
                    start_y = max(0, end_y - self.tile_size)
                if end_x - start_x < self.tile_size:
                    start_x = max(0, end_x - self.tile_size)
                    
                tiles.append((start_y, min(start_y + self.tile_size, image_height),
                             start_x, min(start_x + self.tile_size, image_width)))
        
        return tiles
    
    def extract_tile(self, image: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a single tile from an image with padding if necessary.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            tile_coords: Tuple of (start_y, end_y, start_x, end_x)
            
        Returns:
            Extracted tile of size (tile_size, tile_size, C) or (tile_size, tile_size)
        """
        start_y, end_y, start_x, end_x = tile_coords
        
        # Extract the tile
        if len(image.shape) == 3:
            tile = image[start_y:end_y, start_x:end_x, :]
        else:
            tile = image[start_y:end_y, start_x:end_x]
        
        # Pad if necessary
        pad_y = self.tile_size - tile.shape[0]
        pad_x = self.tile_size - tile.shape[1]
        
        if pad_y > 0 or pad_x > 0:
            if self.padding_mode == 'reflect':
                if len(image.shape) == 3:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode='reflect')
                else:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode='reflect')
            elif self.padding_mode == 'constant':
                if len(image.shape) == 3:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
                else:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode='constant')
        
        return tile


class TileStitcher:
    """Stitch tiles back together with overlap handling using Gaussian blending."""
    
    def __init__(self, tile_size: int = 512, stride: int = 256, gaussian_sigma: float = 1.0):
        """
        Initialize tile stitcher.
        
        Args:
            tile_size: Size of each tile
            stride: Stride used during tiling
            gaussian_sigma: Sigma for Gaussian weight calculation
        """
        self.tile_size = tile_size
        self.stride = stride
        self.gaussian_sigma = gaussian_sigma
        self._weight_cache = {}
        
    def _get_gaussian_weights(self, tile_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate Gaussian weights for a tile.
        
        Args:
            tile_shape: Shape of the tile (H, W)
            
        Returns:
            Weight matrix with same spatial dimensions as tile
        """
        cache_key = tile_shape
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]
            
        h, w = tile_shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Calculate distance from center
        center_y, center_x = h // 2, w // 2
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        
        # Apply Gaussian
        weights = np.exp(-(distances ** 2) / (2 * (self.gaussian_sigma * min(h, w) / 4) ** 2))
        
        self._weight_cache[cache_key] = weights
        return weights
    
    def stitch_tiles(self, 
                     tiles: List[np.ndarray], 
                     tile_coords: List[Tuple[int, int, int, int]], 
                     output_shape: Tuple[int, int],
                     num_classes: Optional[int] = None) -> np.ndarray:
        """
        Stitch tiles back into a full image using Gaussian weighted blending.
        
        Args:
            tiles: List of tile predictions
            tile_coords: List of tile coordinates
            output_shape: Shape of the output image (H, W)
            num_classes: Number of classes (for segmentation masks)
            
        Returns:
            Stitched image
        """
        h, w = output_shape
        
        if num_classes is not None:
            # For segmentation masks
            output = np.zeros((h, w, num_classes), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
        else:
            # For single channel or RGB images
            if len(tiles[0].shape) == 3:
                output = np.zeros((h, w, tiles[0].shape[2]), dtype=np.float32)
            else:
                output = np.zeros((h, w), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
        
        for tile, (start_y, end_y, start_x, end_x) in zip(tiles, tile_coords):
            # Get actual tile dimensions (may be smaller than tile_size at edges)
            actual_h = min(self.tile_size, end_y - start_y)
            actual_w = min(self.tile_size, end_x - start_x)
            
            # Crop tile to actual size
            if len(tile.shape) == 3:
                tile_crop = tile[:actual_h, :actual_w, :]
            else:
                tile_crop = tile[:actual_h, :actual_w]
            
            # Get weights for this tile size
            weights = self._get_gaussian_weights((actual_h, actual_w))
            
            # Calculate actual placement coordinates
            place_start_y = start_y
            place_end_y = start_y + actual_h
            place_start_x = start_x
            place_end_x = start_x + actual_w
            
            # Accumulate weighted predictions
            if num_classes is not None:
                for c in range(num_classes):
                    # Get the tile slice and output slice
                    tile_slice = tile_crop[:, :, c]
                    output_slice = output[place_start_y:place_end_y, place_start_x:place_end_x, c]
                    
                    # Both tile_slice and weights need to match output_slice shape
                    if tile_slice.shape != output_slice.shape:
                        tile_slice = np.resize(tile_slice, output_slice.shape)
                    
                    if weights.shape != output_slice.shape:
                        weights_resized = np.resize(weights, output_slice.shape)
                    else:
                        weights_resized = weights
                    
                    output[place_start_y:place_end_y, place_start_x:place_end_x, c] += \
                        tile_slice * weights_resized
            else:
                if len(tile.shape) == 3:
                    for c in range(tile.shape[2]):
                        # Get the tile slice and output slice
                        tile_slice = tile_crop[:, :, c]
                        output_slice = output[place_start_y:place_end_y, place_start_x:place_end_x, c]
                        
                        # Both tile_slice and weights need to match output_slice shape
                        if tile_slice.shape != output_slice.shape:
                            tile_slice = np.resize(tile_slice, output_slice.shape)
                        
                        if weights.shape != output_slice.shape:
                            weights_resized = np.resize(weights, output_slice.shape)
                        else:
                            weights_resized = weights
                        
                        output[place_start_y:place_end_y, place_start_x:place_end_x, c] += \
                            tile_slice * weights_resized
                else:
                    # 2D case
                    output_slice = output[place_start_y:place_end_y, place_start_x:place_end_x]
                    
                    if tile_crop.shape != output_slice.shape:
                        tile_crop = np.resize(tile_crop, output_slice.shape)
                    
                    if weights.shape != output_slice.shape:
                        weights_resized = np.resize(weights, output_slice.shape)
                    else:
                        weights_resized = weights
                    
                    output[place_start_y:place_end_y, place_start_x:place_end_x] += \
                        tile_crop * weights_resized
            
            # Accumulate weights
            target_h = place_end_y - place_start_y
            target_w = place_end_x - place_start_x
            weight_slice = weight_sum[place_start_y:place_end_y, place_start_x:place_end_x]
            
            if weights.shape != weight_slice.shape:
                weights_for_sum = np.resize(weights, weight_slice.shape)
            else:
                weights_for_sum = weights
            weight_sum[place_start_y:place_end_y, place_start_x:place_end_x] += weights_for_sum
        
        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-8)  # Avoid division by zero
        
        if num_classes is not None:
            for c in range(num_classes):
                output[:, :, c] /= weight_sum
        else:
            if len(output.shape) == 3:
                for c in range(output.shape[2]):
                    output[:, :, c] /= weight_sum
            else:
                output /= weight_sum
        
        return output


def test_perfect_reconstruction():
    """Test that tiling and stitching produces perfect reconstruction on non-overlapping regions."""
    # Create test image
    test_image = np.random.rand(1000, 1000, 3).astype(np.float32)
    
    # Initialize tiler and stitcher
    tiler = TileGenerator(tile_size=512, stride=512)  # No overlap for perfect reconstruction
    stitcher = TileStitcher(tile_size=512, stride=512)
    
    # Generate tiles
    tile_coords = tiler.get_tile_indices(test_image.shape[0], test_image.shape[1])
    tiles = [tiler.extract_tile(test_image, coords) for coords in tile_coords]
    
    # Stitch back
    reconstructed = stitcher.stitch_tiles(tiles, tile_coords, test_image.shape[:2])
    
    # Check reconstruction error
    mse = np.mean((test_image - reconstructed) ** 2)
    print(f"Perfect reconstruction MSE: {mse}")
    assert mse < 1e-6, f"Reconstruction error too high: {mse}"
    
    return True


if __name__ == "__main__":
    test_perfect_reconstruction()
    print("Tiling tests passed!")
