"""
Tiling utilities for large image segmentation with deterministic tile extraction and stitching.
Includes SAM acoustic microscopy adaptive overlap functionality.
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Optional, Union
import math


class ConfigurableTileGenerator:
    """Advanced tile generator with configurable stride for different use cases."""

    def __init__(
        self,
        tile_size: int = 512,
        train_stride: int = 384,  # Faster training stride
        val_stride: int = 256,  # Better boundaries for validation
        inference_stride: int = 384,  # Fast inference
        padding_mode: str = "reflect",
        use_gaussian_weights: bool = True,
    ):
        """
        Initialize configurable tile generator.

        Args:
            tile_size: Size of each tile (square)
            train_stride: Stride for training (can be larger for speed)
            val_stride: Stride for validation (smaller for accuracy)
            inference_stride: Stride for inference
            padding_mode: Padding mode for edge tiles
            use_gaussian_weights: Whether to use Gaussian weighting for stitching
        """
        self.tile_size = tile_size
        self.train_stride = train_stride
        self.val_stride = val_stride
        self.inference_stride = inference_stride
        self.padding_mode = padding_mode
        self.use_gaussian_weights = use_gaussian_weights

        # Pre-compute Gaussian weights for blending
        if use_gaussian_weights:
            self._gaussian_weights = self._create_gaussian_weights()
        else:
            self._gaussian_weights = np.ones((tile_size, tile_size), dtype=np.float32)

    def _create_gaussian_weights(self) -> np.ndarray:
        """Create Gaussian weights for tile blending."""
        center = self.tile_size // 2
        sigma = self.tile_size / 6.0  # 3-sigma covers most of the tile

        y, x = np.ogrid[: self.tile_size, : self.tile_size]
        gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

        # Normalize to 0-1 range
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        return gaussian.astype(np.float32)

    def get_stride_for_mode(self, mode: str) -> int:
        """Get appropriate stride for different modes."""
        stride_map = {
            "train": self.train_stride,
            "val": self.val_stride,
            "validation": self.val_stride,
            "inference": self.inference_stride,
            "test": self.val_stride,  # Use validation stride for testing
        }
        return stride_map.get(mode, self.val_stride)

    def get_tile_indices(
        self, image_height: int, image_width: int, mode: str = "val"
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get all tile indices for an image with mode-specific stride.

        Args:
            image_height: Height of the input image
            image_width: Width of the input image
            mode: Processing mode ('train', 'val', 'inference')

        Returns:
            List of tuples (start_y, end_y, start_x, end_x) for each tile
        """
        stride = self.get_stride_for_mode(mode)
        tiles = []

        # Calculate number of tiles needed
        n_tiles_y = math.ceil((image_height - self.tile_size) / stride) + 1
        n_tiles_x = math.ceil((image_width - self.tile_size) / stride) + 1

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                start_y = i * stride
                start_x = j * stride

                # Ensure we don't go beyond image boundaries
                end_y = min(start_y + self.tile_size, image_height)
                end_x = min(start_x + self.tile_size, image_width)

                # Adjust start positions if tile would be smaller than expected
                if end_y - start_y < self.tile_size:
                    start_y = max(0, end_y - self.tile_size)
                if end_x - start_x < self.tile_size:
                    start_x = max(0, end_x - self.tile_size)

                tiles.append((start_y, end_y, start_x, end_x))

        return tiles


class TileGenerator:
    """Generate tiles from large images with overlap and padding."""

    def __init__(
        self, tile_size: int = 512, stride: int = 256, padding_mode: str = "reflect"
    ):
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

    def get_tile_indices(
        self, image_height: int, image_width: int
    ) -> List[Tuple[int, int, int, int]]:
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

                tiles.append(
                    (
                        start_y,
                        min(start_y + self.tile_size, image_height),
                        start_x,
                        min(start_x + self.tile_size, image_width),
                    )
                )

        return tiles

    def extract_tile(
        self, image: np.ndarray, tile_coords: Tuple[int, int, int, int]
    ) -> np.ndarray:
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
            if self.padding_mode == "reflect":
                if len(image.shape) == 3:
                    tile = np.pad(
                        tile, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect"
                    )
                else:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode="reflect")
            elif self.padding_mode == "constant":
                if len(image.shape) == 3:
                    tile = np.pad(
                        tile, ((0, pad_y), (0, pad_x), (0, 0)), mode="constant"
                    )
                else:
                    tile = np.pad(tile, ((0, pad_y), (0, pad_x)), mode="constant")

        return tile


class TileStitcher:
    """Stitch tiles back together with overlap handling using Gaussian blending."""

    def __init__(
        self, tile_size: int = 512, stride: int = 256, gaussian_sigma: float = 1.0
    ):
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
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        # Calculate distance from center
        center_y, center_x = h // 2, w // 2
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

        # Apply Gaussian
        weights = np.exp(
            -(distances**2) / (2 * (self.gaussian_sigma * min(h, w) / 4) ** 2)
        )

        self._weight_cache[cache_key] = weights
        return weights

    def stitch_tiles(
        self,
        tiles: List[np.ndarray],
        tile_coords: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int],
        num_classes: Optional[int] = None,
    ) -> np.ndarray:
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
                    output_slice = output[
                        place_start_y:place_end_y, place_start_x:place_end_x, c
                    ]

                    # Both tile_slice and weights need to match output_slice shape
                    if tile_slice.shape != output_slice.shape:
                        tile_slice = np.resize(tile_slice, output_slice.shape)

                    if weights.shape != output_slice.shape:
                        weights_resized = np.resize(weights, output_slice.shape)
                    else:
                        weights_resized = weights

                    output[place_start_y:place_end_y, place_start_x:place_end_x, c] += (
                        tile_slice * weights_resized
                    )
            else:
                if len(tile.shape) == 3:
                    for c in range(tile.shape[2]):
                        # Get the tile slice and output slice
                        tile_slice = tile_crop[:, :, c]
                        output_slice = output[
                            place_start_y:place_end_y, place_start_x:place_end_x, c
                        ]

                        # Both tile_slice and weights need to match output_slice shape
                        if tile_slice.shape != output_slice.shape:
                            tile_slice = np.resize(tile_slice, output_slice.shape)

                        if weights.shape != output_slice.shape:
                            weights_resized = np.resize(weights, output_slice.shape)
                        else:
                            weights_resized = weights

                        output[
                            place_start_y:place_end_y, place_start_x:place_end_x, c
                        ] += (tile_slice * weights_resized)
                else:
                    # 2D case
                    output_slice = output[
                        place_start_y:place_end_y, place_start_x:place_end_x
                    ]

                    if tile_crop.shape != output_slice.shape:
                        tile_crop = np.resize(tile_crop, output_slice.shape)

                    if weights.shape != output_slice.shape:
                        weights_resized = np.resize(weights, output_slice.shape)
                    else:
                        weights_resized = weights

                    output[place_start_y:place_end_y, place_start_x:place_end_x] += (
                        tile_crop * weights_resized
                    )

            # Accumulate weights
            target_h = place_end_y - place_start_y
            target_w = place_end_x - place_start_x
            weight_slice = weight_sum[
                place_start_y:place_end_y, place_start_x:place_end_x
            ]

            if weights.shape != weight_slice.shape:
                weights_for_sum = np.resize(weights, weight_slice.shape)
            else:
                weights_for_sum = weights
            weight_sum[
                place_start_y:place_end_y, place_start_x:place_end_x
            ] += weights_for_sum

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


class SAMAdaptiveTileGenerator:
    """SAM-specific tile generator with adaptive overlap (50-75%) and defect-aware positioning."""

    def __init__(
        self,
        tile_size: int = 512,
        overlap_range: Tuple[float, float] = (0.5, 0.75),  # 50-75% overlap as specified
        padding_mode: str = "reflect",
        defect_avoidance: bool = True,
    ):
        """
        Initialize SAM adaptive tile generator.

        Args:
            tile_size: Size of each tile (square)
            overlap_range: Range of overlap percentages (min, max)
            padding_mode: Padding mode for edge tiles
            defect_avoidance: Whether to avoid cutting through large defects
        """
        self.tile_size = tile_size
        self.overlap_range = overlap_range
        self.padding_mode = padding_mode
        self.defect_avoidance = defect_avoidance

    def _calculate_adaptive_stride(
        self, image_shape: Tuple[int, int], defect_map: Optional[np.ndarray] = None
    ) -> Tuple[int, int]:
        """
        Calculate adaptive stride based on image characteristics and defects.

        Args:
            image_shape: Shape of the input image (H, W)
            defect_map: Binary map of defect locations (optional)

        Returns:
            Tuple of (stride_y, stride_x)
        """
        h, w = image_shape

        # Start with random overlap within specified range
        overlap_y = np.random.uniform(self.overlap_range[0], self.overlap_range[1])
        overlap_x = np.random.uniform(self.overlap_range[0], self.overlap_range[1])

        base_stride_y = int(self.tile_size * (1 - overlap_y))
        base_stride_x = int(self.tile_size * (1 - overlap_x))

        # Ensure minimum stride
        stride_y = max(self.tile_size // 4, base_stride_y)
        stride_x = max(self.tile_size // 4, base_stride_x)

        return stride_y, stride_x

    def _analyze_defect_density(
        self, defect_map: np.ndarray, region: Tuple[int, int, int, int]
    ) -> float:
        """
        Analyze defect density in a given region.

        Args:
            defect_map: Binary defect map
            region: Tuple of (start_y, end_y, start_x, end_x)

        Returns:
            Defect density (0.0 to 1.0)
        """
        start_y, end_y, start_x, end_x = region
        roi = defect_map[start_y:end_y, start_x:end_x]
        return np.mean(roi) if roi.size > 0 else 0.0

    def get_adaptive_tile_indices(
        self,
        image_height: int,
        image_width: int,
        defect_map: Optional[np.ndarray] = None,
        mode: str = "sam",
    ) -> List[Tuple[int, int, int, int]]:
        """
        Get adaptive tile indices with SAM-specific overlap and defect awareness.

        Args:
            image_height: Height of the input image
            image_width: Width of the input image
            defect_map: Optional binary defect map for guided tiling
            mode: Tiling mode (affects overlap strategy)

        Returns:
            List of tuples (start_y, end_y, start_x, end_x) for each tile
        """
        stride_y, stride_x = self._calculate_adaptive_stride(
            (image_height, image_width), defect_map
        )

        tiles = []

        # Calculate number of tiles with adaptive stride
        n_tiles_y = math.ceil((image_height - self.tile_size) / stride_y) + 1
        n_tiles_x = math.ceil((image_width - self.tile_size) / stride_x) + 1

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # Base position
                start_y = i * stride_y
                start_x = j * stride_x

                # Defect-aware adjustment
                if self.defect_avoidance and defect_map is not None:
                    start_y, start_x = self._adjust_for_defects(
                        start_y, start_x, defect_map, image_height, image_width
                    )

                # Ensure we don't go beyond image boundaries
                end_y = min(start_y + self.tile_size, image_height)
                end_x = min(start_x + self.tile_size, image_width)

                # Adjust start positions if tile would be smaller than expected
                if end_y - start_y < self.tile_size:
                    start_y = max(0, end_y - self.tile_size)
                if end_x - start_x < self.tile_size:
                    start_x = max(0, end_x - self.tile_size)

                tiles.append((start_y, end_y, start_x, end_x))

        return tiles

    def _adjust_for_defects(
        self,
        start_y: int,
        start_x: int,
        defect_map: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> Tuple[int, int]:
        """
        Adjust tile position to avoid cutting through large defects.

        Args:
            start_y: Initial Y position
            start_x: Initial X position
            defect_map: Binary defect map
            image_height: Image height
            image_width: Image width

        Returns:
            Adjusted (start_y, start_x) position
        """
        adjustment_range = 32  # Maximum adjustment in pixels
        best_y, best_x = start_y, start_x
        min_edge_defects = float("inf")

        # Try small adjustments around the initial position
        for dy in range(-adjustment_range, adjustment_range + 1, 8):
            for dx in range(-adjustment_range, adjustment_range + 1, 8):
                adj_y = max(0, min(start_y + dy, image_height - self.tile_size))
                adj_x = max(0, min(start_x + dx, image_width - self.tile_size))

                # Count defects near tile edges
                edge_defects = self._count_edge_defects(
                    adj_y, adj_x, defect_map, image_height, image_width
                )

                if edge_defects < min_edge_defects:
                    min_edge_defects = edge_defects
                    best_y, best_x = adj_y, adj_x

        return best_y, best_x

    def _count_edge_defects(
        self,
        start_y: int,
        start_x: int,
        defect_map: np.ndarray,
        image_height: int,
        image_width: int,
        edge_width: int = 16,
    ) -> float:
        """
        Count defects near tile edges to minimize boundary artifacts.

        Args:
            start_y: Tile Y position
            start_x: Tile X position
            defect_map: Binary defect map
            image_height: Image height
            image_width: Image width
            edge_width: Width of edge region to analyze

        Returns:
            Number of edge defects
        """
        end_y = min(start_y + self.tile_size, image_height)
        end_x = min(start_x + self.tile_size, image_width)

        edge_defects = 0

        # Top and bottom edges
        if start_y > 0:
            top_edge = defect_map[max(0, start_y - edge_width) : start_y, start_x:end_x]
            edge_defects += np.sum(top_edge)

        if end_y < image_height:
            bottom_edge = defect_map[
                end_y : min(end_y + edge_width, image_height), start_x:end_x
            ]
            edge_defects += np.sum(bottom_edge)

        # Left and right edges
        if start_x > 0:
            left_edge = defect_map[
                start_y:end_y, max(0, start_x - edge_width) : start_x
            ]
            edge_defects += np.sum(left_edge)

        if end_x < image_width:
            right_edge = defect_map[
                start_y:end_y, end_x : min(end_x + edge_width, image_width)
            ]
            edge_defects += np.sum(right_edge)

        return edge_defects


class SAMAdaptiveStitcher(TileStitcher):
    """Enhanced tile stitcher for SAM with adaptive overlap handling."""

    def __init__(
        self,
        tile_size: int = 512,
        min_overlap: float = 0.5,
        gaussian_sigma: float = 1.2,  # Slightly wider Gaussian for SAM
        confidence_weighting: bool = True,
    ):
        """
        Initialize SAM adaptive stitcher.

        Args:
            tile_size: Size of each tile
            min_overlap: Minimum expected overlap ratio
            gaussian_sigma: Sigma for Gaussian weight calculation
            confidence_weighting: Whether to use prediction confidence for weighting
        """
        # Calculate effective stride from minimum overlap
        effective_stride = int(tile_size * (1 - min_overlap))
        super().__init__(tile_size, effective_stride, gaussian_sigma)

        self.min_overlap = min_overlap
        self.confidence_weighting = confidence_weighting

    def _get_confidence_weights(
        self, tile: np.ndarray, tile_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Calculate confidence-based weights from prediction uncertainty.

        Args:
            tile: Tile prediction array
            tile_shape: Shape of the tile (H, W)

        Returns:
            Confidence weight matrix
        """
        # For binary segmentation, confidence is distance from 0.5
        if len(tile.shape) == 2:
            confidence = np.abs(tile - 0.5) * 2
        elif len(tile.shape) == 3 and tile.shape[2] == 1:
            confidence = np.abs(tile[:, :, 0] - 0.5) * 2
        else:
            # For multi-class, use max probability as confidence
            confidence = np.max(tile, axis=2) if len(tile.shape) == 3 else tile

        # Apply smoothing
        confidence = cv2.GaussianBlur(confidence.astype(np.float32), (5, 5), 1.0)

        return confidence

    def stitch_tiles_adaptive(
        self,
        tiles: List[np.ndarray],
        tile_coords: List[Tuple[int, int, int, int]],
        output_shape: Tuple[int, int],
        num_classes: Optional[int] = None,
        confidence_maps: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Stitch tiles with adaptive overlap handling and confidence weighting.

        Args:
            tiles: List of tile predictions
            tile_coords: List of tile coordinates
            output_shape: Shape of the output image (H, W)
            num_classes: Number of classes (for segmentation masks)
            confidence_maps: Optional confidence maps for each tile

        Returns:
            Stitched image
        """
        h, w = output_shape

        if num_classes is not None:
            output = np.zeros((h, w, num_classes), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
        else:
            if len(tiles[0].shape) == 3:
                output = np.zeros((h, w, tiles[0].shape[2]), dtype=np.float32)
            else:
                output = np.zeros((h, w), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)

        for idx, (tile, coords) in enumerate(zip(tiles, tile_coords)):
            start_y, end_y, start_x, end_x = coords

            # Get actual tile dimensions
            actual_h = min(self.tile_size, end_y - start_y)
            actual_w = min(self.tile_size, end_x - start_x)

            # Crop tile to actual size
            if len(tile.shape) == 3:
                tile_crop = tile[:actual_h, :actual_w, :]
            else:
                tile_crop = tile[:actual_h, :actual_w]

            # Get Gaussian weights
            gaussian_weights = self._get_gaussian_weights((actual_h, actual_w))

            # Get confidence weights if enabled
            if self.confidence_weighting:
                if confidence_maps and idx < len(confidence_maps):
                    conf_weights = confidence_maps[idx][:actual_h, :actual_w]
                else:
                    conf_weights = self._get_confidence_weights(
                        tile_crop, (actual_h, actual_w)
                    )

                # Combine Gaussian and confidence weights
                combined_weights = gaussian_weights * conf_weights
            else:
                combined_weights = gaussian_weights

            # Calculate placement coordinates
            place_start_y = start_y
            place_end_y = start_y + actual_h
            place_start_x = start_x
            place_end_x = start_x + actual_w

            # Accumulate weighted predictions
            if num_classes is not None:
                for c in range(num_classes):
                    tile_slice = tile_crop[:, :, c]
                    output_slice = output[
                        place_start_y:place_end_y, place_start_x:place_end_x, c
                    ]

                    if tile_slice.shape != output_slice.shape:
                        tile_slice = np.resize(tile_slice, output_slice.shape)

                    if combined_weights.shape != output_slice.shape:
                        weights_resized = np.resize(
                            combined_weights, output_slice.shape
                        )
                    else:
                        weights_resized = combined_weights

                    output[place_start_y:place_end_y, place_start_x:place_end_x, c] += (
                        tile_slice * weights_resized
                    )
            else:
                if len(tile.shape) == 3:
                    for c in range(tile.shape[2]):
                        tile_slice = tile_crop[:, :, c]
                        output_slice = output[
                            place_start_y:place_end_y, place_start_x:place_end_x, c
                        ]

                        if tile_slice.shape != output_slice.shape:
                            tile_slice = np.resize(tile_slice, output_slice.shape)

                        if combined_weights.shape != output_slice.shape:
                            weights_resized = np.resize(
                                combined_weights, output_slice.shape
                            )
                        else:
                            weights_resized = combined_weights

                        output[
                            place_start_y:place_end_y, place_start_x:place_end_x, c
                        ] += (tile_slice * weights_resized)
                else:
                    output_slice = output[
                        place_start_y:place_end_y, place_start_x:place_end_x
                    ]

                    if tile_crop.shape != output_slice.shape:
                        tile_crop = np.resize(tile_crop, output_slice.shape)

                    if combined_weights.shape != output_slice.shape:
                        weights_resized = np.resize(
                            combined_weights, output_slice.shape
                        )
                    else:
                        weights_resized = combined_weights

                    output[place_start_y:place_end_y, place_start_x:place_end_x] += (
                        tile_crop * weights_resized
                    )

            # Accumulate weights
            weight_slice = weight_sum[
                place_start_y:place_end_y, place_start_x:place_end_x
            ]

            if combined_weights.shape != weight_slice.shape:
                weights_for_sum = np.resize(combined_weights, weight_slice.shape)
            else:
                weights_for_sum = combined_weights

            weight_sum[
                place_start_y:place_end_y, place_start_x:place_end_x
            ] += weights_for_sum

        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-8)

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
    tiler = TileGenerator(
        tile_size=512, stride=512
    )  # No overlap for perfect reconstruction
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
