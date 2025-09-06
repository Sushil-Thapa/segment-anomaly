"""
Tiled inference implementation for large image segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time
import psutil
import logging

from ..data.tiling import TileGenerator, TileStitcher
from ..data.transforms import get_test_transform
from ..models.swin_unet import SwinUNet

logger = logging.getLogger(__name__)


class TiledInference:
    """
    Tiled inference class for processing large images with memory management.
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 tile_size: int = 512,
                 stride: int = 256,
                 batch_size: int = 4,
                 num_classes: int = 2,
                 device: torch.device = None,
                 fp16: bool = True,
                 max_memory_gb: float = 10.0):
        """
        Initialize tiled inference.
        
        Args:
            model: Trained segmentation model
            tile_size: Size of tiles for inference
            stride: Stride between tiles
            batch_size: Batch size for tile processing
            num_classes: Number of output classes
            device: Device to run inference on
            fp16: Whether to use FP16 precision
            max_memory_gb: Maximum GPU memory to use in GB
        """
        self.model = model
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.fp16 = fp16
        self.max_memory_gb = max_memory_gb
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device and set precision
        self.model = self.model.to(self.device)
        if self.fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        
        self.model.eval()
        
        # Initialize tiling components
        self.tiler = TileGenerator(tile_size=tile_size, stride=stride)
        self.stitcher = TileStitcher(tile_size=tile_size, stride=stride)
        
        # Initialize transform
        self.transform = get_test_transform()
        
        # Memory monitoring
        self.memory_stats = {
            'peak_memory_mb': 0,
            'current_memory_mb': 0,
            'inference_times': []
        }
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024**2
        else:
            return psutil.Process().memory_info().rss / 1024**2
    
    def _check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        current_memory_gb = self._get_memory_usage() / 1024
        return current_memory_gb < self.max_memory_gb
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Preprocessed image
        """
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR to RGB conversion if needed
            pass
        
        return image
    
    def _postprocess_predictions(self, 
                                predictions: np.ndarray,
                                original_shape: Tuple[int, int],
                                min_area: int = 100) -> np.ndarray:
        """
        Post-process segmentation predictions.
        
        Args:
            predictions: Raw predictions (H, W, C)
            original_shape: Original image shape (H, W)
            min_area: Minimum area for connected components
            
        Returns:
            Post-processed binary mask
        """
        # Convert to binary mask
        if predictions.shape[-1] > 1:
            # Multi-class predictions
            pred_mask = np.argmax(predictions, axis=-1).astype(np.uint8)
        else:
            # Binary predictions
            pred_mask = (predictions.squeeze() > 0.5).astype(np.uint8)
        
        # Resize to original shape if needed
        if pred_mask.shape != original_shape:
            pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Closing: fill small holes
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        
        # Opening: remove small noise
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        
        # Connected components filtering
        if min_area > 0:
            pred_mask = self._filter_small_components(pred_mask, min_area)
        
        return pred_mask
    
    def _filter_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Filter out small connected components.
        
        Args:
            mask: Binary mask
            min_area: Minimum area in pixels
            
        Returns:
            Filtered mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # Filter components by area
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 1
        
        return filtered_mask.astype(np.uint8)
    
    def _process_tile_batch(self, tiles: list) -> np.ndarray:
        """
        Process a batch of tiles through the model.
        
        Args:
            tiles: List of preprocessed tile tensors
            
        Returns:
            Batch predictions array
        """
        if len(tiles) == 0:
            return np.array([])
        
        # Stack tiles into batch
        batch_tensor = torch.stack(tiles).to(self.device)
        
        if self.fp16 and self.device.type == 'cuda':
            batch_tensor = batch_tensor.half()
        
        # Forward pass
        with torch.no_grad():
            if self.fp16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
            
            # Handle multiple outputs (e.g., deep supervision)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            
            # Convert to probabilities
            if self.num_classes > 1:
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = torch.sigmoid(outputs)
            
            # Move to CPU and convert to numpy
            predictions = outputs.cpu().float().numpy()
        
        return predictions
    
    def predict_image(self, 
                     image: np.ndarray,
                     return_probabilities: bool = False,
                     min_area: int = 100) -> Dict[str, Any]:
        """
        Predict segmentation for a single image.
        
        Args:
            image: Input image (H, W, C)
            return_probabilities: Whether to return probability maps
            min_area: Minimum area for connected component filtering
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        original_shape = image.shape[:2]
        
        # Preprocess image
        image = self._preprocess_image(image)
        
        # Generate tile coordinates
        tile_coords = self.tiler.get_tile_indices(image.shape[0], image.shape[1])
        
        logger.info(f"Processing {len(tile_coords)} tiles of size {self.tile_size}x{self.tile_size}")
        
        # Process tiles in batches
        all_predictions = []
        
        for i in range(0, len(tile_coords), self.batch_size):
            # Check memory usage
            if not self._check_memory_limit():
                logger.warning("Memory limit reached, reducing batch size")
                self.batch_size = max(1, self.batch_size // 2)
            
            batch_coords = tile_coords[i:i + self.batch_size]
            batch_tiles = []
            
            # Extract and preprocess tiles
            for coords in batch_coords:
                tile = self.tiler.extract_tile(image, coords)
                
                # Apply transforms
                transformed = self.transform(image=tile)
                tile_tensor = transformed['image']
                
                batch_tiles.append(tile_tensor)
            
            # Process batch
            batch_predictions = self._process_tile_batch(batch_tiles)
            all_predictions.extend(batch_predictions)
            
            # Update memory stats
            current_memory = self._get_memory_usage()
            self.memory_stats['current_memory_mb'] = current_memory
            self.memory_stats['peak_memory_mb'] = max(
                self.memory_stats['peak_memory_mb'], 
                current_memory
            )
            
            # Log progress
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_coords)}/{len(tile_coords)} tiles")
        
        # Stitch tiles back together
        logger.info("Stitching tiles back together...")
        
        if self.num_classes > 1:
            # Multi-class segmentation
            stitched_probs = self.stitcher.stitch_tiles(
                tiles=[pred.transpose(1, 2, 0) for pred in all_predictions],
                tile_coords=tile_coords,
                output_shape=original_shape,
                num_classes=self.num_classes
            )
        else:
            # Binary segmentation
            stitched_probs = self.stitcher.stitch_tiles(
                tiles=[pred.squeeze(0) for pred in all_predictions],
                tile_coords=tile_coords,
                output_shape=original_shape,
                num_classes=None
            )
        
        # Post-process predictions
        pred_mask = self._postprocess_predictions(stitched_probs, original_shape, min_area)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.memory_stats['inference_times'].append(inference_time)
        
        logger.info(f"Inference completed in {inference_time:.2f}s")
        logger.info(f"Peak memory usage: {self.memory_stats['peak_memory_mb']:.1f} MB")
        
        # Prepare results
        results = {
            'prediction': pred_mask,
            'inference_time': inference_time,
            'memory_stats': self.memory_stats.copy(),
            'num_tiles': len(tile_coords)
        }
        
        if return_probabilities:
            results['probabilities'] = stitched_probs
        
        return results
    
    def predict_image_file(self, 
                          image_path: str,
                          output_path: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Predict segmentation for an image file.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output mask
            **kwargs: Additional arguments for predict_image
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        results = self.predict_image(image, **kwargs)
        
        # Save output if requested
        if output_path is not None:
            output_mask = results['prediction'] * 255
            cv2.imwrite(output_path, output_mask)
            logger.info(f"Saved prediction to: {output_path}")
        
        return results
    
    def benchmark(self, 
                 image_shape: Tuple[int, int, int] = (2048, 2048, 3),
                 num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            image_shape: Shape of test image (H, W, C)
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark statistics
        """
        logger.info(f"Benchmarking inference on {image_shape} image for {num_runs} runs")
        
        # Create random test image
        test_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
        
        # Warmup
        _ = self.predict_image(test_image)
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            self.memory_stats = {'peak_memory_mb': 0, 'current_memory_mb': 0, 'inference_times': []}
            
            result = self.predict_image(test_image)
            times.append(result['inference_time'])
            memory_usage.append(result['memory_stats']['peak_memory_mb'])
            
            logger.info(f"Run {i+1}/{num_runs}: {result['inference_time']:.2f}s")
        
        # Calculate statistics
        stats = {
            'avg_time_s': np.mean(times),
            'std_time_s': np.std(times),
            'min_time_s': np.min(times),
            'max_time_s': np.max(times),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'throughput_mpx_per_s': (image_shape[0] * image_shape[1] / 1e6) / np.mean(times)
        }
        
        logger.info("Benchmark results:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return stats


def create_inference_engine(checkpoint_path: str,
                          config: Dict[str, Any],
                          device: Optional[torch.device] = None) -> TiledInference:
    """
    Create inference engine from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to run inference on
        
    Returns:
        TiledInference engine
    """
    # Load model
    model = SwinUNet(
        backbone_name=config['model']['backbone'],
        pretrained=False,
        decoder_channels=config['model']['decoder_channels'],
        num_classes=2,
        dropout=config['model']['dropout']
    )
    
    # Load checkpoint
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create inference engine
    inference_engine = TiledInference(
        model=model,
        tile_size=config['data']['tile_size'],
        stride=config['data']['stride'],
        device=device,
        fp16=config['inference']['fp16'],
        max_memory_gb=config['inference']['max_memory_gb']
    )
    
    return inference_engine


def test_inference():
    """Test inference implementation."""
    # Create dummy model
    model = torch.nn.Conv2d(3, 2, 1)
    
    # Create inference engine
    inference_engine = TiledInference(
        model=model,
        tile_size=256,
        stride=128,
        device=torch.device('cpu'),
        fp16=False
    )
    
    # Test with dummy image
    test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    
    results = inference_engine.predict_image(test_image)
    
    print(f"Prediction shape: {results['prediction'].shape}")
    print(f"Inference time: {results['inference_time']:.2f}s")
    print(f"Number of tiles: {results['num_tiles']}")
    
    return True


if __name__ == "__main__":
    test_inference()
    print("Inference tests passed!")
