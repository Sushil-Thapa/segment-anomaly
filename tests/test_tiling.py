"""
Test tiling functionality for perfect reconstruction.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.tiling import TileGenerator, TileStitcher


def test_perfect_reconstruction():
    """Test that tiling and stitching produces perfect reconstruction."""
    print("Testing perfect reconstruction with no overlap...")
    
    # Create test image
    test_image = np.random.rand(1000, 1200, 3).astype(np.float32)
    
    # Initialize tiler and stitcher with no overlap
    tile_size = 256
    stride = 256  # No overlap
    
    tiler = TileGenerator(tile_size=tile_size, stride=stride)
    stitcher = TileStitcher(tile_size=tile_size, stride=stride)
    
    # Generate tiles
    tile_coords = tiler.get_tile_indices(test_image.shape[0], test_image.shape[1])
    tiles = [tiler.extract_tile(test_image, coords) for coords in tile_coords]
    
    print(f"Generated {len(tiles)} tiles")
    
    # Stitch back
    reconstructed = stitcher.stitch_tiles(tiles, tile_coords, test_image.shape[:2])
    
    # Check reconstruction error
    mse = np.mean((test_image - reconstructed) ** 2)
    print(f"Perfect reconstruction MSE: {mse}")
    
    assert mse < 1e-6, f"Reconstruction error too high: {mse}"
    print("✓ Perfect reconstruction test passed!")
    
    assert True  # Test passed


def test_overlapping_reconstruction():
    """Test reconstruction with overlap and Gaussian blending."""
    print("Testing reconstruction with overlap...")
    
    # Create test image with distinct regions
    test_image = np.zeros((512, 512, 3), dtype=np.float32)
    test_image[100:200, 100:200, :] = 1.0
    test_image[300:400, 300:400, :] = 0.5
    
    # Initialize tiler and stitcher with overlap
    tile_size = 256
    stride = 128  # 50% overlap
    
    tiler = TileGenerator(tile_size=tile_size, stride=stride)
    stitcher = TileStitcher(tile_size=tile_size, stride=stride)
    
    # Generate tiles
    tile_coords = tiler.get_tile_indices(test_image.shape[0], test_image.shape[1])
    tiles = [tiler.extract_tile(test_image, coords) for coords in tile_coords]
    
    print(f"Generated {len(tiles)} tiles with {stride}/{tile_size} overlap")
    
    # Stitch back
    reconstructed = stitcher.stitch_tiles(tiles, tile_coords, test_image.shape[:2])
    
    # Check that reconstruction preserves the original structure
    diff = np.abs(test_image - reconstructed)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    assert max_diff < 0.1, f"Maximum difference too high: {max_diff}"
    assert mean_diff < 0.01, f"Mean difference too high: {mean_diff}"
    print("✓ Overlapping reconstruction test passed!")
    
    assert True  # Test passed


def test_edge_cases():
    """Test edge cases like small images and unusual dimensions."""
    print("Testing edge cases...")
    
    # Test small image - use smaller tile size that fits the image
    small_image = np.random.rand(100, 150, 3).astype(np.float32)
    
    tiler = TileGenerator(tile_size=64, stride=32)  # Smaller tiles for small image
    stitcher = TileStitcher(tile_size=64, stride=32)
    
    tile_coords = tiler.get_tile_indices(small_image.shape[0], small_image.shape[1])
    
    # Only proceed if we can generate valid tiles
    if tile_coords:
        tiles = [tiler.extract_tile(small_image, coords) for coords in tile_coords]
        reconstructed = stitcher.stitch_tiles(tiles, tile_coords, small_image.shape[:2])
        
        # Check reconstruction
        mse = np.mean((small_image - reconstructed) ** 2)
        assert mse < 0.1, f"Small image reconstruction error: {mse}"
    else:
        print("No valid tiles generated for small image - this is expected")
    
    mse = np.mean((small_image - reconstructed) ** 2)
    assert mse < 0.01, f"Small image reconstruction error: {mse}"
    print("✓ Small image test passed!")
    
    # Test non-square image
    rect_image = np.random.rand(300, 800, 3).astype(np.float32)
    
    tile_coords = tiler.get_tile_indices(rect_image.shape[0], rect_image.shape[1])
    tiles = [tiler.extract_tile(rect_image, coords) for coords in tile_coords]
    reconstructed = stitcher.stitch_tiles(tiles, tile_coords, rect_image.shape[:2])
    
    mse = np.mean((rect_image - reconstructed) ** 2)
    assert mse < 0.01, f"Rectangular image reconstruction error: {mse}"
    print("✓ Non-square image test passed!")
    
    assert True  # Test passed


def test_segmentation_masks():
    """Test tiling with segmentation masks."""
    print("Testing segmentation mask tiling...")
    
    # Create binary mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    mask[300:400, 300:400] = 1
    
    tiler = TileGenerator(tile_size=256, stride=128)
    stitcher = TileStitcher(tile_size=256, stride=128)
    
    # Generate tiles
    tile_coords = tiler.get_tile_indices(mask.shape[0], mask.shape[1])
    tiles = [tiler.extract_tile(mask, coords) for coords in tile_coords]
    
    # Stitch back (binary mask doesn't need multiple classes)
    reconstructed = stitcher.stitch_tiles(tiles, tile_coords, mask.shape[:2])
    
    # For binary masks, check that values are preserved
    diff = np.abs(mask.astype(np.float32) - reconstructed)
    max_diff = np.max(diff)
    
    print(f"Mask reconstruction max difference: {max_diff}")
    assert max_diff < 0.1, f"Mask reconstruction error too high: {max_diff}"
    print("✓ Segmentation mask test passed!")
    
    assert True  # Test passed


def run_all_tests():
    """Run all tiling tests."""
    print("Running tiling tests...\n")
    
    tests = [
        test_perfect_reconstruction,
        test_overlapping_reconstruction,
        test_edge_cases,
        test_segmentation_masks
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
        print()
    
    print(f"Tiling tests completed: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
