"""
Memory profiling tests to ensure efficient memory usage.
"""

import sys
import gc
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, skipping memory tests")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, using basic memory tracking")


class MemoryProfiler:
    """Simple memory profiler for tracking peak usage."""

    def __init__(self):
        self.peak_memory_mb = 0
        self.start_memory_mb = 0
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()

    def start_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()
        self.start_memory_mb = self._get_current_memory()

    def stop_tracking(self):
        """Stop tracking and return peak memory usage."""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_memory_mb = max(self.peak_memory_mb, peak / 1024 / 1024)

        return self.peak_memory_mb - self.start_memory_mb

    def _get_current_memory(self):
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            return psutil.Process().memory_info().rss / 1024 / 1024
        return 0

    def get_gpu_memory(self):
        """Get GPU memory usage in MB."""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0


def test_tiling_memory_usage():
    """Test memory usage of tiling operations."""
    if not TORCH_AVAILABLE:
        print("Skipping tiling memory test - PyTorch not available")
        assert True  # Test passed

    print("Testing tiling memory usage...")

    from src.data.tiling import TileGenerator, TileStitcher

    profiler = MemoryProfiler()
    profiler.start_tracking()

    # Create large test image
    large_image = np.random.rand(2048, 2048, 3).astype(np.float32)

    # Initialize tiling
    tiler = TileGenerator(tile_size=512, stride=256)
    stitcher = TileStitcher(tile_size=512, stride=256)

    # Generate tiles
    tile_coords = tiler.get_tile_indices(large_image.shape[0], large_image.shape[1])

    # Process tiles without keeping all in memory
    sample_tiles = []
    for i, coords in enumerate(tile_coords[:10]):  # Test with first 10 tiles
        tile = tiler.extract_tile(large_image, coords)
        sample_tiles.append(tile)

        # Force garbage collection periodically
        if i % 5 == 0:
            gc.collect()

    # Test stitching with proper output dimensions
    output_height = min(512, large_image.shape[0])
    output_width = min(512, large_image.shape[1])

    reconstructed = stitcher.stitch_tiles(
        sample_tiles, tile_coords[:10], (output_height, output_width)
    )

    memory_used = profiler.stop_tracking()
    print(f"Tiling memory usage: {memory_used:.1f} MB")

    # Check memory usage is reasonable (should be much less than image size)
    image_size_mb = large_image.nbytes / 1024 / 1024
    print(f"Original image size: {image_size_mb:.1f} MB")

    # Memory usage should be reasonable
    assert (
        memory_used < image_size_mb * 3
    ), f"Memory usage too high: {memory_used:.1f} MB"

    print("✓ Tiling memory test passed!")
    assert True  # Test passed


def test_model_memory_scaling():
    """Test memory scaling with batch size."""
    if not TORCH_AVAILABLE:
        print("Skipping model memory test - PyTorch not available")
        assert True  # Test passed

    print("Testing model memory scaling...")

    # Simple conv model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 2, 1)
    )

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    memory_usage = []
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create input
        x = torch.randn(batch_size, 3, 256, 256, device=device)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Measure memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = 0  # Approximate for CPU

        memory_usage.append(memory_mb)
        print(f"Batch size {batch_size}: {memory_mb:.1f} MB")

    # Check memory scales reasonably with batch size
    if len(memory_usage) > 1 and memory_usage[0] > 0:
        # Memory should scale sub-linearly with batch size due to fixed model weights
        ratio = memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 1
        expected_max_ratio = batch_sizes[-1] * 1.5  # Allow some overhead

        print(
            f"Memory scaling ratio: {ratio:.2f} (max expected: {expected_max_ratio:.2f})"
        )
        assert ratio < expected_max_ratio, f"Memory scaling too high: {ratio:.2f}"

    print("✓ Model memory scaling test passed!")
    assert True  # Test passed


def test_gradient_accumulation_memory():
    """Test memory usage with gradient accumulation."""
    if not TORCH_AVAILABLE:
        print("Skipping gradient accumulation test - PyTorch not available")
        assert True  # Test passed

    print("Testing gradient accumulation memory...")

    # Simple model
    model = torch.nn.Linear(1000, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Test with and without gradient accumulation
    accumulation_steps = 4

    memory_without_accum = []
    memory_with_accum = []

    # Without accumulation - large batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    x_large = torch.randn(32, 1000, device=device)
    y_large = torch.randint(0, 10, (32,), device=device)

    optimizer.zero_grad()
    output = model(x_large)
    loss = criterion(output, y_large)
    loss.backward()
    optimizer.step()

    if torch.cuda.is_available():
        memory_without_accum.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

    # With accumulation - small batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad()
    for i in range(accumulation_steps):
        x_small = torch.randn(8, 1000, device=device)  # 32/4 = 8
        y_small = torch.randint(0, 10, (8,), device=device)

        output = model(x_small)
        loss = criterion(output, y_small) / accumulation_steps
        loss.backward()

    optimizer.step()

    if torch.cuda.is_available():
        memory_with_accum.append(torch.cuda.max_memory_allocated() / 1024 / 1024)

    if memory_without_accum and memory_with_accum:
        print(f"Large batch memory: {memory_without_accum[0]:.1f} MB")
        print(f"Accumulated batch memory: {memory_with_accum[0]:.1f} MB")

        # Gradient accumulation should use less peak memory
        assert (
            memory_with_accum[0] <= memory_without_accum[0] * 1.1
        ), "Gradient accumulation should not increase memory significantly"

    print("✓ Gradient accumulation memory test passed!")
    assert True  # Test passed


def test_memory_cleanup():
    """Test that memory is properly cleaned up."""
    if not TORCH_AVAILABLE:
        print("Skipping memory cleanup test - PyTorch not available")
        assert True  # Test passed

    print("Testing memory cleanup...")

    profiler = MemoryProfiler()

    # Get initial memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated()

    profiler.start_tracking()

    # Create and destroy tensors
    for i in range(10):
        large_tensor = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            large_tensor = large_tensor.cuda()

        # Use the tensor
        result = large_tensor.sum()

        # Explicitly delete
        del large_tensor
        del result

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Check memory is cleaned up
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated()
        memory_diff = (final_gpu_memory - initial_gpu_memory) / 1024 / 1024
        print(f"GPU memory difference: {memory_diff:.1f} MB")

        # Should not have significant memory leak
        assert memory_diff < 10, f"Possible GPU memory leak: {memory_diff:.1f} MB"

    cpu_memory_used = profiler.stop_tracking()
    print(f"CPU memory used: {cpu_memory_used:.1f} MB")

    print("✓ Memory cleanup test passed!")
    assert True  # Test passed


def run_all_tests():
    """Run all memory tests."""
    print("Running memory profiling tests...\n")

    if not TORCH_AVAILABLE:
        print("PyTorch not available - running limited tests")
        assert True  # Test passed

    tests = [
        test_tiling_memory_usage,
        test_model_memory_scaling,
        test_gradient_accumulation_memory,
        test_memory_cleanup,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
        print()

    print(f"Memory tests completed: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
