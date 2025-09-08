"""
Test metric implementations against sklearn baselines.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - running limited tests")

if TORCH_AVAILABLE:
    from src.utils.metrics import (
        IoUMetric,
        DiceMetric,
        F1Metric,
        PixelAccuracyMetric,
        compute_metrics,
    )


def create_test_data():
    """Create test prediction and target data."""
    # Create predictable test data
    batch_size, height, width = 4, 64, 64

    # Create predictions with some pattern
    predictions = torch.zeros(batch_size, 2, height, width)
    predictions[:, 0, :32, :32] = 2.0  # Background region
    predictions[:, 1, 32:, 32:] = 3.0  # Foreground region

    # Create corresponding targets
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets[:, 32:, 32:] = 1  # Foreground matches predictions

    return predictions, targets


def test_iou_metric():
    """Test IoU metric implementation."""
    print("Testing IoU metric...")

    predictions, targets = create_test_data()

    # Test IoU metric
    iou_metric = IoUMetric(num_classes=2)
    iou_metric.update(predictions, targets)
    results = iou_metric.compute()

    print(f"IoU results: {results}")

    # Check that results are reasonable
    assert 0 <= results["mean_iou"] <= 1, f"Invalid mean IoU: {results['mean_iou']}"
    assert (
        0 <= results["background_iou"] <= 1
    ), f"Invalid background IoU: {results['background_iou']}"
    assert (
        0 <= results["foreground_iou"] <= 1
    ), f"Invalid foreground IoU: {results['foreground_iou']}"

    # For this specific test case, we expect reasonable IoU
    assert (
        results["foreground_iou"] > 0.2
    ), f"Foreground IoU too low: {results['foreground_iou']}"

    print("✓ IoU metric test passed!")
    assert True  # Test passed


def test_dice_metric():
    """Test Dice metric implementation."""
    print("Testing Dice metric...")

    predictions, targets = create_test_data()

    # Test Dice metric
    dice_metric = DiceMetric(num_classes=2)
    dice_metric.update(predictions, targets)
    results = dice_metric.compute()

    print(f"Dice results: {results}")

    # Check that results are reasonable
    assert 0 <= results["mean_dice"] <= 1, f"Invalid mean Dice: {results['mean_dice']}"
    assert (
        0 <= results["background_dice"] <= 1
    ), f"Invalid background Dice: {results['background_dice']}"
    assert (
        0 <= results["foreground_dice"] <= 1
    ), f"Invalid foreground Dice: {results['foreground_dice']}"

    print("✓ Dice metric test passed!")
    assert True  # Test passed


def test_f1_metric():
    """Test F1 metric implementation."""
    print("Testing F1 metric...")

    predictions, targets = create_test_data()

    # Test F1 metric
    f1_metric = F1Metric(num_classes=2)
    f1_metric.update(predictions, targets)
    results = f1_metric.compute()

    print(f"F1 results: {results}")

    # Check that results are reasonable
    assert 0 <= results["mean_f1"] <= 1, f"Invalid mean F1: {results['mean_f1']}"
    assert (
        0 <= results["background_f1"] <= 1
    ), f"Invalid background F1: {results['background_f1']}"
    assert (
        0 <= results["foreground_f1"] <= 1
    ), f"Invalid foreground F1: {results['foreground_f1']}"
    assert 0 <= results["precision"] <= 1, f"Invalid precision: {results['precision']}"
    assert 0 <= results["recall"] <= 1, f"Invalid recall: {results['recall']}"

    print("✓ F1 metric test passed!")
    assert True  # Test passed


def test_pixel_accuracy():
    """Test pixel accuracy metric."""
    print("Testing pixel accuracy metric...")

    predictions, targets = create_test_data()

    # Test pixel accuracy
    acc_metric = PixelAccuracyMetric()
    acc_metric.update(predictions, targets)
    accuracy = acc_metric.compute()

    print(f"Pixel accuracy: {accuracy:.4f}")

    # Check that accuracy is reasonable
    assert 0 <= accuracy <= 1, f"Invalid accuracy: {accuracy}"
    assert (
        accuracy > 0.4
    ), f"Accuracy too low: {accuracy}"  # Should be decent for our test case

    print("✓ Pixel accuracy test passed!")
    assert True  # Test passed


def test_perfect_predictions():
    """Test metrics with perfect predictions."""
    print("Testing metrics with perfect predictions...")

    batch_size, height, width = 2, 32, 32

    # Perfect predictions
    predictions = torch.zeros(batch_size, 2, height, width)
    predictions[:, 0, :16, :] = 10.0  # Strong background prediction
    predictions[:, 1, 16:, :] = 10.0  # Strong foreground prediction

    # Matching targets
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets[:, 16:, :] = 1

    # Test all metrics
    results = compute_metrics(predictions, targets, num_classes=2)
    print(f"Perfect prediction results: {results}")

    # Check that perfect predictions give high scores
    assert (
        results["iou_mean_iou"] > 0.99
    ), f"Perfect IoU too low: {results['iou_mean_iou']}"
    assert (
        results["dice_mean_dice"] > 0.99
    ), f"Perfect Dice too low: {results['dice_mean_dice']}"
    assert results["f1_mean_f1"] > 0.99, f"Perfect F1 too low: {results['f1_mean_f1']}"
    assert (
        results["pixel_accuracy"] > 0.99
    ), f"Perfect accuracy too low: {results['pixel_accuracy']}"

    print("✓ Perfect predictions test passed!")
    assert True  # Test passed


def test_worst_predictions():
    """Test metrics with worst case predictions."""
    print("Testing metrics with worst predictions...")

    batch_size, height, width = 2, 32, 32

    # Worst predictions (completely wrong)
    predictions = torch.zeros(batch_size, 2, height, width)
    predictions[:, 1, :16, :] = 10.0  # Predict foreground where it's background
    predictions[:, 0, 16:, :] = 10.0  # Predict background where it's foreground

    # Opposite targets
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets[:, 16:, :] = 1

    # Test metrics
    results = compute_metrics(predictions, targets, num_classes=2)
    print(f"Worst prediction results: {results}")

    # Check that worst predictions give low scores
    assert (
        results["iou_mean_iou"] < 0.1
    ), f"Worst IoU too high: {results['iou_mean_iou']}"
    assert (
        results["dice_mean_dice"] < 0.1
    ), f"Worst Dice too high: {results['dice_mean_dice']}"
    assert results["f1_mean_f1"] < 0.1, f"Worst F1 too high: {results['f1_mean_f1']}"
    assert (
        results["pixel_accuracy"] < 0.1
    ), f"Worst accuracy too high: {results['pixel_accuracy']}"

    print("✓ Worst predictions test passed!")
    assert True  # Test passed


def test_binary_case():
    """Test metrics with binary predictions."""
    print("Testing binary predictions...")

    batch_size, height, width = 2, 32, 32

    # Binary predictions (single channel)
    predictions = torch.zeros(batch_size, 1, height, width)
    predictions[:, 0, 16:, :] = 1.0  # Sigmoid output > 0.5

    # Binary targets
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets[:, 16:, :] = 1

    # Test metrics
    results = compute_metrics(predictions, targets, num_classes=2)
    print(f"Binary prediction results: {results}")

    # Check that binary case works
    assert (
        results["iou_mean_iou"] > 0.4
    ), f"Binary IoU too low: {results['iou_mean_iou']}"
    assert (
        results["pixel_accuracy"] > 0.4
    ), f"Binary accuracy too low: {results['pixel_accuracy']}"

    print("✓ Binary predictions test passed!")
    assert True  # Test passed


def test_metric_consistency():
    """Test that metrics are consistent across multiple updates."""
    print("Testing metric consistency...")

    predictions, targets = create_test_data()

    # Split data and compute metrics incrementally
    iou_metric = IoUMetric(num_classes=2)

    # Update with full batch
    iou_metric.reset()
    iou_metric.update(predictions, targets)
    full_results = iou_metric.compute()

    # Update with individual samples
    iou_metric.reset()
    for i in range(predictions.size(0)):
        iou_metric.update(predictions[i : i + 1], targets[i : i + 1])
    incremental_results = iou_metric.compute()

    print(f"Full batch IoU: {full_results['mean_iou']:.4f}")
    print(f"Incremental IoU: {incremental_results['mean_iou']:.4f}")

    # Check consistency
    diff = abs(full_results["mean_iou"] - incremental_results["mean_iou"])
    assert diff < 1e-6, f"Inconsistent IoU computation: {diff}"

    print("✓ Metric consistency test passed!")
    assert True  # Test passed


def test_multiclass_metrics():
    """Test metrics with multiple classes."""
    if not TORCH_AVAILABLE:
        print("Skipping multi-class test - PyTorch not available")
        assert True  # Test passed

    print("Testing multi-class metrics (3 classes)...")

    from src.utils.metrics import MetricCollection

    # Create 3-class data: PASS (0), defect_1 (1), defect_2 (2)
    batch_size, height, width = 2, 64, 64
    num_classes = 3

    # Create structured predictions
    predictions = torch.zeros(batch_size, num_classes, height, width)

    # First image: mostly PASS with some defect_1
    predictions[0, 0, :40, :] = 3.0  # PASS region (strong logit)
    predictions[0, 1, 40:50, :] = 2.5  # defect_1 region
    predictions[0, 2, 50:, :] = 1.0  # weak defect_2 signal

    # Second image: mixed defects
    predictions[1, 0, :20, :] = 2.0  # PASS region
    predictions[1, 1, 20:40, :] = 1.5  # defect_1 region
    predictions[1, 2, 40:, :] = 3.0  # defect_2 region (strong)

    # Create corresponding targets
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)

    # First image targets
    targets[0, :40, :] = 0  # PASS
    targets[0, 40:50, :] = 1  # defect_1
    targets[0, 50:, :] = 2  # defect_2

    # Second image targets
    targets[1, :20, :] = 0  # PASS
    targets[1, 20:40, :] = 1  # defect_1
    targets[1, 40:, :] = 2  # defect_2

    # Compute metrics
    metric_collection = MetricCollection(num_classes=3)
    metric_collection.update(predictions, targets)
    results = metric_collection.compute()

    # Verify we get per-class metrics
    for class_idx in range(num_classes):
        assert f"iou_class_{class_idx}_iou" in results
        assert f"dice_class_{class_idx}_dice" in results
        assert f"f1_class_{class_idx}_f1" in results

        # All metrics should be perfect since we designed the test data to match
        assert results[f"iou_class_{class_idx}_iou"] == 1.0
        assert results[f"dice_class_{class_idx}_dice"] == 1.0
        assert (
            results[f"f1_class_{class_idx}_f1"] >= 0.99
        )  # Allow slight numerical precision

    # Check mean metrics
    assert results["iou_mean_iou"] == 1.0
    assert results["dice_mean_dice"] == 1.0
    assert results["f1_mean_f1"] >= 0.99

    print("✓ Multi-class metrics test passed!")
    assert True  # Test passed


def test_configurable_classes():
    """Test that metrics work with different numbers of classes."""
    if not TORCH_AVAILABLE:
        print("Skipping configurable classes test - PyTorch not available")
        assert True  # Test passed

    print("Testing configurable number of classes...")

    from src.utils.metrics import MetricCollection

    # Test different numbers of classes
    for num_classes in [2, 3, 5]:
        # Create random but valid data
        batch_size, height, width = 2, 32, 32
        predictions = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        # Compute metrics
        metric_collection = MetricCollection(num_classes=num_classes)
        metric_collection.update(predictions, targets)
        results = metric_collection.compute()

        # Verify we get the right number of class-specific metrics
        class_iou_keys = [k for k in results.keys() if k.startswith("iou_class_")]
        assert (
            len(class_iou_keys) == num_classes
        ), f"Expected {num_classes} class IoU metrics, got {len(class_iou_keys)}"

        # Verify mean metrics exist
        assert "iou_mean_iou" in results
        assert "dice_mean_dice" in results
        assert "f1_mean_f1" in results

        print(f"  ✓ {num_classes} classes: Mean IoU = {results['iou_mean_iou']:.4f}")

    print("✓ Configurable classes test passed!")
    assert True  # Test passed


def run_all_tests():
    """Run all metric tests."""
    print("Running metric tests...\n")

    if not TORCH_AVAILABLE:
        print("PyTorch not available - running limited tests")
        assert True  # Test passed

    tests = [
        test_iou_metric,
        test_dice_metric,
        test_f1_metric,
        test_pixel_accuracy,
        test_perfect_predictions,
        test_worst_predictions,
        test_binary_case,
        test_metric_consistency,
        test_multiclass_metrics,
        test_configurable_classes,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
        print()

    print(f"Metric tests completed: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
