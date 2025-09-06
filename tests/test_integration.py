"""
Integration tests to verify the complete training pipeline works.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Required packages not available for integration tests")


def create_dummy_data(data_dir, num_samples=10):
    """Create dummy wafer data for testing."""
    if not TORCH_AVAILABLE:
        return False
        
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Generate dummy images and masks
    for split in ['train', 'val', 'test']:
        n_samples = num_samples if split == 'train' else max(2, num_samples // 3)
        
        for i in range(n_samples):
            # Create dummy wafer image (circular wafer on dark background)
            img = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            # Add circular wafer
            center = (512, 512)
            radius = 400
            y, x = np.ogrid[:1024, :1024]
            mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask_circle] = np.random.randint(100, 200, size=(mask_circle.sum(), 3))
            
            # Add some noise
            img += np.random.randint(0, 30, size=img.shape, dtype=np.uint8)
            
            # Create corresponding binary mask with some defects
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            
            # Add random defects (small circles)
            if np.random.random() > 0.5:  # 50% chance of having defects
                n_defects = np.random.randint(1, 5)
                for _ in range(n_defects):
                    # Random position within wafer
                    angle = np.random.random() * 2 * np.pi
                    r = np.random.random() * 300  # Within wafer radius
                    def_x = int(center[0] + r * np.cos(angle))
                    def_y = int(center[1] + r * np.sin(angle))
                    def_radius = np.random.randint(10, 30)
                    
                    y, x = np.ogrid[:1024, :1024]
                    defect_mask = (x - def_x)**2 + (y - def_y)**2 <= def_radius**2
                    mask[defect_mask] = 255
            
            # Save images
            Image.fromarray(img).save(data_dir / split / 'images' / f'wafer_{i:03d}.png')
            Image.fromarray(mask).save(data_dir / split / 'masks' / f'wafer_{i:03d}.png')
    
    return True


def test_data_loading():
    """Test data loading pipeline."""
    if not TORCH_AVAILABLE:
        print("Skipping data loading test - dependencies not available")
        return True
        
    print("Testing data loading pipeline...")
    
    from src.data.dataset import WaferTileDataset
    from src.data.transforms import get_train_transforms, get_val_transforms
    
    # Create temporary data
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        if not create_dummy_data(data_dir, num_samples=5):
            print("Could not create dummy data")
            return False
        
        # Test dataset creation
        train_transforms = get_train_transforms(512)
        train_dataset = WaferTileDataset(
            data_dir=data_dir / 'train',
            tile_size=512,
            stride=256,
            transforms=train_transforms,
            cache_tiles=False  # Disable caching for test
        )
        
        print(f"Created dataset with {len(train_dataset)} tiles")
        
        # Test data loading
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        images, masks = batch
        
        print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
        
        # Verify shapes and types
        assert images.shape == (2, 3, 512, 512), f"Unexpected image shape: {images.shape}"
        assert masks.shape == (2, 512, 512), f"Unexpected mask shape: {masks.shape}"
        assert images.dtype == torch.float32, f"Unexpected image dtype: {images.dtype}"
        assert masks.dtype == torch.long, f"Unexpected mask dtype: {masks.dtype}"
        
        # Verify value ranges
        assert images.min() >= -3 and images.max() <= 3, "Images not properly normalized"
        assert masks.min() >= 0 and masks.max() <= 1, "Masks not in range [0, 1]"
        
        print("✓ Data loading test passed!")
        return True


def test_model_creation():
    """Test model creation and forward pass."""
    if not TORCH_AVAILABLE:
        print("Skipping model test - dependencies not available")
        return True
        
    print("Testing model creation...")
    
    try:
        from src.models.swin_unet import create_model
        
        # Create small model for testing (avoiding large downloads)
        model = create_model(
            backbone_name='swin_tiny_patch4_window7_224',
            num_classes=2,
            pretrained=False  # Don't download weights for test
        )
        
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512)
            output = model(x)
        
        print(f"Output shape: {output.shape}")
        
        # Verify output shape
        assert output.shape == (1, 2, 512, 512), f"Unexpected output shape: {output.shape}"
        
        print("✓ Model creation test passed!")
        return True
        
    except ImportError as e:
        print(f"Could not import model components: {e}")
        return False


def test_loss_computation():
    """Test loss function computation."""
    if not TORCH_AVAILABLE:
        print("Skipping loss test - dependencies not available")
        return True
        
    print("Testing loss computation...")
    
    try:
        from src.losses.combined import CombinedLoss
        
        # Create loss function
        loss_fn = CombinedLoss(
            ce_weight=0.7,
            dice_weight=0.3,
            class_weights=None
        )
        
        # Create dummy predictions and targets
        batch_size, height, width = 2, 64, 64
        predictions = torch.randn(batch_size, 2, height, width)
        targets = torch.randint(0, 2, (batch_size, height, width))
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        
        print(f"Loss value: {loss.item():.4f}")
        
        # Verify loss properties
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        
        # Test gradient flow
        loss.backward()
        
        print("✓ Loss computation test passed!")
        return True
        
    except ImportError as e:
        print(f"Could not import loss components: {e}")
        return False


def test_metrics_computation():
    """Test metrics computation."""
    if not TORCH_AVAILABLE:
        print("Skipping metrics test - dependencies not available")
        return True
        
    print("Testing metrics computation...")
    
    try:
        from src.utils.metrics import MetricCollection
        
        # Create metrics
        metrics = MetricCollection()
        
        # Create dummy predictions and targets
        batch_size, height, width = 2, 64, 64
        predictions = torch.randn(batch_size, 2, height, width)
        targets = torch.randint(0, 2, (batch_size, height, width))
        
        # Apply sigmoid and threshold to get binary predictions
        probs = torch.softmax(predictions, dim=1)
        binary_preds = (probs[:, 1] > 0.5).long()
        
        # Update metrics
        metrics.update(binary_preds, targets)
        
        # Compute metrics
        results = metrics.compute()
        
        print("Computed metrics:")
        for name, value in results.items():
            print(f"  {name}: {value:.4f}")
        
        # Verify all metrics are computed
        expected_metrics = ['iou', 'dice', 'f1', 'pixel_accuracy']
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"
            assert 0 <= results[metric] <= 1, f"Invalid metric value for {metric}: {results[metric]}"
        
        print("✓ Metrics computation test passed!")
        return True
        
    except ImportError as e:
        print(f"Could not import metrics components: {e}")
        return False


def test_trainer_setup():
    """Test trainer setup without actual training."""
    if not TORCH_AVAILABLE:
        print("Skipping trainer test - dependencies not available")
        return True
        
    print("Testing trainer setup...")
    
    try:
        from src.training.trainer import Trainer
        from src.models.swin_unet import create_model
        from src.losses.combined import CombinedLoss
        
        # Create model and loss
        model = create_model(
            backbone_name='swin_tiny_patch4_window7_224',
            num_classes=2,
            pretrained=False
        )
        
        loss_fn = CombinedLoss(ce_weight=0.7, dice_weight=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device('cpu'),
            mixed_precision=False
        )
        
        print(f"Trainer created successfully")
        print(f"Device: {trainer.device}")
        print(f"Mixed precision: {trainer.mixed_precision}")
        
        # Test saving/loading state
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'test_checkpoint.pth'
            
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path, epoch=0)
            assert checkpoint_path.exists(), "Checkpoint not saved"
            
            # Load checkpoint
            loaded_epoch = trainer.load_checkpoint(checkpoint_path)
            assert loaded_epoch == 0, f"Incorrect epoch loaded: {loaded_epoch}"
        
        print("✓ Trainer setup test passed!")
        return True
        
    except ImportError as e:
        print(f"Could not import trainer components: {e}")
        return False


def test_inference_setup():
    """Test inference engine setup."""
    if not TORCH_AVAILABLE:
        print("Skipping inference test - dependencies not available")
        return True
        
    print("Testing inference setup...")
    
    try:
        from src.inference import TiledInference
        from src.models.swin_unet import create_model
        
        # Create model
        model = create_model(
            backbone_name='swin_tiny_patch4_window7_224',
            num_classes=2,
            pretrained=False
        )
        
        # Create inference engine
        inference = TiledInference(
            model=model,
            tile_size=512,
            stride=256,
            device=torch.device('cpu')
        )
        
        print(f"Inference engine created")
        print(f"Tile size: {inference.tile_size}")
        print(f"Stride: {inference.stride}")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Predict (this will test the tiling logic)
        with torch.no_grad():
            prediction = inference.predict_image(dummy_image)
        
        print(f"Prediction shape: {prediction.shape}")
        
        # Verify prediction
        assert prediction.shape == (1024, 1024), f"Unexpected prediction shape: {prediction.shape}"
        assert prediction.dtype == np.uint8, f"Unexpected prediction dtype: {prediction.dtype}"
        assert set(np.unique(prediction)).issubset({0, 255}), "Prediction should be binary"
        
        print("✓ Inference setup test passed!")
        return True
        
    except ImportError as e:
        print(f"Could not import inference components: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("Running integration tests...\n")
    
    if not TORCH_AVAILABLE:
        print("Required packages not available - skipping most tests")
        return True
    
    tests = [
        test_data_loading,
        test_model_creation,
        test_loss_computation,
        test_metrics_computation,
        test_trainer_setup,
        test_inference_setup
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print(f"Integration tests completed: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
