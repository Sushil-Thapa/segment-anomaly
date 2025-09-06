"""
Training callbacks for model training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import cv2
from typing import Dict, List, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self,
                 patience: int = 15,
                 min_delta: float = 1e-4,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: 'min' or 'max' - direction of improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_value = np.inf
        else:
            self.monitor_op = np.greater
            self.best_value = -np.inf
    
    def __call__(self, 
                 epoch: int, 
                 metrics: Dict[str, float], 
                 model: torch.nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            model: Model being trained
            
        Returns:
            True if training should stop, False otherwise
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {name: param.clone() 
                                   for name, param in model.state_dict().items()}
        else:
            self.wait += 1
            
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(f"Restored best weights from epoch {epoch - self.wait}")
            
            logger.info(f"Early stopping at epoch {epoch} with {self.monitor}={self.best_value:.4f}")
            return True
        
        return False


class ModelCheckpoint:
    """Model checkpoint callback to save best models."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_top_k: int = 1,
                 verbose: bool = True):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path template for saving checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' - direction of improvement
            save_best_only: Whether to save only improved models
            save_top_k: Number of best models to keep
            verbose: Whether to print save messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_top_k = save_top_k
        self.verbose = verbose
        
        self.best_values = []
        self.saved_files = []
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
    
    def __call__(self,
                 epoch: int,
                 metrics: Dict[str, float],
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[Any] = None) -> None:
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
        """
        current_value = metrics.get(self.monitor, None)
        
        if current_value is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return
        
        # Check if we should save
        should_save = True
        if self.save_best_only:
            if len(self.best_values) < self.save_top_k:
                should_save = True
            else:
                if self.mode == 'min':
                    should_save = current_value < max(self.best_values)
                else:
                    should_save = current_value > min(self.best_values)
        
        if should_save:
            # Format filepath
            filepath = self.filepath.format(
                epoch=epoch,
                **{key: value for key, value in metrics.items()}
            )
            
            # Create state dictionary
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'monitor_value': current_value
            }
            
            if scheduler is not None:
                state['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save checkpoint
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, filepath)
            
            if self.verbose:
                logger.info(f"Saved checkpoint: {filepath} ({self.monitor}={current_value:.4f})")
            
            # Update best values list
            self.best_values.append(current_value)
            self.saved_files.append(filepath)
            
            # Remove old checkpoints if necessary
            if len(self.best_values) > self.save_top_k:
                if self.mode == 'min':
                    worst_idx = np.argmax(self.best_values)
                else:
                    worst_idx = np.argmin(self.best_values)
                
                # Remove worst checkpoint file
                old_file = self.saved_files.pop(worst_idx)
                old_value = self.best_values.pop(worst_idx)
                
                try:
                    Path(old_file).unlink()
                    if self.verbose:
                        logger.info(f"Removed old checkpoint: {old_file} ({self.monitor}={old_value:.4f})")
                except FileNotFoundError:
                    pass


class HardNegativeMining:
    """Hard negative mining callback for training."""
    
    def __init__(self,
                 start_epoch: int = 30,
                 neg_pos_ratio: float = 3.0,
                 min_hard_ratio: float = 0.25):
        """
        Initialize hard negative mining callback.
        
        Args:
            start_epoch: Epoch to start hard negative mining
            neg_pos_ratio: Ratio of negative to positive samples
            min_hard_ratio: Minimum ratio of hard samples to keep
        """
        self.start_epoch = start_epoch
        self.neg_pos_ratio = neg_pos_ratio
        self.min_hard_ratio = min_hard_ratio
        self.active = False
    
    def should_activate(self, epoch: int) -> bool:
        """Check if hard negative mining should be activated."""
        if epoch >= self.start_epoch and not self.active:
            self.active = True
            logger.info(f"Activated hard negative mining at epoch {epoch}")
        return self.active
    
    def mine_hard_negatives(self, 
                          predictions: torch.Tensor,
                          targets: torch.Tensor,
                          loss_per_pixel: torch.Tensor) -> torch.Tensor:
        """
        Mine hard negative samples.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_per_pixel: Per-pixel loss values
            
        Returns:
            Mask for hard samples
        """
        batch_size = targets.size(0)
        device = targets.device
        
        hard_mask = torch.zeros_like(targets, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            target_b = targets[b].view(-1)
            loss_b = loss_per_pixel[b].view(-1)
            
            # Identify positive and negative samples
            pos_mask = target_b > 0
            neg_mask = target_b == 0
            
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            if num_pos == 0:
                # No positive samples, select hardest negatives
                num_hard = max(int(num_neg * self.min_hard_ratio), 100)
                if num_hard > 0 and num_neg > 0:
                    _, hard_indices = torch.topk(loss_b[neg_mask], min(num_hard, num_neg))
                    neg_indices = torch.where(neg_mask)[0]
                    selected_indices = neg_indices[hard_indices]
                    hard_mask[b].view(-1)[selected_indices] = True
            else:
                # Select all positive samples
                hard_mask[b].view(-1)[pos_mask] = True
                
                # Select hard negative samples
                if num_neg > 0:
                    neg_losses = loss_b[neg_mask]
                    num_hard_neg = min(int(num_pos * self.neg_pos_ratio), num_neg)
                    
                    if num_hard_neg > 0:
                        _, hard_neg_indices = torch.topk(neg_losses, num_hard_neg)
                        neg_indices = torch.where(neg_mask)[0]
                        selected_neg_indices = neg_indices[hard_neg_indices]
                        hard_mask[b].view(-1)[selected_neg_indices] = True
        
        return hard_mask


class VisualizePredictions:
    """Callback to visualize predictions during training."""
    
    def __init__(self,
                 save_dir: str,
                 num_samples: int = 4,
                 save_every_n_epochs: int = 10,
                 max_files_per_class: int = 50):
        """
        Initialize visualization callback.
        
        Args:
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            save_every_n_epochs: Save visualizations every N epochs
            max_files_per_class: Maximum files to keep per class
        """
        self.save_dir = Path(save_dir)
        self.num_samples = num_samples
        self.save_every_n_epochs = save_every_n_epochs
        self.max_files_per_class = max_files_per_class
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create colormap for segmentation
        colors = ['black', 'red']
        self.cmap = ListedColormap(colors)
    
    def __call__(self,
                 epoch: int,
                 model: torch.nn.Module,
                 val_loader: torch.utils.data.DataLoader) -> None:
        """
        Create and save visualizations.
        
        Args:
            epoch: Current epoch
            model: Model to use for predictions
            val_loader: Validation data loader
        """
        if epoch % self.save_every_n_epochs != 0:
            return
        
        model.eval()
        
        # Get samples from validation loader
        samples = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if len(samples) >= self.num_samples:
                    break
                
                if isinstance(batch, dict):
                    images = batch['image']
                    masks = batch['mask']
                else:
                    images, masks = batch[:2]
                
                if torch.cuda.is_available():
                    images = images.cuda()
                
                # Get predictions
                predictions = model(images)
                if isinstance(predictions, (list, tuple)):
                    predictions = predictions[0]
                
                predictions = F.softmax(predictions, dim=1)
                pred_masks = torch.argmax(predictions, dim=1)
                
                # Move to CPU
                images = images.cpu()
                masks = masks.cpu()
                pred_masks = pred_masks.cpu()
                
                # Store samples
                for j in range(min(images.size(0), self.num_samples - len(samples))):
                    samples.append({
                        'image': images[j],
                        'mask': masks[j],
                        'prediction': pred_masks[j]
                    })
        
        # Create visualization
        self._create_visualization(epoch, samples)
        
        model.train()
    
    def _create_visualization(self, epoch: int, samples: List[Dict]) -> None:
        """Create and save visualization figure."""
        n_samples = len(samples)
        if n_samples == 0:
            return
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples):
            image = sample['image']
            mask = sample['mask']
            prediction = sample['prediction']
            
            # Denormalize image for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()
            pred_np = prediction.numpy()
            
            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap=self.cmap, vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np, cmap=self.cmap, vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / f'predictions_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved predictions visualization: {save_path}")
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """Remove old visualization files to save space."""
        viz_files = list(self.save_dir.glob('predictions_epoch_*.png'))
        
        if len(viz_files) > self.max_files_per_class:
            # Sort by creation time and remove oldest
            viz_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = viz_files[:-self.max_files_per_class]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except FileNotFoundError:
                    pass


class LearningRateScheduler:
    """Learning rate scheduler callback."""
    
    def __init__(self,
                 scheduler,
                 monitor: str = 'val_loss',
                 patience: int = 5,
                 factor: float = 0.5,
                 min_lr: float = 1e-7,
                 verbose: bool = True):
        """
        Initialize learning rate scheduler callback.
        
        Args:
            scheduler: PyTorch scheduler instance
            monitor: Metric to monitor for ReduceLROnPlateau
            patience: Patience for ReduceLROnPlateau
            factor: Factor to reduce LR by
            min_lr: Minimum learning rate
            verbose: Whether to print LR changes
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
    
    def __call__(self, 
                 epoch: int, 
                 metrics: Dict[str, float],
                 optimizer: torch.optim.Optimizer) -> None:
        """
        Update learning rate.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            optimizer: Optimizer to update
        """
        if hasattr(self.scheduler, 'step'):
            if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                metric_value = metrics.get(self.monitor, None)
                if metric_value is not None:
                    old_lr = optimizer.param_groups[0]['lr']
                    self.scheduler.step(metric_value)
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    if self.verbose and old_lr != new_lr:
                        logger.info(f"Reduced learning rate from {old_lr:.2e} to {new_lr:.2e}")
            else:
                self.scheduler.step()


def test_callbacks():
    """Test callback implementations."""
    # Create dummy model and data
    model = torch.nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, monitor='val_loss')
    
    metrics = {'val_loss': 1.0}
    stop = early_stopping(1, metrics, model)
    print(f"Early stopping epoch 1: {stop}")
    
    metrics = {'val_loss': 0.8}
    stop = early_stopping(2, metrics, model)
    print(f"Early stopping epoch 2: {stop}")
    
    # Test model checkpoint
    checkpoint = ModelCheckpoint(
        filepath='test_epoch_{epoch:02d}_{val_loss:.4f}.pth',
        monitor='val_loss'
    )
    
    checkpoint(1, {'val_loss': 1.0}, model, optimizer)
    checkpoint(2, {'val_loss': 0.8}, model, optimizer)
    
    # Test hard negative mining
    hnm = HardNegativeMining(start_epoch=1)
    should_activate = hnm.should_activate(2)
    print(f"Hard negative mining active: {should_activate}")
    
    return True


if __name__ == "__main__":
    test_callbacks()
    print("Callback tests passed!")
