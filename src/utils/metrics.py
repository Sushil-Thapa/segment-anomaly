"""
Evaluation metrics for segmentation tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class IoUMetric:
    """Intersection over Union (IoU) metric."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 ignore_index: int = -100,
                 smooth: float = 1e-6):
        """
        Initialize IoU metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            smooth: Smoothing factor
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions and targets.
        
        Args:
            predictions: Predicted logits (B, C, H, W) or probabilities
            targets: Ground truth labels (B, H, W)
        """
        # Convert to predicted classes
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = (predictions > 0.5).long().squeeze(1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored indices
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Move to CPU for computation
        predictions = predictions.cpu()
        targets = targets.cpu()
        
        # Compute intersection and union for each class
        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union
    
    def compute(self) -> Dict[str, float]:
        """
        Compute IoU scores.
        
        Returns:
            Dictionary with IoU scores
        """
        # Compute per-class IoU
        iou_per_class = (self.intersection + self.smooth) / (self.union + self.smooth)
        
        # Compute mean IoU
        valid_classes = self.union > 0
        mean_iou = iou_per_class[valid_classes].mean().item()
        
        results = {'mean_iou': mean_iou}
        
        # Add per-class IoU scores
        for class_idx in range(self.num_classes):
            results[f'class_{class_idx}_iou'] = iou_per_class[class_idx].item()
        
        # Keep backward compatibility for binary case
        if self.num_classes == 2:
            results['background_iou'] = iou_per_class[0].item()
            results['foreground_iou'] = iou_per_class[1].item()
        
        return results


class DiceMetric:
    """Dice coefficient metric."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 ignore_index: int = -100,
                 smooth: float = 1e-6):
        """
        Initialize Dice metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            smooth: Smoothing factor
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions and targets.
        
        Args:
            predictions: Predicted logits (B, C, H, W) or probabilities
            targets: Ground truth labels (B, H, W)
        """
        # Convert to predicted classes
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = (predictions > 0.5).long().squeeze(1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored indices
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Move to CPU for computation
        predictions = predictions.cpu()
        targets = targets.cpu()
        
        # Compute intersection and total for each class
        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).sum().float()
            total = pred_mask.sum().float() + target_mask.sum().float()
            
            self.intersection[class_idx] += intersection
            self.total[class_idx] += total
    
    def compute(self) -> Dict[str, float]:
        """
        Compute Dice scores.
        
        Returns:
            Dictionary with Dice scores
        """
        # Compute per-class Dice
        dice_per_class = (2.0 * self.intersection + self.smooth) / (self.total + self.smooth)
        
        # Compute mean Dice
        valid_classes = self.total > 0
        mean_dice = dice_per_class[valid_classes].mean().item()
        
        results = {'mean_dice': mean_dice}
        
        # Add per-class Dice scores
        for class_idx in range(self.num_classes):
            results[f'class_{class_idx}_dice'] = dice_per_class[class_idx].item()
        
        # Keep backward compatibility for binary case
        if self.num_classes == 2:
            results['background_dice'] = dice_per_class[0].item()
            results['foreground_dice'] = dice_per_class[1].item()
        
        return results


class F1Metric:
    """F1 score metric."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 ignore_index: int = -100,
                 average: str = 'macro'):
        """
        Initialize F1 metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
            average: Averaging method ('macro', 'micro', 'weighted')
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions and targets.
        
        Args:
            predictions: Predicted logits (B, C, H, W) or probabilities
            targets: Ground truth labels (B, H, W)
        """
        # Convert to predicted classes
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = (predictions > 0.5).long().squeeze(1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored indices
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Move to CPU for computation
        predictions = predictions.cpu()
        targets = targets.cpu()
        
        # Compute confusion matrix components
        for class_idx in range(self.num_classes):
            pred_pos = (predictions == class_idx)
            target_pos = (targets == class_idx)
            
            tp = (pred_pos & target_pos).sum().float()
            fp = (pred_pos & ~target_pos).sum().float()
            fn = (~pred_pos & target_pos).sum().float()
            
            self.tp[class_idx] += tp
            self.fp[class_idx] += fp
            self.fn[class_idx] += fn
    
    def compute(self) -> Dict[str, float]:
        """
        Compute F1 scores.
        
        Returns:
            Dictionary with F1 scores
        """
        # Compute precision and recall
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        
        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        if self.average == 'macro':
            mean_f1 = f1.mean().item()
        elif self.average == 'micro':
            tp_sum = self.tp.sum()
            fp_sum = self.fp.sum()
            fn_sum = self.fn.sum()
            
            micro_precision = tp_sum / (tp_sum + fp_sum + 1e-6)
            micro_recall = tp_sum / (tp_sum + fn_sum + 1e-6)
            mean_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-6)).item()
        else:  # weighted
            support = self.tp + self.fn
            weights = support / support.sum()
            mean_f1 = (f1 * weights).sum().item()
        
        results = {'mean_f1': mean_f1}
        
        # Add per-class F1, precision, recall
        for class_idx in range(self.num_classes):
            results[f'class_{class_idx}_f1'] = f1[class_idx].item()
            results[f'class_{class_idx}_precision'] = precision[class_idx].item()
            results[f'class_{class_idx}_recall'] = recall[class_idx].item()
        
        # Keep backward compatibility for binary case
        if self.num_classes == 2:
            results['background_f1'] = f1[0].item()
            results['foreground_f1'] = f1[1].item()
            results['precision'] = precision[1].item()  # Positive class precision
            results['recall'] = recall[1].item()        # Positive class recall
        
        return results


class PixelAccuracyMetric:
    """Pixel-wise accuracy metric."""
    
    def __init__(self, ignore_index: int = -100):
        """
        Initialize pixel accuracy metric.
        
        Args:
            ignore_index: Index to ignore in computation
        """
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with new predictions and targets.
        
        Args:
            predictions: Predicted logits (B, C, H, W) or probabilities
            targets: Ground truth labels (B, H, W)
        """
        # Convert to predicted classes
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = torch.argmax(predictions, dim=1)
        elif predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = (predictions > 0.5).long().squeeze(1)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Remove ignored indices
        mask = (targets != self.ignore_index)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update counters
        self.correct += (predictions == targets).sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        """
        Compute pixel accuracy.
        
        Returns:
            Pixel accuracy
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class MetricCollection:
    """Collection of metrics for easy management."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 ignore_index: int = -100):
        """
        Initialize metric collection.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
        """
        self.metrics = {
            'iou': IoUMetric(num_classes, ignore_index),
            'dice': DiceMetric(num_classes, ignore_index),
            'f1': F1Metric(num_classes, ignore_index),
            'pixel_accuracy': PixelAccuracyMetric(ignore_index)
        }
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update all metrics.
        
        Args:
            predictions: Predicted logits or probabilities
            targets: Ground truth labels
        """
        for metric in self.metrics.values():
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with all metric results
        """
        results = {}
        
        for name, metric in self.metrics.items():
            metric_result = metric.compute()
            
            if isinstance(metric_result, dict):
                for key, value in metric_result.items():
                    results[f"{name}_{key}" if name != key else key] = value
            else:
                results[name] = metric_result
        
        return results


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   num_classes: int = 2) -> Dict[str, float]:
    """
    Compute all metrics for given predictions and targets.
    
    Args:
        predictions: Predicted logits or probabilities
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with all metric results
    """
    metric_collection = MetricCollection(num_classes)
    metric_collection.update(predictions, targets)
    return metric_collection.compute()


def test_metrics():
    """Test metric implementations."""
    # Create dummy data
    batch_size, num_classes, height, width = 2, 2, 32, 32
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test individual metrics
    iou_metric = IoUMetric(num_classes)
    iou_metric.update(predictions, targets)
    iou_results = iou_metric.compute()
    print(f"IoU results: {iou_results}")
    
    dice_metric = DiceMetric(num_classes)
    dice_metric.update(predictions, targets)
    dice_results = dice_metric.compute()
    print(f"Dice results: {dice_results}")
    
    f1_metric = F1Metric(num_classes)
    f1_metric.update(predictions, targets)
    f1_results = f1_metric.compute()
    print(f"F1 results: {f1_results}")
    
    pixel_acc = PixelAccuracyMetric()
    pixel_acc.update(predictions, targets)
    acc_result = pixel_acc.compute()
    print(f"Pixel accuracy: {acc_result:.4f}")
    
    # Test metric collection
    metric_collection = MetricCollection(num_classes)
    metric_collection.update(predictions, targets)
    all_results = metric_collection.compute()
    print(f"All metrics: {all_results}")
    
    # Test compute_metrics function
    quick_results = compute_metrics(predictions, targets, num_classes)
    print(f"Quick results: {quick_results}")
    
    return True


if __name__ == "__main__":
    test_metrics()
    print("Metric tests passed!")
