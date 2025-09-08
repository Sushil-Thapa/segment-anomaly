"""
Combined loss functions for segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(
        self, smooth: float = 1e-6, ignore_index: int = -100, reduction: str = "mean"
    ):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth labels (B, H, W)

        Returns:
            Dice loss
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)

        # Convert targets to one-hot if needed
        if targets.dim() == 3:  # (B, H, W)
            num_classes = predictions.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets

        # Handle ignore index
        if self.ignore_index != -100:
            mask = (targets != self.ignore_index).float()
            mask = mask.unsqueeze(1).expand_as(targets_one_hot)
            predictions = predictions * mask
            targets_one_hot = targets_one_hot * mask

        # Compute Dice coefficient
        intersection = torch.sum(predictions * targets_one_hot, dim=(2, 3))
        union = torch.sum(predictions, dim=(2, 3)) + torch.sum(
            targets_one_hot, dim=(2, 3)
        )

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(
        self,
        alpha: Union[float, List[float]] = 1.0,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        """
        Initialize Focal loss.

        Args:
            alpha: Weighting factor for rare class (default 1.0)
            gamma: Focusing parameter (default 2.0)
            ignore_index: Index to ignore in loss computation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()

        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth labels (B, H, W)

        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            predictions, targets, ignore_index=self.ignore_index, reduction="none"
        )

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss with automatic class weight computation."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        """
        Initialize weighted cross entropy loss.

        Args:
            class_weights: Manual class weights
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()

        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def compute_class_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights based on frequency.

        Args:
            targets: Ground truth labels

        Returns:
            Class weights tensor
        """
        # Get class frequencies
        unique, counts = torch.unique(
            targets[targets != self.ignore_index], return_counts=True
        )

        # Compute weights as inverse log frequency
        total_samples = counts.sum().float()
        frequencies = counts.float() / total_samples
        weights = 1.0 / torch.log(1.02 + frequencies)  # As specified in requirements

        # Create weight tensor for all classes
        num_classes = targets.max().item() + 1
        class_weights = torch.ones(num_classes, device=targets.device)
        class_weights[unique] = weights

        return class_weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth labels (B, H, W)

        Returns:
            Weighted cross entropy loss
        """
        # Compute class weights if not provided
        if self.class_weights is None:
            weights = self.compute_class_weights(targets)
        else:
            weights = self.class_weights
            if weights.device != predictions.device:
                weights = weights.to(predictions.device)

        # Compute weighted cross entropy
        loss = F.cross_entropy(
            predictions,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        return loss


class DynamicCombinedLoss(nn.Module):
    """Dynamic Combined Loss with adaptive weighting and hard negative mining."""

    def __init__(
        self,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: Union[float, List[float]] = 1.0,
        ignore_index: int = -100,
        # Dynamic weighting parameters
        dynamic_weighting: bool = True,
        weight_update_freq: int = 10,
        # Hard negative mining parameters
        hard_negative_mining: bool = True,
        neg_pos_ratio: float = 5.0,  # Start at 5x as requested
        neg_pos_decay: float = 0.95,  # Decay to 3x over time
        hnm_start_epoch: int = 5,
        # Dynamic HNM parameters
        dynamic_hnm: bool = True,
        fp_threshold: float = 0.1,  # False positive rate threshold
        hnm_adaptation_rate: float = 0.1,
    ):
        """
        Initialize dynamic combined loss.

        Args:
            ce_weight: Weight for cross entropy loss
            dice_weight: Weight for dice loss
            focal_weight: Weight for focal loss
            class_weights: Initial class weights
            label_smoothing: Label smoothing factor
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            ignore_index: Index to ignore in loss computation
            dynamic_weighting: Enable dynamic class weight updates
            weight_update_freq: Frequency of weight updates (epochs)
            hard_negative_mining: Enable hard negative mining
            neg_pos_ratio: Initial negative to positive ratio
            neg_pos_decay: Decay factor for neg_pos_ratio
            hnm_start_epoch: Epoch to start hard negative mining
            dynamic_hnm: Enable dynamic HNM adaptation
            fp_threshold: False positive rate threshold for HNM adaptation
            hnm_adaptation_rate: Rate of HNM parameter adaptation
        """
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dynamic_weighting = dynamic_weighting
        self.weight_update_freq = weight_update_freq
        self.hard_negative_mining = hard_negative_mining
        self.neg_pos_ratio = neg_pos_ratio
        self.initial_neg_pos_ratio = neg_pos_ratio
        self.neg_pos_decay = neg_pos_decay
        self.hnm_start_epoch = hnm_start_epoch
        self.dynamic_hnm = dynamic_hnm
        self.fp_threshold = fp_threshold
        self.hnm_adaptation_rate = hnm_adaptation_rate

        # Training state
        self.current_epoch = 0
        self.last_weight_update = 0

        # Running statistics for dynamic weighting
        self.register_buffer(
            "running_class_counts", torch.zeros(2)
        )  # Binary segmentation
        self.register_buffer("running_total", torch.tensor(0.0))
        self.register_buffer("running_fp_rate", torch.tensor(0.0))

        # Initialize class weights
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone())
        else:
            self.class_weights = None

        # Initialize loss functions
        if ce_weight > 0:
            self.ce_loss = WeightedCrossEntropyLoss(
                class_weights=self.class_weights,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
            )

        if dice_weight > 0:
            self.dice_loss = DiceLoss(ignore_index=ignore_index)

        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index
            )

    def update_dynamic_weights(self, targets: torch.Tensor):
        """Update class weights based on running statistics."""
        if not self.dynamic_weighting:
            return

        # Update running counts
        unique, counts = torch.unique(targets[targets != -100], return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < len(self.running_class_counts):
                self.running_class_counts[cls] += count
        self.running_total += targets.numel()

        # Update weights every N epochs
        if (self.current_epoch - self.last_weight_update) >= self.weight_update_freq:
            if self.running_total > 0:
                # Compute dynamic weights - Critical for wafer defects
                class_freqs = self.running_class_counts / self.running_total
                # Use inverse frequency with logarithmic smoothing
                weights = 1.0 / torch.log(1.02 + class_freqs + 1e-6)
                weights = weights / weights.sum() * len(weights)  # Normalize

                # Update weights
                self.class_weights = weights
                if hasattr(self.ce_loss, "class_weights"):
                    self.ce_loss.class_weights = weights

                self.last_weight_update = self.current_epoch
                print(f"Updated dynamic weights: {weights.cpu().numpy()}")

    def update_hnm_parameters(self, fp_rate: float):
        """Update hard negative mining parameters based on false positive rate."""
        if not self.dynamic_hnm or self.current_epoch < self.hnm_start_epoch:
            return

        # Update running FP rate with momentum
        self.running_fp_rate = 0.9 * self.running_fp_rate + 0.1 * fp_rate

        # Adapt neg_pos_ratio based on FP rate
        if self.running_fp_rate > self.fp_threshold:
            # High FP rate - increase negative sampling
            self.neg_pos_ratio = min(
                self.neg_pos_ratio * (1 + self.hnm_adaptation_rate), 8.0
            )
        else:
            # Low FP rate - can reduce negative sampling
            target_ratio = max(
                3.0,
                self.initial_neg_pos_ratio * (self.neg_pos_decay**self.current_epoch),
            )
            self.neg_pos_ratio = max(
                self.neg_pos_ratio * (1 - self.hnm_adaptation_rate), target_ratio
            )

    def hard_negative_mining(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_per_pixel: torch.Tensor,
    ) -> torch.Tensor:
        """Apply hard negative mining to loss."""
        if not self.hard_negative_mining or self.current_epoch < self.hnm_start_epoch:
            return loss_per_pixel.mean()

        # Flatten tensors
        flat_losses = loss_per_pixel.view(-1)
        flat_targets = targets.view(-1)
        flat_preds = predictions.view(-1, predictions.size(1))

        # Get predicted classes for FP calculation
        pred_classes = torch.argmax(flat_preds, dim=1)

        # Identify positive and negative samples
        pos_mask = flat_targets > 0
        neg_mask = flat_targets == 0

        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        # Calculate false positive rate for dynamic adaptation
        if num_neg > 0:
            fp_mask = (pred_classes > 0) & neg_mask
            fp_rate = fp_mask.sum().float() / num_neg
            self.update_hnm_parameters(fp_rate.item())

        if num_pos == 0:
            # No positive samples - select hardest negatives
            num_keep = min(int(self.neg_pos_ratio * 100), num_neg)
            if num_keep > 0:
                _, indices = torch.topk(flat_losses[neg_mask], num_keep)
                return flat_losses[neg_mask][indices].mean()
            else:
                return flat_losses.mean()

        # Keep all positive samples
        pos_losses = flat_losses[pos_mask]

        # Select hard negative samples
        neg_losses = flat_losses[neg_mask]
        num_neg_keep = min(int(num_pos * self.neg_pos_ratio), num_neg)

        if num_neg_keep > 0:
            _, neg_indices = torch.topk(neg_losses, num_neg_keep)
            hard_neg_losses = neg_losses[neg_indices]
            combined_losses = torch.cat([pos_losses, hard_neg_losses])
        else:
            combined_losses = pos_losses

        return combined_losses.mean()

    def set_epoch(self, epoch: int):
        """Set current epoch for dynamic updates."""
        self.current_epoch = epoch

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Forward pass with dynamic weighting and hard negative mining.

        Args:
            predictions: Predicted logits (B, C, H, W) or list for deep supervision
            targets: Ground truth labels (B, H, W)

        Returns:
            Dictionary containing total loss and individual components
        """
        loss_dict = {}
        total_loss = 0.0

        # Update dynamic weights
        self.update_dynamic_weights(targets)

        # Handle deep supervision
        if isinstance(predictions, (list, tuple)):
            main_pred = predictions[0]
            aux_preds = predictions[1:] if len(predictions) > 1 else []
        else:
            main_pred = predictions
            aux_preds = []

        # Main loss computation with hard negative mining
        if self.ce_weight > 0:
            # Compute per-pixel CE loss for HNM
            ce_loss_pixel = F.cross_entropy(
                main_pred,
                targets,
                weight=self.class_weights,
                ignore_index=-100,
                reduction="none",
            )
            ce_loss = self.hard_negative_mining(main_pred, targets, ce_loss_pixel)
            loss_dict["ce_loss"] = ce_loss
            total_loss += self.ce_weight * ce_loss

        if self.dice_weight > 0:
            dice_loss = self.dice_loss(main_pred, targets)
            loss_dict["dice_loss"] = dice_loss
            total_loss += self.dice_weight * dice_loss

        if self.focal_weight > 0:
            focal_loss = self.focal_loss(main_pred, targets)
            loss_dict["focal_loss"] = focal_loss
            total_loss += self.focal_weight * focal_loss

        # Auxiliary losses (deep supervision)
        aux_loss_total = 0.0
        for i, aux_pred in enumerate(aux_preds):
            aux_loss = 0.0

            if self.ce_weight > 0:
                aux_ce_pixel = F.cross_entropy(
                    aux_pred,
                    targets,
                    weight=self.class_weights,
                    ignore_index=-100,
                    reduction="none",
                )
                aux_ce = self.hard_negative_mining(aux_pred, targets, aux_ce_pixel)
                aux_loss += self.ce_weight * aux_ce

            if self.dice_weight > 0:
                aux_dice = self.dice_loss(aux_pred, targets)
                aux_loss += self.dice_weight * aux_dice

            aux_loss_total += aux_loss * 0.4
            loss_dict[f"aux_loss_{i}"] = aux_loss

        total_loss += aux_loss_total
        loss_dict["aux_loss_total"] = aux_loss_total
        loss_dict["total_loss"] = total_loss

        # Add diagnostic information
        loss_dict["current_neg_pos_ratio"] = self.neg_pos_ratio
        if hasattr(self, "running_fp_rate"):
            loss_dict["running_fp_rate"] = self.running_fp_rate.item()

        return loss_dict


class CombinedLoss(nn.Module):
    """Combined loss function with multiple loss terms."""

    def __init__(
        self,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: Union[float, List[float]] = 1.0,
        ignore_index: int = -100,
    ):
        """
        Initialize combined loss.

        Args:
            ce_weight: Weight for cross entropy loss
            dice_weight: Weight for dice loss
            focal_weight: Weight for focal loss
            class_weights: Class weights for weighted CE
            label_smoothing: Label smoothing factor
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Initialize loss functions
        if ce_weight > 0:
            self.ce_loss = WeightedCrossEntropyLoss(
                class_weights=class_weights,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
            )

        if dice_weight > 0:
            self.dice_loss = DiceLoss(ignore_index=ignore_index)

        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index
            )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            predictions: Predicted logits (B, C, H, W) or list for deep supervision
            targets: Ground truth labels (B, H, W)

        Returns:
            Dictionary containing total loss and individual components
        """
        loss_dict = {}
        total_loss = 0.0

        # Handle deep supervision (multiple predictions)
        if isinstance(predictions, (list, tuple)):
            main_pred = predictions[0]
            aux_preds = predictions[1:] if len(predictions) > 1 else []
        else:
            main_pred = predictions
            aux_preds = []

        # Main loss computation
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(main_pred, targets)
            loss_dict["ce_loss"] = ce_loss
            total_loss += self.ce_weight * ce_loss

        if self.dice_weight > 0:
            dice_loss = self.dice_loss(main_pred, targets)
            loss_dict["dice_loss"] = dice_loss
            total_loss += self.dice_weight * dice_loss

        if self.focal_weight > 0:
            focal_loss = self.focal_loss(main_pred, targets)
            loss_dict["focal_loss"] = focal_loss
            total_loss += self.focal_weight * focal_loss

        # Auxiliary losses (deep supervision)
        aux_loss_total = 0.0
        for i, aux_pred in enumerate(aux_preds):
            aux_loss = 0.0

            if self.ce_weight > 0:
                aux_ce = self.ce_loss(aux_pred, targets)
                aux_loss += self.ce_weight * aux_ce

            if self.dice_weight > 0:
                aux_dice = self.dice_loss(aux_pred, targets)
                aux_loss += self.dice_weight * aux_dice

            aux_loss_total += aux_loss * 0.4  # Reduced weight for auxiliary losses
            loss_dict[f"aux_loss_{i}"] = aux_loss

        total_loss += aux_loss_total
        loss_dict["aux_loss_total"] = aux_loss_total
        loss_dict["total_loss"] = total_loss

        return loss_dict


class HardNegativeMiningLoss(nn.Module):
    """Hard negative mining wrapper for any loss function."""

    def __init__(
        self, base_loss: nn.Module, neg_pos_ratio: float = 3.0, min_kept: int = 100
    ):
        """
        Initialize hard negative mining loss.

        Args:
            base_loss: Base loss function to apply
            neg_pos_ratio: Ratio of negative to positive samples to keep
            min_kept: Minimum number of samples to keep
        """
        super().__init__()
        self.base_loss = base_loss
        self.neg_pos_ratio = neg_pos_ratio
        self.min_kept = min_kept

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard negative mining.

        Args:
            predictions: Predicted logits
            targets: Ground truth labels

        Returns:
            Loss computed on hard samples
        """
        # Compute per-pixel losses
        if hasattr(self.base_loss, "reduction"):
            original_reduction = self.base_loss.reduction
            self.base_loss.reduction = "none"

        pixel_losses = self.base_loss(predictions, targets)

        if hasattr(self.base_loss, "reduction"):
            self.base_loss.reduction = original_reduction

        # Flatten losses
        flat_losses = pixel_losses.view(-1)
        flat_targets = targets.view(-1)

        # Count positive and negative samples
        pos_mask = flat_targets > 0
        neg_mask = flat_targets == 0

        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        if num_pos == 0:
            # No positive samples, select hardest negatives
            num_keep = min(self.min_kept, num_neg)
            _, indices = torch.topk(flat_losses, num_keep)
            return flat_losses[indices].mean()

        # Keep all positive samples
        pos_losses = flat_losses[pos_mask]

        # Select hard negative samples
        neg_losses = flat_losses[neg_mask]
        num_neg_keep = min(int(num_pos * self.neg_pos_ratio), num_neg)

        if num_neg_keep > 0:
            _, neg_indices = torch.topk(neg_losses, num_neg_keep)
            hard_neg_losses = neg_losses[neg_indices]

            # Combine positive and hard negative losses
            combined_losses = torch.cat([pos_losses, hard_neg_losses])
        else:
            combined_losses = pos_losses

        return combined_losses.mean()


def create_loss_function(config: dict) -> CombinedLoss:
    """
    Create loss function from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Combined loss function
    """
    loss_config = config["loss"]

    return CombinedLoss(
        ce_weight=loss_config["ce_weight"],
        dice_weight=loss_config["dice_weight"],
        focal_weight=loss_config.get("focal_weight", 0.0),
        label_smoothing=loss_config["label_smoothing"],
        focal_gamma=loss_config.get("focal_gamma", 2.0),
    )


def test_losses():
    """Test loss functions."""
    # Create dummy data
    batch_size, num_classes, height, width = 2, 2, 64, 64
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    # Test individual losses
    dice_loss = DiceLoss()
    dice_result = dice_loss(predictions, targets)
    print(f"Dice loss: {dice_result.item():.4f}")

    focal_loss = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)
    focal_result = focal_loss(predictions, targets)
    print(f"Focal loss: {focal_result.item():.4f}")

    ce_loss = WeightedCrossEntropyLoss(label_smoothing=0.1)
    ce_result = ce_loss(predictions, targets)
    print(f"Weighted CE loss: {ce_result.item():.4f}")

    # Test combined loss
    combined_loss = CombinedLoss(
        ce_weight=0.7, dice_weight=0.3, focal_weight=0.0, label_smoothing=0.1
    )

    combined_result = combined_loss(predictions, targets)
    print(f"Combined loss: {combined_result}")

    # Test with deep supervision
    aux_predictions = [predictions, predictions * 0.8, predictions * 0.6]
    deep_result = combined_loss(aux_predictions, targets)
    print(f"Deep supervision loss: {deep_result}")

    # Test hard negative mining
    hnm_loss = HardNegativeMiningLoss(ce_loss, neg_pos_ratio=3.0)
    hnm_result = hnm_loss(predictions, targets)
    print(f"Hard negative mining loss: {hnm_result.item():.4f}")

    return True


if __name__ == "__main__":
    test_losses()
    print("Loss function tests passed!")
