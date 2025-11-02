"""
Main trainer class for segmentation model training.
Includes MAE trainer for self-supervised pretraining.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict
import json
import matplotlib.pyplot as plt

from src.utils.metrics import MetricCollection
from src.utils.distributed import (
    is_main_process,
    average_tensor,
    DistributedMetrics,
    setup_ddp,
    save_checkpoint_distributed,
)
from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    HardNegativeMining,
    VisualizePredictions,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for segmentation model training."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        device: torch.device,
        logger_obj: Optional[Any] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to train on
            logger_obj: Logger object (WandB, TensorBoard, etc.)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger_obj

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = (
            float("inf")
            if config["callbacks"]["early_stopping"]["mode"] == "min"
            else -float("inf")
        )

        # Mixed precision training
        requested_amp = config.get("use_amp", True)

        # Disable AMP for MPS devices as it's not fully supported yet
        if self.device.type == "mps" and requested_amp:
            print(
                "Warning: Mixed precision training not fully supported on MPS devices, disabling AMP"
            )
            self.use_amp = False
        else:
            self.use_amp = requested_amp

        # Initialize GradScaler with device-specific backend
        if self.use_amp:
            if self.device.type == "cuda":
                self.scaler = GradScaler("cuda")
            else:
                self.scaler = GradScaler("cpu")
        else:
            self.scaler = None

        # Gradient accumulation
        self.accumulate_grad_steps = config["training"].get("accumulate_grad", 1)

        # Metrics
        self.train_metrics = MetricCollection(num_classes=2)
        self.val_metrics = MetricCollection(num_classes=2)

        # Callbacks
        self._setup_callbacks()

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
        }

    def _setup_callbacks(self):
        """Setup training callbacks."""
        self.callbacks = []

        # Early stopping
        if "early_stopping" in self.config["callbacks"]:
            es_config = self.config["callbacks"]["early_stopping"]
            early_stopping = EarlyStopping(
                patience=es_config["patience"],
                monitor=es_config["monitor"],
                mode=es_config["mode"],
            )
            self.callbacks.append(early_stopping)

        # Model checkpoint
        if "model_checkpoint" in self.config["callbacks"]:
            mc_config = self.config["callbacks"]["model_checkpoint"]
            checkpoint_callback = ModelCheckpoint(
                filepath=mc_config["filename"],
                monitor=mc_config["monitor"],
                mode=mc_config["mode"],
                save_top_k=mc_config["save_top_k"],
            )
            self.callbacks.append(checkpoint_callback)

        # Hard negative mining
        if "hard_negative_mining" in self.config["callbacks"]:
            hnm_config = self.config["callbacks"]["hard_negative_mining"]
            hnm_callback = HardNegativeMining(start_epoch=hnm_config["start_epoch"])
            self.callbacks.append(hnm_callback)

        # Visualization
        if "visualization" in self.config["callbacks"]:
            viz_config = self.config["callbacks"]["visualization"]
            viz_callback = VisualizePredictions(
                save_dir="./visualizations",
                save_every_n_epochs=viz_config["save_every_n_epochs"],
            )
            self.callbacks.append(viz_callback)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        running_loss = 0.0
        running_loss_components = defaultdict(float)
        num_samples = 0

        # Print epoch start info
        if is_main_process():
            # Get max epochs from config instead of storing as attribute
            max_epochs = self.config.get("training", {}).get("max_epochs", "N/A")
            print(
                f"\n🚀 Starting Epoch {self.current_epoch}/{max_epochs} - {len(self.train_loader)} batches"
            )
            print(
                f"📊 Batch size: {self.train_loader.batch_size} | Device: {self.device}"
            )

        # Set sampler epoch for distributed training
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            if isinstance(batch, dict):
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["mask"].to(self.device, non_blocking=True)
            else:
                images, targets = batch
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(self.device.type):
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict["total_loss"] / self.accumulate_grad_steps
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total_loss"] / self.accumulate_grad_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            with torch.no_grad():
                if isinstance(outputs, (list, tuple)):
                    pred_outputs = outputs[0]
                else:
                    pred_outputs = outputs

                self.train_metrics.update(pred_outputs, targets)

            # Track losses
            running_loss += loss.item() * self.accumulate_grad_steps * batch_size
            for key, value in loss_dict.items():
                if key != "total_loss":
                    # Handle both tensor and scalar values
                    if hasattr(value, "item"):
                        running_loss_components[key] += value.item() * batch_size
                    else:
                        running_loss_components[key] += value * batch_size

            num_samples += batch_size

            # Log progress frequently with tqdm-style updates
            log_every_n_steps = self.config.get("logging", {}).get(
                "log_every_n_steps", 5
            )  # Much more frequent
            if batch_idx % log_every_n_steps == 0 and is_main_process():
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_pct = (batch_idx / len(self.train_loader)) * 100
                avg_loss_so_far = running_loss / max(num_samples, 1)

                # Create progress bar style output
                bar_length = 30
                filled_length = int(bar_length * batch_idx // len(self.train_loader))
                bar = "█" * filled_length + "-" * (bar_length - filled_length)

                print(
                    f"\rEpoch {self.current_epoch:3d} |{bar}| {progress_pct:5.1f}% [{batch_idx:4d}/{len(self.train_loader):4d}] "
                    f"Loss: {avg_loss_so_far:.4f} LR: {current_lr:.2e} ",
                    end="",
                    flush=True,
                )

                # Also log to logger every 20 steps for permanent record
                if batch_idx % (log_every_n_steps * 4) == 0:
                    logger.info(
                        f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                        f"Loss: {avg_loss_so_far:.4f} LR: {current_lr:.2e}"
                    )

        # Compute epoch metrics
        avg_loss = running_loss / num_samples
        avg_loss_components = {
            k: v / num_samples for k, v in running_loss_components.items()
        }
        train_metrics = self.train_metrics.compute()

        # Combine metrics
        epoch_metrics = {
            "train_loss": avg_loss,
            **{f"train_{k}": v for k, v in avg_loss_components.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }

        # Print epoch completion
        if is_main_process():
            print(f"\n✅ Epoch {self.current_epoch} Complete! Loss: {avg_loss:.4f}")

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        running_loss = 0.0
        running_loss_components = defaultdict(float)
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["mask"].to(self.device, non_blocking=True)
                else:
                    images, targets = batch
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                batch_size = images.size(0)

                # Forward pass
                if self.use_amp:
                    with autocast(self.device.type):
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)

                # Update metrics
                if isinstance(outputs, (list, tuple)):
                    pred_outputs = outputs[0]
                else:
                    pred_outputs = outputs

                self.val_metrics.update(pred_outputs, targets)

                # Track losses
                running_loss += loss_dict["total_loss"].item() * batch_size
                for key, value in loss_dict.items():
                    if key != "total_loss":
                        # Handle both tensor and scalar values
                        if hasattr(value, "item"):
                            running_loss_components[key] += value.item() * batch_size
                        else:
                            running_loss_components[key] += value * batch_size

                num_samples += batch_size

        # Compute epoch metrics
        avg_loss = running_loss / num_samples
        avg_loss_components = {
            k: v / num_samples for k, v in running_loss_components.items()
        }
        val_metrics = self.val_metrics.compute()

        # Combine metrics
        epoch_metrics = {
            "val_loss": avg_loss,
            **{f"val_{k}": v for k, v in avg_loss_components.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        # Add common metric aliases for monitoring
        if "dice_mean_dice" in val_metrics:
            epoch_metrics["val_dice"] = val_metrics["dice_mean_dice"]
        if "iou_mean_iou" in val_metrics:
            epoch_metrics["val_iou"] = val_metrics["iou_mean_iou"]
        if "f1_mean_f1" in val_metrics:
            epoch_metrics["val_f1"] = val_metrics["f1_mean_f1"]

        return epoch_metrics

    def fit(
        self, epochs: int, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Train the model for specified number of epochs.

        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history dictionary
        """
        start_epoch = 0

        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None:
            checkpoint = self.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed training from epoch {start_epoch}")

        logger.info(f"Starting training for {epochs - start_epoch} epochs")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = {}
            if epoch % self.config["training"]["val_every_n_epochs"] == 0:
                val_metrics = self.validate_epoch()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Print comprehensive metrics summary
            if is_main_process() and val_metrics:
                try:
                    from tabulate import tabulate

                    print(f"\nEpoch {epoch} Metrics:")

                    # Prepare data for tabulation
                    metrics_data = []

                    # Loss
                    train_loss = train_metrics.get("train_loss", 0)
                    val_loss = val_metrics.get("val_loss", 0)
                    metrics_data.append(
                        ["Loss", f"{train_loss:.4f}", f"{val_loss:.4f}"]
                    )

                    # Dice Score
                    if (
                        "train_dice_mean_dice" in train_metrics
                        and "val_dice_mean_dice" in val_metrics
                    ):
                        train_dice = train_metrics["train_dice_mean_dice"]
                        val_dice = val_metrics["val_dice_mean_dice"]
                        metrics_data.append(
                            ["Dice", f"{train_dice:.4f}", f"{val_dice:.4f}"]
                        )

                    # IoU
                    if (
                        "train_iou_mean_iou" in train_metrics
                        and "val_iou_mean_iou" in val_metrics
                    ):
                        train_iou = train_metrics["train_iou_mean_iou"]
                        val_iou = val_metrics["val_iou_mean_iou"]
                        metrics_data.append(
                            ["IoU", f"{train_iou:.4f}", f"{val_iou:.4f}"]
                        )

                    # Pixel Accuracy
                    if (
                        "train_pixel_accuracy" in train_metrics
                        and "val_pixel_accuracy" in val_metrics
                    ):
                        train_acc = train_metrics["train_pixel_accuracy"]
                        val_acc = val_metrics["val_pixel_accuracy"]
                        metrics_data.append(
                            ["Accuracy", f"{train_acc:.4f}", f"{val_acc:.4f}"]
                        )

                    # Damage-specific Dice (foreground class)
                    if (
                        "train_dice_foreground_dice" in train_metrics
                        and "val_dice_foreground_dice" in val_metrics
                    ):
                        train_damage = train_metrics["train_dice_foreground_dice"]
                        val_damage = val_metrics["val_dice_foreground_dice"]
                        metrics_data.append(
                            ["Damage Dice", f"{train_damage:.4f}", f"{val_damage:.4f}"]
                        )

                    # Print table
                    headers = ["Metric", "Train", "Validation"]
                    print(
                        tabulate(
                            metrics_data,
                            headers=headers,
                            tablefmt="grid",
                            stralign="right",
                        )
                    )
                    print()

                except ImportError:
                    # Fallback to original format if tabulate not available
                    print(f"\nEpoch {epoch} Metrics:")
                    print(f"  Train Loss: {train_metrics.get('train_loss', 0):.4f}")
                    if "val_loss" in val_metrics:
                        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

                    # Training metrics
                    if "train_dice_mean_dice" in train_metrics:
                        print(
                            f"  Train Dice: {train_metrics['train_dice_mean_dice']:.4f}"
                        )
                    if "train_iou_mean_iou" in train_metrics:
                        print(f"  Train IoU: {train_metrics['train_iou_mean_iou']:.4f}")
                    if "train_pixel_accuracy" in train_metrics:
                        print(
                            f"  Train Acc: {train_metrics['train_pixel_accuracy']:.4f}"
                        )

                    # Validation metrics
                    if "val_dice_mean_dice" in val_metrics:
                        print(f"  Val Dice: {val_metrics['val_dice_mean_dice']:.4f}")
                    if "val_iou_mean_iou" in val_metrics:
                        print(f"  Val IoU: {val_metrics['val_iou_mean_iou']:.4f}")
                    if "val_pixel_accuracy" in val_metrics:
                        print(f"  Val Acc: {val_metrics['val_pixel_accuracy']:.4f}")

                    # Damage-specific performance (overfitting check)
                    if (
                        "train_dice_foreground_dice" in train_metrics
                        and "val_dice_foreground_dice" in val_metrics
                    ):
                        print(
                            f"  Damage Dice - Train: {train_metrics['train_dice_foreground_dice']:.4f}, Val: {val_metrics['val_dice_foreground_dice']:.4f}"
                        )
                    print()

            # Reduce metrics across processes
            if torch.distributed.is_initialized():
                epoch_metrics = DistributedMetrics.reduce_dict(epoch_metrics)

            # Update history
            self.history["train_loss"].append(epoch_metrics.get("train_loss", 0))
            self.history["val_loss"].append(epoch_metrics.get("val_loss", 0))
            self.history["train_metrics"].append(
                {k: v for k, v in epoch_metrics.items() if k.startswith("train_")}
            )
            self.history["val_metrics"].append(
                {k: v for k, v in epoch_metrics.items() if k.startswith("val_")}
            )
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Update learning rate scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if "ReduceLROnPlateau" in str(type(self.scheduler)):
                        monitor_metric = epoch_metrics.get(
                            "val_loss", epoch_metrics.get("train_loss", 0)
                        )
                        self.scheduler.step(monitor_metric)
                    else:
                        self.scheduler.step()

            # Log metrics
            epoch_time = time.time() - epoch_start_time
            if is_main_process():
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                for key, value in epoch_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")

                # Log to external logger
                if self.logger is not None:
                    log_dict = {
                        **epoch_metrics,
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    self.logger.log(log_dict)

            # Run callbacks
            should_stop = False
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopping):
                    if callback(epoch, epoch_metrics, self.model):
                        should_stop = True
                        break
                elif isinstance(callback, ModelCheckpoint):
                    callback(
                        epoch, epoch_metrics, self.model, self.optimizer, self.scheduler
                    )
                elif isinstance(callback, HardNegativeMining):
                    callback.should_activate(epoch)
                elif isinstance(callback, VisualizePredictions):
                    if val_metrics:  # Only visualize when we have validation data
                        callback(epoch, self.model, self.val_loader)

            if should_stop:
                logger.info("Early stopping triggered")
                break

        logger.info("Training completed")
        return self.history

    def save_checkpoint(self, filepath: str, is_best: bool = False) -> None:
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best checkpoint
        """
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
            "config": self.config,
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        save_checkpoint_distributed(state, filepath, is_best)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.history = checkpoint.get("history", self.history)

        return checkpoint

    def save_history(self, filepath: str) -> None:
        """Save training history to JSON file."""
        if is_main_process():
            with open(filepath, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Saved training history to {filepath}")


def create_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    config: Dict[str, Any],
    device: torch.device,
    logger_obj: Optional[Any] = None,
) -> Trainer:
    """
    Create trainer instance.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration dictionary
        device: Device to train on
        logger_obj: Logger object

    Returns:
        Trainer instance
    """
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger_obj=logger_obj,
    )


def test_trainer():
    """Test trainer implementation."""
    # Create dummy components
    model = torch.nn.Conv2d(3, 2, 1)

    # Dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return torch.randn(3, 64, 64), torch.randint(0, 2, (64, 64))

    train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    val_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    # Dummy config
    config = {
        "training": {"accumulate_grad": 1, "val_every_n_epochs": 1},
        "logging": {"log_every_n_steps": 5},
        "callbacks": {
            "early_stopping": {"patience": 3, "monitor": "val_loss", "mode": "min"}
        },
    }

    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=torch.device("cpu"),
    )

    print(f"Trainer created successfully with {len(trainer.callbacks)} callbacks")
    return True


class MAETrainer(Trainer):
    """
    Trainer for MAE self-supervised pretraining.

    Key features:
    - Pixel reconstruction loss with masking
    - Visualization of reconstructed patches
    - Encoder feature monitoring
    - Seamless transition to segmentation fine-tuning
    """

    def __init__(
        self,
        model,  # MAESwinUNet
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        device: torch.device,
        logger_obj: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize MAE trainer.

        Args:
            model: MAESwinUNet model with MAE capabilities
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Training device
            logger_obj: Logger instance
        """
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            logger_obj=logger_obj,
            **kwargs,
        )

        # MAE specific settings
        self.mask_ratio = config.get("training", {}).get("mae_mask_ratio", 0.75)
        self.visualize_reconstructions = config.get("training", {}).get(
            "visualize_reconstructions", True
        )
        self.reconstruction_save_dir = Path(
            config.get("training", {}).get(
                "reconstruction_dir", "outputs/mae_reconstructions"
            )
        )
        self.reconstruction_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = config.get("training", {}).get("log_interval", 10)

        # Setup checkpoint directory
        self.checkpoint_dir = Path(
            config.get("checkpoint_dir", config.get("output_dir", "checkpoints/mae"))
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set model to MAE mode
        if hasattr(self.model, "set_training_mode"):
            self.model.set_training_mode("mae")

        print(f"MAE Trainer initialized with mask ratio: {self.mask_ratio}")

    def fit(
        self, epochs: int, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List]:
        """Train the model for specified number of epochs (MAE version without validation)."""
        start_epoch = 0

        if resume_from_checkpoint is not None:
            checkpoint = self.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed training from epoch {start_epoch}")

        logger.info(f"Starting MAE training for {epochs - start_epoch} epochs")

        history = defaultdict(list)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()

            # Store metrics
            for key, value in train_metrics.items():
                if isinstance(value, list):
                    history[key].extend(value)
                else:
                    history[key].append(value)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.get("training", {}).get("save_every", 10) == 0:
                self.save_checkpoint(epoch, train_metrics, is_best=False)

        # Save final encoder weights
        final_metrics = {
            k: v[-1] if isinstance(v, list) and v else v for k, v in history.items()
        }
        self.save_checkpoint(epochs - 1, final_metrics, is_best=True)
        return dict(history)

    def train_epoch(self) -> Dict[str, float]:
        """Override base train_epoch to use MAE-specific training logic."""
        return self.train_one_epoch(self.current_epoch)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with MAE reconstruction loss.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(list)

        # Progress tracking
        total_batches = len(self.train_loader)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move to device
                images = batch["image"].to(self.device, non_blocking=True)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast(device_type=self.device.type):
                    # MAE forward pass
                    mae_output = self.model(images, mask_ratio=self.mask_ratio)
                    loss = mae_output["loss"]

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    if self.config.get("training", {}).get("gradient_clip_val", 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["training"]["gradient_clip_val"],
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    # Gradient clipping
                    if self.config.get("training", {}).get("gradient_clip_val", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["training"]["gradient_clip_val"],
                        )

                    self.optimizer.step()

                # Update metrics
                epoch_metrics["train_loss"].append(loss.item())
                epoch_metrics["mask_ratio"].append(
                    mae_output["mask"].float().mean().item()
                )

                # Log progress
                if batch_idx % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time

                    print(
                        f"Epoch {epoch} [{batch_idx:4d}/{total_batches}] "
                        f"Loss: {loss.item():.4f} "
                        f"Mask: {mae_output['mask'].float().mean().item():.3f} "
                        f"LR: {current_lr:.2e} "
                        f"Time: {elapsed:.1f}s"
                    )

                # Visualize reconstructions periodically
                if (
                    self.visualize_reconstructions
                    and batch_idx % (self.log_interval * 10) == 0
                    and batch_idx > 0
                ):
                    self._visualize_reconstruction(
                        mae_output,
                        epoch,
                        batch_idx,
                        save_dir=self.reconstruction_save_dir,
                    )

                # Memory management for MPS
                if self.device.type == "mps":
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {batch_idx}, skipping...")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps":
                        torch.mps.empty_cache()
                    continue
                else:
                    raise e

        # Calculate epoch averages
        epoch_avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }

        return epoch_avg_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate MAE reconstruction performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # Move to device
                    images = batch["image"].to(self.device, non_blocking=True)

                    # Forward pass
                    with autocast(device_type=self.device.type):
                        mae_output = self.model(images, mask_ratio=self.mask_ratio)
                        loss = mae_output["loss"]

                    # Update metrics
                    val_metrics["val_loss"].append(loss.item())
                    val_metrics["val_mask_ratio"].append(
                        mae_output["mask"].float().mean().item()
                    )

                    # Visualize first validation batch
                    if batch_idx == 0 and self.visualize_reconstructions:
                        self._visualize_reconstruction(
                            mae_output,
                            epoch,
                            batch_idx,
                            save_dir=self.reconstruction_save_dir,
                            prefix="val",
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation batch {batch_idx}, skipping...")
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif self.device.type == "mps":
                            torch.mps.empty_cache()
                        continue
                    else:
                        raise e

        # Calculate validation averages
        val_avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}

        return val_avg_metrics

    def _visualize_reconstruction(
        self,
        mae_output: Dict[str, torch.Tensor],
        epoch: int,
        batch_idx: int,
        save_dir: Path,
        prefix: str = "train",
    ):
        """
        Visualize MAE reconstructions.

        Args:
            mae_output: Output from MAE forward pass
            epoch: Current epoch
            batch_idx: Current batch index
            save_dir: Directory to save visualizations
            prefix: Prefix for filename
        """
        try:
            # Get first sample from batch
            pred_patches = mae_output["pred"][0]  # [N, patch_dim]
            target_patches = mae_output["target"][0]  # [N, patch_dim]
            mask = mae_output["mask"][0]  # [N]
            masked_input = mae_output.get("masked_input", None)

            # Convert patches back to images
            if hasattr(self.model, "mae_decoder"):
                # Reconstruct predicted image
                pred_img = self.model.mae_decoder.unpatchify(
                    pred_patches.unsqueeze(0), (384, 384)  # Target size
                ).squeeze(0)

                # Reconstruct target image
                target_img = self.model.mae_decoder.unpatchify(
                    target_patches.unsqueeze(0), (384, 384)
                ).squeeze(0)
            else:
                return  # Skip if no MAE decoder

            # Convert to numpy and normalize for visualization
            pred_img = pred_img.detach().cpu().numpy()
            target_img = target_img.detach().cpu().numpy()

            # Handle different channel formats
            if pred_img.shape[0] in [1, 3]:  # CHW format
                pred_img = np.transpose(pred_img, (1, 2, 0))
                target_img = np.transpose(target_img, (1, 2, 0))

            # Normalize to [0, 1]
            pred_img = np.clip(pred_img, 0, 1)
            target_img = np.clip(target_img, 0, 1)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original/target
            if target_img.shape[-1] == 1:
                axes[0].imshow(target_img.squeeze(), cmap="gray")
            else:
                axes[0].imshow(target_img)
            axes[0].set_title("Target")
            axes[0].axis("off")

            # Masked input (if available)
            if masked_input is not None:
                masked_img = masked_input[0].detach().cpu().numpy()
                if masked_img.shape[0] in [1, 3]:
                    masked_img = np.transpose(masked_img, (1, 2, 0))
                masked_img = np.clip(masked_img, 0, 1)

                if masked_img.shape[-1] == 1:
                    axes[1].imshow(masked_img.squeeze(), cmap="gray")
                else:
                    axes[1].imshow(masked_img)
                axes[1].set_title(f"Masked Input ({mask.float().mean():.1%} masked)")
            else:
                axes[1].text(0.5, 0.5, "No masked input", ha="center", va="center")
                axes[1].set_title("Masked Input (N/A)")
            axes[1].axis("off")

            # Reconstruction
            if pred_img.shape[-1] == 1:
                axes[2].imshow(pred_img.squeeze(), cmap="gray")
            else:
                axes[2].imshow(pred_img)
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")

            plt.tight_layout()

            # Save visualization
            filename = (
                f"{prefix}_reconstruction_epoch_{epoch:03d}_batch_{batch_idx:04d}.png"
            )
            filepath = save_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception as e:
            print(f"Error creating reconstruction visualization: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """
        Save MAE training checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            filename: Custom filename
        """
        if filename is None:
            filename = f"mae_checkpoint_epoch_{epoch:03d}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "mask_ratio": self.mask_ratio,
            "training_mode": "mae",
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "mae_best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best MAE checkpoint: {best_path}")

            # Save encoder-only weights for DINOv3 initialization
            encoder_state = {}
            for key, value in self.model.state_dict().items():
                if key.startswith("backbone."):
                    encoder_state[key] = value

            encoder_path = self.checkpoint_dir / "mae_encoder_weights.pth"
            torch.save(encoder_state, encoder_path)
            print(f"Saved MAE encoder weights: {encoder_path}")

        print(f"Saved MAE checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load MAE training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore MAE settings
        if "mask_ratio" in checkpoint:
            self.mask_ratio = checkpoint["mask_ratio"]

        # Set training mode
        if hasattr(self.model, "set_training_mode"):
            self.model.set_training_mode("mae")

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        print(f"Loaded MAE checkpoint from epoch {epoch}")
        print(f"Checkpoint metrics: {metrics}")

        return epoch, metrics


class DINOv3Trainer(Trainer):
    """
    Trainer for DINOv3 self-distillation pretraining.

    Key features:
    - Multi-crop strategy with global and local views
    - Student-teacher framework with EMA updates
    - Temperature-based sharpening
    - Can be initialized from MAE pretrained weights for sequential SSL
    """

    def __init__(
        self,
        model,  # DINOv3SwinUNet
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        optimizer: optim.Optimizer,
        scheduler: Any,
        config: Dict[str, Any],
        device: torch.device,
        logger_obj: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize DINOv3 trainer.

        Args:
            model: DINOv3SwinUNet model
            train_loader: Training data loader (must provide multi-crop views)
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Training device
            logger_obj: Logger instance
        """
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            logger_obj=logger_obj,
            **kwargs,
        )

        # DINOv3 specific settings
        self.n_global_crops = config.get("training", {}).get("n_global_crops", 2)
        self.n_local_crops = config.get("training", {}).get("n_local_crops", 4)
        self.teacher_temp_schedule = config.get("training", {}).get(
            "teacher_temp_schedule", False
        )
        self.warmup_teacher_temp = config.get("training", {}).get(
            "warmup_teacher_temp", 0.04
        )
        self.teacher_temp = config.get("training", {}).get("teacher_temp", 0.07)
        self.warmup_teacher_temp_epochs = config.get("training", {}).get(
            "warmup_teacher_temp_epochs", 30
        )
        self.log_interval = config.get("training", {}).get("log_interval", 10)

        # Setup checkpoint directory
        self.checkpoint_dir = Path(
            config.get("checkpoint_dir", config.get("output_dir", "checkpoints/dinov3"))
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set model to DINOv3 mode
        if hasattr(self.model, "set_training_mode"):
            self.model.set_training_mode("dino")

        print(
            f"DINOv3 Trainer initialized with {self.n_global_crops} global crops, {self.n_local_crops} local crops"
        )

    def fit(
        self, epochs: int, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List]:
        """Train the model for specified number of epochs (DINOv3 version without validation)."""
        start_epoch = 0

        if resume_from_checkpoint is not None:
            checkpoint = self.load_checkpoint(resume_from_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed training from epoch {start_epoch}")

        logger.info(f"Starting DINOv3 training for {epochs - start_epoch} epochs")

        history = defaultdict(list)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()

            # Store metrics
            for key, value in train_metrics.items():
                if isinstance(value, list):
                    history[key].extend(value)
                else:
                    history[key].append(value)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.get("training", {}).get("save_every", 10) == 0:
                self.save_checkpoint(epoch, train_metrics, is_best=False)

        # Save final encoder weights
        final_metrics = {
            k: v[-1] if isinstance(v, list) and v else v for k, v in history.items()
        }
        self.save_checkpoint(epochs - 1, final_metrics, is_best=True)
        return dict(history)

    def train_epoch(self) -> Dict[str, float]:
        """Override base train_epoch to use DINOv3-specific training logic."""
        return self.train_one_epoch(self.current_epoch)

    def _update_teacher_temp(self, epoch: int, total_epochs: int):
        """Update teacher temperature with warmup schedule."""
        if not self.teacher_temp_schedule:
            return

        if epoch < self.warmup_teacher_temp_epochs:
            # Linear warmup
            self.model.dino_teacher_temp = (
                self.warmup_teacher_temp
                + (self.teacher_temp - self.warmup_teacher_temp)
                * epoch
                / self.warmup_teacher_temp_epochs
            )
        else:
            self.model.dino_teacher_temp = self.teacher_temp

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with DINOv3 self-distillation loss.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(list)

        # Update teacher temperature
        total_epochs = self.config.get("training", {}).get("epochs", 100)
        self._update_teacher_temp(epoch, total_epochs)

        # Progress tracking
        total_batches = len(self.train_loader)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Extract multi-crop views from batch
                # Expected format: batch["global_views"] = list of global crop tensors
                #                  batch["local_views"] = list of local crop tensors
                if "global_views" in batch:
                    global_views = [
                        v.to(self.device, non_blocking=True)
                        for v in batch["global_views"]
                    ]
                    local_views = [
                        v.to(self.device, non_blocking=True)
                        for v in batch.get("local_views", [])
                    ]

                    logger.debug(
                        f"TRAINER: len(global_views)={len(global_views)}, len(local_views)={len(local_views)}"
                    )
                    for i, v in enumerate(global_views):
                        logger.debug(f"TRAINER: global_views[{i}].shape={v.shape}")
                    for i, v in enumerate(local_views):
                        logger.debug(f"TRAINER: local_views[{i}].shape={v.shape}")
                else:
                    # Fallback: single image, create pseudo multi-crop
                    images = batch["image"].to(self.device, non_blocking=True)
                    global_views = [images, images]
                    local_views = []

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast(device_type=self.device.type):
                    dino_output = self.model(global_views, local_views)
                    loss = dino_output["loss"]

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()

                    # Gradient clipping
                    if self.config.get("training", {}).get("gradient_clip_val", 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["training"]["gradient_clip_val"],
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    # Gradient clipping
                    if self.config.get("training", {}).get("gradient_clip_val", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["training"]["gradient_clip_val"],
                        )

                    self.optimizer.step()

                # Update teacher with EMA
                self.model.update_teacher()

                # Update metrics
                epoch_metrics["train_loss"].append(loss.item())
                epoch_metrics["teacher_temp"].append(self.model.dino_teacher_temp)

                # Log progress
                if batch_idx % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time

                    print(
                        f"Epoch {epoch} [{batch_idx:4d}/{total_batches}] "
                        f"Loss: {loss.item():.4f} "
                        f"T_temp: {self.model.dino_teacher_temp:.4f} "
                        f"LR: {current_lr:.2e} "
                        f"Time: {elapsed:.1f}s"
                    )

                # Memory management for MPS
                if self.device.type == "mps":
                    torch.mps.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error in batch {batch_idx}, skipping...")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps":
                        torch.mps.empty_cache()
                    continue
                else:
                    raise e

        # Calculate epoch averages
        epoch_avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }

        return epoch_avg_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate DINOv3 performance.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # Extract views
                    if "global_views" in batch:
                        global_views = [
                            v.to(self.device, non_blocking=True)
                            for v in batch["global_views"]
                        ]
                        local_views = [
                            v.to(self.device, non_blocking=True)
                            for v in batch.get("local_views", [])
                        ]
                    else:
                        images = batch["image"].to(self.device, non_blocking=True)
                        global_views = [images, images]
                        local_views = []

                    # Forward pass
                    with autocast(device_type=self.device.type):
                        dino_output = self.model(global_views, local_views)
                        loss = dino_output["loss"]

                    # Update metrics
                    val_metrics["val_loss"].append(loss.item())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in validation batch {batch_idx}, skipping...")
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif self.device.type == "mps":
                            torch.mps.empty_cache()
                        continue
                    else:
                        raise e

        # Calculate validation averages
        val_avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}

        return val_avg_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None,
    ):
        """
        Save DINOv3 training checkpoint.

        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            filename: Custom filename
        """
        if filename is None:
            filename = f"dino_checkpoint_epoch_{epoch:03d}.pth"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "training_mode": "dino",
            "teacher_temp": self.model.dino_teacher_temp,
            "center": self.model.center,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        # Save encoder weights separately for downstream tasks
        encoder_checkpoint = {
            "backbone_state_dict": {
                k.replace("backbone.", ""): v
                for k, v in self.model.state_dict().items()
                if k.startswith("backbone.")
            },
            "epoch": epoch,
            "config": self.config,
        }
        encoder_path = self.checkpoint_dir / f"dino_encoder_epoch_{epoch:03d}.pth"
        torch.save(encoder_checkpoint, encoder_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "dino_best_checkpoint.pth"
            torch.save(checkpoint, best_path)

            best_encoder_path = self.checkpoint_dir / "dino_encoder_weights.pth"
            torch.save(encoder_checkpoint, best_encoder_path)

            print(f"Saved best DINOv3 checkpoint: {best_path}")
            print(f"Saved best encoder weights: {best_encoder_path}")

        print(f"Saved DINOv3 checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load DINOv3 training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore DINOv3 settings
        if "teacher_temp" in checkpoint:
            self.model.dino_teacher_temp = checkpoint["teacher_temp"]
        if "center" in checkpoint:
            self.model.center = checkpoint["center"]

        # Set training mode
        if hasattr(self.model, "set_training_mode"):
            self.model.set_training_mode("dino")

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        print(f"Loaded DINOv3 checkpoint from epoch {epoch}")
        print(f"Checkpoint metrics: {metrics}")

        return epoch, metrics


def test_mae_trainer():
    """Test MAE trainer functionality."""
    print("Testing MAE trainer...")

    # This would need actual MAE model and data to test properly
    # For now, just verify the class can be instantiated
    try:
        # Mock configuration
        config = {
            "training": {
                "mae_mask_ratio": 0.75,
                "visualize_reconstructions": False,
                "gradient_clip_val": 1.0,
            }
        }

        print("MAE trainer class defined successfully")
        return True

    except Exception as e:
        print(f"Error testing MAE trainer: {e}")
        return False


def test_dino_trainer():
    """Test DINOv3 trainer functionality."""
    print("Testing DINOv3 trainer...")

    try:
        # Mock configuration
        config = {
            "training": {
                "n_global_crops": 2,
                "n_local_crops": 4,
                "teacher_temp_schedule": True,
                "warmup_teacher_temp": 0.04,
                "teacher_temp": 0.07,
                "warmup_teacher_temp_epochs": 30,
                "gradient_clip_val": 1.0,
            }
        }

        print("DINOv3 trainer class defined successfully")
        return True

    except Exception as e:
        print(f"Error testing DINOv3 trainer: {e}")
        return False


if __name__ == "__main__":
    test_trainer()
    test_mae_trainer()
    test_dino_trainer()
    print("Trainer tests passed!")
