#!/usr/bin/env python3
"""
DINOv3 Pretraining Script

Self-distillation pretraining using DINOv3 for Swin-UNet backbone.
Can be used standalone or after MAE pretraining for sequential SSL.

Usage:
    python src/train_dinov3.py --config configs/dinov3_pretraining.yaml
    python src/train_dinov3.py --config configs/dinov3_pretraining.yaml --debug
    python src/train_dinov3.py --config configs/dinov3_pretraining.yaml --mae_checkpoint checkpoints/mae/mae_encoder_weights.pth
"""

import argparse
import logging
import yaml
import torch
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.swin_unet import create_dino_model
from src.data.dataset import create_dino_dataset
from src.training.trainer import DINOv3Trainer
from src.utils.distributed import setup_ddp, cleanup_ddp


def setup_logging(output_dir: Path, debug: bool = False):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train_dinov3.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def setup_wandb(config: dict):
    """Setup Weights & Biases logging."""
    if not config.get("logging", {}).get("use_wandb", False):
        return None

    try:
        import wandb

        wandb.init(
            project=config["logging"]["project_name"],
            entity=config["logging"].get("entity"),
            config=config,
            tags=config["logging"].get("tags", []),
            name=config.get("experiment_name"),
        )
        return wandb
    except ImportError:
        logging.warning("wandb not installed, skipping W&B logging")
        return None


def create_dino_dataloaders(config: dict, num_workers: int = 4):
    """Create DINOv3 data loaders."""
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = create_dino_dataset(config, debug_mode=config.get("debug", False))

    # Create data loader
    batch_size = config.get("training", {}).get("batch_size", 32)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("system", {}).get("pin_memory", True),
        drop_last=True,
        persistent_workers=config.get("system", {}).get("persistent_workers", True),
    )

    return loader


def create_dino_trainer(model, train_loader, config: dict, device, logger_obj=None):
    """Create DINOv3 trainer."""
    # Create optimizer with layer-wise learning rate decay
    optimizer_config = config.get("optimizer", {})

    # Group parameters by layer for layer-wise LR decay
    param_groups = []
    layer_decay = optimizer_config.get("layer_decay", 0.65)
    base_lr = optimizer_config.get("learning_rate", 5e-4)

    # Simple approach: all parameters with same LR (can be enhanced later)
    param_groups = [{"params": model.parameters(), "lr": base_lr}]

    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=optimizer_config.get("weight_decay", 0.04),
        betas=optimizer_config.get("betas", [0.9, 0.95]),
    )

    # Create scheduler
    scheduler_config = config.get("scheduler", {})
    num_epochs = config.get("training", {}).get("epochs", 100)

    if scheduler_config.get("type") == "cosine":
        warmup_epochs = scheduler_config.get("warmup_epochs", 10)
        total_steps = num_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        # Cosine with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            min_lr = scheduler_config.get("min_lr", 1e-6)
            return max(min_lr / base_lr, 0.5 * (1.0 + torch.cos(torch.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Create trainer
    trainer = DINOv3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger_obj=logger_obj,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(description="DINOv3 Pretraining for Swin-UNet")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default=None,
        help="Path to MAE checkpoint for sequential SSL",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override with debug mode
    if args.debug:
        config["debug"] = True
        config["training"]["epochs"] = 3
        config["training"]["batch_size"] = min(config["training"]["batch_size"], 4)

    # Setup logging
    output_dir = Path(config["output_dir"])
    logger = setup_logging(output_dir, args.debug)

    logger.info("=" * 80)
    logger.info("DINOv3 PRETRAINING")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Debug mode: {args.debug}")
    if args.mae_checkpoint:
        logger.info(f"Sequential SSL from MAE checkpoint: {args.mae_checkpoint}")

    # Setup device
    device_str = config.get("system", {}).get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create model
    logger.info("Creating DINOv3 model...")
    model = create_dino_model(config, mae_checkpoint=args.mae_checkpoint)
    model = model.to(device)

    # Print model info
    model_info = model.get_model_size()
    logger.info(f"Model created: {model_info}")
    logger.info(f"Model backbone input size: {model.backbone_input_size}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_dino_dataloaders(
        config, num_workers=config.get("system", {}).get("num_workers", 4)
    )

    logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Number of training batches: {len(train_loader)}")

    # Setup logging backend
    wandb_logger = setup_wandb(config)

    # Create trainer
    logger.info("Creating DINOv3 trainer...")
    trainer = create_dino_trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger_obj=wandb_logger,
    )

    # Training loop
    num_epochs = config.get("training", {}).get("epochs", 100)
    logger.info(f"Starting training for {num_epochs} epochs")

    try:
        # Train the model
        if args.resume:
            history = trainer.fit(num_epochs, resume_from_checkpoint=args.resume)
        else:
            history = trainer.fit(num_epochs)

        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")

        # Save final results
        results_path = Path(config["output_dir"]) / "training_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(
                {
                    "experiment_name": config.get("experiment_name"),
                    "config_path": args.config,
                    "mae_checkpoint": args.mae_checkpoint,
                    "final_metrics": {
                        k: v[-1] if v else None for k, v in history.items()
                    },
                    "total_epochs": len(history.get("train_loss", [])),
                    "model_info": model_info,
                },
                f,
            )

        logger.info(f"Training results saved to: {results_path}")

        # Print next steps
        print("\n" + "=" * 80)
        print("DINOV3 PRETRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. The encoder weights have been saved for segmentation fine-tuning")
        print(
            "2. You can now proceed with segmentation training using the pretrained encoder"
        )
        print("3. Use the saved encoder weights in your segmentation config")
        print("\nKey files:")
        print(f"- Encoder weights: {trainer.checkpoint_dir}/dino_encoder_weights.pth")
        print(f"- Best checkpoint: {trainer.checkpoint_dir}/dino_best_checkpoint.pth")
        print(f"- Training results: {results_path}")

        print("\nTo start segmentation fine-tuning:")
        print(
            "python train.py --config configs/config_vehicle_csam_proxy.yaml --dino_encoder checkpoints/dinov3/dino_encoder_weights.pth"
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        # Cleanup
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
