#!/usr/bin/env python3
"""
MAE Pretraining Script

Self-supervised pretraining using Masked Autoencoder on unlabeled SAM images.
This script implements the first stage of the advanced pipeline described in
ADDITIONAL_PRETRAINING.md.

Usage:
    python src/train_mae.py --config configs/mae_pretraining.yaml
    python src/train_mae.py --config configs/mae_pretraining.yaml --debug  # Quick test
    python src/train_mae.py --config configs/mae_pretraining.yaml --resume checkpoints/mae_latest.pth
"""

import argparse
import logging
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.swin_unet import create_mae_model
from src.data.dataset import create_mae_dataset, MAEPretrainingDataset
from src.training.trainer import MAETrainer
from src.utils.distributed import setup_ddp, cleanup_ddp


def create_mae_dataloaders(config: dict, num_workers: int = 4):
    """Create MAE data loaders."""
    import torch
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = create_mae_dataset(config, debug_mode=config.get("debug", False))

    # Create data loader
    batch_size = config.get("training", {}).get("batch_size", 32)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def create_mae_trainer(model, train_loader, config: dict, device, logger_obj=None):
    """Create MAE trainer."""
    import torch.optim as optim

    # Create optimizer
    optimizer_config = config.get("optimizer", {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get("learning_rate", 1e-4),
        weight_decay=optimizer_config.get("weight_decay", 0.05),
        betas=optimizer_config.get("betas", [0.9, 0.95]),
    )

    # Create scheduler
    scheduler_config = config.get("scheduler", {})
    if scheduler_config.get("type") == "cosine":
        num_epochs = config.get("mae", {}).get("epochs", 100)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=scheduler_config.get("min_lr", 1e-6),
        )
    else:
        scheduler = None

    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # No validation for MAE pretraining
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger_obj=logger_obj,
    )

    return trainer


def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(config["log_dir"]) / "mae_training.log"),
        ],
    )


def create_output_dirs(config: dict):
    """Create output directories."""
    for dir_key in ["output_dir", "checkpoint_dir", "log_dir"]:
        if dir_key in config:
            Path(config[dir_key]).mkdir(parents=True, exist_ok=True)

    # Create tensorboard dir if enabled
    tb_config = config.get("logging", {}).get("tensorboard", {})
    if tb_config.get("enabled", False):
        Path(tb_config["log_dir"]).mkdir(parents=True, exist_ok=True)


def setup_device(config: dict) -> torch.device:
    """Setup training device."""
    device_config = config.get("system", {}).get("device", "auto")

    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)

    print(f"Using device: {device}")
    return device


def setup_wandb(config: dict):
    """Setup Weights & Biases logging if enabled."""
    wandb_config = config.get("logging", {}).get("wandb", {})

    if not wandb_config.get("enabled", False):
        return None

    try:
        import wandb

        wandb.init(
            project=wandb_config.get("project", "segment-anomaly-mae"),
            entity=wandb_config.get("entity"),
            config=config,
            tags=wandb_config.get("tags", []),
            name=config.get("experiment_name", "mae_pretraining"),
        )

        return wandb

    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")
        return None


def main():
    parser = argparse.ArgumentParser(description="MAE Pretraining for Segment Anomaly")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override debug settings if flag is set
    if args.debug:
        config["debug"]["enabled"] = True
        config["mae"]["debug_mode"] = True
        config["mae"]["epochs"] = 2  # Quick test
        config["mae"]["save_reconstructions"] = False
        print(
            "🐛 DEBUG MODE ENABLED - Running quick test with reduced dataset and epochs"
        )

    # Create output directories
    create_output_dirs(config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info(
        f"Starting MAE pretraining experiment: {config.get('experiment_name', 'mae_pretraining')}"
    )
    logger.info(f"Config: {args.config}")

    # Setup distributed training if enabled
    if config.get("training", {}).get("ddp", False):
        setup_ddp(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = setup_device(config)

    # Set random seed for reproducibility
    seed = config.get("system", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Setup optimization flags
    opt_config = config.get("optimization", {})
    if opt_config.get("benchmark", True):
        torch.backends.cudnn.benchmark = True
    if opt_config.get("deterministic", False):
        torch.backends.cudnn.deterministic = True

    # Create model
    logger.info("Creating MAE model...")
    model = create_mae_model(config)
    model = model.to(device)

    # Print model info
    model_info = model.get_model_size()
    logger.info(f"Model created: {model_info}")
    logger.info(f"Model backbone input size: {model.backbone_input_size}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_mae_dataloaders(
        config, num_workers=config.get("system", {}).get("num_workers", 4)
    )

    logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Number of training batches: {len(train_loader)}")

    # Setup logging backend
    wandb_logger = setup_wandb(config)

    # Create trainer
    logger.info("Creating MAE trainer...")
    trainer = create_mae_trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        device=device,
        logger_obj=wandb_logger,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    elif config.get("resume", {}).get("checkpoint_path"):
        checkpoint_path = config["resume"]["checkpoint_path"]
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch = trainer.load_checkpoint(checkpoint_path)

    # Training loop
    num_epochs = config.get("mae", {}).get("epochs", 100)
    logger.info(
        f"Starting training for {num_epochs} epochs (starting from epoch {start_epoch})"
    )

    try:
        # Train the model
        history = trainer.train(num_epochs, start_epoch)

        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")

        # Save final results
        results_path = Path(config["output_dir"]) / "training_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(
                {
                    "experiment_name": config.get("experiment_name"),
                    "config_path": args.config,
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
        print("🎉 MAE PRETRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. The encoder weights have been saved for segmentation fine-tuning")
        print(
            "2. You can now proceed with segmentation training using the pretrained encoder"
        )
        print("3. Use the saved encoder weights in your segmentation config")
        print("\nKey files:")
        print(f"- Encoder weights: {trainer.checkpoint_dir}/mae_encoder_weights.pth")
        print(f"- Best checkpoint: {trainer.checkpoint_dir}/mae_best.pth")
        print(f"- Training results: {results_path}")

        if config.get("mae", {}).get("save_reconstructions", True):
            print(f"- Reconstruction visualizations: {trainer.vis_dir}")

        print("\nTo start segmentation fine-tuning:")
        print(
            "python train.py --config configs/config_vehicle_csam_proxy.yaml --mae_encoder checkpoints/mae_encoder_weights.pth"
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

        if config.get("training", {}).get("ddp", False):
            cleanup_ddp()


if __name__ == "__main__":
    main()
