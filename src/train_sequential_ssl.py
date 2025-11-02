#!/usr/bin/env python3
"""
Sequential Self-Supervised Learning Pipeline

Orchestrates MAE → DINOv3 sequential pretraining for Swin-UNet backbone.
First runs MAE reconstruction pretraining, then uses those weights to initialize DINOv3.

Usage:
    python src/train_sequential_ssl.py --mae_config configs/mae_pretraining.yaml --dino_config configs/dinov3_pretraining.yaml
    python src/train_sequential_ssl.py --mae_config configs/mae_pretraining_mps.yaml --dino_config configs/dinov3_pretraining_mps.yaml --debug
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
from datetime import datetime
import mlflow

sys.path.append(str(Path(__file__).parent.parent))

from src.models.swin_unet import create_mae_model, create_dino_model
from src.data.dataset import create_mae_dataset, create_dino_dataset
from src.training.trainer import MAETrainer, DINOv3Trainer
import torch.optim as optim


def setup_logging(output_dir: Path, debug: bool = False):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train_sequential_ssl.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def run_mae_pretraining(config: dict, device, logger, mlflow_run=None):
    """Run MAE pretraining phase."""
    logger.info("=" * 80)
    logger.info("PHASE 1: MAE PRETRAINING")
    logger.info("=" * 80)

    # Start MLflow nested run for MAE
    if mlflow_run:
        mlflow.start_run(run_id=mlflow_run.info.run_id, nested=True)
        mlflow.set_tag("phase", "mae_pretraining")

    # Create MAE model
    logger.info("Creating MAE model...")
    model = create_mae_model(config)
    model = model.to(device)

    model_info = model.get_model_size()
    logger.info(f"MAE model created: {model_info}")

    # Create dataset
    logger.info("Creating MAE dataset...")
    from torch.utils.data import DataLoader

    dataset = create_mae_dataset(
        config, debug_mode=config.get("debug", {}).get("enabled", False)
    )

    # Get batch size from either mae or training section
    batch_size = config.get("mae", {}).get(
        "batch_size", config.get("training", {}).get("batch_size", 32)
    )
    num_workers = config.get("system", {}).get("num_workers", 4)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("system", {}).get("pin_memory", True),
        drop_last=True,
        persistent_workers=(
            config.get("system", {}).get("persistent_workers", True)
            if num_workers > 0
            else False
        ),
    )

    logger.info(f"MAE training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Number of training batches: {len(train_loader)}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("mae", {}).get(
            "learning_rate", config.get("optimizer", {}).get("learning_rate", 1.5e-4)
        ),
        weight_decay=config.get("mae", {}).get(
            "weight_decay", config.get("optimizer", {}).get("weight_decay", 0.05)
        ),
        betas=config.get("mae", {}).get(
            "betas", config.get("optimizer", {}).get("betas", [0.9, 0.95])
        ),
    )

    # Create scheduler
    scheduler_config = config.get("scheduler", {})
    num_epochs = config.get("mae", {}).get(
        "epochs", config.get("training", {}).get("epochs", 300)
    )

    if scheduler_config.get("type") == "cosine":
        warmup_epochs = scheduler_config.get(
            "warmup_epochs", config.get("mae", {}).get("warmup_epochs", 40)
        )
        total_steps = num_epochs * len(train_loader)
        warmup_steps = warmup_epochs * len(train_loader)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            min_lr = scheduler_config.get(
                "min_lr", config.get("mae", {}).get("min_lr", 1e-6)
            )
            base_lr = config.get("mae", {}).get(
                "learning_rate",
                config.get("optimizer", {}).get("learning_rate", 1.5e-4),
            )
            return max(min_lr / base_lr, 0.5 * (1.0 + torch.cos(torch.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Log MAE hyperparameters to MLflow
    if mlflow_run:
        mlflow.log_params(
            {
                "mae_epochs": num_epochs,
                "mae_batch_size": batch_size,
                "mae_learning_rate": config.get("mae", {}).get("learning_rate", 1.5e-4),
                "mae_weight_decay": config.get("mae", {}).get("weight_decay", 0.05),
                "mae_mask_ratio": config.get("mae", {}).get("mask_ratio", 0.75),
                "mae_warmup_epochs": config.get("mae", {}).get("warmup_epochs", 40),
                "mae_dataset_size": len(train_loader.dataset),
                "mae_num_batches": len(train_loader),
            }
        )

    # Create MAE trainer
    logger.info("Creating MAE trainer...")
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=None,  # MAE computes loss internally
        config=config,
        device=device,

        mlflow_run=mlflow_run,
    )

    # Train MAE
    logger.info(f"Starting MAE training for {num_epochs} epochs...")

    # Check if we need to resume
    resume_checkpoint = config.get("resume", {}).get("checkpoint_path")
    if resume_checkpoint:
        logger.info(f"Resuming MAE training from: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint, load_optimizer=True)

    history = trainer.fit(num_epochs)

    logger.info("MAE pretraining completed!")
    logger.info(f"Final MAE loss: {history['train_loss'][-1]:.4f}")

    # Log final MAE metrics to MLflow
    if mlflow_run:
        mlflow.log_metrics(
            {
                "mae_final_loss": history["train_loss"][-1],
                "mae_total_epochs": len(history["train_loss"]),
            }
        )
        mlflow.end_run()

    # Get encoder checkpoint path
    encoder_path = trainer.checkpoint_dir / "mae_encoder_weights.pth"
    logger.info(f"MAE encoder weights saved to: {encoder_path}")

    return encoder_path, history


def run_dino_pretraining(
    config: dict, mae_checkpoint: str, device, logger, mlflow_run=None
):
    """Run DINOv3 pretraining phase."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: DINOV3 PRETRAINING")
    logger.info("=" * 80)
    logger.info(f"Initializing from MAE checkpoint: {mae_checkpoint}")

    # Start MLflow nested run for DINOv3
    if mlflow_run:
        mlflow.start_run(run_id=mlflow_run.info.run_id, nested=True)
        mlflow.set_tag("phase", "dinov3_pretraining")

    # Create DINOv3 model with MAE initialization
    logger.info("Creating DINOv3 model...")
    model = create_dino_model(config, mae_checkpoint=str(mae_checkpoint))
    model = model.to(device)

    model_info = model.get_model_size()
    logger.info(f"DINOv3 model created: {model_info}")

    # Create dataset
    logger.info("Creating DINOv3 dataset...")
    from torch.utils.data import DataLoader

    dataset = create_dino_dataset(
        config, debug_mode=config.get("debug", {}).get("enabled", False)
    )

    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("system", {}).get("num_workers", 4)

    # Custom collate function for DINOv3 multi-crop batches
    def dino_collate_fn(batch):
        """
        Collate function that properly handles lists of crops from DINOv3 dataset.
        Each sample returns {'global_views': [crop1, crop2], 'local_views': [crop1, ..., cropN]}
        We need to stack each crop type across the batch.
        """
        global_views = []
        local_views = []

        # Get number of crops from first sample
        n_global = len(batch[0]["global_views"])
        n_local = len(batch[0]["local_views"])

        # Stack each crop type across batch dimension
        for crop_idx in range(n_global):
            crops = torch.stack([sample["global_views"][crop_idx] for sample in batch])
            global_views.append(crops)

        for crop_idx in range(n_local):
            crops = torch.stack([sample["local_views"][crop_idx] for sample in batch])
            local_views.append(crops)

        return {"global_views": global_views, "local_views": local_views}

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("system", {}).get("pin_memory", True),
        drop_last=True,
        persistent_workers=(
            config.get("system", {}).get("persistent_workers", True)
            if num_workers > 0
            else False
        ),
        collate_fn=dino_collate_fn,
    )

    logger.info(f"DINOv3 training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Number of training batches: {len(train_loader)}")

    # Create optimizer with layer-wise LR decay
    optimizer_config = config.get("optimizer", {})
    base_lr = optimizer_config.get("learning_rate", 5e-4)

    optimizer = optim.AdamW(
        model.parameters(),
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

    # Log DINOv3 hyperparameters to MLflow
    if mlflow_run:
        mlflow.log_params(
            {
                "dino_epochs": num_epochs,
                "dino_batch_size": batch_size,
                "dino_learning_rate": base_lr,
                "dino_weight_decay": optimizer_config.get("weight_decay", 0.04),
                "dino_warmup_epochs": scheduler_config.get("warmup_epochs", 10),
                "dino_teacher_temp": config.get("dino", {}).get("teacher_temp", 0.07),
                "dino_student_temp": config.get("dino", {}).get("student_temp", 0.1),
                "dino_momentum_teacher": config.get("dino", {}).get(
                    "momentum_teacher", 0.996
                ),
                "dino_n_global_crops": config.get("dino", {}).get("n_global_crops", 2),
                "dino_n_local_crops": config.get("dino", {}).get("n_local_crops", 4),
                "dino_dataset_size": len(train_loader.dataset),
                "dino_num_batches": len(train_loader),
            }
        )

    # Create DINOv3 trainer
    logger.info("Creating DINOv3 trainer...")
    trainer = DINOv3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=None,  # DINOv3 computes loss internally
        config=config,
        device=device,
        mlflow_run=mlflow_run,
    )

    # Train DINOv3
    logger.info(f"Starting DINOv3 training for {num_epochs} epochs...")

    # Check if we need to resume
    resume_checkpoint = config.get("resume", {}).get("checkpoint_path")
    if resume_checkpoint:
        logger.info(f"Resuming DINOv3 training from: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint, load_optimizer=True)

    history = trainer.fit(num_epochs)

    logger.info("DINOv3 pretraining completed!")
    logger.info(f"Final DINOv3 loss: {history['train_loss'][-1]:.4f}")

    # Log final DINOv3 metrics to MLflow
    if mlflow_run:
        mlflow.log_metrics(
            {
                "dino_final_loss": history["train_loss"][-1],
                "dino_total_epochs": len(history["train_loss"]),
            }
        )
        mlflow.end_run()

    # Get encoder checkpoint path
    encoder_path = trainer.checkpoint_dir / "dino_encoder_weights.pth"
    logger.info(f"DINOv3 encoder weights saved to: {encoder_path}")

    return encoder_path, history


def main():
    parser = argparse.ArgumentParser(
        description="Sequential SSL: MAE → DINOv3 Pretraining"
    )
    parser.add_argument(
        "--mae_config", type=str, required=True, help="Path to MAE config file"
    )
    parser.add_argument(
        "--dino_config", type=str, required=True, help="Path to DINOv3 config file"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--skip_mae",
        type=str,
        default=None,
        help="Skip MAE training and use this checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous sequential SSL directory (e.g., checkpoints/sequential_ssl_20251101_191502)",
    )
    parser.add_argument(
        "--continue-training",
        type=str,
        default=None,
        dest="continue_training",
        help="Continue training from a completed checkpoint with new data (loads final weights, resets epochs)",
    )

    args = parser.parse_args()

    # Load configs
    with open(args.mae_config, "r") as f:
        mae_config = yaml.safe_load(f)

    with open(args.dino_config, "r") as f:
        dino_config = yaml.safe_load(f)

    # Override with debug mode
    if args.debug:
        mae_config["debug"] = True
        # Handle different config structures
        if "mae" in mae_config and "epochs" in mae_config["mae"]:
            mae_config["mae"]["epochs"] = 3
            mae_config["mae"]["batch_size"] = min(
                mae_config["mae"].get("batch_size", 8), 4
            )
        elif "training" in mae_config:
            mae_config["training"]["epochs"] = 3
            mae_config["training"]["batch_size"] = min(
                mae_config["training"].get("batch_size", 8), 4
            )

        dino_config["debug"] = True
        if "training" in dino_config:
            dino_config["training"]["epochs"] = 3
            dino_config["training"]["batch_size"] = min(
                dino_config["training"].get("batch_size", 8), 4
            )

    # Setup main output directory
    if args.continue_training:
        # Continue training mode: load from completed checkpoint but create new directory
        source_dir = Path(args.continue_training)
        if not source_dir.exists():
            raise ValueError(
                f"Continue training source directory does not exist: {source_dir}"
            )

        # Create new directory for continued training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = Path(f"checkpoints/sequential_ssl_{timestamp}_continued")
        main_output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging early for continue mode
        logger = setup_logging(main_output_dir, args.debug)
        logger.info(f"Continue training from: {source_dir}")
        logger.info(f"New output directory: {main_output_dir}")

    elif args.resume:
        # Resume from existing directory
        main_output_dir = Path(args.resume)
        if not main_output_dir.exists():
            raise ValueError(f"Resume directory does not exist: {main_output_dir}")
        timestamp = main_output_dir.name.replace("sequential_ssl_", "")
        logger = setup_logging(main_output_dir, args.debug)
        logger.info(f"Resuming from: {main_output_dir}")

    else:
        # Create new directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = Path(f"checkpoints/sequential_ssl_{timestamp}")
        main_output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(main_output_dir, args.debug)

    logger.info("=" * 80)
    logger.info("SEQUENTIAL SELF-SUPERVISED LEARNING")
    logger.info("MAE → DINOv3 → Segmentation Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"MAE config: {args.mae_config}")
    logger.info(f"DINOv3 config: {args.dino_config}")
    logger.info(f"Output directory: {main_output_dir}")
    logger.info(f"Debug mode: {args.debug}")

    # Setup MLflow
    mlflow_tracking_uri = Path("logs/mlruns").absolute().as_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("sequential_ssl")
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    logger.info("View experiments with: mlflow ui --backend-store-uri logs/mlruns")

    # Setup device
    device_str = mae_config.get("system", {}).get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    try:
        # Start MLflow run for the entire sequential SSL pipeline
        with mlflow.start_run(run_name=f"sequential_ssl_{timestamp}") as run:
            # Log overall pipeline parameters
            mlflow.log_params(
                {
                    "pipeline": "MAE → DINOv3",
                    "mae_config_file": args.mae_config,
                    "dino_config_file": args.dino_config,
                    "debug_mode": args.debug,
                    "device": str(device),
                    "resume_mode": args.resume is not None,
                    "continue_training": args.continue_training is not None,
                }
            )

            # Check what phases are already complete or should be continued
        mae_complete = False
        dino_complete = False
        mae_encoder_path = None
        dino_encoder_path = None
        mae_resume_checkpoint = None
        dino_resume_checkpoint = None
        mae_init_weights = None  # For continue-training mode
        dino_init_weights = None  # For continue-training mode

        if args.continue_training:
            # Continue training mode: load final weights as initialization, don't skip phases
            source_dir = Path(args.continue_training)

            # Look for MAE encoder to use as initialization
            mae_source_paths = [
                source_dir / "mae" / "mae_encoder_weights.pth",
                source_dir
                / "dinov3"
                / "dino_encoder_weights.pth",  # DINOv3 encoder works too
            ]
            for path in mae_source_paths:
                if path.exists():
                    mae_init_weights = path
                    logger.info(f"→ Will initialize MAE from: {mae_init_weights}")
                    break

            # Look for DINOv3 checkpoint to use as initialization
            dino_source_path = source_dir / "dinov3" / "dino_best_checkpoint.pth"
            if dino_source_path.exists():
                dino_init_weights = dino_source_path
                logger.info(f"→ Will initialize DINOv3 from: {dino_init_weights}")
            elif mae_init_weights:
                # Can start DINOv3 from MAE weights
                logger.info(f"→ Will initialize DINOv3 from MAE encoder")

            # In continue mode, we don't mark anything as complete - we train fresh with new data
            mae_complete = False
            dino_complete = False

        elif args.resume:
            # Check for existing checkpoints in resume mode
            mae_output_dir = main_output_dir / "mae"

            # Check if MAE is complete
            potential_mae_encoder = mae_output_dir / "mae_encoder_weights.pth"
            if potential_mae_encoder.exists():
                mae_complete = True
                mae_encoder_path = potential_mae_encoder
                logger.info(f"✓ Found completed MAE checkpoint: {mae_encoder_path}")
            else:
                # Check for incomplete MAE checkpoint to resume from
                mae_checkpoints = (
                    list(mae_output_dir.glob("mae_checkpoint_epoch_*.pth"))
                    if mae_output_dir.exists()
                    else []
                )
                if mae_checkpoints:
                    # Get the latest checkpoint
                    mae_resume_checkpoint = sorted(mae_checkpoints)[-1]
                    logger.info(
                        f"→ Found incomplete MAE training, will resume from: {mae_resume_checkpoint}"
                    )

            # Check for DINOv3 checkpoints
            dino_output_dir = main_output_dir / "dinov3"

            # Check if DINOv3 is complete
            potential_dino_encoder = dino_output_dir / "dino_encoder_weights.pth"
            if potential_dino_encoder.exists():
                dino_complete = True
                dino_encoder_path = potential_dino_encoder
                logger.info(f"✓ Found completed DINOv3 checkpoint: {dino_encoder_path}")
            else:
                # Check for incomplete DINOv3 checkpoint to resume from
                dino_checkpoints = (
                    list(dino_output_dir.glob("dino_checkpoint_epoch_*.pth"))
                    if dino_output_dir.exists()
                    else []
                )
                if dino_checkpoints:
                    # Get the latest checkpoint
                    dino_resume_checkpoint = sorted(dino_checkpoints)[-1]
                    logger.info(
                        f"→ Found incomplete DINOv3 training, will resume from: {dino_resume_checkpoint}"
                    )

        # Phase 1: MAE Pretraining
        if args.skip_mae:
            logger.info(f"Skipping MAE training, using checkpoint: {args.skip_mae}")
            mae_encoder_path = Path(args.skip_mae)
            mae_history = {}
        elif mae_complete:
            logger.info(f"MAE already completed, using checkpoint: {mae_encoder_path}")
            mae_history = {}
        else:
            # Update MAE output directory
            mae_output_dir = main_output_dir / "mae"
            mae_config["output_dir"] = str(mae_output_dir)

            # If we have initialization weights from continue-training mode
            if mae_init_weights:
                logger.info(f"Initializing MAE with weights from: {mae_init_weights}")
                if "resume" not in mae_config:
                    mae_config["resume"] = {}
                mae_config["resume"]["checkpoint_path"] = str(mae_init_weights)
                mae_config["resume"][
                    "load_optimizer"
                ] = False  # Don't load optimizer for fresh training
                mae_config["resume"][
                    "load_scheduler"
                ] = False  # Don't load scheduler for fresh training
            # If we have a resume checkpoint, add it to config
            elif mae_resume_checkpoint:
                logger.info(f"Resuming MAE training from: {mae_resume_checkpoint}")
                if "resume" not in mae_config:
                    mae_config["resume"] = {}
                mae_config["resume"]["checkpoint_path"] = str(mae_resume_checkpoint)
                mae_config["resume"]["load_optimizer"] = True
                mae_config["resume"]["load_scheduler"] = True

            mae_encoder_path, mae_history = run_mae_pretraining(
                mae_config, device, logger, run
            )

        # Phase 2: DINOv3 Pretraining
        if dino_complete:
            logger.info(
                f"DINOv3 already completed, using checkpoint: {dino_encoder_path}"
            )
            dino_history = {}
        else:
            dino_output_dir = main_output_dir / "dinov3"
            dino_config["output_dir"] = str(dino_output_dir)

            # If we have initialization weights from continue-training mode
            if dino_init_weights:
                logger.info(
                    f"Initializing DINOv3 with weights from: {dino_init_weights}"
                )
                if "resume" not in dino_config:
                    dino_config["resume"] = {}
                dino_config["resume"]["checkpoint_path"] = str(dino_init_weights)
                dino_config["resume"][
                    "load_optimizer"
                ] = False  # Don't load optimizer for fresh training
                dino_config["resume"][
                    "load_scheduler"
                ] = False  # Don't load scheduler for fresh training
            # If we have a resume checkpoint, add it to config
            elif dino_resume_checkpoint:
                logger.info(f"Resuming DINOv3 training from: {dino_resume_checkpoint}")
                if "resume" not in dino_config:
                    dino_config["resume"] = {}
                dino_config["resume"]["checkpoint_path"] = str(dino_resume_checkpoint)
                dino_config["resume"]["load_optimizer"] = True
                dino_config["resume"]["load_scheduler"] = True

            # Check for DINOv3 checkpoint to resume from
            dino_checkpoint_dir = dino_output_dir / "checkpoints"
            if dino_checkpoint_dir.exists():
                checkpoints = list(dino_checkpoint_dir.glob("dinov3_epoch_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(
                        checkpoints, key=lambda p: p.stat().st_mtime
                    )
                    logger.info(
                        f"Found DINOv3 checkpoint to resume from: {latest_checkpoint}"
                    )
                    dino_config["resume"] = {"checkpoint_path": str(latest_checkpoint)}

            dino_encoder_path, dino_history = run_dino_pretraining(
                dino_config, mae_encoder_path, device, logger, run
            )

        # Save final results
        results = {
            "timestamp": timestamp,
            "mae_config": args.mae_config,
            "dino_config": args.dino_config,
            "debug_mode": args.debug,
            "phases": {
                "mae": {
                    "encoder_path": str(mae_encoder_path),
                    "final_loss": (
                        mae_history.get("train_loss", [None])[-1]
                        if mae_history
                        else None
                    ),
                    "epochs": (
                        len(mae_history.get("train_loss", [])) if mae_history else 0
                    ),
                },
                "dinov3": {
                    "encoder_path": str(dino_encoder_path),
                    "final_loss": dino_history.get("train_loss", [None])[-1],
                    "epochs": len(dino_history.get("train_loss", [])),
                },
            },
        }

        results_path = main_output_dir / "sequential_ssl_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f)

            # Log final results to MLflow
            mlflow.log_artifact(str(results_path))

            # Print success message
            print("\n" + "=" * 80)
            print("SEQUENTIAL SSL COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print("\nTraining Pipeline:")
            print("1. ✓ MAE Pretraining - Masked image reconstruction")
            print("2. ✓ DINOv3 Pretraining - Self-distillation with MAE initialization")
            print("3. → Next: Segmentation Fine-tuning")

            print("\nKey Files:")
            if not args.skip_mae:
                print(f"- MAE Encoder: {mae_encoder_path}")
            print(f"- DINOv3 Encoder: {dino_encoder_path}")
            print(f"- Results: {results_path}")

            print("\nTo start segmentation fine-tuning:")
            print(
                f"python train.py --config configs/config_vehicle_csam_proxy.yaml --pretrained_encoder {dino_encoder_path}"
            )

            print("\nSequential SSL Benefits:")
            print("- MAE learns low-level visual reconstruction")
            print("- DINOv3 refines with semantic self-distillation")
            print("- Combined approach captures both local and global features")

            print("\nMLflow Tracking:")
            print("- View experiments: mlflow ui --backend-store-uri logs/mlruns")
            print(f"- Run ID: {run.info.run_id}")

            logger.info(f"Sequential SSL results saved to: {results_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Sequential SSL failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
