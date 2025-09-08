import argparse
import yaml
import torch
from pathlib import Path
from src.models.swin_unet import SwinUNet
from src.data.dataset import DynamicOversamplingDataset
from src.training.trainer import Trainer
from src.losses.combined import CombinedLoss
from torch.utils.data import DataLoader
import logging


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Swin-UNet segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (sample small dataset)"
    )
    parser.add_argument(
        "--debug-ratio",
        type=float,
        default=0.01,
        help="Debug sample ratio (default: 0.01 = 1 percent)",
    )
    parser.add_argument("--debug-max", type=int, help="Debug max images per split")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override config with debug settings if debug flag is used
    if args.debug:
        if "debug" not in config:
            config["debug"] = {}
        config["debug"]["enabled"] = True
        config["debug"]["sample_ratio"] = args.debug_ratio
        if args.debug_max:
            config["debug"]["max_images"] = args.debug_max
        print(
            f"🐛 DEBUG MODE ENABLED: sample_ratio={args.debug_ratio*100:.1f}%, max_images={args.debug_max}"
        )

    # Device selection - prioritize MPS for Mac M-series chips
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders) on Mac M-series GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    print(f"Selected device: {device}")

    data_dir = config["data"].get("data_dir", "data/vehicle_damage_csam_proxy")
    dataset_format = config["data"].get("format", "coco")

    train_dataset = DynamicOversamplingDataset(
        data_root=data_dir, split="train", dataset_format=dataset_format, config=config
    )
    val_dataset = DynamicOversamplingDataset(
        data_root=data_dir, split="val", dataset_format=dataset_format, config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Model
    model = SwinUNet(
        backbone_name=config["model"].get("backbone", "swin_large_patch4_window12_384"),
        num_classes=config["model"].get("num_classes", 2),
        in_channels=config["model"].get("in_channels", 3),
    ).to(device)

    # Loss, optimizer, scheduler
    loss_config = config["training"]["loss"]
    class_weights = loss_config.get("class_weights", [1.0, 2.5])
    if class_weights:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = CombinedLoss(
        dice_weight=loss_config.get("dice_weight", 0.5),
        ce_weight=loss_config.get("ce_weight", 0.3),
        focal_weight=loss_config.get("focal_weight", 0.2),
        focal_alpha=loss_config.get("focal_alpha", 0.25),
        focal_gamma=loss_config.get("focal_gamma", 2.0),
        class_weights=class_weights,
    )
    learning_rate = float(
        config["training"].get("learning_rate", config["training"].get("lr", 1e-4))
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )
    trainer.fit(epochs=config["training"]["max_epochs"])


if __name__ == "__main__":
    main()
