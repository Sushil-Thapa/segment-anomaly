# Vehicle Damage Segmentation

Swin-UNet based semantic segmentation for vehicle damage detection supporting COCO, YOLO, and Pascal VOC formats.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or use uv (recommended)
uv sync
```

## Pre-commit Setup

This project uses pre-commit hooks to ensure code quality and consistency.

```bash
# Add pre-commit as dev dependency
uv add --dev pre-commit

# Install hooks locally
uv run pre-commit install

# (Optional) run hooks on all files
uv run pre-commit run --all-files
```

**Note:** A `.secrets.baseline` file is already provided. If you need to update it with new secrets to ignore, run:
```bash
uv run detect-secrets scan --baseline .secrets.baseline
```

## Data Preparation

### Download Public Datasets
```bash
# Download and prepare public datasets (requires appropriate config)
uv run python src/data/download.py configs/your_download_config.yaml

# Prepare existing dataset (convert to COCO format)
uv run python src/data/prepare.py data/your_dataset
```

### Manual Dataset Setup
For custom datasets, organize your data in one of these formats:

**COCO Format:**
```
data/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

**YOLO Format:**
```
data/
├── train/
│   ├── images/
│   └── labels/  # .txt files
└── val/
    ├── images/
    └── labels/
```

**Pascal VOC Format:**
```
data/
├── train/
│   ├── images/
│   └── masks/  # .png files
└── val/
    ├── images/
    └── masks/
```

## Training

### Quick Start
```bash
# Full training
uv run python src/train.py --config configs/config_vehicle_csam_proxy.yaml

# Debug mode (1% data, 3 epochs) - for pipeline testing
uv run python src/train.py --config configs/config_vehicle_csam_proxy.yaml --debug
```

### Self-Supervised Learning (SSL) Pretraining

Pretrain the encoder with MAE → DINOv3 sequential pipeline before fine-tuning:

```bash
# Full sequential SSL (MAE → DINOv3)
uv run python src/train_sequential_ssl.py \
    --mae_config configs/mae_pretraining_mps.yaml \
    --dino_config configs/dinov3_pretraining_mps.yaml

# Resume interrupted training
uv run python src/train_sequential_ssl.py \
    --mae_config configs/mae_pretraining_mps.yaml \
    --dino_config configs/dinov3_pretraining_mps.yaml \
    --resume checkpoints/sequential_ssl_20251101_191502

# Continue training on new dataset (keeps learned weights, resets optimizer)
uv run python src/train_sequential_ssl.py \
    --mae_config configs/mae_pretraining_mps.yaml \
    --dino_config configs/dinov3_pretraining_mps.yaml \
    --continue-training checkpoints/sequential_ssl_20251101_191502

# View MLflow experiments
mlflow ui --backend-store-uri logs/mlruns
```

**Resume vs Continue-Training:**

- `--resume`: Continue interrupted training (same dataset, same checkpoint dir, resumes from epoch N)
- `--continue-training`: Start fresh training with learned weights (new dataset, new checkpoint dir, starts from epoch 0)

See [SEQUENTIAL_SSL.md](docs/SEQUENTIAL_SSL.md) and [RESUME_TRAINING.md](docs/RESUME_TRAINING.md) for details.

### Dataset Format Support
The unified dataset loader supports three formats via `dataset_format` config:

Set `dataset_format: coco|yolo|voc` in your config file.

## Inference

```bash
python src/inference.py --config configs/your_config.yaml --input path/to/images --output path/to/results
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python tests/test_integration.py
python tests/test_metrics.py
python tests/test_tiling.py
```

## Key Features

- **GPU Support**: Automatic MPS (Mac M-series) and CUDA detection
- **Tiled Processing**: Handles large images via intelligent tiling
- **Dynamic Oversampling**: Balances positive/negative samples during training
- **Multi-format Support**: COCO, YOLO, Pascal VOC in single codebase
- **Debug Mode**: Quick pipeline testing with 1% data sampling
- **Progress Tracking**: Real-time training progress with tqdm-style updates

## Project Structure

## Project Structure

```text
src/
├── data/           # Dataset handling, transforms, and data scripts
├── models/         # Swin-UNet architecture
├── training/       # Training loop and callbacks
├── losses/         # Combined loss functions
└── utils/          # Metrics, distributed training, export
```

````

Copyright © 2025 Sushil Thapa