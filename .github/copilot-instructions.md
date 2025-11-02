## Project Style Guidelines

### Code Organization
- Integrate new features into existing files rather than creating separate modules
- Testing files separate from production code for easy cleanup
- No emojis in code/docs, keep tone professional and concise
- Write as if the user wrote it themselves, not "because you asked"

### Development Workflow
- Use `uv run python` for all script execution
- Git: Add files specifically, never `git add .` or `git add -A`
- Avoid adding .md documentation files to git

### File Consolidation Pattern
When implementing features:
1. Add to existing appropriate file (e.g., trainer.py, swin_unet.py)
2. Create temporary test file if needed
3. Once stable, merge into existing files and remove test file
4. Training scripts (train_*.py) serve as entry points only

## Self-Supervised Learning Pipeline

### Architecture Overview
**Sequential Pipeline**: MAE → DINOv3 → Segmentation Fine-tuning
- MAE: Learns pixel-level features via masked reconstruction (75% mask ratio)
- DINOv3: Learns semantic features via self-distillation (initialized from MAE)
- Both phases consolidated into existing files per project style

### File Structure
```
src/
├── models/swin_unet.py         # MAESwinUNet, DINOv3SwinUNet, DINOHead
├── models/decoder.py           # MAEDecoder
├── training/trainer.py         # MAETrainer, DINOv3Trainer (with MLflow support)
├── data/dataset.py             # MAEPretrainingDataset, DINOv3PretrainingDataset
├── data/transforms.py          # get_dino_multicrop_transform
├── train_mae.py                # Standalone MAE entry point
├── train_dinov3.py             # Standalone DINOv3 entry point
└── train_sequential_ssl.py     # Sequential MAE→DINOv3 pipeline

configs/
├── mae_pretraining.yaml        # Production MAE config
├── mae_pretraining_mps.yaml    # Apple Silicon MAE
├── dinov3_pretraining.yaml     # Production DINOv3 config
└── dinov3_pretraining_mps.yaml # Apple Silicon DINOv3
```

### MAE Pretraining Details
- Mask ratio: 75%
- Loss: Pixel reconstruction (MSE)
- Decoder: Lightweight with encoder feature dims [96, 192, 384, 768]
- Seamless mode switching: set_training_mode("mae")

### DINOv3 Pretraining Details
- Student-teacher EMA: momentum=0.996
- Multi-crop strategy:
  - Global: 2 crops @ 384x384 (scale 0.4-1.0)
  - Local: 4-6 crops @ 192x192 (scale 0.05-0.4)
- DINOHead: encoder_dim → 2048 → 256 → output_dim
- Teacher temp: warmup 0.04→0.07 over 30 epochs
- Center momentum: 0.9 (prevents mode collapse)
- Loss: Cross-entropy(student, sharpened_teacher)
- Initialization: load_mae_pretrained_encoder() for sequential SSL

### Apple Silicon (MPS) Optimizations
- Reduced batch sizes and model dimensions
- Automatic cache management: torch.mps.empty_cache()
- Gradient accumulation: batch_size=8, accumulation=4
- FP32 only (no mixed precision on MPS)
- DINOv3 output_dim: 8192 (MPS) vs 65536 (CUDA)

### Training Commands
```bash
# Standalone MAE
uv run python src/train_mae.py --config configs/mae_pretraining_mps.yaml

# Standalone DINOv3 (optionally from MAE checkpoint)
uv run python src/train_dinov3.py --config configs/dinov3_pretraining_mps.yaml [--mae_checkpoint PATH]

# Sequential SSL (recommended)
uv run python src/train_sequential_ssl.py \
  --mae_config configs/mae_pretraining_mps.yaml \
  --dino_config configs/dinov3_pretraining_mps.yaml

# Resume interrupted training
uv run python src/train_sequential_ssl.py ... --resume checkpoints/sequential_ssl_TIMESTAMP

# Continue with new dataset (keeps weights, resets optimizer)
uv run python src/train_sequential_ssl.py ... --continue-training checkpoints/sequential_ssl_TIMESTAMP

# Segmentation fine-tuning
uv run python src/train.py \
  --config configs/config_vehicle_csam_proxy.yaml \
  --pretrained_encoder checkpoints/sequential_ssl_TIMESTAMP/dinov3/dino_encoder_weights.pth
```

### Resume vs Continue-Training
| Flag | Purpose | Checkpoint Dir | Optimizer | Epoch | Use Case |
|------|---------|----------------|-----------|-------|----------|
| --resume | Continue interrupted training | Same | Loaded | Resume from N | Power outage, crash |
| --continue-training | Train on new data | New (timestamped) | Fresh | Start from 0 | Multi-dataset pretraining |
| --skip_mae | Use existing MAE | Same/new | N/A | N/A | Already have MAE weights |

### MLflow Experiment Tracking
**Setup**: Automatic in train_sequential_ssl.py
- Tracking URI: `logs/mlruns`
- Experiment: `sequential_ssl`
- View: `mlflow ui --backend-store-uri logs/mlruns`

**Logged Parameters** (once per run):
- Pipeline: mae/dino config paths, device, debug mode
- MAE: epochs, batch_size, lr, weight_decay, mask_ratio, warmup_epochs, dataset_size
- DINOv3: epochs, batch_size, lr, teacher_temp, student_temp, momentum, n_crops, dataset_size

**Logged Metrics** (per epoch):
- MAE: mae_epoch_loss, mae_learning_rate
- DINOv3: dino_epoch_loss, dino_learning_rate, dino_teacher_temp

**Logged Artifacts**:
- sequential_ssl_results.yaml (final checkpoint paths and losses)

**Design**: Epoch-level only (<0.1% overhead), no per-batch logging to avoid slowdown on 25,566 image dataset

### Checkpoint Structure
```
checkpoints/sequential_ssl_20251101_194949/
├── sequential_ssl_results.yaml
├── train_sequential_ssl.log
├── mae/
│   ├── mae_encoder_weights.pth          # Final encoder (for DINOv3 init)
│   ├── mae_best_checkpoint.pth          # Best model
│   └── mae_checkpoint_epoch_*.pth       # Per-epoch checkpoints
└── dinov3/
    ├── dino_encoder_weights.pth         # Final encoder (for fine-tuning)
    ├── dino_best_checkpoint.pth
    └── dino_checkpoint_epoch_*.pth
```

### Known Issues & Solutions
**DINOv3 Learning Rate = 0 at start**: Expected during warmup
- With warmup_epochs=1 and 1452 batches/epoch, LR increases over 1452 steps
- At batch 30, only 2% through warmup → LR ≈ 0
- Solution: Increase total epochs to 5+ or disable warmup (warmup_epochs=0)

**Trainer log_interval mismatch**: 
- Trainer reads from `config.training.log_interval` (default 10)
- If in `config.logging.log_interval`, trainer won't find it
- Solution: Place log_interval in training section of YAML

**MAE→DINOv3 key mismatch warning**: Non-critical
- Naming difference: `layers_0` vs `layers.0`
- Weights load correctly despite warning
- Happens during strict=False loading

**Reconstruction visualization errors**: Non-critical
- Minor bug in visualization code
- Does not affect training

### Documentation
- Sequential SSL details: docs/SEQUENTIAL_SSL.md
- Resume/continue guide: docs/RESUME_TRAINING.md
