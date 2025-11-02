Style:
While running the scripts, please make sure to ### MLflow tracking**: Integrated experiment tracking for all SSL phases
  - Epoch-level metrics only: final_loss, total_epochs per phase (efficient, <0.1% overhead)
  - Logs all hyperparameters: lr, batch_size, mask_ratio, teacher_temp, momentum, etc.
  - Pipeline params: device, configs, resume/continue modes
  - Nested runs: separate MAE and DINOv3 phases
  - Artifacts: sequential_ssl_results.yaml
  - View with: mlflow ui --backend-store-uri logs/mlruns
  - No per-batch logging to avoid slowdown on large datasets (25,566 images) run python (recommended).

While making a new feature or implementing something, please refrain from creating a new file each time if you dont think it is very ovious or critical. if additional classes in the same file makes more sense please do that instead. you could make new file for testing purposes if it is easier for you but once the feature is stable, please merge it with the existing files or remove the one time testing files. If I were you I'd just implement all core or script that will stay within the project files and styles and keeping the testing files separate so it is easier to remove them later and transitioning is not as difficult once you have tested it.

Please do not include emojis in the code comments or documentations. Please do not make it seem like it is overly verbose or written just for the sake of writing it. Dont make it too casual and mention you are doing something because i asked it. Make it seem like this is what I'd write if I were to write it myself.

Remember:
I do not want to add files with git add . or git add -A. I want to add them specifically. So please do not suggest commands that add everything. I for eg. do not want to add .md files that I keep creating to document things.

Update the sections below with the findings so that it might be useful for you n the future.
# Learnings and documentations  (To be filled by Github Copilot below)

## Self-Supervised Learning (SSL) Pipeline

### MAE (Masked Autoencoder) Pretraining
- MAE pipeline fully implemented and consolidated into existing architecture files following project style guidelines
- MAE components integrated into: decoder.py (MAEDecoder), swin_unet.py (MAESwinUNet), dataset.py (MAEPretrainingDataset), trainer.py (MAETrainer)
- Separate MAE files (mae_decoder.py, mae_swin_unet.py, mae_dataset.py, mae_trainer.py) removed after consolidation to maintain clean codebase
- Apple Silicon MPS acceleration validated and optimized for MAE training with memory management and error handling
- MAE system supports 75% masking ratio, pixel reconstruction loss, seamless transition between pretraining and segmentation modes
- Core training script src/train_mae.py for standalone MAE pretraining
- Configuration files (mae_pretraining.yaml, mae_pretraining_mps.yaml) for production and MPS-optimized training
- All MAE functionality consolidated per user style guidelines preferring feature integration over separate file creation

### DINOv3 Self-Distillation Pretraining
- DINOv3 pipeline implemented following same consolidation pattern as MAE
- DINOv3 components integrated into: swin_unet.py (DINOv3SwinUNet, DINOHead), trainer.py (DINOv3Trainer), dataset.py (DINOv3PretrainingDataset), transforms.py (get_dino_multicrop_transform)
- Student-teacher framework with EMA momentum updates (momentum=0.996) for stable self-distillation
- Multi-crop augmentation strategy: 2 global crops (384x384, scale 0.4-1.0) + 4-6 local crops (192x192, scale 0.05-0.4)
- DINOHead projection: 3-layer MLP with bottleneck (encoder_dim → 2048 → 256 → output_dim)
- Teacher temperature scheduling: linear warmup from 0.04 to 0.07 over 30 epochs for output sharpening
- Center momentum (0.9) prevents mode collapse by maintaining running center of teacher outputs
- Cross-entropy loss between student predictions and sharpened teacher outputs
- Core training script src/train_dinov3.py for standalone DINOv3 pretraining
- Configuration files (dinov3_pretraining.yaml, dinov3_pretraining_mps.yaml) for production and Apple Silicon
- Can initialize from MAE checkpoint via load_mae_pretrained_encoder() method for sequential SSL

### Sequential SSL Pipeline (MAE → DINOv3)
- Full sequential pretraining pipeline implemented in src/train_sequential_ssl.py
- Phase 1: MAE learns low-level visual features through masked reconstruction
- Phase 2: DINOv3 refines semantic understanding through self-distillation, initialized from MAE weights
- Sequential approach combines complementary strengths: MAE for pixel-level features, DINOv3 for semantic relationships
- Pipeline orchestrates both phases automatically, handles checkpoint passing between stages
- Supports skipping MAE phase if checkpoint already exists (--skip_mae flag)
- **Resume capability**: Use --resume flag to continue interrupted training from existing checkpoint directory
  - Automatically detects completed phases (MAE/DINOv3) and skips them
  - Resumes incomplete phases from latest checkpoint with full state restoration (model, optimizer, scheduler)
  - Supports changing dataset mid-training for additional pretraining
  - Documentation in docs/RESUME_TRAINING.md
- **Continue training**: Use --continue-training flag to train on new dataset with learned weights
  - Loads final encoder weights from completed checkpoint
  - Creates new checkpoint directory (preserves old checkpoints)
  - Starts from epoch 0 with fresh optimizer/scheduler
  - Perfect for multi-dataset pretraining workflows
- **MLflow tracking**: Integrated experiment tracking for all SSL phases
  - Tracks key metrics: loss, learning rate, epoch time
  - Logs hyperparameters and model architecture
  - Stores reconstruction samples (MAE) and training curves
  - View with: mlflow ui --backend-store-uri logs/mlruns
  - Efficient logging: only essential metrics to avoid slowdown
- Outputs organized in timestamped directories: checkpoints/sequential_ssl_<timestamp>/mae/ and /dinov3/
- Final encoder weights from DINOv3 phase used for downstream segmentation fine-tuning
- Comprehensive documentation in docs/SEQUENTIAL_SSL.md covering architecture, usage, troubleshooting

### Apple Silicon (MPS) Optimizations
- MPS-specific configs reduce batch sizes, model dimensions, and crop counts for memory constraints
- Automatic cache management before tensor allocation
- Gradient accumulation compensates for smaller batch sizes (e.g., batch_size=8, gradient_accumulation=4)
- FP32 precision used (MPS doesn't support mixed precision)
- Reduced DINOv3 output dimensions: 8192 (MPS) vs 65536 (CUDA) to fit memory

### Training Workflow
- Standalone MAE: uv run python src/train_mae.py --config configs/mae_pretraining.yaml
- Standalone DINOv3: uv run python src/train_dinov3.py --config configs/dinov3_pretraining.yaml [--mae_checkpoint PATH]
- Sequential SSL: uv run python src/train_sequential_ssl.py --mae_config configs/mae_pretraining.yaml --dino_config configs/dinov3_pretraining.yaml
- Segmentation fine-tuning: uv run python train.py --config configs/config_vehicle_csam_proxy.yaml --pretrained_encoder checkpoints/dinov3/dino_encoder_weights.pth

### Code Organization Principles
- All SSL functionality consolidated into existing files (swin_unet.py, trainer.py, dataset.py, transforms.py)
- No separate module files (e.g., no mae.py or dino.py modules)
- Training scripts (train_mae.py, train_dinov3.py, train_sequential_ssl.py) serve as entry points
- Configuration files define pretraining parameters
- Follow existing codebase style: integrate features within appropriate existing files rather than creating new modules
