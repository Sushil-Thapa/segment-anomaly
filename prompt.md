# Swin-UNet Segmentation Training Repository

Build a production PyTorch repository for binary segmentation (defect detection) using Swin-Large backbone + UNet decoder.

## Core Requirements

### Architecture
- Backbone: `swin_large_patch4_window12_384` from timm (pretrained=True)
- UNet decoder with skip connections: channels [1024, 512, 256, 128, 64]
- Output: logits (B, 2, H, W) for binary segmentation

### Data Pipeline
- **Tiling**: 512x512 tiles, stride 256 (50% overlap), reflect padding
- **Dataset**: Lazy-load images/masks, precompute tile indices
- **Sampling**: WeightedRandomSampler with 3:1 positive:negative ratio
- **Augmentations** (train): 
  - Geometric: RandomRotate90, Flip, ShiftScaleRotate(±0.1, ±30°)
  - Intensity: RandomBrightnessContrast, CLAHE(clip_limit=2)
  - Heavy: ElasticTransform(alpha=120), CoarseDropout(max_holes=8)
- **Normalization**: ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]

### Training
- Loss: 0.7 * WeightedCE + 0.3 * Dice (class weights: 1/log(1.02 + freq))
- Optimizer: AdamW(lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR(T_max=epochs) with 5-epoch linear warmup
- Mixed precision: torch.cuda.amp.GradScaler + autocast
- Gradient accumulation: 2 steps
- Early stopping: patience=15 on val_iou
- Checkpointing: save top-3 by val_iou
- DDP: torch.nn.parallel.DistributedDataParallel

### Inference Optimizations (12GB GPU)
- Tile-wise inference with overlap blending (Gaussian weights)
- FP16 inference with autocast
- Connected components post-processing (min_area=100px)
- Morphological ops: closing(3x3) → opening(3x3)

### File Structure
src/
├── data/
│   ├── dataset.py      # WaferTileDataset(torch.utils.data.Dataset)
│   ├── tiling.py       # deterministic tile/stitch with Gaussian blending
│   └── transforms.py   # get_train_transform(), get_val_transform()
├── models/
│   ├── swin_unet.py    # SwinUNet(nn.Module)
│   └── decoder.py      # UNetDecoder with attention gates
├── losses/
│   └── combined.py     # CombinedLoss(nn.Module) with optional focal
├── training/
│   ├── trainer.py      # Trainer class with train/val loops
│   ├── callbacks.py    # HardNegativeMining, VisualizePredictions
│   └── checkpoint.py   # CheckpointManager, EarlyStopping
├── utils/
│   ├── metrics.py      # IoU, Dice, F1 (pure torch)
│   ├── distributed.py  # setup_ddp(), cleanup_ddp()
│   └── export.py       # to_onnx(), to_torchscript() with dynamic axes
├── inference.py        # TiledInference class with memory profiling
└── train.py           # main training script with argparse

### Config (configs/default.yaml)
```yaml
data:
  root: ./data
  tile_size: 512
  stride: 256
  oversample_ratio: 3
  
model:
  backbone: swin_large_patch4_window12_384
  decoder_channels: [1024, 512, 256, 128, 64]
  dropout: 0.1
  
training:
  batch_size: 8  # per GPU
  accumulate_grad: 2
  epochs: 120
  val_every_n_epochs: 2
  num_workers: 4
  
optimizer:
  lr: 1e-4
  weight_decay: 0.01
  
loss:
  ce_weight: 0.7
  dice_weight: 0.3
  label_smoothing: 0.1
Export & Deployment

ONNX: opset=14, dynamic batch/height/width, FP16 weights
TorchScript: traced with example input, quantization-ready
TensorRT: INT8 calibration dataset included

Testing

test_tiling.py: verify perfect reconstruction
test_metrics.py: validate against sklearn
test_memory.py: profile peak memory usage

CLI Commands
bash# Train single GPU
python train.py --config configs/default.yaml

# Train multi-GPU with DDP
torchrun --nproc_per_node=4 train.py --config configs/default.yaml --distributed

# Inference with memory limit
python inference.py --checkpoint best.pth --image wafer.png \
    --max_memory_gb 10 --fp16

# Export
python -m src.utils.export --checkpoint best.pth --format onnx \
    --optimize --fp16
Training Loop Implementation
python# Pseudocode structure for train.py
class Trainer:
    def __init__(self, model, config):
        self.scaler = torch.cuda.amp.GradScaler()
        self.setup_optimizers()
        self.setup_callbacks()
    
    def train_epoch(self):
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                loss = self.training_step(batch)
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulate_grad == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
    
    def validation_epoch(self):
        # Compute metrics, save best checkpoints
Special Requirements

Hard negative mining after epoch 30
Class activation maps visualization with matplotlib
Automatic mixed precision with GradScaler
Reproducible (deterministic mode with seed)
W&B integration for logging (optional via flag)
TensorBoard logging by default

Generate complete, working code with docstrings. Prioritize inference speed and memory efficiency. Include README with benchmarks table.