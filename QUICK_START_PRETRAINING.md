# Quick Start: Sequential SSL Pretraining with Your Data

## Your Data Structure
```
your_folder/
├── folderA/
│   ├── folderB/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── folderC/
│       └── ...
└── ...
```

## Step 1: Update Config Files

Edit both config files to point to your data:

### `configs/mae_pretraining_cuda.yaml`
```yaml
data:
  root: "path/to/your_folder"  # e.g., "D:/datasets/my_images" or "data/raw_images"
```

### `configs/dinov3_pretraining_cuda.yaml`
```yaml
data:
  data_root: "path/to/your_folder"  # Same as MAE config
```

**Note:** Use the same path for both configs. The script will automatically find all `.jpg` and `.png` files recursively in all subdirectories.

## Step 2: Run Sequential Pretraining

### Full Training (100 epochs each)
```bash
uv run python src/train_sequential_ssl.py \
    --mae_config configs/mae_pretraining_cuda.yaml \
    --dino_config configs/dinov3_pretraining_cuda.yaml
```

### Quick Test (2 epochs to verify setup)
Edit the configs temporarily:
- In `mae_pretraining_cuda.yaml`: Set `epochs: 2`
- In `dinov3_pretraining_cuda.yaml`: Set `epochs: 2`

Then run:
```bash
uv run python src/train_sequential_ssl.py \
    --mae_config configs/mae_pretraining_cuda.yaml \
    --dino_config configs/dinov3_pretraining_cuda.yaml
```

### With Smaller Batch Size (if GPU memory issues)
If you get OOM errors, reduce batch sizes in the configs:
- MAE: `batch_size: 32` → `batch_size: 16`
- DINOv3: `batch_size: 32` → `batch_size: 16`

## Step 3: Monitor Training

### Watch GPU Usage
```bash
nvidia-smi -l 1
```

### View MLflow Dashboard
```bash
mlflow ui --backend-store-uri logs/mlruns
```
Then open: http://localhost:5000

## Step 4: Use Pretrained Encoder

After training completes, use the pretrained encoder for fine-tuning:

```bash
uv run python src/train.py \
    --config configs/config_vehicle_csam_proxy.yaml \
    --pretrained_encoder checkpoints/sequential_ssl_TIMESTAMP/dinov3/dino_encoder_weights.pth
```

## Expected Behavior

### MAE Phase
- **GPU Memory:** ~8-12 GB (depends on batch size)
- **Speed:** ~2-5 seconds/batch on RTX 4090
- **Output:** Reconstruction loss should decrease

### DINOv3 Phase
- **GPU Memory:** ~12-16 GB (multi-crop increases memory)
- **Speed:** ~3-7 seconds/batch on RTX 4090
- **Output:** DINOv3 loss should stabilize after warmup

## Checkpoint Structure
```
checkpoints/sequential_ssl_20251102_HHMMSS/
├── sequential_ssl_results.yaml
├── train_sequential_ssl.log
├── mae/
│   ├── mae_encoder_weights.pth          ← Use for DINOv3 init
│   ├── mae_best_checkpoint.pth
│   └── mae_checkpoint_epoch_*.pth
└── dinov3/
    ├── dino_encoder_weights.pth         ← Use for fine-tuning
    ├── dino_best_checkpoint.pth
    └── dino_checkpoint_epoch_*.pth
```

## Troubleshooting

### "Directory not found" error
- Check the path in your config matches your actual folder
- Use absolute paths if relative paths don't work: `D:/datasets/images`
- On Windows, use forward slashes `/` or escaped backslashes `\\`

### "CUDA out of memory" error
- Reduce `batch_size` in configs (try 16, 8, or 4)
- Reduce model size: Use `swin_tiny` instead of `swin_base` in MAE config
- Reduce `n_local_crops` in DINOv3 config (try 4 instead of 6)

### "No images found" error
- Ensure your images have `.jpg`, `.png`, `.jpeg`, or `.JPG`, `.PNG` extensions
- Check that files exist in subdirectories
- The script searches recursively, so any depth is fine

### Training is too slow
- Increase `batch_size` if GPU memory allows
- Increase `num_workers` for faster data loading (try 12-16)
- Ensure `use_amp: true` for mixed precision training

## Performance Tips

**For RTX 4090 (24GB VRAM):**
- MAE: batch_size 64-128
- DINOv3: batch_size 32-48
- num_workers: 8-12

**For smaller GPUs (8-12GB):**
- MAE: batch_size 16-32
- DINOv3: batch_size 8-16
- Reduce model to `swin_tiny`
- Set `use_amp: true`
