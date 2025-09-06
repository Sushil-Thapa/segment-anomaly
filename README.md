# Swin-UNet Wafer Defect Segmentation

PyTorch implementation for wafer defect segmentation using Swin Transformer backbone with UNet decoder. Supports both binary and multi-class segmentation.

## 🏗️ Architecture

- **Backbone**: Swin-Large (swin_large_patch4_window12_384) from timm
- **Decoder**: UNet with skip connections and attention gates
- **Input**: 512×512 tiles with 50% overlap (256 stride)
- **Output**: Configurable segmentation masks (binary or multi-class)

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd segment-anomaly

# Setup with uv (recommended)
./setup.sh  # Mac/Linux
# OR setup.bat  # Windows

# Or use pip
pip install -r requirements.txt

# Verify installation
uv run python tests/run_tests.py
```

## 📁 Data Structure

Organize your wafer data as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── wafer_001.png
│   │   └── ...
│   └── masks/
│       ├── wafer_001.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Requirements:**

- Images: RGB wafer images (any size, will be tiled)
- Masks: Grayscale masks with class indices
  - Binary: 0=background, 1=defect
  - Multi-class: 0=background, 1=defect_type1, 2=defect_type2, etc.
- Matching filenames between images and masks

## 🚀 Quick Start

### Training

```bash
# Single GPU training
python train.py --config configs/default.yaml

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 train.py --config configs/default.yaml --distributed

# Resume from checkpoint
python train.py --config configs/default.yaml --resume checkpoints/best_model.pth
```

### Inference

```python
from src.inference import TiledInference
from src.models.swin_unet import create_model
import numpy as np
from PIL import Image

# Load model
model = create_model('swin_large_patch4_window12_384', num_classes=2)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create inference engine
inference = TiledInference(model, tile_size=512, stride=256)

# Process image
image = np.array(Image.open('wafer.png'))
prediction = inference.predict_image(image)

# Save result
Image.fromarray(prediction).save('prediction.png')
```

## ⚙️ Configuration

Key configuration options in `configs/default.yaml`:

```yaml
# Data settings
data:
  data_dir: "data"
  tile_size: 512
  stride: 256
  positive_ratio: 0.75  # For WeightedRandomSampler

# Model settings
model:
  backbone: "swin_large_patch4_window12_384"
  pretrained: true
  num_classes: 2

# Training settings
training:
  batch_size: 8
  accumulate_grad_batches: 4  # Effective batch size: 32
  num_epochs: 100
  mixed_precision: true

# Loss settings
loss:
  ce_weight: 0.7
  dice_weight: 0.3
  hard_negative_mining: true
  start_epoch: 30

# Optimizer
optimizer:
  name: "adamw"
  lr: 1e-4
  weight_decay: 0.01
```

## 🔬 Features

### Data Pipeline
- **Tiling**: 512×512 tiles with Gaussian-weighted stitching
- **Augmentation**: Geometric, intensity, and texture augmentations
- **Sampling**: WeightedRandomSampler for class balance (3:1 positive:negative)
- **Normalization**: ImageNet statistics

### Model Architecture
- **Backbone**: Pre-trained Swin Transformer (384M parameters)
- **Decoder**: UNet with [1024, 512, 256, 128, 64] channels
- **Features**: Skip connections, attention gates, deep supervision
- **CAM**: Class Activation Maps for interpretability

### Training Strategy
- **Loss**: Combined CE (70%) + Dice (30%) with class weighting
- **Optimizer**: AdamW with different LR for backbone/decoder
- **Scheduler**: CosineAnnealingLR with 5-epoch warmup
- **Precision**: Mixed precision (FP16) with gradient scaling
- **Mining**: Hard negative mining after epoch 30

### Advanced Features
- **Distributed**: Multi-GPU training with DistributedDataParallel
- **Callbacks**: Early stopping, checkpointing, visualization
- **Export**: ONNX and TorchScript with optimization
- **Monitoring**: Memory usage, training metrics, visualizations

## 📊 Performance

### Training Metrics
- **IoU**: Intersection over Union for segmentation quality
- **Dice**: Dice coefficient for overlap measurement
- **F1**: F1-score for classification performance
- **Pixel Accuracy**: Overall pixel classification accuracy

### Memory Optimization
- **Tiling**: Process large images without memory constraints
- **Gradient Accumulation**: Simulate large batches on limited memory
- **Mixed Precision**: 50% memory reduction with minimal quality loss
- **Caching**: Optional tile caching for faster training

### Inference Optimization
- **Batched Processing**: Process multiple tiles simultaneously
- **Post-processing**: Connected components, morphological operations
- **Export Formats**: ONNX (deployment) and TorchScript (production)

## 📁 Project Structure

```
segment-anomaly/
├── configs/
│   └── default.yaml           # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py         # Dataset and data loading
│   │   ├── tiling.py          # Tiling utilities
│   │   └── transforms.py      # Data augmentation
│   ├── models/
│   │   ├── decoder.py         # UNet decoder
│   │   └── swin_unet.py       # Complete model
│   ├── losses/
│   │   └── combined.py        # Loss functions
│   ├── training/
│   │   ├── callbacks.py       # Training callbacks
│   │   └── trainer.py         # Training loop
│   └── utils/
│       ├── metrics.py         # Evaluation metrics
│       ├── distributed.py     # Multi-GPU utilities
│       └── export.py          # Model export utilities
├── tests/
│   ├── test_*.py              # Unit tests
│   └── run_tests.py           # Test runner
├── train.py                   # Main training script
├── inference.py               # Inference engine
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python tests/run_tests.py

# Run specific tests
python tests/test_tiling.py      # Tiling operations
python tests/test_metrics.py     # Metric calculations
python tests/test_memory.py      # Memory profiling
python tests/test_integration.py # End-to-end pipeline
```

## 🔧 Advanced Usage

### Custom Backbone

```python
from src.models.swin_unet import create_model

# Use different Swin variant
model = create_model('swin_base_patch4_window7_224', num_classes=2)

# Use EfficientNet backbone
model = create_model('efficientnet_b5', num_classes=2)
```

### Custom Loss Function

```python
from src.losses.combined import CombinedLoss

# Custom loss weights
loss_fn = CombinedLoss(
    ce_weight=0.5,
    dice_weight=0.3,
    focal_weight=0.2,
    class_weights=[1.0, 5.0]  # Heavily weight defect class
)
```

### Export for Production

```python
from src.utils.export import to_onnx, to_torchscript

# Export to ONNX
to_onnx(model, 'model.onnx', input_shape=(1, 3, 512, 512))

# Export to TorchScript
to_torchscript(model, 'model.pt', input_shape=(1, 3, 512, 512))
```

## 📈 Monitoring

Training progress is logged to:
- **Console**: Real-time metrics and progress
- **Files**: Detailed logs in `logs/` directory
- **Checkpoints**: Model states in `checkpoints/` directory
- **Visualizations**: Sample predictions saved during training

## 🐛 Troubleshooting

### Memory Issues
```bash
# Reduce batch size
# Increase gradient accumulation
# Disable tile caching
# Use CPU inference for very large images
```

### Training Issues
```bash
# Check data paths in config
# Verify image/mask correspondence
# Monitor loss curves for convergence
# Adjust learning rate if needed
```

### Performance Issues
```bash
# Enable mixed precision
# Use multiple workers for data loading
# Consider distributed training
# Profile memory usage with tests
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check existing issues on GitHub
- Run `python tests/run_tests.py` to verify setup
- Review configuration in `configs/default.yaml`
- Check data format and structure

---

**Happy segmenting! 🔬✨**
