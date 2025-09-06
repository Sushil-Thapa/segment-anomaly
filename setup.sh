#!/bin/bash
# Cross-platform setup script for segment-anomaly

set -e

echo "🚀 Setting up segment-anomaly project..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="Windows"
else
    PLATFORM="Unknown"
fi

echo "🖥️  Detected platform: $PLATFORM"

# Check for GPU support
GPU_SUPPORT=""
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        echo "🎮 NVIDIA GPU detected: $GPU_INFO"
        GPU_SUPPORT="cuda"
    fi
elif [[ "$PLATFORM" == "macOS" ]]; then
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo "🍎 Apple Silicon detected - MPS support available"
        GPU_SUPPORT="mps"
    fi
fi

# Install dependencies with uv
echo "📦 Installing dependencies with uv..."

# Base installation
uv sync

# Platform-specific optimizations
if [[ "$GPU_SUPPORT" == "cuda" ]]; then
    echo "🔥 Installing CUDA-optimized PyTorch..."
    uv add --index https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    
    # Install apex for mixed precision on Linux/CUDA
    if [[ "$PLATFORM" == "Linux" ]]; then
        echo "⚡ Installing NVIDIA Apex for mixed precision..."
        uv add --optional gpu apex || echo "⚠️  Could not install apex - mixed precision will use native PyTorch AMP"
    fi
    
elif [[ "$GPU_SUPPORT" == "mps" ]]; then
    echo "🍎 Installing MPS-optimized PyTorch for Apple Silicon..."
    # macOS with Apple Silicon - use default PyTorch with MPS support
    uv sync
    
else
    echo "💻 Installing CPU-only PyTorch..."
    uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cpu
fi

# Install development dependencies
echo "🛠️  Installing development dependencies..."
uv sync --group dev

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Prepare your data in the following structure:"
echo "   data/"
echo "   ├── train/"
echo "   │   ├── images/"
echo "   │   └── masks/"
echo "   ├── val/"
echo "   │   ├── images/" 
echo "   │   └── masks/"
echo "   └── test/"
echo "       ├── images/"
echo "       └── masks/"
echo ""
echo "2. Run training:"
if [[ "$GPU_SUPPORT" == "cuda" ]]; then
    echo "   # Single GPU"
    echo "   uv run python train.py --config configs/default.yaml"
    echo ""
    echo "   # Multi-GPU (4 GPUs)"
    echo "   uv run torchrun --nproc_per_node=4 train.py --config configs/default.yaml --distributed"
else
    echo "   uv run python train.py --config configs/default.yaml"
fi
echo ""
echo "3. Run tests:"
echo "   uv run python tests/run_tests.py"
echo ""
echo "4. Test multi-class support:"
echo "   uv run python test_multiclass.py"
