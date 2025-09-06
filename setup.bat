@echo off
REM Windows setup script for segment-anomaly

echo 🚀 Setting up segment-anomaly project...

REM Check if uv is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ❌ Failed to install uv
        exit /b 1
    )
)

echo 🖥️  Platform: Windows

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo 🎮 NVIDIA GPU detected
    echo 🔥 Installing CUDA-optimized PyTorch...
    
    REM Install CUDA version of PyTorch
    uv sync
    uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121
) else (
    echo 💻 Installing CPU-only PyTorch...
    uv sync
    uv add torch torchvision torchaudio --index https://download.pytorch.org/whl/cpu
)

REM Install development dependencies
echo 🛠️  Installing development dependencies...
uv sync --group dev

echo ✅ Setup complete!
echo.
echo 🎯 Next steps:
echo 1. Prepare your data in the following structure:
echo    data/
echo    ├── train/
echo    │   ├── images/
echo    │   └── masks/
echo    ├── val/
echo    │   ├── images/
echo    │   └── masks/
echo    └── test/
echo        ├── images/
echo        └── masks/
echo.
echo 2. Run training:
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo    # Single GPU
    echo    uv run python train.py --config configs/default.yaml
    echo.
    echo    # Multi-GPU ^(adjust number^)
    echo    uv run torchrun --nproc_per_node=2 train.py --config configs/default.yaml --distributed
) else (
    echo    uv run python train.py --config configs/default.yaml
)
echo.
echo 3. Run tests:
echo    uv run python tests/run_tests.py
echo.
echo 4. Test multi-class support:
echo    uv run python test_multiclass.py

pause
