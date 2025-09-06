#!/usr/bin/env python3
"""
Cross-platform device detection and optimization utilities.
"""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def detect_platform() -> Dict[str, str]:
    """Detect the current platform and architecture."""
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    
    # Normalize architecture names
    if machine in ['x86_64', 'AMD64']:
        arch = 'x64'
    elif machine in ['arm64', 'aarch64']:
        arch = 'arm64'
    else:
        arch = machine
    
    return {
        'system': system,
        'architecture': arch,
        'python_version': python_version,
        'platform_string': f"{system.lower()}_{arch}"
    }


def detect_gpu() -> Dict[str, any]:
    """Detect available GPU hardware and capabilities."""
    gpu_info = {
        'has_gpu': False,
        'gpu_type': 'none',
        'gpu_count': 0,
        'gpu_names': [],
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'memory_gb': 0,
        'compute_capability': None
    }
    
    # NVIDIA CUDA detection
    if gpu_info['cuda_available']:
        gpu_info['has_gpu'] = True
        gpu_info['gpu_type'] = 'nvidia'
        gpu_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(gpu_info['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            gpu_info['gpu_names'].append(props.name)
            gpu_info['memory_gb'] += props.total_memory / (1024**3)
            if gpu_info['compute_capability'] is None:
                gpu_info['compute_capability'] = f"{props.major}.{props.minor}"
    
    # Apple MPS detection  
    elif gpu_info['mps_available']:
        gpu_info['has_gpu'] = True
        gpu_info['gpu_type'] = 'apple'
        gpu_info['gpu_count'] = 1
        gpu_info['gpu_names'] = ['Apple Silicon GPU']
        # MPS memory is shared with system RAM
        gpu_info['memory_gb'] = 8  # Conservative estimate
    
    # Try to detect NVIDIA via nvidia-smi even if PyTorch CUDA isn't available
    elif not gpu_info['cuda_available']:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                names = result.stdout.strip().split('\n')
                if names and names[0]:
                    gpu_info['gpu_names'] = names
                    gpu_info['gpu_count'] = len(names)
                    gpu_info['gpu_type'] = 'nvidia_no_cuda'  # Detected but not accessible
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return gpu_info


def get_optimal_device() -> torch.device:
    """Get the optimal PyTorch device for the current system."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_recommended_settings() -> Dict[str, any]:
    """Get recommended training settings based on hardware."""
    platform_info = detect_platform()
    gpu_info = detect_gpu()
    
    settings = {
        'device': get_optimal_device(),
        'mixed_precision': False,
        'batch_size': 4,
        'num_workers': min(4, max(1, torch.get_num_threads() // 2)),
        'pin_memory': gpu_info['has_gpu'],
        'compile_model': False,
        'distributed': gpu_info['gpu_count'] > 1,
    }
    
    # GPU-specific optimizations
    if gpu_info['gpu_type'] == 'nvidia':
        settings['mixed_precision'] = True
        settings['compile_model'] = True  # PyTorch 2.0+ compile
        
        # Memory-based batch size recommendations
        total_memory = gpu_info['memory_gb']
        if total_memory >= 40:  # A100 80GB, H100
            settings['batch_size'] = 16
        elif total_memory >= 20:  # A100 40GB, RTX 6000
            settings['batch_size'] = 12
        elif total_memory >= 10:  # RTX 4090, RTX 3090
            settings['batch_size'] = 8
        elif total_memory >= 6:   # RTX 3060, RTX 4060
            settings['batch_size'] = 6
        else:  # Lower-end GPUs
            settings['batch_size'] = 4
            
    elif gpu_info['gpu_type'] == 'apple':
        settings['mixed_precision'] = False  # MPS doesn't support FP16 yet
        settings['batch_size'] = 8  # Apple Silicon has unified memory
        settings['compile_model'] = False  # Not fully supported on MPS
        
    # CPU fallback
    else:
        settings['batch_size'] = 2  # Conservative for CPU
        settings['num_workers'] = max(1, torch.get_num_threads() // 4)
        settings['pin_memory'] = False
    
    # Platform-specific adjustments
    if platform_info['system'] == 'Windows':
        settings['num_workers'] = min(settings['num_workers'], 0)  # Windows multiprocessing issues
    
    return settings


def print_system_info():
    """Print comprehensive system information."""
    platform_info = detect_platform()
    gpu_info = detect_gpu()
    settings = get_recommended_settings()
    
    print("🖥️  System Information")
    print("=" * 50)
    print(f"Platform: {platform_info['system']} ({platform_info['architecture']})")
    print(f"Python: {platform_info['python_version']}")
    print(f"PyTorch: {torch.__version__}")
    
    print(f"\n🎮 GPU Information")
    print("=" * 50)
    print(f"GPU Available: {gpu_info['has_gpu']}")
    print(f"GPU Type: {gpu_info['gpu_type']}")
    print(f"GPU Count: {gpu_info['gpu_count']}")
    
    if gpu_info['gpu_names']:
        print("GPU Names:")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"  [{i}] {name}")
    
    if gpu_info['gpu_type'] == 'nvidia':
        print(f"Total GPU Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        if gpu_info['compute_capability']:
            print(f"Compute Capability: {gpu_info['compute_capability']}")
    
    print(f"\n⚙️  Recommended Settings")
    print("=" * 50)
    print(f"Device: {settings['device']}")
    print(f"Mixed Precision: {settings['mixed_precision']}")
    print(f"Batch Size: {settings['batch_size']}")
    print(f"Num Workers: {settings['num_workers']}")
    print(f"Pin Memory: {settings['pin_memory']}")
    print(f"Model Compile: {settings['compile_model']}")
    print(f"Distributed: {settings['distributed']}")


def validate_installation():
    """Validate that all required packages are properly installed."""
    required_packages = [
        'torch',
        'torchvision', 
        'timm',
        'opencv-python',
        'albumentations',
        'numpy',
        'matplotlib',
        'PyYAML',
        'tqdm',
        'psutil'
    ]
    
    missing_packages = []
    working_packages = []
    
    print("📦 Package Validation")
    print("=" * 50)
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'opencv-python':
                import cv2
                working_packages.append(f"{package} ({cv2.__version__})")
            elif package == 'PyYAML':
                import yaml
                working_packages.append(f"{package} ({yaml.__version__})")
            else:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    version = module.__version__
                else:
                    version = "unknown"
                working_packages.append(f"{package} ({version})")
                
        except ImportError:
            missing_packages.append(package)
    
    for package in working_packages:
        print(f"✅ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages:")
        for package in missing_packages:
            print(f"   - {package}")
        return False
    
    print(f"\n🎉 All {len(working_packages)} required packages are installed!")
    return True


def main():
    """Main function to run system diagnostics."""
    print("Swin-UNet Segmentation - System Diagnostics")
    print("=" * 60)
    
    print_system_info()
    
    print(f"\n📋 Installation Check")
    print("=" * 50)
    if validate_installation():
        print("\n🚀 System ready for training!")
    else:
        print("\n⚠️  Please install missing packages with:")
        print("   uv sync")
        sys.exit(1)


if __name__ == "__main__":
    main()
