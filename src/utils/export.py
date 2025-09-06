"""
Model export utilities for ONNX and TorchScript.
"""

import torch
import torch.onnx
import onnx
import onnxsim
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_onnx(model: torch.nn.Module,
            example_input: torch.Tensor,
            output_path: str,
            input_names: List[str] = ["input"],
            output_names: List[str] = ["output"],
            dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
            opset_version: int = 14,
            fp16: bool = False,
            optimize: bool = True) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        example_input: Example input tensor for tracing
        output_path: Path to save ONNX model
        input_names: Names for input nodes
        output_names: Names for output nodes
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version
        fp16: Whether to use FP16 precision
        optimize: Whether to optimize the model
    """
    model.eval()
    
    # Set up dynamic axes for variable input sizes
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    
    # Convert model to half precision if requested
    if fp16:
        model = model.half()
        example_input = example_input.half()
    
    # Export to ONNX
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")
        
        # Optimize the model
        if optimize:
            logger.info("Optimizing ONNX model...")
            try:
                optimized_model, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(optimized_model, output_path)
                    logger.info("ONNX model optimization completed")
                else:
                    logger.warning("ONNX model optimization failed, using unoptimized version")
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")
        
        # Print model info
        logger.info(f"ONNX model saved to: {output_path}")
        logger.info(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
        logger.info(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        raise


def to_torchscript(model: torch.nn.Module,
                  example_input: torch.Tensor,
                  output_path: str,
                  method: str = 'trace',
                  fp16: bool = False,
                  optimize: bool = True) -> None:
    """
    Export model to TorchScript.
    
    Args:
        model: PyTorch model to export
        example_input: Example input tensor for tracing
        output_path: Path to save TorchScript model
        method: Export method ('trace' or 'script')
        fp16: Whether to use FP16 precision
        optimize: Whether to optimize the model
    """
    model.eval()
    
    # Convert to half precision if requested
    if fp16:
        model = model.half()
        example_input = example_input.half()
    
    logger.info(f"Exporting model to TorchScript using {method} method: {output_path}")
    
    try:
        with torch.no_grad():
            if method == 'trace':
                traced_model = torch.jit.trace(model, example_input)
            elif method == 'script':
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown export method: {method}")
        
        # Optimize if requested
        if optimize:
            logger.info("Optimizing TorchScript model...")
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save model
        traced_model.save(output_path)
        logger.info(f"TorchScript model saved to: {output_path}")
        
        # Verify by loading and running
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded_model(example_input)
            logger.info(f"TorchScript model verification passed, output shape: {output.shape}")
            
    except Exception as e:
        logger.error(f"Failed to export TorchScript model: {e}")
        raise


def create_calibration_dataset(data_loader: torch.utils.data.DataLoader,
                              num_batches: int = 100) -> List[torch.Tensor]:
    """
    Create calibration dataset for INT8 quantization.
    
    Args:
        data_loader: DataLoader for calibration data
        num_batches: Number of batches to use for calibration
        
    Returns:
        List of calibration tensors
    """
    calibration_data = []
    
    logger.info(f"Creating calibration dataset with {num_batches} batches")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            calibration_data.append(images.cpu())
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{num_batches} calibration batches")
    
    logger.info(f"Calibration dataset created with {len(calibration_data)} samples")
    return calibration_data


def quantize_model(model: torch.nn.Module,
                  calibration_data: List[torch.Tensor],
                  output_path: str,
                  quantization_backend: str = 'fbgemm') -> None:
    """
    Quantize model to INT8.
    
    Args:
        model: PyTorch model to quantize
        calibration_data: Calibration dataset
        output_path: Path to save quantized model
        quantization_backend: Quantization backend ('fbgemm', 'qnnpack')
    """
    try:
        import torch.quantization as quantization
        
        model.eval()
        
        logger.info(f"Quantizing model with backend: {quantization_backend}")
        
        # Set quantization backend
        torch.backends.quantized.engine = quantization_backend
        
        # Prepare model for quantization
        model.qconfig = quantization.get_default_qconfig(quantization_backend)
        quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        logger.info("Calibrating model...")
        with torch.no_grad():
            for data in calibration_data:
                if torch.cuda.is_available():
                    data = data.cuda()
                model(data)
        
        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        logger.info(f"Quantized model saved to: {output_path}")
        
    except ImportError:
        logger.error("Quantization requires PyTorch with quantization support")
        raise
    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        raise


def benchmark_model(model_path: str,
                   input_shape: Tuple[int, ...],
                   num_runs: int = 100,
                   warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark exported model performance.
    
    Args:
        model_path: Path to exported model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Benchmark results dictionary
    """
    import time
    
    # Determine model type and load
    if model_path.endswith('.onnx'):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            def run_inference(data):
                return session.run(None, {input_name: data.numpy()})
                
        except ImportError:
            logger.error("ONNX Runtime not available")
            return {}
            
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        model = torch.jit.load(model_path)
        model.eval()
        
        def run_inference(data):
            with torch.no_grad():
                return model(data)
    else:
        logger.error(f"Unsupported model format: {model_path}")
        return {}
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    logger.info(f"Benchmarking model: {model_path}")
    logger.info(f"Input shape: {input_shape}")
    
    # Warmup
    for _ in range(warmup_runs):
        _ = run_inference(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = run_inference(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps = 1.0 / avg_time
    
    results = {
        'avg_inference_time_ms': avg_time * 1000,
        'min_inference_time_ms': min_time * 1000,
        'max_inference_time_ms': max_time * 1000,
        'fps': fps,
        'throughput_imgs_per_sec': fps
    }
    
    logger.info("Benchmark results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.3f}")
    
    return results


def export_model(checkpoint_path: str,
                output_dir: str,
                format: str = 'onnx',
                input_size: Tuple[int, int] = (512, 512),
                batch_size: int = 1,
                fp16: bool = False,
                optimize: bool = True,
                benchmark: bool = True) -> None:
    """
    Export trained model to specified format.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save exported models
        format: Export format ('onnx', 'torchscript', 'both')
        input_size: Input image size (H, W)
        batch_size: Batch size for export
        fp16: Whether to use FP16 precision
        optimize: Whether to optimize exported model
        benchmark: Whether to benchmark exported model
    """
    from ..models.swin_unet import SwinUNet
    
    # Load model
    logger.info(f"Loading model from: {checkpoint_path}")
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = SwinUNet(
        backbone_name='swin_large_patch4_window12_384',
        pretrained=False,
        decoder_channels=[1024, 512, 256, 128, 64],
        num_classes=2
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create example input
    example_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    if torch.cuda.is_available():
        model = model.cuda()
        example_input = example_input.cuda()
    
    # Export based on format
    exported_files = []
    
    if format in ['onnx', 'both']:
        onnx_path = output_path / 'model.onnx'
        to_onnx(
            model=model,
            example_input=example_input,
            output_path=str(onnx_path),
            fp16=fp16,
            optimize=optimize
        )
        exported_files.append(str(onnx_path))
    
    if format in ['torchscript', 'both']:
        torchscript_path = output_path / 'model.pt'
        to_torchscript(
            model=model,
            example_input=example_input,
            output_path=str(torchscript_path),
            fp16=fp16,
            optimize=optimize
        )
        exported_files.append(str(torchscript_path))
    
    # Benchmark exported models
    if benchmark:
        for file_path in exported_files:
            logger.info(f"\nBenchmarking {file_path}")
            results = benchmark_model(
                model_path=file_path,
                input_shape=(batch_size, 3, input_size[0], input_size[1])
            )
            
            # Save benchmark results
            import json
            benchmark_path = Path(file_path).with_suffix('.benchmark.json')
            with open(benchmark_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    logger.info(f"Model export completed. Files saved to: {output_dir}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Export trained model to ONNX/TorchScript')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'],
                       default='onnx', help='Export format')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                       help='Input image size (height width)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Optimize exported model')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Benchmark exported model')
    
    args = parser.parse_args()
    
    export_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        format=args.format,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        fp16=args.fp16,
        optimize=args.optimize,
        benchmark=args.benchmark
    )


if __name__ == '__main__':
    main()
