"""
Main training script for wafer defect segmentation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
from pathlib import Path
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import create_dataloaders
from src.models.swin_unet import create_model
from src.losses.combined import create_loss_function
from src.training.trainer import create_trainer
from src.utils.distributed import setup_ddp, cleanup_ddp, is_main_process
from src.utils.metrics import MetricCollection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process any environment variable substitutions
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str) and sub_value.startswith('${') and sub_value.endswith('}'):
                    env_var = sub_value[2:-1]
                    if env_var in os.environ:
                        config[key][sub_key] = type(sub_value)(os.environ[env_var])
    
    return config


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from configuration."""
    optimizer_config = config['optimizer']
    
    # Group parameters for weight decay
    param_groups = []
    
    # No weight decay for bias, batch norm, and layer norm parameters
    no_decay = ['bias', 'bn', 'norm']
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': optimizer_config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    """Create learning rate scheduler from configuration."""
    scheduler_config = config.get('scheduler', {})
    
    if not scheduler_config:
        return None
    
    # Cosine annealing with warmup
    T_max = scheduler_config.get('T_max', config['training']['epochs'])
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    
    if warmup_epochs > 0:
        # Create warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs)
        )
        
        # Create main scheduler
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            eta_min=1e-7
        )
        
        # Combine schedulers
        from torch.optim.lr_scheduler import SequentialLR
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-7
        )
    
    return scheduler


def create_logger(config: Dict[str, Any]):
    """Create experiment logger (WandB or TensorBoard)."""
    logging_config = config.get('logging', {})
    
    if logging_config.get('use_wandb', False):
        try:
            import wandb
            
            wandb.init(
                project=logging_config.get('project', 'wafer-segmentation'),
                entity=logging_config.get('entity', None),
                config=config,
                name=f"run_{config.get('run_id', 'default')}"
            )
            return wandb
            
        except ImportError:
            logger.warning("WandB not available, falling back to TensorBoard")
    
    if logging_config.get('use_tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = Path('./runs') / f"run_{config.get('run_id', 'default')}"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            return SummaryWriter(log_dir=str(log_dir))
            
        except ImportError:
            logger.warning("TensorBoard not available")
    
    return None


def train_worker(rank: int, world_size: int, config: Dict[str, Any], resume_from: str = None):
    """Training worker function for distributed training."""
    
    # Setup distributed training
    if world_size > 1:
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    if config['training'].get('deterministic', True):
        set_seed(config['training'].get('seed', 42) + rank)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, 
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Wrap model for distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Create loss function
    criterion = create_loss_function(config)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create logger (only on main process)
    experiment_logger = None
    if is_main_process():
        experiment_logger = create_logger(config)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger_obj=experiment_logger
    )
    
    # Log model info
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    try:
        history = trainer.fit(
            epochs=config['training']['epochs'],
            resume_from_checkpoint=resume_from
        )
        
        # Save final checkpoint
        if is_main_process():
            final_checkpoint_path = f"final_model_epoch_{config['training']['epochs']}.pth"
            trainer.save_checkpoint(final_checkpoint_path)
            trainer.save_history('training_history.json')
            logger.info(f"Training completed. Final model saved to {final_checkpoint_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if is_main_process():
            trainer.save_checkpoint("interrupted_checkpoint.pth")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if experiment_logger and hasattr(experiment_logger, 'finish'):
            experiment_logger.finish()
        
        if world_size > 1:
            cleanup_ddp()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Swin-UNet for wafer defect segmentation')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                       help='World size for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up distributed training
    if args.distributed or 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
        
        if 'RANK' in os.environ:
            # torchrun setup
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            train_worker(rank, world_size, config, args.resume)
        else:
            # Manual distributed setup
            mp.spawn(
                train_worker,
                args=(world_size, config, args.resume),
                nprocs=world_size,
                join=True
            )
    else:
        # Single GPU/CPU training
        train_worker(0, 1, config, args.resume)


def test_training_setup():
    """Test training setup with dummy configuration."""
    config = {
        'data': {
            'root': './dummy_data',
            'tile_size': 512,
            'stride': 256,
            'oversample_ratio': 3.0,
            'num_workers': 2,
            'pin_memory': True
        },
        'model': {
            'backbone': 'swin_large_patch4_window12_384',
            'decoder_channels': [1024, 512, 256, 128, 64],
            'dropout': 0.1
        },
        'training': {
            'batch_size': 4,
            'accumulate_grad': 2,
            'epochs': 5,
            'val_every_n_epochs': 1,
            'num_workers': 2,
            'deterministic': True,
            'seed': 42
        },
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 0.01
        },
        'scheduler': {
            'T_max': 5,
            'warmup_epochs': 2
        },
        'loss': {
            'ce_weight': 0.7,
            'dice_weight': 0.3,
            'label_smoothing': 0.1
        },
        'callbacks': {
            'early_stopping': {
                'patience': 3,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'model_checkpoint': {
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3,
                'filename': 'epoch-{epoch:02d}-val_loss-{val_loss:.4f}.pth'
            }
        },
        'logging': {
            'use_wandb': False,
            'use_tensorboard': True,
            'log_every_n_steps': 10
        }
    }
    
    # Test configuration loading and processing
    print("Testing configuration processing...")
    
    # Test optimizer creation
    model = torch.nn.Linear(10, 2)
    optimizer = create_optimizer(model, config)
    print(f"Optimizer created: {type(optimizer).__name__}")
    
    # Test scheduler creation
    scheduler = create_scheduler(optimizer, config)
    print(f"Scheduler created: {type(scheduler).__name__}")
    
    print("Training setup test completed successfully!")
    return True


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Run tests if no arguments provided
        test_training_setup()
    else:
        main()
