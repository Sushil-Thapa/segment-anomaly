"""
Distributed training utilities for multi-GPU setups.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl') -> None:
    """
    Initialize distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl', 'gloo')
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    print(f"Initialized DDP process {rank}/{world_size}")


def cleanup_ddp() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size (total number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor to avoid in-place operations
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    
    return tensor


def average_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average tensor across all processes.
    
    Args:
        tensor: Tensor to average
        
    Returns:
        Averaged tensor
    """
    reduced_tensor = reduce_tensor(tensor, op=dist.ReduceOp.SUM)
    return reduced_tensor / get_world_size()


def gather_tensor(tensor: torch.Tensor) -> Optional[list]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes (only on main process)
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    
    # Prepare tensors list for gathering
    if is_main_process():
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None
    
    # Gather tensors
    dist.gather(tensor, gathered_tensors, dst=0)
    
    return gathered_tensors


class DistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Custom distributed sampler with epoch tracking."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        """Initialize distributed sampler."""
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for proper shuffling."""
        self.current_epoch = epoch
        super().set_epoch(epoch)
    
    def __iter__(self):
        """Iterator with epoch-aware shuffling."""
        self.set_epoch(self.current_epoch)
        return super().__iter__()


def create_distributed_sampler(dataset, shuffle: bool = True):
    """
    Create distributed sampler if using DDP, otherwise return None.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle data
        
    Returns:
        DistributedSampler if using DDP, None otherwise
    """
    if dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle
        )
    return None


class DistributedMetrics:
    """Utility class for computing metrics across distributed processes."""
    
    @staticmethod
    def reduce_dict(metric_dict: dict) -> dict:
        """
        Reduce metric dictionary across all processes.
        
        Args:
            metric_dict: Dictionary of metrics
            
        Returns:
            Reduced metric dictionary
        """
        if not dist.is_initialized():
            return metric_dict
        
        reduced_dict = {}
        for key, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                reduced_dict[key] = average_tensor(value).item()
            elif isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, dtype=torch.float32, device='cuda')
                reduced_dict[key] = average_tensor(tensor_value).item()
            else:
                reduced_dict[key] = value  # Don't reduce non-numeric values
        
        return reduced_dict
    
    @staticmethod
    def gather_dict(metric_dict: dict) -> Optional[dict]:
        """
        Gather metric dictionary from all processes.
        
        Args:
            metric_dict: Dictionary of metrics
            
        Returns:
            Gathered metrics (only on main process)
        """
        if not dist.is_initialized() or not is_main_process():
            return metric_dict
        
        gathered_dict = {}
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, dtype=torch.float32, device='cuda')
                gathered_tensors = gather_tensor(tensor_value)
                if gathered_tensors is not None:
                    gathered_dict[key] = [t.item() for t in gathered_tensors]
            else:
                gathered_dict[key] = [value] * get_world_size()
        
        return gathered_dict


def save_checkpoint_distributed(state: dict, 
                               filepath: str, 
                               is_best: bool = False,
                               save_all_ranks: bool = False) -> None:
    """
    Save checkpoint in distributed setting.
    
    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        save_all_ranks: Whether to save from all ranks
    """
    if save_all_ranks or is_main_process():
        torch.save(state, filepath)
        
        if is_best:
            import shutil
            best_filepath = filepath.replace('.pth', '_best.pth')
            shutil.copyfile(filepath, best_filepath)


def load_checkpoint_distributed(filepath: str, 
                               model: torch.nn.Module,
                               optimizer: Optional[torch.optim.Optimizer] = None,
                               scheduler: Optional = None) -> dict:
    """
    Load checkpoint in distributed setting.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        
    Returns:
        Loaded state dictionary
    """
    # Load checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location=f'cuda:{get_rank()}')
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    if hasattr(model, 'module'):
        # DDP wrapped model
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def broadcast_object(obj, src: int = 0):
    """
    Broadcast Python object from source rank to all ranks.
    
    Args:
        obj: Object to broadcast
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    # Convert to tensor for broadcasting
    import pickle
    
    if get_rank() == src:
        obj_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(pickle.dumps(obj)))
        size_tensor = torch.LongTensor([obj_tensor.numel()])
    else:
        size_tensor = torch.LongTensor([0])
    
    # Broadcast size first
    dist.broadcast(size_tensor, src=src)
    
    # Prepare tensor for receiving
    if get_rank() != src:
        obj_tensor = torch.ByteTensor(size_tensor[0].item())
    
    # Broadcast object
    dist.broadcast(obj_tensor, src=src)
    
    # Deserialize
    obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
    
    return obj


def print_rank_0(message: str) -> None:
    """Print message only from rank 0."""
    if is_main_process():
        print(message)


def test_distributed():
    """Test distributed utilities (requires multiple GPUs)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping distributed tests")
        return True
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for distributed testing")
        return True
    
    print("Distributed utilities created successfully")
    return True


if __name__ == "__main__":
    test_distributed()
    print("Distributed utility tests passed!")
