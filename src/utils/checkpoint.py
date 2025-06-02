"""Checkpoint management utilities"""

import os
import torch
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil
from datetime import datetime


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   step: int,
                   loss: float,
                   metrics: Dict[str, Any],
                   checkpoint_dir: str,
                   filename: Optional[str] = None,
                   is_best: bool = False) -> str:
    """Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        loss: Current loss
        metrics: Training metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Custom filename (optional)
        is_best: Whether this is the best checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        shutil.copy2(checkpoint_path, best_path)
    
    # Save latest checkpoint link
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    shutil.copy2(checkpoint_path, latest_path)
    
    # Save checkpoint metadata
    metadata = {
        'checkpoint_path': checkpoint_path,
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', '')
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Check for latest checkpoint link
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # Find latest checkpoint by modification time
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and file.startswith('checkpoint_'):
            file_path = os.path.join(checkpoint_dir, file)
            checkpoint_files.append((file_path, os.path.getmtime(file_path)))
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])
    return latest_checkpoint[0]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint in directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to best checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    if os.path.exists(best_path):
        return best_path
    
    return None


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 5, keep_best: bool = True):
    """Clean up old checkpoints, keeping only the most recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and file.startswith('checkpoint_'):
            # Skip special checkpoints
            if file in ['latest_checkpoint.pt', 'best_checkpoint.pt']:
                continue
            
            file_path = os.path.join(checkpoint_dir, file)
            checkpoint_files.append((file_path, os.path.getmtime(file_path)))
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # Remove old checkpoints
    for i, (file_path, _) in enumerate(checkpoint_files):
        if i >= keep_last_n:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {file_path}")
            except OSError as e:
                print(f"Error removing checkpoint {file_path}: {e}")


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get information about a checkpoint without loading the full model
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load only the metadata
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'timestamp': checkpoint.get('timestamp', ''),
        'file_size': os.path.getsize(checkpoint_path),
        'file_path': checkpoint_path
    }
    
    return info


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """List all checkpoints in directory with their information
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        List of checkpoint information dictionaries
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and 'checkpoint' in file:
            file_path = os.path.join(checkpoint_dir, file)
            try:
                info = get_checkpoint_info(file_path)
                checkpoints.append(info)
            except Exception as e:
                print(f"Error reading checkpoint {file_path}: {e}")
    
    # Sort by step/epoch
    checkpoints.sort(key=lambda x: (x['epoch'], x['step']))
    return checkpoints