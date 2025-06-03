"""Logging utilities for RL Learning Framework"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import coloredlogs
    HAS_COLOREDLOGS = True
except ImportError:
    coloredlogs = None
    HAS_COLOREDLOGS = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: str):
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            print(f"Warning: TensorBoard not available. Logs would be saved to: {log_dir}")
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value"""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: Optional[int] = None):
        """Log multiple scalar values"""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log a histogram"""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text"""
        if self.writer is None:
            return
        if step is None:
            step = self.step
        self.writer.add_text(tag, text, step)
    
    def increment_step(self):
        """Increment the global step"""
        self.step += 1
    
    def close(self):
        """Close the writer"""
        if self.writer is not None:
            self.writer.close()


def setup_logger(
    name: str = "rlhf",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """Setup logger with colored output and file logging
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file path
        log_dir: Log directory (if log_file not provided)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    if isinstance(level, int):
        console_handler.setLevel(level)
    else:
        console_handler.setLevel(getattr(logging, level.upper()))
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Setup colored logs if available
    if HAS_COLOREDLOGS:
        coloredlogs.install(
            level=level.upper(),
            logger=logger,
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # File handler
    if log_file or log_dir:
        if log_file is None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "rl_learning") -> logging.Logger:
    """Get existing logger instance"""
    return logging.getLogger(name)