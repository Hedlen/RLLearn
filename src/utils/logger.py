"""Logging utilities for RL Learning Framework"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import coloredlogs
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: Optional[int] = None):
        """Log multiple scalar values"""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram of values"""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text"""
        if step is None:
            step = self.step
        self.writer.add_text(tag, text, step)
    
    def increment_step(self):
        """Increment global step"""
        self.step += 1
    
    def close(self):
        """Close the writer"""
        self.writer.close()


def setup_logger(
    name: str = "rl_learning",
    level: str = "INFO",
    use_tensorboard: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """Setup logger with colored output and optional TensorBoard logging
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_tensorboard: Whether to enable TensorBoard logging
        log_dir: Directory for log files and TensorBoard logs
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup colored console logging
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    coloredlogs.install(
        level=level.upper(),
        logger=logger,
        fmt=console_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_dir is specified
    if log_dir:
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"),
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Setup TensorBoard logger if requested
    if use_tensorboard and log_dir:
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        tb_logger = TensorBoardLogger(tb_log_dir)
        logger.tb_logger = tb_logger
        logger.info(f"TensorBoard logging enabled. Log dir: {tb_log_dir}")
    else:
        logger.tb_logger = None
    
    return logger


def get_logger(name: str = "rl_learning") -> logging.Logger:
    """Get existing logger instance"""
    return logging.getLogger(name)