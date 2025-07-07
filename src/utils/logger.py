"""
Logging configuration for MidiFly.

This module provides a standardized logging setup with rotating file handlers,
structured output, and different log levels for development and production.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Standard log format with timestamp, level, module, and message
    log_format = "[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(log_format, datefmt=date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, base_dir: str = "logs") -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        
    Returns:
        Logger configured for the experiment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{base_dir}/experiments/{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=f"midifly.{experiment_name}",
        log_file=log_file,
        level="INFO"
    )


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for debugging purposes."""
    import platform
    import torch
    
    logger.info("="*50)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*50)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA capability: {torch.cuda.get_device_capability()}")
    else:
        logger.info("CUDA available: False")
    
    # Memory info
    import psutil
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info("="*50)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    losses: dict,
    metrics: dict,
    learning_rate: float,
    files_processed: int = 0,
    augmented_count: int = 0,
    memory_usage: Optional[dict] = None,
    sample_path: Optional[str] = None,
) -> None:
    """
    Log training progress with structured format.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        total_epochs: Total number of epochs
        batch: Current batch
        total_batches: Total batches in epoch
        losses: Dictionary of loss values
        metrics: Dictionary of metric values
        learning_rate: Current learning rate
        files_processed: Number of files processed
        augmented_count: Number of augmented samples
        memory_usage: Memory usage information
        sample_path: Path to generated sample (if any)
    """
    # Format losses as string
    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
    
    # Format metrics as string
    metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    # Main progress line
    logger.info(f"Epoch: {epoch}/{total_epochs}, Batch: {batch}/{total_batches}")
    logger.info(f"  - Files processed: {files_processed} ({augmented_count} augmented)")
    logger.info(f"  - Losses: {{{loss_str}}}")
    logger.info(f"  - Metrics: {{{metric_str}}}")
    logger.info(f"  - Learning rate: {learning_rate:.6f}")
    
    if memory_usage:
        if 'gpu' in memory_usage:
            logger.info(f"  - Memory: GPU {memory_usage['gpu']:.2f}GB, RAM {memory_usage['ram']:.2f}GB")
        else:
            logger.info(f"  - Memory: RAM {memory_usage['ram']:.2f}GB")
    
    if sample_path:
        logger.info(f"  - Sample saved: {sample_path}")


def log_model_info(logger: logging.Logger, model, config: dict) -> None:
    """Log model architecture and configuration information."""
    logger.info("="*50)
    logger.info("MODEL INFORMATION")
    logger.info("="*50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / (1024**2):.1f} MB (float32)")
    
    # Log configuration
    logger.info("Model configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for subkey, subvalue in value.items():
                logger.info(f"    {subkey}: {subvalue}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("="*50)


# Create a default logger for the package
default_logger = setup_logger("midifly", level="INFO")