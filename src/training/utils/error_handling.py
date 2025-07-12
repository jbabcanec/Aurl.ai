"""
Enhanced error handling and recovery for training pipeline.

This module provides robust error handling, automatic recovery mechanisms,
and detailed diagnostics for common training failures.
"""

import torch
import traceback
import sys
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
from datetime import datetime
from functools import wraps

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class GradientError(TrainingError):
    """Exception for gradient-related issues."""
    pass


class MemoryError(TrainingError):
    """Exception for memory-related issues."""
    pass


class ModelError(TrainingError):
    """Exception for model-related issues."""
    pass


class ErrorHandler:
    """Comprehensive error handling for training pipeline."""
    
    def __init__(self, 
                 checkpoint_dir: Optional[Path] = None,
                 max_retries: int = 3,
                 save_error_states: bool = True):
        """
        Initialize error handler.
        
        Args:
            checkpoint_dir: Directory to save error states
            max_retries: Maximum retries for recoverable errors
            save_error_states: Whether to save model state on errors
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self.save_error_states = save_error_states
        self.error_history = []
        
        if checkpoint_dir:
            self.error_log_path = checkpoint_dir / "error_log.json"
        else:
            self.error_log_path = None
    
    def handle_gradient_error(self, e: Exception, model: torch.nn.Module, 
                            batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle gradient computation errors.
        
        Args:
            e: The exception
            model: The model that failed
            batch_data: The batch that caused the error
            
        Returns:
            Recovery suggestions and diagnostics
        """
        error_info = {
            "type": "gradient_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc()
        }
        
        # Check for common gradient issues
        diagnostics = []
        
        # 1. Check for NaN/Inf in model parameters
        nan_params = []
        inf_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(name)
                if torch.isinf(param.grad).any():
                    inf_params.append(name)
        
        if nan_params:
            diagnostics.append(f"NaN gradients in: {', '.join(nan_params[:5])}")
        if inf_params:
            diagnostics.append(f"Inf gradients in: {', '.join(inf_params[:5])}")
        
        # 2. Check for in-place operations
        if "version" in str(e) and "expected version" in str(e):
            diagnostics.append("In-place operation detected - check for += or tensor modifications")
            
            # Try to identify the specific tensor
            error_msg = str(e)
            if "torch.FloatTensor" in error_msg:
                # Extract tensor shape from error message
                import re
                shape_match = re.search(r'\[([^\]]+)\]', error_msg)
                if shape_match:
                    diagnostics.append(f"Problematic tensor shape: [{shape_match.group(1)}]")
        
        # 3. Check gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > 100:
            diagnostics.append(f"Large gradient norm: {total_norm:.2f} - consider gradient clipping")
        
        error_info["diagnostics"] = diagnostics
        
        # Recovery suggestions
        recovery = []
        
        if nan_params or inf_params:
            recovery.append("Reduce learning rate")
            recovery.append("Enable gradient clipping")
            recovery.append("Check for numerical instability in loss computation")
        
        if "version" in str(e):
            recovery.append("Enable anomaly detection: torch.autograd.set_detect_anomaly(True)")
            recovery.append("Review model for in-place operations")
            recovery.append("Use .clone() on tensors that will be modified")
        
        error_info["recovery_suggestions"] = recovery
        
        # Save error state
        if self.save_error_states and self.checkpoint_dir:
            self._save_error_state(error_info, model, batch_data)
        
        self.error_history.append(error_info)
        self._log_error(error_info)
        
        return error_info
    
    def handle_memory_error(self, e: Exception, model: torch.nn.Module,
                          batch_size: int, seq_length: int) -> Dict[str, Any]:
        """
        Handle out-of-memory errors.
        
        Args:
            e: The exception
            model: The model
            batch_size: Current batch size
            seq_length: Current sequence length
            
        Returns:
            Recovery suggestions
        """
        error_info = {
            "type": "memory_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "sequence_length": seq_length
        }
        
        # Calculate model memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # Get current GPU memory if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            error_info["gpu_allocated_mb"] = allocated
            error_info["gpu_reserved_mb"] = reserved
        
        error_info["model_params_mb"] = param_memory
        
        # Recovery suggestions
        recovery = []
        
        if batch_size > 1:
            recovery.append(f"Reduce batch size from {batch_size} to {batch_size // 2}")
        
        if seq_length > 256:
            recovery.append(f"Reduce sequence length from {seq_length} to {seq_length // 2}")
        
        recovery.append("Enable gradient accumulation")
        recovery.append("Enable gradient checkpointing")
        recovery.append("Use mixed precision training")
        recovery.append("Clear GPU cache: torch.cuda.empty_cache()")
        
        error_info["recovery_suggestions"] = recovery
        
        self.error_history.append(error_info)
        self._log_error(error_info)
        
        return error_info
    
    def handle_model_error(self, e: Exception, model: torch.nn.Module,
                         stage: str = "unknown") -> Dict[str, Any]:
        """
        Handle model-related errors (shape mismatches, etc.).
        
        Args:
            e: The exception
            model: The model
            stage: Training stage where error occurred
            
        Returns:
            Diagnostics and recovery suggestions
        """
        error_info = {
            "type": "model_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "traceback": traceback.format_exc()
        }
        
        diagnostics = []
        
        # Check for shape mismatches
        if "size mismatch" in str(e) or "shape" in str(e).lower():
            diagnostics.append("Tensor shape mismatch detected")
            
            # Try to extract shapes from error
            import re
            sizes = re.findall(r'size \(([^\)]+)\)', str(e))
            if sizes:
                diagnostics.append(f"Shapes involved: {sizes}")
        
        # Check for missing keys in state dict
        if "missing keys" in str(e).lower() or "unexpected keys" in str(e).lower():
            diagnostics.append("Model checkpoint incompatibility detected")
        
        error_info["diagnostics"] = diagnostics
        
        # Recovery suggestions
        recovery = []
        
        if "shape" in str(e).lower():
            recovery.append("Check model configuration matches data dimensions")
            recovery.append("Verify vocab_size, hidden_dim, and sequence_length")
        
        if "missing keys" in str(e).lower():
            recovery.append("Use load_weights_only=True when loading checkpoints")
            recovery.append("Check model architecture compatibility between stages")
        
        error_info["recovery_suggestions"] = recovery
        
        self.error_history.append(error_info)
        self._log_error(error_info)
        
        return error_info
    
    def _save_error_state(self, error_info: Dict[str, Any], 
                         model: torch.nn.Module,
                         batch_data: Optional[Dict[str, Any]] = None):
        """Save model and data state when error occurs."""
        if not self.checkpoint_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_dir = self.checkpoint_dir / f"error_{timestamp}"
        error_dir.mkdir(exist_ok=True)
        
        # Save error info
        with open(error_dir / "error_info.json", "w") as f:
            json.dump(error_info, f, indent=2)
        
        # Save model state
        torch.save({
            "model_state_dict": model.state_dict(),
            "error_info": error_info
        }, error_dir / "model_state.pt")
        
        # Save batch data if available
        if batch_data:
            torch.save(batch_data, error_dir / "error_batch.pt")
        
        logger.info(f"Error state saved to: {error_dir}")
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error information."""
        logger.error(f"\n{'='*60}")
        logger.error(f"ERROR: {error_info['type'].upper()}")
        logger.error(f"{'='*60}")
        logger.error(f"Error: {error_info['error']}")
        
        if "diagnostics" in error_info:
            logger.error("\nDiagnostics:")
            for diag in error_info["diagnostics"]:
                logger.error(f"  - {diag}")
        
        if "recovery_suggestions" in error_info:
            logger.error("\nRecovery Suggestions:")
            for suggestion in error_info["recovery_suggestions"]:
                logger.error(f"  ✓ {suggestion}")
        
        logger.error(f"{'='*60}\n")
        
        # Save to error log
        if self.error_log_path:
            self.error_log_path.parent.mkdir(exist_ok=True)
            
            # Load existing errors
            if self.error_log_path.exists():
                with open(self.error_log_path, "r") as f:
                    errors = json.load(f)
            else:
                errors = []
            
            # Append new error
            errors.append(error_info)
            
            # Save updated log
            with open(self.error_log_path, "w") as f:
                json.dump(errors, f, indent=2)


def safe_training_step(error_handler: ErrorHandler):
    """
    Decorator for safe training steps with automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < error_handler.max_retries:
                try:
                    return func(*args, **kwargs)
                
                except RuntimeError as e:
                    error_str = str(e)
                    
                    # Handle different error types
                    if "out of memory" in error_str.lower():
                        # Extract batch info from args/kwargs
                        batch_size = kwargs.get("batch_size", 32)
                        seq_length = kwargs.get("seq_length", 512)
                        
                        error_info = error_handler.handle_memory_error(
                            e, args[0] if args else None, batch_size, seq_length
                        )
                        
                        # Try to recover
                        if retries < error_handler.max_retries - 1:
                            torch.cuda.empty_cache()
                            # Reduce batch size for retry
                            if "batch_size" in kwargs:
                                kwargs["batch_size"] = kwargs["batch_size"] // 2
                                logger.info(f"Retrying with batch_size={kwargs['batch_size']}")
                    
                    elif "gradient" in error_str or "version" in error_str:
                        model = args[0] if args else kwargs.get("model")
                        batch_data = kwargs.get("batch_data", {})
                        
                        error_info = error_handler.handle_gradient_error(
                            e, model, batch_data
                        )
                        
                        # For gradient errors, don't retry automatically
                        raise GradientError(f"Gradient computation failed: {error_info}")
                    
                    else:
                        # General model error
                        model = args[0] if args else kwargs.get("model")
                        error_info = error_handler.handle_model_error(
                            e, model, stage=kwargs.get("stage", "unknown")
                        )
                        
                        raise ModelError(f"Model error: {error_info}")
                    
                    last_error = e
                    retries += 1
                
                except Exception as e:
                    # Unexpected error
                    logger.error(f"Unexpected error in training: {e}")
                    logger.error(traceback.format_exc())
                    raise
            
            # Max retries exceeded
            raise TrainingError(f"Training failed after {retries} retries. Last error: {last_error}")
        
        return wrapper
    return decorator


class GradientMonitor:
    """Monitor gradients for anomalies during training."""
    
    def __init__(self, 
                 threshold_norm: float = 100.0,
                 check_nan: bool = True,
                 check_inf: bool = True):
        """
        Initialize gradient monitor.
        
        Args:
            threshold_norm: Maximum allowed gradient norm
            check_nan: Check for NaN values
            check_inf: Check for Inf values
        """
        self.threshold_norm = threshold_norm
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.gradient_history = []
    
    def check_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Check model gradients for anomalies.
        
        Args:
            model: Model to check
            
        Returns:
            Dictionary with gradient statistics and any issues found
        """
        issues = []
        stats = {
            "total_norm": 0.0,
            "max_norm": 0.0,
            "min_norm": float('inf'),
            "num_params_with_grad": 0,
            "num_nan": 0,
            "num_inf": 0
        }
        
        param_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                param_norm = grad.norm(2).item()
                param_norms[name] = param_norm
                
                stats["total_norm"] += param_norm ** 2
                stats["max_norm"] = max(stats["max_norm"], param_norm)
                stats["min_norm"] = min(stats["min_norm"], param_norm)
                stats["num_params_with_grad"] += 1
                
                # Check for NaN
                if self.check_nan and torch.isnan(grad).any():
                    stats["num_nan"] += 1
                    issues.append(f"NaN gradient in {name}")
                
                # Check for Inf
                if self.check_inf and torch.isinf(grad).any():
                    stats["num_inf"] += 1
                    issues.append(f"Inf gradient in {name}")
        
        stats["total_norm"] = stats["total_norm"] ** 0.5
        
        # Check total norm threshold
        if stats["total_norm"] > self.threshold_norm:
            issues.append(f"Gradient norm {stats['total_norm']:.2f} exceeds threshold {self.threshold_norm}")
        
        # Find parameters with largest gradients
        if param_norms:
            sorted_params = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)
            stats["top_gradient_params"] = sorted_params[:5]
        
        stats["issues"] = issues
        stats["healthy"] = len(issues) == 0
        
        self.gradient_history.append({
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        })
        
        return stats