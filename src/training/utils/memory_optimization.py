"""
Memory optimization utilities for efficient training.

This module provides tools for:
- Activation checkpointing
- Memory profiling and monitoring
- Gradient checkpointing configuration
- Model parallelism setup
- Memory-efficient data loading
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import psutil
import gc
from functools import wraps
import time
import logging

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MemoryProfiler:
    """Profile and monitor memory usage during training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.checkpoints = []
        
    def checkpoint(self, name: str):
        """Record a memory checkpoint."""
        memory_info = self._get_memory_info()
        memory_info['name'] = name
        memory_info['timestamp'] = time.time()
        self.checkpoints.append(memory_info)
        
        logger.debug(f"Memory checkpoint '{name}': {memory_info}")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        info = {}
        
        # GPU memory
        if self.device.type == "cuda":
            info['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            info['gpu_reserved'] = torch.cuda.memory_reserved(self.device) / 1024**3
            info['gpu_max_allocated'] = torch.cuda.max_memory_allocated(self.device) / 1024**3
            
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        info['cpu_rss'] = memory_info.rss / 1024**3  # GB
        info['cpu_vms'] = memory_info.vms / 1024**3
        
        return info
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.checkpoints:
            return {}
        
        summary = {
            'total_checkpoints': len(self.checkpoints),
            'peak_gpu_allocated': max(cp.get('gpu_allocated', 0) for cp in self.checkpoints),
            'peak_cpu_rss': max(cp.get('cpu_rss', 0) for cp in self.checkpoints),
            'checkpoints': self.checkpoints[-10:]  # Last 10 checkpoints
        }
        
        return summary
    
    def clear_checkpoints(self):
        """Clear checkpoint history."""
        self.checkpoints.clear()


def memory_efficient_checkpoint(module: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply activation checkpointing to a module to save memory.
    
    This trades compute for memory by recomputing activations during backward pass.
    """
    return torch.utils.checkpoint.checkpoint(module, input_tensor)


class GradientCheckpointing:
    """Manage gradient checkpointing configuration."""
    
    def __init__(self, model: nn.Module, segments: Optional[List[str]] = None):
        self.model = model
        self.segments = segments or self._auto_detect_segments()
        self.enabled = False
        
    def _auto_detect_segments(self) -> List[str]:
        """Auto-detect model segments suitable for checkpointing."""
        segments = []
        
        # Look for transformer blocks or similar repeated structures
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in ['block', 'layer', 'encoder', 'decoder']):
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    segments.append(name)
        
        # If no specific patterns found, use top-level modules
        if not segments:
            segments = [name for name, _ in self.model.named_children()]
        
        logger.info(f"Auto-detected checkpointing segments: {segments}")
        return segments
    
    def enable(self):
        """Enable gradient checkpointing on specified segments."""
        for segment_name in self.segments:
            module = self._get_module_by_name(segment_name)
            if module and hasattr(module, 'forward'):
                # Wrap forward method with checkpointing
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                
                module.forward = checkpointed_forward
        
        self.enabled = True
        logger.info(f"Enabled gradient checkpointing on {len(self.segments)} segments")
    
    def disable(self):
        """Disable gradient checkpointing."""
        # Note: This is a simplified implementation
        # In practice, you'd need to store original forward methods
        self.enabled = False
        logger.info("Disabled gradient checkpointing")
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by its name path."""
        try:
            module = self.model
            for part in name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError:
            logger.warning(f"Module '{name}' not found for checkpointing")
            return None


class ModelParallelism:
    """Handle model parallelism for very large models."""
    
    def __init__(self, model: nn.Module, device_map: Optional[Dict[str, str]] = None):
        self.model = model
        self.device_map = device_map or self._auto_device_map()
        self.applied = False
        
    def _auto_device_map(self) -> Dict[str, str]:
        """Automatically create device mapping for model parallelism."""
        device_map = {}
        
        if not torch.cuda.is_available():
            return device_map
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            return device_map
        
        # Simple strategy: distribute major components across GPUs
        components = []
        for name, module in self.model.named_children():
            components.append(name)
        
        # Distribute components across available GPUs
        for i, component in enumerate(components):
            gpu_id = i % num_gpus
            device_map[component] = f"cuda:{gpu_id}"
        
        logger.info(f"Auto-generated device map for {num_gpus} GPUs: {device_map}")
        return device_map
    
    def apply(self):
        """Apply model parallelism based on device map."""
        if not self.device_map:
            logger.info("No device map provided, skipping model parallelism")
            return
        
        for component_name, device in self.device_map.items():
            component = getattr(self.model, component_name, None)
            if component:
                component.to(device)
                logger.info(f"Moved {component_name} to {device}")
        
        self.applied = True
        logger.info("Applied model parallelism")
    
    def get_device_for_component(self, component_name: str) -> str:
        """Get device for a specific component."""
        return self.device_map.get(component_name, "cuda:0")


def memory_efficient_forward(func: Callable) -> Callable:
    """Decorator for memory-efficient forward passes."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Perform forward pass
        result = func(*args, **kwargs)
        
        # Optional: Clear cache after forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    return wrapper


class MemoryEfficientDataLoader:
    """Memory-efficient data loading strategies."""
    
    def __init__(self, 
                 dataloader: torch.utils.data.DataLoader,
                 prefetch_factor: int = 2,
                 pin_memory_device: str = "cuda"):
        self.dataloader = dataloader
        self.prefetch_factor = prefetch_factor
        self.pin_memory_device = pin_memory_device
        
    def __iter__(self):
        """Iterate with memory-efficient prefetching."""
        for batch in self.dataloader:
            # Move to device efficiently
            if isinstance(batch, dict):
                batch = {k: v.to(self.pin_memory_device, non_blocking=True) 
                        if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.pin_memory_device, non_blocking=True)
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)


class MemoryOptimizer:
    """Comprehensive memory optimization manager."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 enable_gradient_checkpointing: bool = False,
                 enable_model_parallelism: bool = False,
                 memory_threshold: float = 0.9):
        
        self.model = model
        self.device = device
        self.memory_threshold = memory_threshold
        
        # Initialize components
        self.profiler = MemoryProfiler(device)
        self.gradient_checkpointing = GradientCheckpointing(model) if enable_gradient_checkpointing else None
        self.model_parallelism = ModelParallelism(model) if enable_model_parallelism else None
        
        # Memory monitoring
        self.peak_memory = 0.0
        self.memory_warnings = []
        
        logger.info(f"Initialized MemoryOptimizer with threshold {memory_threshold}")
    
    def optimize_for_training(self):
        """Apply all memory optimizations for training."""
        logger.info("Applying memory optimizations...")
        
        # Enable gradient checkpointing
        if self.gradient_checkpointing:
            self.gradient_checkpointing.enable()
        
        # Apply model parallelism
        if self.model_parallelism:
            self.model_parallelism.apply()
        
        # Set memory efficient settings
        if torch.cuda.is_available():
            # Enable memory efficient attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory pool settings
            torch.cuda.empty_cache()
            
        logger.info("Memory optimizations applied")
    
    def monitor_step(self, step_name: str):
        """Monitor memory usage at a training step."""
        self.profiler.checkpoint(step_name)
        
        # Check memory usage
        if self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated(self.device) / torch.cuda.max_memory_allocated(self.device)
            self.peak_memory = max(self.peak_memory, current_memory)
            
            if current_memory > self.memory_threshold:
                warning = f"High memory usage at {step_name}: {current_memory:.2%}"
                self.memory_warnings.append(warning)
                logger.warning(warning)
                
                # Emergency cleanup
                self.emergency_cleanup()
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Performed emergency memory cleanup")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        report = {
            'peak_memory_usage': self.peak_memory,
            'memory_warnings': len(self.memory_warnings),
            'recent_warnings': self.memory_warnings[-5:] if self.memory_warnings else [],
            'profiler_summary': self.profiler.get_memory_summary(),
            'optimizations_enabled': {
                'gradient_checkpointing': self.gradient_checkpointing.enabled if self.gradient_checkpointing else False,
                'model_parallelism': self.model_parallelism.applied if self.model_parallelism else False,
            }
        }
        
        return report


def optimize_model_for_memory(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Apply model-level memory optimizations."""
    
    # Convert to memory efficient formats
    if config.get('use_half_precision', False):
        model = model.half()
        logger.info("Converted model to half precision")
    
    # Enable gradient checkpointing on transformer blocks
    if config.get('gradient_checkpointing', False):
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.checkpoint = True
                logger.info(f"Enabled checkpointing on {name}")
    
    # Optimize embedding layers
    if config.get('optimize_embeddings', False):
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                # Enable sparse gradients for large embeddings
                if module.num_embeddings > 10000:
                    module.sparse = True
                    logger.info(f"Enabled sparse gradients for {name}")
    
    return model