"""
Training utilities and components for Aurl.ai.

This package contains supporting utilities for training:
- Memory optimization and profiling
- Checkpoint management
- Early stopping mechanisms
- Regularization techniques
- Learning rate scheduling
- Pipeline state management
"""

from .memory_optimization import MemoryProfiler, MemoryOptimizer
from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopping, EarlyStoppingConfig
from .regularization import ComprehensiveRegularizer, RegularizationConfig
from .lr_scheduler import AdaptiveLRScheduler, StochasticWeightAveraging
from .pipeline_state_manager import PipelineStateManager
from .experiment_comparison import ExperimentComparator

__all__ = [
    "MemoryProfiler",
    "MemoryOptimizer",
    "CheckpointManager",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ComprehensiveRegularizer",
    "RegularizationConfig",
    "AdaptiveLRScheduler",
    "StochasticWeightAveraging",
    "PipelineStateManager",
    "ExperimentComparator"
]