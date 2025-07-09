"""
Core training infrastructure for Aurl.ai.

This package contains the essential components for training:
- Training orchestration and loops
- Loss function frameworks
- Logging and experiment tracking
"""

from .trainer import AdvancedTrainer
from .losses import ComprehensiveLossFramework
from .training_logger import EnhancedTrainingLogger
from .experiment_tracker import (
    ComprehensiveExperimentTracker, DataUsageInfo, EpochSummary, 
    ExperimentConfig, MetricsVisualizer
)

__all__ = [
    "AdvancedTrainer",
    "ComprehensiveLossFramework", 
    "EnhancedTrainingLogger",
    "ComprehensiveExperimentTracker",
    "DataUsageInfo",
    "EpochSummary",
    "ExperimentConfig",
    "MetricsVisualizer"
]