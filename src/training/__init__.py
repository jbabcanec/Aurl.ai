"""
Training infrastructure for Aurl.ai music generation.

This package contains organized training components:
- core/: Essential training infrastructure (trainer, losses, logging)
- monitoring/: Visualization and monitoring tools
- utils/: Training utilities and components

The training system is designed for professional music generation with:
- Advanced training orchestration
- Comprehensive loss frameworks
- Real-time monitoring and visualization
- Professional checkpointing and state management
- Early stopping and regularization
- Musical quality tracking
"""

# Import key components for easy access
from .core import (
    AdvancedTrainer,
    ComprehensiveLossFramework,
    EnhancedTrainingLogger,
    ComprehensiveExperimentTracker,
    DataUsageInfo,
    EpochSummary
)

from .monitoring import (
    LossMonitor,
    TrainingStabilityMonitor,
    MusicalQualityTracker,
    EnhancedAnomalyDetector
)

from .utils import (
    CheckpointManager,
    EarlyStopping,
    EarlyStoppingConfig,
    ComprehensiveRegularizer,
    AdaptiveLRScheduler,
    StochasticWeightAveraging,
    ExperimentComparator
)

__all__ = [
    # Core components
    "AdvancedTrainer",
    "ComprehensiveLossFramework",
    "EnhancedTrainingLogger",
    "ComprehensiveExperimentTracker",
    "DataUsageInfo",
    "EpochSummary",
    
    # Monitoring
    "LossMonitor",
    "TrainingStabilityMonitor",
    "MusicalQualityTracker",
    "EnhancedAnomalyDetector",
    
    # Utils
    "CheckpointManager",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ComprehensiveRegularizer",
    "AdaptiveLRScheduler",
    "StochasticWeightAveraging",
    "ExperimentComparator"
]