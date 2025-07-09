"""
Training utilities and components for Aurl.ai.

This package contains supporting utilities for training:
- Memory optimization and profiling
- Checkpoint management
- Early stopping mechanisms
- Regularization techniques
- Learning rate scheduling
- Pipeline state management
- Advanced curriculum learning
- Knowledge distillation
- Advanced optimizers
- Multi-stage training protocols
- Reproducibility management
- Real-time quality evaluation
- Musical domain-specific strategies
"""

from .memory_optimization import MemoryProfiler, MemoryOptimizer
from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopping, EarlyStoppingConfig
from .regularization import ComprehensiveRegularizer, RegularizationConfig
from .lr_scheduler import AdaptiveLRScheduler, StochasticWeightAveraging
from .pipeline_state_manager import PipelineStateManager
from .experiment_comparison import ExperimentComparator

# Phase 4.5 Advanced Training Techniques
from .curriculum_learning import (
    ProgressiveCurriculumScheduler, CurriculumConfig, CurriculumStrategy,
    create_musical_curriculum_config
)
from .knowledge_distillation import (
    KnowledgeDistillationTrainer, DistillationConfig, DistillationStrategy,
    create_musical_distillation_config
)
from .advanced_optimizers import (
    LionOptimizer, AdamWEnhanced, SophiaOptimizer, AdaFactorOptimizer,
    MusicalAdaptiveOptimizer, OptimizerConfig, create_musical_optimizer_config,
    create_advanced_optimizer
)
from .multi_stage_training import (
    MultiStageTrainingOrchestrator, MultiStageConfig, StageConfig,
    create_musical_multistage_config, create_rapid_prototyping_config
)
from .reproducibility import (
    ReproducibilityManager, ReproducibilityConfig, ReproducibilityLevel,
    create_production_reproducibility_config, setup_reproducible_training
)
from .realtime_evaluation import (
    RealTimeQualityEvaluator, QualityConfig, QualityDimension,
    create_production_quality_config
)
from .musical_strategies import (
    MusicalDomainTrainer, MusicalTrainingConfig, MusicalDomain,
    create_classical_training_config, create_jazz_training_config,
    create_pop_training_config
)
from .hyperparameter_optimization import (
    HyperparameterOptimizer, OptimizationConfig, OptimizationStrategy,
    ParameterSpec, ParameterType, TrialResult, create_musical_parameter_space,
    create_quick_search_config, create_comprehensive_search_config,
    create_multi_objective_config, optimize_hyperparameters
)
from .training_efficiency import (
    TrainingEfficiencyOptimizer, EfficiencyConfig, PerformanceMetric,
    BottleneckType, OptimizationStrategy as EfficiencyOptimizationStrategy,
    PerformanceSnapshot, OptimizationResult, PerformanceProfiler,
    create_production_efficiency_config, create_research_efficiency_config,
    optimize_training_efficiency
)
from .scaling_laws import (
    ModelScalingAnalyzer, ScalingExperiment, ScalingResult, ScalingLaw,
    ScalingDimension, PerformanceMetric as ScalingPerformanceMetric,
    ScalingLawType, create_scaling_experiment, analyze_model_scaling
)

__all__ = [
    # Phase 4.1-4.4 Components
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
    "ExperimentComparator",
    
    # Phase 4.5 Advanced Training Techniques
    "ProgressiveCurriculumScheduler",
    "CurriculumConfig",
    "CurriculumStrategy",
    "create_musical_curriculum_config",
    
    "KnowledgeDistillationTrainer",
    "DistillationConfig",
    "DistillationStrategy",
    "create_musical_distillation_config",
    
    "LionOptimizer",
    "AdamWEnhanced",
    "SophiaOptimizer",
    "AdaFactorOptimizer",
    "MusicalAdaptiveOptimizer",
    "OptimizerConfig",
    "create_musical_optimizer_config",
    "create_advanced_optimizer",
    
    "MultiStageTrainingOrchestrator",
    "MultiStageConfig",
    "StageConfig",
    "create_musical_multistage_config",
    "create_rapid_prototyping_config",
    
    "ReproducibilityManager",
    "ReproducibilityConfig",
    "ReproducibilityLevel",
    "create_production_reproducibility_config",
    "setup_reproducible_training",
    
    "RealTimeQualityEvaluator",
    "QualityConfig",
    "QualityDimension",
    "create_production_quality_config",
    
    "MusicalDomainTrainer",
    "MusicalTrainingConfig",
    "MusicalDomain",
    "create_classical_training_config",
    "create_jazz_training_config",
    "create_pop_training_config",
    
    # Phase 4.5 Additional Components
    "HyperparameterOptimizer",
    "OptimizationConfig",
    "OptimizationStrategy",
    "ParameterSpec",
    "ParameterType",
    "TrialResult",
    "create_musical_parameter_space",
    "create_quick_search_config",
    "create_comprehensive_search_config",
    "create_multi_objective_config",
    "optimize_hyperparameters",
    
    "TrainingEfficiencyOptimizer",
    "EfficiencyConfig",
    "PerformanceMetric",
    "BottleneckType",
    "EfficiencyOptimizationStrategy",
    "PerformanceSnapshot",
    "OptimizationResult",
    "PerformanceProfiler",
    "create_production_efficiency_config",
    "create_research_efficiency_config",
    "optimize_training_efficiency",
    
    "ModelScalingAnalyzer",
    "ScalingExperiment",
    "ScalingResult",
    "ScalingLaw",
    "ScalingDimension",
    "ScalingPerformanceMetric",
    "ScalingLawType",
    "create_scaling_experiment",
    "analyze_model_scaling"
]