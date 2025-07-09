"""
Multi-Stage Training Protocols for Aurl.ai Music Generation.

This module implements sophisticated multi-stage training strategies:
- Pretrain → Finetune → Polish training pipeline
- Stage-specific configurations and objectives
- Progressive complexity and quality refinement
- Musical domain-specific stage transitions
- Adaptive stage duration and criteria
- Stage-specific data selection and augmentation

Designed for optimal music generation model development with structured learning phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
from collections import defaultdict, deque
import copy

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class TrainingStage(Enum):
    """Training stages in the multi-stage protocol."""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    POLISH = "polish"
    MUSICAL_REFINEMENT = "musical_refinement"


class StageTransitionCriteria(Enum):
    """Criteria for transitioning between stages."""
    EPOCH_BASED = "epoch_based"
    LOSS_BASED = "loss_based"
    MUSICAL_QUALITY_BASED = "musical_quality_based"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"


class DataSelectionStrategy(Enum):
    """Data selection strategies for different stages."""
    ALL_DATA = "all_data"
    CURRICULUM_FILTERED = "curriculum_filtered"
    QUALITY_FILTERED = "quality_filtered"
    GENRE_SPECIFIC = "genre_specific"
    COMPLEXITY_STAGED = "complexity_staged"


@dataclass
class StageConfig:
    """Configuration for a specific training stage."""
    
    # Stage identification
    stage: TrainingStage
    name: str
    description: str
    
    # Duration and criteria
    max_epochs: int
    min_epochs: int = 5
    transition_criteria: StageTransitionCriteria = StageTransitionCriteria.EPOCH_BASED
    
    # Learning objectives
    primary_objectives: List[str] = field(default_factory=list)
    loss_weights: Dict[str, float] = field(default_factory=dict)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model configuration
    enable_layers: List[str] = field(default_factory=lambda: ["all"])
    freeze_layers: List[str] = field(default_factory=list)
    learning_rate_multipliers: Dict[str, float] = field(default_factory=dict)
    
    # Data configuration
    data_selection: DataSelectionStrategy = DataSelectionStrategy.ALL_DATA
    data_filters: Dict[str, Any] = field(default_factory=dict)
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training specifics
    batch_size: Optional[int] = None
    gradient_accumulation: Optional[int] = None
    mixed_precision: bool = True
    
    # Musical specifics
    musical_focus: List[str] = field(default_factory=list)  # ["rhythm", "harmony", "melody"]
    style_emphasis: List[str] = field(default_factory=list)
    genre_focus: List[str] = field(default_factory=list)
    
    # Transition criteria values
    loss_improvement_threshold: float = 0.05
    musical_quality_threshold: float = 0.6
    patience_epochs: int = 10
    
    # Regularization
    dropout_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    
    # Stage-specific optimizations
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    lr_schedule_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiStageConfig:
    """Configuration for multi-stage training protocol."""
    
    # Overall protocol
    protocol_name: str = "musical_progressive"
    stages: List[StageConfig] = field(default_factory=list)
    
    # Global settings
    save_stage_checkpoints: bool = True
    stage_checkpoint_dir: str = "stage_checkpoints"
    
    # Transition settings
    smooth_transitions: bool = True
    transition_epochs: int = 2
    
    # Evaluation settings
    evaluate_between_stages: bool = True
    stage_evaluation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptation settings
    enable_adaptive_stages: bool = True
    stage_adaptation_frequency: int = 5
    performance_history_window: int = 20
    
    # Musical domain settings
    musical_progression_validation: bool = True
    cross_stage_musical_consistency: bool = True
    
    # Recovery settings
    enable_stage_rollback: bool = True
    rollback_criteria: Dict[str, float] = field(default_factory=lambda: {
        "loss_increase_threshold": 0.2,
        "musical_quality_drop": 0.1
    })


@dataclass
class StageState:
    """Current state of multi-stage training."""
    
    current_stage: TrainingStage = TrainingStage.PRETRAIN
    stage_index: int = 0
    stage_epoch: int = 0
    global_epoch: int = 0
    
    # Stage history
    completed_stages: List[TrainingStage] = field(default_factory=list)
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    stage_performance: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    best_stage_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Transition tracking
    last_transition_epoch: int = 0
    transition_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Adaptation tracking
    stage_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    last_adaptation_epoch: int = 0
    
    # Model state tracking
    stage_model_states: Dict[str, str] = field(default_factory=dict)  # stage -> checkpoint path
    
    # Data statistics
    stage_data_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class StageDataSelector:
    """Handles data selection and filtering for different training stages."""
    
    def __init__(self, config: MultiStageConfig):
        self.config = config
        
    def select_data_for_stage(self,
                            stage_config: StageConfig,
                            available_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select appropriate data for the given stage."""
        
        if stage_config.data_selection == DataSelectionStrategy.ALL_DATA:
            return available_data
        
        elif stage_config.data_selection == DataSelectionStrategy.CURRICULUM_FILTERED:
            return self._curriculum_filter_data(stage_config, available_data)
        
        elif stage_config.data_selection == DataSelectionStrategy.QUALITY_FILTERED:
            return self._quality_filter_data(stage_config, available_data)
        
        elif stage_config.data_selection == DataSelectionStrategy.GENRE_SPECIFIC:
            return self._genre_filter_data(stage_config, available_data)
        
        elif stage_config.data_selection == DataSelectionStrategy.COMPLEXITY_STAGED:
            return self._complexity_filter_data(stage_config, available_data)
        
        else:
            logger.warning(f"Unknown data selection strategy: {stage_config.data_selection}")
            return available_data
    
    def _curriculum_filter_data(self,
                              stage_config: StageConfig,
                              available_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on curriculum learning principles."""
        
        # Get complexity and difficulty thresholds from config
        complexity_threshold = stage_config.data_filters.get("max_complexity", 1.0)
        min_length = stage_config.data_filters.get("min_sequence_length", 0)
        max_length = stage_config.data_filters.get("max_sequence_length", float('inf'))
        
        filtered_data = []
        for sample in available_data:
            # Check complexity
            sample_complexity = sample.get("complexity", 0.5)
            if sample_complexity > complexity_threshold:
                continue
            
            # Check sequence length
            sample_length = sample.get("sequence_length", 0)
            if sample_length < min_length or sample_length > max_length:
                continue
            
            filtered_data.append(sample)
        
        logger.info(f"Curriculum filtering: {len(filtered_data)}/{len(available_data)} samples selected")
        return filtered_data
    
    def _quality_filter_data(self,
                           stage_config: StageConfig,
                           available_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on quality metrics."""
        
        min_quality = stage_config.data_filters.get("min_musical_quality", 0.0)
        
        filtered_data = []
        for sample in available_data:
            sample_quality = sample.get("musical_quality", 0.5)
            if sample_quality >= min_quality:
                filtered_data.append(sample)
        
        logger.info(f"Quality filtering: {len(filtered_data)}/{len(available_data)} samples selected")
        return filtered_data
    
    def _genre_filter_data(self,
                         stage_config: StageConfig,
                         available_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on genre focus."""
        
        target_genres = stage_config.genre_focus
        if not target_genres:
            return available_data
        
        filtered_data = []
        for sample in available_data:
            sample_genre = sample.get("genre", "unknown")
            if sample_genre in target_genres or "all" in target_genres:
                filtered_data.append(sample)
        
        logger.info(f"Genre filtering ({target_genres}): {len(filtered_data)}/{len(available_data)} samples selected")
        return filtered_data
    
    def _complexity_filter_data(self,
                              stage_config: StageConfig,
                              available_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on complexity staging."""
        
        # Define complexity ranges for different stages
        complexity_ranges = {
            TrainingStage.PRETRAIN: (0.0, 0.4),
            TrainingStage.FINETUNE: (0.3, 0.8),
            TrainingStage.POLISH: (0.6, 1.0),
            TrainingStage.MUSICAL_REFINEMENT: (0.4, 1.0)
        }
        
        min_complexity, max_complexity = complexity_ranges.get(
            stage_config.stage, (0.0, 1.0)
        )
        
        filtered_data = []
        for sample in available_data:
            sample_complexity = sample.get("complexity", 0.5)
            if min_complexity <= sample_complexity <= max_complexity:
                filtered_data.append(sample)
        
        logger.info(f"Complexity filtering ({min_complexity:.1f}-{max_complexity:.1f}): "
                   f"{len(filtered_data)}/{len(available_data)} samples selected")
        return filtered_data


class StageTransitionManager:
    """Manages transitions between training stages."""
    
    def __init__(self, config: MultiStageConfig):
        self.config = config
        
    def should_transition(self,
                         current_stage_config: StageConfig,
                         state: StageState,
                         performance_metrics: Dict[str, float]) -> bool:
        """Determine if stage transition should occur."""
        
        # Check minimum epochs requirement
        if state.stage_epoch < current_stage_config.min_epochs:
            return False
        
        # Check maximum epochs
        if state.stage_epoch >= current_stage_config.max_epochs:
            logger.info(f"Stage transition triggered by max epochs: {state.stage_epoch}/{current_stage_config.max_epochs}")
            return True
        
        # Check transition criteria
        if current_stage_config.transition_criteria == StageTransitionCriteria.EPOCH_BASED:
            return state.stage_epoch >= current_stage_config.max_epochs
        
        elif current_stage_config.transition_criteria == StageTransitionCriteria.LOSS_BASED:
            return self._check_loss_based_transition(current_stage_config, state, performance_metrics)
        
        elif current_stage_config.transition_criteria == StageTransitionCriteria.MUSICAL_QUALITY_BASED:
            return self._check_musical_quality_transition(current_stage_config, state, performance_metrics)
        
        elif current_stage_config.transition_criteria == StageTransitionCriteria.ADAPTIVE:
            return self._check_adaptive_transition(current_stage_config, state, performance_metrics)
        
        else:  # MANUAL
            return False
    
    def _check_loss_based_transition(self,
                                   stage_config: StageConfig,
                                   state: StageState,
                                   performance_metrics: Dict[str, float]) -> bool:
        """Check if loss-based transition criteria are met."""
        
        if len(state.stage_performance["total_loss"]) < stage_config.patience_epochs:
            return False
        
        recent_losses = state.stage_performance["total_loss"][-stage_config.patience_epochs:]
        best_loss = min(state.stage_performance["total_loss"])
        current_loss = recent_losses[-1]
        
        # Check if loss improvement is below threshold
        improvement = (best_loss - current_loss) / max(abs(best_loss), 1e-8)
        
        if improvement < stage_config.loss_improvement_threshold:
            logger.info(f"Stage transition triggered by loss plateau: improvement={improvement:.4f} < {stage_config.loss_improvement_threshold}")
            return True
        
        return False
    
    def _check_musical_quality_transition(self,
                                        stage_config: StageConfig,
                                        state: StageState,
                                        performance_metrics: Dict[str, float]) -> bool:
        """Check if musical quality-based transition criteria are met."""
        
        musical_quality = performance_metrics.get("musical_quality", 0.0)
        
        if musical_quality >= stage_config.musical_quality_threshold:
            logger.info(f"Stage transition triggered by musical quality: {musical_quality:.3f} >= {stage_config.musical_quality_threshold}")
            return True
        
        return False
    
    def _check_adaptive_transition(self,
                                 stage_config: StageConfig,
                                 state: StageState,
                                 performance_metrics: Dict[str, float]) -> bool:
        """Check adaptive transition criteria combining multiple factors."""
        
        # Combine loss and musical quality criteria
        loss_ready = self._check_loss_based_transition(stage_config, state, performance_metrics)
        quality_ready = self._check_musical_quality_transition(stage_config, state, performance_metrics)
        
        # Adaptive logic: either criteria can trigger, but prefer quality
        musical_quality = performance_metrics.get("musical_quality", 0.0)
        if musical_quality >= stage_config.musical_quality_threshold * 0.8:  # 80% of target
            return loss_ready or quality_ready
        
        # If quality is low, require both criteria
        return loss_ready and quality_ready


class MultiStageTrainingOrchestrator:
    """
    Orchestrates multi-stage training with stage-specific configurations and transitions.
    
    Features:
    - Stage-specific model configurations and objectives
    - Automatic stage transitions based on criteria
    - Data selection and filtering per stage
    - Stage checkpointing and recovery
    - Musical domain-specific stage progression
    """
    
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.state = StageState()
        self.data_selector = StageDataSelector(config)
        self.transition_manager = StageTransitionManager(config)
        
        # Initialize with first stage
        if config.stages:
            self.state.current_stage = config.stages[0].stage
            self.state.stage_index = 0
        
        logger.info(f"Initialized multi-stage training with {len(config.stages)} stages")
        logger.info(f"Stage progression: {[stage.stage.value for stage in config.stages]}")
    
    def get_current_stage_config(self) -> Optional[StageConfig]:
        """Get configuration for current stage."""
        
        if 0 <= self.state.stage_index < len(self.config.stages):
            return self.config.stages[self.state.stage_index]
        return None
    
    def configure_model_for_stage(self, model: nn.Module, stage_config: StageConfig):
        """Configure model for the current stage."""
        
        # Freeze/unfreeze layers
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            
            if stage_config.freeze_layers and any(frozen in name for frozen in stage_config.freeze_layers):
                param.requires_grad = False
                logger.debug(f"Frozen parameter: {name}")
            elif "all" in stage_config.enable_layers or any(enabled in name for enabled in stage_config.enable_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Configure stage-specific model settings
        if hasattr(model, 'configure_for_stage'):
            model.configure_for_stage(stage_config)
        
        logger.info(f"Configured model for stage: {stage_config.stage.value}")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def get_stage_data_config(self, stage_config: StageConfig) -> Dict[str, Any]:
        """Get data configuration for current stage."""
        
        return {
            "data_selection": stage_config.data_selection,
            "data_filters": stage_config.data_filters,
            "augmentation_config": stage_config.augmentation_config,
            "batch_size": stage_config.batch_size,
            "musical_focus": stage_config.musical_focus,
            "genre_focus": stage_config.genre_focus
        }
    
    def get_stage_loss_weights(self, stage_config: StageConfig) -> Dict[str, float]:
        """Get loss weights for current stage."""
        
        # Default weights
        default_weights = {
            "reconstruction_loss": 1.0,
            "kl_divergence": 1.0,
            "adversarial_loss": 0.5,
            "musical_quality": 0.3
        }
        
        # Override with stage-specific weights
        stage_weights = default_weights.copy()
        stage_weights.update(stage_config.loss_weights)
        
        return stage_weights
    
    def update_stage_progress(self,
                            epoch: int,
                            performance_metrics: Dict[str, float]):
        """Update stage progress and check for transitions."""
        
        self.state.global_epoch = epoch
        self.state.stage_epoch = epoch - self.state.last_transition_epoch
        
        # Update performance tracking
        for metric_name, value in performance_metrics.items():
            self.state.stage_performance[metric_name].append(value)
        
        # Check for stage transition
        current_stage_config = self.get_current_stage_config()
        if current_stage_config and self._should_advance_stage(current_stage_config, performance_metrics):
            self._advance_to_next_stage(performance_metrics)
        
        # Check for stage adaptation
        if (self.config.enable_adaptive_stages and 
            epoch % self.config.stage_adaptation_frequency == 0):
            self._adapt_current_stage(performance_metrics)
    
    def _should_advance_stage(self,
                            stage_config: StageConfig,
                            performance_metrics: Dict[str, float]) -> bool:
        """Check if should advance to next stage."""
        
        # Check if next stage exists
        if self.state.stage_index >= len(self.config.stages) - 1:
            return False
        
        return self.transition_manager.should_transition(
            stage_config, self.state, performance_metrics
        )
    
    def _advance_to_next_stage(self, performance_metrics: Dict[str, float]):
        """Advance to the next training stage."""
        
        current_stage_config = self.get_current_stage_config()
        
        # Record current stage completion
        stage_summary = {
            "stage": self.state.current_stage.value,
            "epochs": self.state.stage_epoch,
            "final_metrics": performance_metrics.copy(),
            "completion_epoch": self.state.global_epoch
        }
        
        self.state.completed_stages.append(self.state.current_stage)
        self.state.stage_history.append(stage_summary)
        
        # Save stage checkpoint if configured
        if self.config.save_stage_checkpoints:
            checkpoint_path = f"{self.config.stage_checkpoint_dir}/{self.state.current_stage.value}_final.pt"
            self.state.stage_model_states[self.state.current_stage.value] = checkpoint_path
        
        # Advance to next stage
        self.state.stage_index += 1
        if self.state.stage_index < len(self.config.stages):
            next_stage_config = self.config.stages[self.state.stage_index]
            self.state.current_stage = next_stage_config.stage
            self.state.last_transition_epoch = self.state.global_epoch
            
            # Clear stage-specific performance history
            self.state.stage_performance = defaultdict(list)
            
            # Record transition
            transition_info = {
                "from_stage": self.state.completed_stages[-1].value,
                "to_stage": self.state.current_stage.value,
                "epoch": self.state.global_epoch,
                "trigger_metrics": performance_metrics.copy()
            }
            self.state.transition_history.append(transition_info)
            
            logger.info(f"Advanced to stage: {self.state.current_stage.value} at epoch {self.state.global_epoch}")
            logger.info(f"Stage {self.state.stage_index + 1}/{len(self.config.stages)}: {next_stage_config.name}")
        else:
            logger.info("All training stages completed!")
    
    def _adapt_current_stage(self, performance_metrics: Dict[str, float]):
        """Adapt current stage based on performance."""
        
        current_stage_config = self.get_current_stage_config()
        if not current_stage_config:
            return
        
        # Simple adaptation: extend stage if performance is still improving
        if len(self.state.stage_performance["total_loss"]) >= self.config.performance_history_window:
            recent_losses = self.state.stage_performance["total_loss"][-self.config.performance_history_window:]
            
            # Check if loss is still decreasing
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            if trend < -0.001:  # Still improving
                # Extend stage by 20%
                extension = max(5, int(current_stage_config.max_epochs * 0.2))
                current_stage_config.max_epochs += extension
                
                adaptation_info = {
                    "epoch": self.state.global_epoch,
                    "stage": self.state.current_stage.value,
                    "action": "extend_stage",
                    "extension_epochs": extension,
                    "reason": "performance_improving"
                }
                self.state.stage_adaptations.append(adaptation_info)
                self.state.last_adaptation_epoch = self.state.global_epoch
                
                logger.info(f"Extended stage {self.state.current_stage.value} by {extension} epochs due to improving performance")
    
    def should_rollback_stage(self, performance_metrics: Dict[str, float]) -> bool:
        """Check if stage should be rolled back."""
        
        if not self.config.enable_stage_rollback:
            return False
        
        if len(self.state.stage_history) == 0:
            return False
        
        # Check for significant performance degradation
        current_loss = performance_metrics.get("total_loss", 0.0)
        previous_stage = self.state.stage_history[-1]
        previous_loss = previous_stage["final_metrics"].get("total_loss", 0.0)
        
        if previous_loss > 0 and current_loss > previous_loss * (1 + self.config.rollback_criteria["loss_increase_threshold"]):
            logger.warning(f"Stage rollback triggered by loss increase: {current_loss:.4f} > {previous_loss:.4f}")
            return True
        
        # Check for musical quality drop
        current_quality = performance_metrics.get("musical_quality", 0.0)
        previous_quality = previous_stage["final_metrics"].get("musical_quality", 0.0)
        
        if current_quality < previous_quality - self.config.rollback_criteria["musical_quality_drop"]:
            logger.warning(f"Stage rollback triggered by quality drop: {current_quality:.3f} < {previous_quality:.3f}")
            return True
        
        return False
    
    def rollback_to_previous_stage(self):
        """Rollback to previous stage."""
        
        if len(self.state.completed_stages) == 0:
            logger.warning("Cannot rollback: no previous stages")
            return False
        
        # Rollback state
        previous_stage = self.state.completed_stages[-1]
        self.state.current_stage = previous_stage
        self.state.stage_index -= 1
        
        # Remove last transition
        if self.state.transition_history:
            self.state.transition_history.pop()
        
        # Remove from completed stages
        self.state.completed_stages.pop()
        stage_summary = self.state.stage_history.pop()
        
        # Reset stage progress
        self.state.last_transition_epoch = stage_summary.get("start_epoch", 0)
        
        logger.info(f"Rolled back to stage: {self.state.current_stage.value}")
        return True
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        
        current_stage_config = self.get_current_stage_config()
        
        return {
            "current_stage": self.state.current_stage.value,
            "stage_index": self.state.stage_index,
            "total_stages": len(self.config.stages),
            "global_epoch": self.state.global_epoch,
            "stage_epoch": self.state.stage_epoch,
            "completed_stages": [s.value for s in self.state.completed_stages],
            "stage_progress": f"{self.state.stage_index + 1}/{len(self.config.stages)}",
            "current_stage_config": {
                "name": current_stage_config.name if current_stage_config else "None",
                "max_epochs": current_stage_config.max_epochs if current_stage_config else 0,
                "objectives": current_stage_config.primary_objectives if current_stage_config else []
            },
            "transitions_count": len(self.state.transition_history),
            "adaptations_count": len(self.state.stage_adaptations),
            "stage_performance": {
                metric: values[-10:] if values else []  # Last 10 values
                for metric, values in self.state.stage_performance.items()
            }
        }
    
    def save_state(self, filepath: Path):
        """Save multi-stage training state."""
        
        state_dict = {
            "config": {
                "protocol_name": self.config.protocol_name,
                "stages": [
                    {
                        "stage": stage.stage.value,
                        "name": stage.name,
                        "max_epochs": stage.max_epochs,
                        "primary_objectives": stage.primary_objectives
                    }
                    for stage in self.config.stages
                ]
            },
            "state": {
                "current_stage": self.state.current_stage.value,
                "stage_index": self.state.stage_index,
                "stage_epoch": self.state.stage_epoch,
                "global_epoch": self.state.global_epoch,
                "completed_stages": [s.value for s in self.state.completed_stages],
                "stage_history": self.state.stage_history,
                "last_transition_epoch": self.state.last_transition_epoch,
                "transition_history": self.state.transition_history,
                "stage_adaptations": self.state.stage_adaptations,
                "stage_model_states": self.state.stage_model_states
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Saved multi-stage training state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load multi-stage training state."""
        
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Restore state
        saved_state = state_dict.get("state", {})
        self.state.current_stage = TrainingStage(saved_state.get("current_stage", "pretrain"))
        self.state.stage_index = saved_state.get("stage_index", 0)
        self.state.stage_epoch = saved_state.get("stage_epoch", 0)
        self.state.global_epoch = saved_state.get("global_epoch", 0)
        self.state.completed_stages = [TrainingStage(s) for s in saved_state.get("completed_stages", [])]
        self.state.stage_history = saved_state.get("stage_history", [])
        self.state.last_transition_epoch = saved_state.get("last_transition_epoch", 0)
        self.state.transition_history = saved_state.get("transition_history", [])
        self.state.stage_adaptations = saved_state.get("stage_adaptations", [])
        self.state.stage_model_states = saved_state.get("stage_model_states", {})
        
        logger.info(f"Loaded multi-stage training state from {filepath}")
        logger.info(f"Resumed at stage {self.state.current_stage.value}, epoch {self.state.global_epoch}")


def create_musical_multistage_config() -> MultiStageConfig:
    """Create multi-stage configuration optimized for music generation."""
    
    # Stage 1: Pretrain - Foundation Learning
    pretrain_stage = StageConfig(
        stage=TrainingStage.PRETRAIN,
        name="Foundation Pretraining",
        description="Learn basic musical patterns and structures",
        max_epochs=30,
        min_epochs=10,
        transition_criteria=StageTransitionCriteria.ADAPTIVE,
        primary_objectives=["reconstruction", "basic_musical_patterns"],
        loss_weights={
            "reconstruction_loss": 1.0,
            "kl_divergence": 0.5,
            "musical_quality": 0.2
        },
        target_metrics={
            "reconstruction_loss": 2.0,
            "musical_quality": 0.3
        },
        data_selection=DataSelectionStrategy.COMPLEXITY_STAGED,
        data_filters={
            "max_complexity": 0.4,
            "max_sequence_length": 512
        },
        musical_focus=["rhythm", "basic_harmony"],
        genre_focus=["simple", "folk"],
        loss_improvement_threshold=0.05,
        musical_quality_threshold=0.3,
        patience_epochs=8
    )
    
    # Stage 2: Finetune - Musical Sophistication
    finetune_stage = StageConfig(
        stage=TrainingStage.FINETUNE,
        name="Musical Sophistication",
        description="Develop complex musical understanding and generation",
        max_epochs=40,
        min_epochs=15,
        transition_criteria=StageTransitionCriteria.ADAPTIVE,
        primary_objectives=["musical_quality", "stylistic_coherence"],
        loss_weights={
            "reconstruction_loss": 0.8,
            "kl_divergence": 0.6,
            "adversarial_loss": 0.4,
            "musical_quality": 0.6
        },
        target_metrics={
            "musical_quality": 0.6,
            "style_consistency": 0.5
        },
        data_selection=DataSelectionStrategy.COMPLEXITY_STAGED,
        data_filters={
            "max_complexity": 0.8,
            "max_sequence_length": 1024
        },
        musical_focus=["rhythm", "harmony", "melody"],
        genre_focus=["pop", "rock", "jazz"],
        loss_improvement_threshold=0.03,
        musical_quality_threshold=0.6,
        patience_epochs=12,
        learning_rate_multipliers={
            "decoder": 1.2,
            "attention": 0.8
        }
    )
    
    # Stage 3: Polish - Refinement and Excellence
    polish_stage = StageConfig(
        stage=TrainingStage.POLISH,
        name="Excellence Polishing",
        description="Fine-tune for exceptional musical quality",
        max_epochs=25,
        min_epochs=10,
        transition_criteria=StageTransitionCriteria.MUSICAL_QUALITY_BASED,
        primary_objectives=["exceptional_quality", "stylistic_mastery"],
        loss_weights={
            "reconstruction_loss": 0.6,
            "kl_divergence": 0.4,
            "adversarial_loss": 0.6,
            "musical_quality": 1.0,
            "style_consistency": 0.4
        },
        target_metrics={
            "musical_quality": 0.8,
            "style_consistency": 0.7
        },
        data_selection=DataSelectionStrategy.QUALITY_FILTERED,
        data_filters={
            "min_musical_quality": 0.6,
            "max_sequence_length": 1024
        },
        musical_focus=["rhythm", "harmony", "melody", "structure"],
        genre_focus=["all"],
        musical_quality_threshold=0.8,
        patience_epochs=15,
        learning_rate_multipliers={
            "all": 0.7  # Slower learning for refinement
        },
        dropout_rate=0.05,  # Reduced dropout for fine-tuning
        weight_decay=0.01
    )
    
    config = MultiStageConfig(
        protocol_name="musical_progressive_3stage",
        stages=[pretrain_stage, finetune_stage, polish_stage],
        save_stage_checkpoints=True,
        stage_checkpoint_dir="multi_stage_checkpoints",
        smooth_transitions=True,
        transition_epochs=2,
        evaluate_between_stages=True,
        enable_adaptive_stages=True,
        stage_adaptation_frequency=5,
        performance_history_window=15,
        musical_progression_validation=True,
        cross_stage_musical_consistency=True,
        enable_stage_rollback=True,
        rollback_criteria={
            "loss_increase_threshold": 0.15,
            "musical_quality_drop": 0.1
        }
    )
    
    return config


def create_rapid_prototyping_config() -> MultiStageConfig:
    """Create multi-stage configuration for rapid prototyping."""
    
    # Fast pretrain
    pretrain_stage = StageConfig(
        stage=TrainingStage.PRETRAIN,
        name="Rapid Pretrain",
        description="Quick foundation learning",
        max_epochs=10,
        min_epochs=5,
        transition_criteria=StageTransitionCriteria.EPOCH_BASED,
        primary_objectives=["basic_reconstruction"],
        loss_weights={"reconstruction_loss": 1.0, "kl_divergence": 0.3},
        data_selection=DataSelectionStrategy.COMPLEXITY_STAGED,
        data_filters={"max_complexity": 0.3, "max_sequence_length": 256},
        musical_focus=["rhythm"],
        genre_focus=["simple"]
    )
    
    # Fast finetune
    finetune_stage = StageConfig(
        stage=TrainingStage.FINETUNE,
        name="Rapid Finetune",
        description="Quick musical development",
        max_epochs=15,
        min_epochs=8,
        transition_criteria=StageTransitionCriteria.EPOCH_BASED,
        primary_objectives=["musical_improvement"],
        loss_weights={"reconstruction_loss": 0.8, "musical_quality": 0.4},
        data_selection=DataSelectionStrategy.ALL_DATA,
        musical_focus=["rhythm", "harmony"]
    )
    
    config = MultiStageConfig(
        protocol_name="rapid_prototyping",
        stages=[pretrain_stage, finetune_stage],
        save_stage_checkpoints=True,
        enable_adaptive_stages=False,
        enable_stage_rollback=False
    )
    
    return config