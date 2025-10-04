"""
Adaptive Constraint Relaxation System

This module provides constraints that adapt based on model performance,
starting strict and relaxing as the model demonstrates competence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from ..generation.optimized_constraints import OptimizedConstraintEngine, OptimizedConstraintConfig
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AdaptiveConstraintConfig:
    """Configuration for adaptive constraint system."""

    # Initial constraint levels (strict)
    initial_harmonic_strictness: float = 1.0
    initial_melodic_strictness: float = 1.0
    initial_rhythmic_strictness: float = 1.0
    initial_dynamic_strictness: float = 1.0

    # Minimum constraint levels (relaxed)
    minimum_harmonic_strictness: float = 0.1
    minimum_melodic_strictness: float = 0.1
    minimum_rhythmic_strictness: float = 0.1
    minimum_dynamic_strictness: float = 0.1

    # Relaxation parameters
    relaxation_rate: float = 0.95  # Exponential decay per epoch
    performance_threshold: float = 0.8  # Performance needed to relax

    # Constraint-specific thresholds
    harmonic_performance_threshold: float = 0.85
    melodic_performance_threshold: float = 0.80
    rhythmic_performance_threshold: float = 0.75
    dynamic_performance_threshold: float = 0.70

    # Safety parameters
    minimum_epochs_before_relaxation: int = 10
    freeze_on_degradation: bool = True
    degradation_threshold: float = 0.1  # 10% performance drop


class ConstraintPerformanceTracker:
    """Tracks performance of each constraint type."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.constraint_violations = defaultdict(int)
        self.constraint_successes = defaultdict(int)

    def update(self, constraint_type: str, success: bool, score: float):
        """Update tracking for a constraint."""
        self.metrics[constraint_type].append(score)
        if success:
            self.constraint_successes[constraint_type] += 1
        else:
            self.constraint_violations[constraint_type] += 1

    def get_performance(self, constraint_type: str, window: int = 10) -> float:
        """Get recent performance for a constraint type."""
        if constraint_type not in self.metrics:
            return 0.0

        recent = self.metrics[constraint_type][-window:]
        if len(recent) == 0:
            return 0.0

        return np.mean(recent)

    def get_success_rate(self, constraint_type: str) -> float:
        """Get success rate for a constraint type."""
        total = (self.constraint_violations[constraint_type] +
                self.constraint_successes[constraint_type])
        if total == 0:
            return 0.0

        return self.constraint_successes[constraint_type] / total


class AdaptiveConstraintEngine(nn.Module):
    """
    Constraint engine that adapts strictness based on model performance.
    """

    def __init__(self, config: AdaptiveConstraintConfig):
        super().__init__()
        self.config = config

        # Initialize base constraint engine
        base_config = OptimizedConstraintConfig(
            precompute_masks=True,
            vectorize_operations=True
        )
        self.base_engine = OptimizedConstraintEngine(base_config)

        # Current strictness levels
        self.strictness = {
            'harmonic': config.initial_harmonic_strictness,
            'melodic': config.initial_melodic_strictness,
            'rhythmic': config.initial_rhythmic_strictness,
            'dynamic': config.initial_dynamic_strictness
        }

        # Performance tracking
        self.performance_tracker = ConstraintPerformanceTracker()
        self.epoch = 0
        self.strictness_history = []

    def forward(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply adaptive constraints to logits.

        Args:
            logits: Model output logits
            sequence: Generated sequence so far
            training: Whether in training mode

        Returns:
            Constrained logits
        """
        # Apply base constraints with current strictness
        constrained_logits = self._apply_adaptive_constraints(
            logits, sequence
        )

        # Track performance if in training
        if training:
            self._track_constraint_performance(
                logits, constrained_logits, sequence
            )

        return constrained_logits

    def _apply_adaptive_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """Apply constraints with adaptive strictness."""
        device = logits.device
        modified_logits = logits.clone()

        # Apply each constraint type with its current strictness
        if self.strictness['harmonic'] > 0:
            modified_logits = self._apply_harmonic_constraint(
                modified_logits, sequence, self.strictness['harmonic']
            )

        if self.strictness['melodic'] > 0:
            modified_logits = self._apply_melodic_constraint(
                modified_logits, sequence, self.strictness['melodic']
            )

        if self.strictness['rhythmic'] > 0:
            modified_logits = self._apply_rhythmic_constraint(
                modified_logits, sequence, self.strictness['rhythmic']
            )

        if self.strictness['dynamic'] > 0:
            modified_logits = self._apply_dynamic_constraint(
                modified_logits, sequence, self.strictness['dynamic']
            )

        return modified_logits

    def _apply_harmonic_constraint(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strictness: float
    ) -> torch.Tensor:
        """Apply harmonic constraints with specified strictness."""
        # Use base engine's harmonic constraint
        base_constrained = self.base_engine._apply_harmonic_constraints_vectorized(
            logits, sequence, logits.device
        )

        # Blend based on strictness
        return logits * (1 - strictness) + base_constrained * strictness

    def _apply_melodic_constraint(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strictness: float
    ) -> torch.Tensor:
        """Apply melodic constraints with specified strictness."""
        base_constrained = self.base_engine._apply_melodic_constraints_vectorized(
            logits, sequence, logits.device
        )

        return logits * (1 - strictness) + base_constrained * strictness

    def _apply_rhythmic_constraint(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strictness: float
    ) -> torch.Tensor:
        """Apply rhythmic constraints with specified strictness."""
        base_constrained = self.base_engine._apply_rhythmic_constraints_vectorized(
            logits, sequence, logits.device
        )

        return logits * (1 - strictness) + base_constrained * strictness

    def _apply_dynamic_constraint(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strictness: float
    ) -> torch.Tensor:
        """Apply dynamic constraints with specified strictness."""
        base_constrained = self.base_engine._apply_dynamic_constraints_vectorized(
            logits, sequence, logits.device
        )

        return logits * (1 - strictness) + base_constrained * strictness

    def _track_constraint_performance(
        self,
        original_logits: torch.Tensor,
        constrained_logits: torch.Tensor,
        sequence: torch.Tensor
    ):
        """Track how well constraints are being followed."""
        # Measure constraint impact
        logit_diff = torch.abs(original_logits - constrained_logits).mean().item()

        # Track overall constraint effectiveness
        self.performance_tracker.update(
            'overall',
            success=logit_diff < 1.0,  # Arbitrary threshold
            score=1.0 / (1.0 + logit_diff)
        )

    def update_epoch(self, epoch: int, performance_metrics: Dict[str, float]):
        """
        Update constraint strictness based on epoch and performance.

        Args:
            epoch: Current training epoch
            performance_metrics: Dictionary of performance metrics
        """
        self.epoch = epoch

        # Don't relax before minimum epochs
        if epoch < self.config.minimum_epochs_before_relaxation:
            return

        # Check each constraint type for relaxation
        for constraint_type in self.strictness:
            self._update_constraint_strictness(
                constraint_type,
                performance_metrics
            )

        # Save history
        self.strictness_history.append({
            'epoch': epoch,
            'strictness': self.strictness.copy(),
            'metrics': performance_metrics.copy()
        })

    def _update_constraint_strictness(
        self,
        constraint_type: str,
        performance_metrics: Dict[str, float]
    ):
        """Update strictness for a specific constraint type."""
        # Get performance threshold for this constraint
        threshold_key = f"{constraint_type}_performance_threshold"
        threshold = getattr(self.config, threshold_key)

        # Get relevant performance metric
        if constraint_type == 'harmonic':
            performance = performance_metrics.get('harmonic_score', 0.0)
        elif constraint_type == 'melodic':
            performance = performance_metrics.get('melodic_score', 0.0)
        elif constraint_type == 'rhythmic':
            performance = performance_metrics.get('rhythmic_score', 0.0)
        elif constraint_type == 'dynamic':
            performance = performance_metrics.get('dynamic_score', 0.0)
        else:
            performance = performance_metrics.get('grammar_score', 0.0)

        # Check if we should relax
        if performance >= threshold:
            # Relax constraint
            new_strictness = self.strictness[constraint_type] * self.config.relaxation_rate

            # Apply minimum
            min_key = f"minimum_{constraint_type}_strictness"
            minimum = getattr(self.config, min_key)
            new_strictness = max(new_strictness, minimum)

            self.strictness[constraint_type] = new_strictness

            logger.info(
                f"Relaxed {constraint_type} constraint: "
                f"{self.strictness[constraint_type]:.3f} "
                f"(performance: {performance:.3f})"
            )

        elif self.config.freeze_on_degradation:
            # Check for performance degradation
            recent_performance = self.performance_tracker.get_performance(
                constraint_type, window=5
            )
            if len(self.strictness_history) > 0:
                prev_performance = self.strictness_history[-1]['metrics'].get(
                    f'{constraint_type}_score', 1.0
                )
                if recent_performance < prev_performance - self.config.degradation_threshold:
                    logger.warning(
                        f"Freezing {constraint_type} constraint due to "
                        f"performance degradation: {recent_performance:.3f}"
                    )
                    # Don't change strictness

    def get_stats(self) -> Dict:
        """Get current constraint statistics."""
        stats = {
            'epoch': self.epoch,
            'strictness': self.strictness.copy()
        }

        # Add performance stats
        for constraint_type in self.strictness:
            stats[f'{constraint_type}_performance'] = (
                self.performance_tracker.get_performance(constraint_type)
            )
            stats[f'{constraint_type}_success_rate'] = (
                self.performance_tracker.get_success_rate(constraint_type)
            )

        return stats


class HybridConstraintSystem(nn.Module):
    """
    Combines dissolvable guidance with adaptive constraints for
    comprehensive training support.
    """

    def __init__(
        self,
        dissolvable_config,
        adaptive_config
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from .dissolvable_guidance import DissolvableGuidanceModule

        # Initialize both systems
        from ..data.representation import VocabularyConfig
        vocab_config = VocabularyConfig()

        self.dissolvable_guidance = DissolvableGuidanceModule(
            vocab_config, dissolvable_config
        )
        self.adaptive_constraints = AdaptiveConstraintEngine(
            adaptive_config
        )

    def forward(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        epoch: int,
        performance_metrics: Optional[Dict[str, float]] = None,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply both guidance and constraints.

        Args:
            logits: Model output logits
            sequence: Generated sequence
            epoch: Current epoch
            performance_metrics: Current performance metrics
            training: Whether in training mode

        Returns:
            Modified logits
        """
        # Apply dissolvable guidance first (stronger early training support)
        if training and performance_metrics is not None:
            performance = performance_metrics.get('note_pairing_score', 0.0)
            logits = self.dissolvable_guidance(
                logits, sequence, epoch, performance
            )

        # Then apply adaptive constraints (fine-tuning)
        logits = self.adaptive_constraints(
            logits, sequence, training
        )

        # Update adaptive constraints if we have metrics
        if training and performance_metrics is not None:
            self.adaptive_constraints.update_epoch(
                epoch, performance_metrics
            )

        return logits

    def get_stats(self) -> Dict:
        """Get combined statistics."""
        return {
            'guidance': self.dissolvable_guidance.get_guidance_stats(),
            'constraints': self.adaptive_constraints.get_stats()
        }