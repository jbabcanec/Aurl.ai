"""
Dissolvable Guidance Architecture for Musical Training

This module implements a training guidance system that provides strong
constraints early in training but gradually dissolves as the model learns,
similar to training wheels that eventually come off.

Key Features:
- Forced note pairing in early epochs
- Gradual relaxation of constraints based on performance
- Automatic detection of when guidance is no longer needed
- Smooth transition from guided to free generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import math

from ..data.representation import VocabularyConfig
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DissolvableGuidanceConfig:
    """Configuration for dissolvable guidance system."""

    # Guidance strength parameters
    initial_guidance_strength: float = 1.0  # Full guidance at start
    minimum_guidance_strength: float = 0.0  # No guidance when fully learned

    # Dissolution parameters
    dissolution_start_epoch: int = 10  # When to start reducing guidance
    dissolution_rate: float = 0.95  # Exponential decay rate per epoch
    performance_threshold: float = 0.85  # Performance level to reduce guidance

    # Forced pairing parameters
    force_note_pairing: bool = True
    pairing_temperature: float = 0.1  # Low temp for forced pairing

    # Adaptive parameters
    adaptive_dissolution: bool = True
    performance_window: int = 5  # Epochs to average performance
    acceleration_factor: float = 1.5  # Speed up dissolution if doing well
    deceleration_factor: float = 0.7  # Slow down if struggling

    # Safety parameters
    minimum_epochs_before_dissolution: int = 20
    rollback_on_collapse: bool = True
    collapse_detection_threshold: float = 0.5

    # Monitoring
    log_guidance_level: bool = True
    save_guidance_history: bool = True


class GuidanceState:
    """Tracks the current state of guidance during training."""

    def __init__(self, config: DissolvableGuidanceConfig):
        self.config = config
        self.current_strength = config.initial_guidance_strength
        self.epoch = 0
        self.performance_history = deque(maxlen=config.performance_window)
        self.guidance_history = []
        self.is_dissolved = False

    def update(self, epoch: int, performance: float):
        """Update guidance state based on training progress."""
        self.epoch = epoch
        self.performance_history.append(performance)
        self.guidance_history.append({
            'epoch': epoch,
            'strength': self.current_strength,
            'performance': performance
        })

        # Check if guidance should be dissolved
        if self.should_dissolve():
            self.dissolve_step()

    def should_dissolve(self) -> bool:
        """Determine if guidance should be reduced."""
        if self.is_dissolved:
            return False

        if self.epoch < self.config.minimum_epochs_before_dissolution:
            return False

        if len(self.performance_history) < self.config.performance_window:
            return False

        avg_performance = np.mean(self.performance_history)
        return avg_performance >= self.config.performance_threshold

    def dissolve_step(self):
        """Reduce guidance strength by one step."""
        if self.config.adaptive_dissolution:
            # Adjust dissolution rate based on performance
            avg_performance = np.mean(self.performance_history)
            if avg_performance > 0.9:
                # Doing great, dissolve faster
                rate = self.config.dissolution_rate * self.config.acceleration_factor
            elif avg_performance < 0.7:
                # Struggling, dissolve slower
                rate = self.config.dissolution_rate * self.config.deceleration_factor
            else:
                rate = self.config.dissolution_rate
        else:
            rate = self.config.dissolution_rate

        self.current_strength *= rate

        if self.current_strength <= self.config.minimum_guidance_strength:
            self.current_strength = self.config.minimum_guidance_strength
            self.is_dissolved = True
            logger.info(f"Guidance fully dissolved at epoch {self.epoch}")


class DissolvableGuidanceModule(nn.Module):
    """
    Neural module that provides dissolvable guidance during training.

    This module modifies logits during generation to enforce musical rules
    with decreasing strength as training progresses.
    """

    def __init__(self, vocab_config: VocabularyConfig, config: DissolvableGuidanceConfig):
        super().__init__()
        self.vocab_config = vocab_config
        self.config = config
        self.state = GuidanceState(config)

        # Create guidance masks
        self._create_guidance_masks()

        # Track active notes for forced pairing
        self.active_notes = {}  # batch_idx -> set of active pitches

    def _create_guidance_masks(self):
        """Create reusable masks for guidance."""
        vocab_size = self.vocab_config.total_size

        # Note token ranges
        self.note_on_start = 13
        self.note_on_end = 141
        self.note_off_start = 141
        self.note_off_end = 269

        # Create pitch mapping
        self.note_on_to_pitch = torch.arange(128)
        self.note_off_to_pitch = torch.arange(128)

    def forward(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        epoch: int,
        performance: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply dissolvable guidance to logits.

        Args:
            logits: Model output logits [batch_size, vocab_size]
            sequence: Generated sequence so far [batch_size, seq_len]
            epoch: Current training epoch
            performance: Current model performance metric

        Returns:
            Modified logits with guidance applied
        """
        # Update guidance state
        if performance is not None:
            self.state.update(epoch, performance)

        # Apply guidance with current strength
        if self.state.current_strength > 0:
            logits = self._apply_guidance(logits, sequence, self.state.current_strength)

        return logits

    def _apply_guidance(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """Apply guidance with specified strength."""
        device = logits.device
        batch_size = logits.shape[0]

        # Track active notes for each batch item
        for batch_idx in range(batch_size):
            if batch_idx not in self.active_notes:
                self.active_notes[batch_idx] = set()

            # Update active notes based on sequence
            self._update_active_notes(sequence[batch_idx], batch_idx)

        # Apply forced note pairing if configured
        if self.config.force_note_pairing and strength > 0:
            logits = self._force_note_pairing(logits, strength)

        # Apply velocity guidance
        logits = self._guide_velocity(logits, sequence, strength)

        # Apply timing guidance
        logits = self._guide_timing(logits, sequence, strength)

        return logits

    def _update_active_notes(self, sequence: torch.Tensor, batch_idx: int):
        """Track which notes are currently active."""
        # Look at recent tokens to track note states
        for token in sequence[-10:]:  # Check last 10 tokens
            token_id = token.item()

            # Check for NOTE_ON
            if self.note_on_start <= token_id < self.note_on_end:
                pitch = token_id - self.note_on_start
                self.active_notes[batch_idx].add(pitch)

            # Check for NOTE_OFF
            elif self.note_off_start <= token_id < self.note_off_end:
                pitch = token_id - self.note_off_start
                self.active_notes[batch_idx].discard(pitch)

    def _force_note_pairing(
        self,
        logits: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """Force generation of NOTE_OFF for active notes."""
        batch_size = logits.shape[0]

        for batch_idx in range(batch_size):
            active = self.active_notes.get(batch_idx, set())

            if len(active) > 0:
                # Boost probability of NOTE_OFF for active notes
                for pitch in active:
                    note_off_token = self.note_off_start + pitch

                    # Apply boost proportional to guidance strength
                    boost = 5.0 * strength  # Strong boost when guidance is high
                    logits[batch_idx, note_off_token] += boost

                # Reduce probability of new NOTE_ON when many notes active
                if len(active) > 4:  # Threshold for "too many" active notes
                    penalty = 3.0 * strength * (len(active) / 4)
                    logits[batch_idx, self.note_on_start:self.note_on_end] -= penalty

        return logits

    def _guide_velocity(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """Guide velocity token generation."""
        # Encourage velocity changes at appropriate times
        velocity_start = 369
        velocity_end = 497

        # Check if recent tokens lack velocity changes
        recent_tokens = sequence[:, -20:] if sequence.shape[1] > 20 else sequence
        has_recent_velocity = ((recent_tokens >= velocity_start) &
                              (recent_tokens < velocity_end)).any(dim=1)

        # Boost velocity tokens if none recent
        for batch_idx in range(logits.shape[0]):
            if not has_recent_velocity[batch_idx]:
                boost = 2.0 * strength
                logits[batch_idx, velocity_start:velocity_end] += boost

        return logits

    def _guide_timing(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """Guide timing token generation."""
        time_shift_start = 269
        time_shift_end = 369

        # Prevent excessive time shifts
        recent_tokens = sequence[:, -5:] if sequence.shape[1] > 5 else sequence
        time_shift_count = ((recent_tokens >= time_shift_start) &
                           (recent_tokens < time_shift_end)).sum(dim=1)

        # Penalize if too many recent time shifts
        for batch_idx in range(logits.shape[0]):
            if time_shift_count[batch_idx] > 3:
                penalty = 2.0 * strength
                logits[batch_idx, time_shift_start:time_shift_end] -= penalty

        return logits

    def get_guidance_stats(self) -> Dict:
        """Get current guidance statistics."""
        return {
            'current_strength': self.state.current_strength,
            'epoch': self.state.epoch,
            'is_dissolved': self.state.is_dissolved,
            'avg_performance': np.mean(self.state.performance_history)
                               if len(self.state.performance_history) > 0 else 0.0,
            'history_length': len(self.state.guidance_history)
        }


class AdaptiveGuidanceScheduler:
    """
    Schedules guidance dissolution based on multiple performance metrics.
    """

    def __init__(self, config: DissolvableGuidanceConfig):
        self.config = config
        self.metrics_history = []

    def compute_dissolution_rate(
        self,
        grammar_score: float,
        note_pairing_score: float,
        loss: float,
        epoch: int
    ) -> float:
        """
        Compute adaptive dissolution rate based on multiple metrics.

        Args:
            grammar_score: Overall grammar quality (0-1)
            note_pairing_score: Note ON/OFF pairing quality (0-1)
            loss: Current training loss
            epoch: Current epoch number

        Returns:
            Dissolution rate multiplier
        """
        # Store metrics
        self.metrics_history.append({
            'epoch': epoch,
            'grammar_score': grammar_score,
            'note_pairing_score': note_pairing_score,
            'loss': loss
        })

        # Compute composite performance score
        performance = self._compute_composite_score(
            grammar_score, note_pairing_score, loss
        )

        # Determine dissolution rate based on performance
        if performance > 0.9:
            # Excellent - accelerate dissolution
            return self.config.acceleration_factor
        elif performance > 0.8:
            # Good - normal dissolution
            return 1.0
        elif performance > 0.6:
            # Okay - slow dissolution
            return self.config.deceleration_factor
        else:
            # Poor - pause dissolution
            return 0.0

    def _compute_composite_score(
        self,
        grammar_score: float,
        note_pairing_score: float,
        loss: float
    ) -> float:
        """Compute weighted composite performance score."""
        # Normalize loss (assuming loss < 1.0 is good)
        loss_score = max(0, 1.0 - loss)

        # Weighted average with note pairing as most important
        weights = {
            'note_pairing': 0.5,
            'grammar': 0.3,
            'loss': 0.2
        }

        composite = (
            weights['note_pairing'] * note_pairing_score +
            weights['grammar'] * grammar_score +
            weights['loss'] * loss_score
        )

        return composite

    def should_rollback(self) -> bool:
        """Check if training should rollback due to collapse."""
        if len(self.metrics_history) < 3:
            return False

        # Check recent metrics for collapse indicators
        recent = self.metrics_history[-3:]

        # Collapse if note pairing drops below threshold
        for metric in recent:
            if metric['note_pairing_score'] < self.config.collapse_detection_threshold:
                logger.warning(f"Collapse detected: note_pairing_score = {metric['note_pairing_score']}")
                return True

        # Collapse if grammar score declining rapidly
        if len(self.metrics_history) > 5:
            recent_grammar = [m['grammar_score'] for m in self.metrics_history[-5:]]
            if all(recent_grammar[i] < recent_grammar[i-1] for i in range(1, len(recent_grammar))):
                logger.warning("Collapse detected: consistent grammar decline")
                return True

        return False