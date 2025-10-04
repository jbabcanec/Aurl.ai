"""
Optimized Musical Constraints Module

This module provides a vectorized, cache-friendly implementation of musical
constraints that avoids O(vocab_size) iterations during generation.

Key optimizations:
- Precomputed constraint masks for all token types
- Vectorized operations instead of loops
- Cached constraint tensors for repeated patterns
- Batch processing for multiple constraints
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import functools
from collections import defaultdict

from ..utils.constants import SPECIAL_TOKENS
from ..data.representation import EventType, VocabularyConfig
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OptimizedConstraintConfig:
    """Configuration for optimized musical constraints."""
    # Enable/disable constraint types
    use_harmonic_constraints: bool = True
    use_melodic_constraints: bool = True
    use_rhythmic_constraints: bool = True
    use_dynamic_constraints: bool = True
    use_structural_constraints: bool = True

    # Optimization settings
    precompute_masks: bool = True
    cache_size: int = 1000  # Number of cached constraint masks
    vectorize_operations: bool = True

    # Constraint parameters (same as original)
    max_leap_interval: int = 12
    dynamic_range: Tuple[int, int] = (20, 100)
    phrase_length_range: Tuple[int, int] = (16, 64)
    dissonance_threshold: float = 0.3
    voice_leading_strictness: float = 0.7


class OptimizedConstraintEngine:
    """
    Optimized engine for applying musical constraints during generation.

    This implementation uses vectorized operations and precomputed masks
    to dramatically reduce computational overhead.
    """

    def __init__(self, config: OptimizedConstraintConfig = None):
        """Initialize the optimized constraint engine."""
        self.config = config or OptimizedConstraintConfig()
        self.vocab_config = VocabularyConfig()
        self.vocab_size = self.vocab_config.total_size

        # Precompute token type masks for fast filtering
        self._precompute_token_masks()

        # Cache for frequently used constraint masks
        self.constraint_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Precompute constraint tensors
        if self.config.precompute_masks:
            self._precompute_constraint_tensors()

    def _precompute_token_masks(self):
        """Precompute boolean masks for different token types."""
        # Create masks for each token type
        self.note_on_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.note_off_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.velocity_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.time_shift_mask = torch.zeros(self.vocab_size, dtype=torch.bool)

        # Fill masks based on vocabulary configuration
        # NOTE_ON tokens: indices 13-140 (128 pitches)
        self.note_on_mask[13:141] = True

        # NOTE_OFF tokens: indices 141-268 (128 pitches)
        self.note_off_mask[141:269] = True

        # TIME_SHIFT tokens: indices 269-368 (100 time values)
        self.time_shift_mask[269:369] = True

        # VELOCITY tokens: indices 369-496 (128 velocity values)
        self.velocity_mask[369:497] = True

        # Special tokens mask
        self.special_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        self.special_mask[:13] = True  # First 13 tokens are special

        # Pitch lookup tensors for fast conversion
        self.note_on_to_pitch = torch.arange(128)  # token_id - 13 = pitch
        self.note_off_to_pitch = torch.arange(128)  # token_id - 141 = pitch
        self.velocity_to_value = torch.arange(128)  # token_id - 369 = velocity

    def _precompute_constraint_tensors(self):
        """Precompute commonly used constraint tensors."""
        # Harmonic constraint tensors
        self._precompute_harmonic_tensors()

        # Melodic constraint tensors
        self._precompute_melodic_tensors()

        # Dynamic constraint tensors
        self._precompute_dynamic_tensors()

        logger.info("Precomputed constraint tensors for fast application")

    def _precompute_harmonic_tensors(self):
        """Precompute harmonic constraint tensors."""
        # Dissonant interval masks (minor 2nd, tritone, etc.)
        dissonant_intervals = {1, 6, 11}  # semitones

        # Create dissonance matrix for all pitch pairs
        self.dissonance_matrix = torch.zeros(128, 128, dtype=torch.float32)
        for pitch1 in range(128):
            for pitch2 in range(128):
                interval = abs(pitch1 - pitch2) % 12
                if interval in dissonant_intervals:
                    self.dissonance_matrix[pitch1, pitch2] = 1.0

    def _precompute_melodic_tensors(self):
        """Precompute melodic constraint tensors."""
        # Interval penalty matrix for large leaps
        self.interval_penalties = torch.zeros(128, 128, dtype=torch.float32)
        for pitch1 in range(128):
            for pitch2 in range(128):
                interval = abs(pitch1 - pitch2)
                if interval > self.config.max_leap_interval:
                    # Exponential penalty for large intervals
                    self.interval_penalties[pitch1, pitch2] = 2.0 * (interval / self.config.max_leap_interval)
                elif interval <= 2:  # Stepwise motion bonus
                    self.interval_penalties[pitch1, pitch2] = -0.5

    def _precompute_dynamic_tensors(self):
        """Precompute dynamic constraint tensors."""
        # Valid velocity mask
        self.valid_velocity_mask = torch.zeros(128, dtype=torch.bool)
        min_vel, max_vel = self.config.dynamic_range
        self.valid_velocity_mask[min_vel:max_vel+1] = True

        # Smooth dynamics penalty
        self.velocity_smoothness = torch.zeros(128, 128, dtype=torch.float32)
        for vel1 in range(128):
            for vel2 in range(128):
                diff = abs(vel1 - vel2)
                if diff > 20:
                    self.velocity_smoothness[vel1, vel2] = diff / 20.0

    def apply_constraints(
        self,
        logits: torch.Tensor,
        generated_sequence: torch.Tensor,
        step: int,
        context: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Apply all active constraints to generation logits.

        This optimized version uses vectorized operations and precomputed
        masks to avoid iterating through the vocabulary.

        Args:
            logits: Raw logits from model [batch_size, vocab_size]
            generated_sequence: Previously generated tokens
            step: Current generation step
            context: Additional context for constraints

        Returns:
            Constrained logits tensor
        """
        # Move tensors to same device as logits
        device = logits.device

        # Apply constraints using vectorized operations
        if self.config.use_harmonic_constraints:
            logits = self._apply_harmonic_constraints_vectorized(
                logits, generated_sequence, device
            )

        if self.config.use_melodic_constraints:
            logits = self._apply_melodic_constraints_vectorized(
                logits, generated_sequence, device
            )

        if self.config.use_rhythmic_constraints:
            logits = self._apply_rhythmic_constraints_vectorized(
                logits, generated_sequence, device
            )

        if self.config.use_dynamic_constraints:
            logits = self._apply_dynamic_constraints_vectorized(
                logits, generated_sequence, device
            )

        if self.config.use_structural_constraints:
            logits = self._apply_structural_constraints_vectorized(
                logits, generated_sequence, step, device
            )

        return logits

    def _apply_harmonic_constraints_vectorized(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply harmonic constraints using vectorized operations."""
        # Extract active notes from sequence
        active_notes = self._get_active_notes_vectorized(sequence)

        if len(active_notes) > 0:
            # Move precomputed tensors to device
            note_on_mask = self.note_on_mask.to(device)
            dissonance_matrix = self.dissonance_matrix.to(device)

            # Calculate dissonance penalties for all note_on tokens at once
            # Shape: [batch_size, 128] for note pitches
            dissonance_penalties = torch.zeros(logits.shape[0], 128, device=device)

            for active_pitch in active_notes:
                # Get dissonance values for this active pitch
                dissonance_penalties += dissonance_matrix[active_pitch]

            # Apply penalties to note_on tokens (indices 13-140)
            # Use advanced indexing to update only note_on tokens
            logits[:, 13:141] -= dissonance_penalties * self.config.dissonance_threshold

        return logits

    def _apply_melodic_constraints_vectorized(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply melodic constraints using vectorized operations."""
        # Get last melodic pitch
        last_pitch = self._get_last_pitch_vectorized(sequence)

        if last_pitch is not None:
            # Move precomputed tensors to device
            interval_penalties = self.interval_penalties.to(device)

            # Get penalties for all possible next pitches
            penalties = interval_penalties[last_pitch]  # Shape: [128]

            # Apply to note_on tokens
            logits[:, 13:141] -= penalties.unsqueeze(0)  # Broadcast to batch

        return logits

    def _apply_rhythmic_constraints_vectorized(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply rhythmic constraints using vectorized operations."""
        # Calculate beat position
        beat_position = self._get_beat_position_vectorized(sequence)

        # Create rhythm emphasis mask
        rhythm_mask = torch.zeros_like(logits)

        # Emphasize notes on strong beats
        if beat_position % 4 == 0:  # Downbeat
            rhythm_mask[:, 13:141] = 0.5  # Boost note_on tokens
        elif beat_position % 2 == 0:  # Mid-bar beat
            rhythm_mask[:, 13:141] = 0.2

        logits += rhythm_mask

        return logits

    def _apply_dynamic_constraints_vectorized(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Apply dynamic constraints using vectorized operations."""
        # Get recent average velocity
        avg_velocity = self._get_average_velocity_vectorized(sequence)

        if avg_velocity is not None:
            # Move precomputed tensors to device
            valid_velocity_mask = self.valid_velocity_mask.to(device)

            # Mask out invalid velocities
            velocity_logits_mask = torch.ones_like(logits)
            velocity_logits_mask[:, 369:497] = 0  # Zero out velocity tokens

            # Add back only valid velocities
            valid_indices = 369 + torch.where(valid_velocity_mask)[0]
            velocity_logits_mask[:, valid_indices] = 1

            # Apply mask (set invalid to -inf)
            logits = torch.where(
                velocity_logits_mask.bool(),
                logits,
                torch.full_like(logits, -float('inf'))
            )

            # Apply smoothness penalties
            if avg_velocity < 128:
                smoothness = self.velocity_smoothness[int(avg_velocity)].to(device)
                logits[:, 369:497] -= smoothness.unsqueeze(0) * 0.5

        return logits

    def _apply_structural_constraints_vectorized(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        device: torch.device
    ) -> torch.Tensor:
        """Apply structural constraints using vectorized operations."""
        # Check phrase length
        min_phrase, max_phrase = self.config.phrase_length_range

        if step >= min_phrase:
            # Increase probability of phrase-ending tokens
            # (e.g., longer time shifts, rests)
            logits[:, 269:369] += 0.3  # Boost time shift tokens

            if step >= max_phrase:
                # Strongly encourage phrase ending
                logits[:, 1] += 2.0  # Boost END token
                logits[:, 269:369] += 0.5  # Further boost time shifts

        return logits

    # Utility methods for vectorized operations

    def _get_active_notes_vectorized(self, sequence: torch.Tensor) -> List[int]:
        """Extract currently active note pitches from sequence."""
        # This would be implemented based on tracking NOTE_ON/OFF events
        # For now, return empty list as placeholder
        return []

    def _get_last_pitch_vectorized(self, sequence: torch.Tensor) -> Optional[int]:
        """Get the last melodic pitch from sequence."""
        # Find last NOTE_ON token in sequence
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)

        # Look for NOTE_ON tokens (13-140)
        note_on_positions = ((sequence >= 13) & (sequence < 141)).nonzero(as_tuple=True)

        if len(note_on_positions[1]) > 0:
            last_note_on_idx = note_on_positions[1][-1].item()
            last_token = sequence[0, last_note_on_idx].item()
            return last_token - 13  # Convert to pitch

        return None

    def _get_beat_position_vectorized(self, sequence: torch.Tensor) -> int:
        """Calculate current beat position from time shifts."""
        # Sum all TIME_SHIFT tokens to get current position
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)

        time_shifts = ((sequence >= 269) & (sequence < 369))
        total_time = time_shifts.sum().item()

        return total_time % 16  # Assume 16 steps per bar

    def _get_average_velocity_vectorized(self, sequence: torch.Tensor) -> Optional[float]:
        """Get average velocity from recent velocity tokens."""
        if len(sequence.shape) == 1:
            sequence = sequence.unsqueeze(0)

        # Find VELOCITY tokens (369-496)
        velocity_mask = ((sequence >= 369) & (sequence < 497))
        velocity_tokens = sequence[velocity_mask]

        if len(velocity_tokens) > 0:
            # Convert tokens to velocity values
            velocities = velocity_tokens - 369
            return velocities.float().mean().item()

        return None

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.constraint_cache)
        }