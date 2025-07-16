"""
Musical Grammar Training Module

This module implements musical grammar loss functions and validation
to ensure generated tokens form proper musical sequences.

Key Components:
- Musical Grammar Loss Functions
- Note On/Off Pairing Validation
- Sequence-level Musical Coherence Scoring
- Real-time Grammar Checking During Training

Phase 5.1 of the Aurl.ai GAMEPLAN - TOP PRIORITY
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from ..data.representation import EventType, VocabularyConfig
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MusicalGrammarConfig:
    """Configuration for musical grammar validation."""
    # Grammar validation weights
    note_pairing_weight: float = 10.0      # Weight for note on/off pairing loss
    sequence_coherence_weight: float = 5.0  # Weight for sequence coherence
    repetition_penalty_weight: float = 3.0  # Weight for repetition penalty
    velocity_quality_weight: float = 5.0    # Weight for velocity parameter quality
    timing_quality_weight: float = 3.0      # Weight for timing parameter quality
    
    # Validation thresholds
    min_note_duration: float = 0.05        # Minimum note duration (seconds)
    max_note_duration: float = 10.0        # Maximum note duration (seconds)
    max_simultaneous_notes: int = 8        # Maximum polyphony
    
    # Training integration
    validate_every_n_batches: int = 100    # Frequency of real-time validation
    grammar_score_threshold: float = 0.8   # Minimum acceptable grammar score
    early_stop_patience: int = 5           # Batches to wait before stopping


class MusicalGrammarLoss(nn.Module):
    """
    Musical Grammar Loss Functions for Training.
    
    Ensures generated tokens form proper musical sequences with:
    - Matching note on/off pairs
    - Reasonable timing relationships
    - Coherent musical structure
    """
    
    def __init__(self, config: MusicalGrammarConfig = None, vocab_config: VocabularyConfig = None):
        super().__init__()
        self.config = config or MusicalGrammarConfig()
        self.vocab_config = vocab_config or VocabularyConfig()
        
        # Track statistics
        self.grammar_stats = {
            "total_sequences": 0,
            "valid_sequences": 0,
            "note_pairing_errors": 0,
            "timing_errors": 0,
            "repetition_errors": 0
        }
    
    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        base_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute musical grammar loss.
        
        Args:
            predicted_tokens: Model predictions [batch_size, seq_len, vocab_size]
            target_tokens: Target token sequence [batch_size, seq_len]
            base_loss: Base language modeling loss
            
        Returns:
            total_loss: Combined loss with grammar penalties
            loss_components: Dictionary of individual loss components
        """
        batch_size = target_tokens.size(0)
        
        # Convert predictions to token sequences for analysis
        pred_token_ids = predicted_tokens.argmax(dim=-1)  # [batch_size, seq_len]
        
        # Calculate grammar losses
        note_pairing_loss = self._note_pairing_loss(pred_token_ids, target_tokens)
        sequence_coherence_loss = self._sequence_coherence_loss(pred_token_ids)
        repetition_penalty = self._repetition_penalty_loss(pred_token_ids)
        
        # Add parameter quality losses (NEW - addresses velocity=1 and duration=0 issues)
        velocity_quality_loss = self._velocity_parameter_loss(pred_token_ids)
        timing_quality_loss = self._timing_parameter_loss(pred_token_ids)
        
        # Combine losses (including new parameter quality losses)
        total_grammar_loss = (
            self.config.note_pairing_weight * note_pairing_loss +
            self.config.sequence_coherence_weight * sequence_coherence_loss +
            self.config.repetition_penalty_weight * repetition_penalty +
            5.0 * velocity_quality_loss +  # High weight - silent notes are unusable
            3.0 * timing_quality_loss      # High weight - zero duration notes are invisible
        )
        
        total_loss = base_loss + total_grammar_loss
        
        # Track statistics
        self.grammar_stats["total_sequences"] += batch_size
        
        loss_components = {
            "base_loss": base_loss.item(),
            "note_pairing_loss": note_pairing_loss.item(),
            "sequence_coherence_loss": sequence_coherence_loss.item(),
            "repetition_penalty": repetition_penalty.item(),
            "velocity_quality_loss": velocity_quality_loss.item(),
            "timing_quality_loss": timing_quality_loss.item(),
            "total_grammar_loss": total_grammar_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _note_pairing_loss(
        self, 
        pred_tokens: torch.Tensor, 
        target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate loss for improper note on/off pairing.
        
        Penalizes sequences where NOTE_ON tokens don't have matching NOTE_OFF tokens
        with the same pitch value, addressing the root cause of mismatched note pairs.
        """
        batch_size, seq_len = pred_tokens.shape
        total_loss = torch.tensor(0.0, device=pred_tokens.device)
        
        for batch_idx in range(batch_size):
            # Analyze both predicted and target sequences
            for tokens in [pred_tokens[batch_idx], target_tokens[batch_idx]]:
                note_states = {}  # {pitch: note_on_count}
                note_sequence = []  # Track order for better matching validation
                
                for token in tokens:
                    token_id = token.item()
                    event_type, value = self.vocab_config.token_to_event_info(token_id)
                    
                    if event_type == EventType.NOTE_ON:
                        note_states[value] = note_states.get(value, 0) + 1
                        note_sequence.append(('on', value))
                    elif event_type == EventType.NOTE_OFF:
                        if value in note_states and note_states[value] > 0:
                            note_states[value] -= 1
                            note_sequence.append(('off', value))
                        else:
                            # NOTE_OFF without matching NOTE_ON (critical error)
                            total_loss += 2.0  # Higher penalty for unmatched off
                            note_sequence.append(('off_orphan', value))
                
                # Penalize unmatched NOTE_ON events (critical error)
                for pitch, count in note_states.items():
                    if count > 0:
                        total_loss += count * 1.5  # Penalty for unmatched on
                        self.grammar_stats["note_pairing_errors"] += count
                
                # Additional penalty for poor note sequence structure
                # Encourage proper note on/off alternation
                pitch_last_event = {}
                for event_type, pitch in note_sequence:
                    if pitch in pitch_last_event:
                        last_event = pitch_last_event[pitch]
                        # Penalize consecutive NOTE_ON or NOTE_OFF for same pitch
                        if (event_type == 'on' and last_event == 'on') or \
                           (event_type == 'off' and last_event == 'off'):
                            total_loss += 0.5
                    pitch_last_event[pitch] = event_type
        
        return total_loss / batch_size
    
    def _sequence_coherence_loss(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for incoherent musical sequences.
        
        Penalizes sequences with poor musical structure.
        """
        batch_size = pred_tokens.size(0)
        total_loss = torch.tensor(0.0, device=pred_tokens.device)
        
        for batch_idx in range(batch_size):
            tokens = pred_tokens[batch_idx]
            
            # Check for basic musical structure
            has_notes = False
            has_timing = False
            
            for token in tokens:
                token_id = token.item()
                event_type, _ = self.vocab_config.token_to_event_info(token_id)
                
                if event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
                    has_notes = True
                elif event_type == EventType.TIME_SHIFT:
                    has_timing = True
            
            # Penalize sequences without basic musical elements
            if not has_notes:
                total_loss += 5.0
            if not has_timing:
                total_loss += 2.0
        
        return total_loss / batch_size
    
    def _repetition_penalty_loss(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for excessive repetition.
        
        Penalizes sequences that repeat the same token excessively.
        """
        batch_size = pred_tokens.size(0)
        total_loss = torch.tensor(0.0, device=pred_tokens.device)
        
        for batch_idx in range(batch_size):
            tokens = pred_tokens[batch_idx].cpu().numpy()
            
            # Count consecutive repetitions
            max_repetition = 1
            current_repetition = 1
            
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i-1]:
                    current_repetition += 1
                    max_repetition = max(max_repetition, current_repetition)
                else:
                    current_repetition = 1
            
            # Penalize excessive repetition (more than 3 consecutive identical tokens)
            if max_repetition > 3:
                repetition_penalty = (max_repetition - 3) ** 2
                total_loss += repetition_penalty
                self.grammar_stats["repetition_errors"] += 1
        
        return total_loss / batch_size
    
    def _velocity_parameter_loss(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for velocity tokens that produce unusable MIDI parameters.
        
        Penalizes tokens that map to velocity=1 (almost silent) or velocity=0.
        Encourages tokens that produce audible velocities (64-127 range).
        """
        batch_size = pred_tokens.size(0)
        total_loss = torch.tensor(0.0, device=pred_tokens.device)
        
        for batch_idx in range(batch_size):
            tokens = pred_tokens[batch_idx]
            
            for token in tokens:
                token_id = token.item()
                event_type, value = self.vocab_config.token_to_event_info(token_id)
                
                if event_type == EventType.VELOCITY_CHANGE:
                    # Convert token value to MIDI velocity using VocabularyConfig logic
                    # Based on diagnosis: velocity = int((token_value / (velocity_bins - 1)) * 126) + 1
                    velocity_bins = getattr(self.vocab_config, 'velocity_bins', 32)
                    midi_velocity = int((value / (velocity_bins - 1)) * 126) + 1
                    
                    # Heavy penalty for silent/near-silent velocities (1-20)
                    if midi_velocity <= 20:
                        total_loss += 3.0  # Critical error - unusable notes
                    # Moderate penalty for very quiet velocities (21-40)
                    elif midi_velocity <= 40:
                        total_loss += 1.0
                    # Small penalty for quiet but usable velocities (41-63)
                    elif midi_velocity <= 63:
                        total_loss += 0.2
                    # No penalty for good velocities (64-127)
        
        return total_loss / batch_size
    
    def _timing_parameter_loss(self, pred_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss for timing tokens that produce unusable MIDI parameters.
        
        Penalizes tokens that map to 0.0s duration (invisible notes).
        Encourages tokens that produce visible durations (0.1-2.0s range).
        """
        batch_size = pred_tokens.size(0)
        total_loss = torch.tensor(0.0, device=pred_tokens.device)
        
        for batch_idx in range(batch_size):
            tokens = pred_tokens[batch_idx]
            
            for token in tokens:
                token_id = token.item()
                event_type, value = self.vocab_config.token_to_event_info(token_id)
                
                if event_type == EventType.TIME_SHIFT:
                    # Convert token value to duration using VocabularyConfig logic
                    # Based on diagnosis: time_delta = token_value * time_shift_ms / 1000.0
                    time_shift_ms = getattr(self.vocab_config, 'time_shift_ms', 15.625)
                    duration_seconds = value * time_shift_ms / 1000.0
                    
                    # Heavy penalty for zero or near-zero durations (invisible notes)
                    if duration_seconds <= 0.01:  # Less than 10ms
                        total_loss += 2.0  # Critical error - invisible notes
                    # Moderate penalty for very short durations (10-50ms)
                    elif duration_seconds <= 0.05:
                        total_loss += 0.5
                    # Small penalty for short but visible durations (50-100ms)
                    elif duration_seconds <= 0.1:
                        total_loss += 0.1
                    # No penalty for good durations (100ms+)
        
        return total_loss / batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get musical grammar statistics."""
        stats = self.grammar_stats.copy()
        if stats["total_sequences"] > 0:
            stats["grammar_success_rate"] = stats["valid_sequences"] / stats["total_sequences"]
        else:
            stats["grammar_success_rate"] = 0.0
        return stats


class MusicalGrammarValidator:
    """
    Real-time musical grammar validation during training.
    
    Provides real-time feedback on musical quality and can trigger
    model rollbacks if grammar quality degrades.
    """
    
    def __init__(self, config: MusicalGrammarConfig = None):
        self.config = config or MusicalGrammarConfig()
        self.vocab_config = VocabularyConfig()
        
        # Validation history
        self.validation_history = []
        self.best_score = 0.0
        self.patience_counter = 0
        
    def validate_sequence(self, tokens: np.ndarray) -> Dict[str, Any]:
        """
        Validate a single token sequence for musical grammar.
        
        Args:
            tokens: Token sequence to validate
            
        Returns:
            Dictionary with validation results and scores
        """
        validation_result = {
            "is_valid": True,
            "grammar_score": 1.0,
            "errors": [],
            "note_count": 0,
            "note_pairing_score": 1.0,
            "timing_score": 1.0,
            "repetition_score": 1.0,
            "velocity_quality_score": 1.0,
            "timing_quality_score": 1.0
        }
        
        # Validate note on/off pairing
        note_pairing_result = self._validate_note_pairing(tokens)
        validation_result.update(note_pairing_result)
        
        # Validate timing structure
        timing_result = self._validate_timing(tokens)
        validation_result.update(timing_result)
        
        # Validate repetition
        repetition_result = self._validate_repetition(tokens)
        validation_result.update(repetition_result)
        
        # NEW: Validate parameter quality (addresses velocity=1 and duration=0 issues)
        velocity_quality_result = self._validate_velocity_parameters(tokens)
        validation_result.update(velocity_quality_result)
        
        timing_quality_result = self._validate_timing_parameters(tokens)
        validation_result.update(timing_quality_result)
        
        # Calculate overall grammar score (includes new parameter quality scores)
        validation_result["grammar_score"] = (
            validation_result["note_pairing_score"] * 0.3 +
            validation_result["timing_score"] * 0.15 +
            validation_result["repetition_score"] * 0.15 +
            validation_result["velocity_quality_score"] * 0.25 +  # High weight - unusable if silent
            validation_result["timing_quality_score"] * 0.15     # Medium weight - unusable if zero duration
        )
        
        validation_result["is_valid"] = (
            validation_result["grammar_score"] >= self.config.grammar_score_threshold
        )
        
        return validation_result
    
    def _validate_note_pairing(self, tokens: np.ndarray) -> Dict[str, Any]:
        """Validate note on/off pairing in sequence."""
        note_states = {}
        total_notes = 0
        pairing_errors = 0
        
        for token in tokens:
            event_type, value = self.vocab_config.token_to_event_info(int(token))
            
            if event_type == EventType.NOTE_ON:
                note_states[value] = note_states.get(value, 0) + 1
                total_notes += 1
            elif event_type == EventType.NOTE_OFF:
                if value in note_states and note_states[value] > 0:
                    note_states[value] -= 1
                else:
                    pairing_errors += 1
        
        # Count unmatched NOTE_ON events
        unmatched_notes = sum(count for count in note_states.values() if count > 0)
        pairing_errors += unmatched_notes
        
        pairing_score = max(0.0, 1.0 - (pairing_errors / max(1, total_notes)))
        
        return {
            "note_count": total_notes,
            "note_pairing_score": pairing_score,
            "note_pairing_errors": pairing_errors
        }
    
    def _validate_timing(self, tokens: np.ndarray) -> Dict[str, Any]:
        """Validate timing structure in sequence."""
        has_timing = False
        timing_errors = 0
        
        for token in tokens:
            event_type, value = self.vocab_config.token_to_event_info(int(token))
            
            if event_type == EventType.TIME_SHIFT:
                has_timing = True
                # Could add more sophisticated timing validation here
        
        timing_score = 1.0 if has_timing else 0.5
        
        return {
            "timing_score": timing_score,
            "timing_errors": timing_errors
        }
    
    def _validate_repetition(self, tokens: np.ndarray) -> Dict[str, Any]:
        """Validate repetition patterns in sequence."""
        if len(tokens) < 2:
            return {"repetition_score": 1.0, "max_repetition": 1}
        
        max_repetition = 1
        current_repetition = 1
        
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                current_repetition += 1
                max_repetition = max(max_repetition, current_repetition)
            else:
                current_repetition = 1
        
        # Score based on maximum repetition
        if max_repetition <= 3:
            repetition_score = 1.0
        elif max_repetition <= 5:
            repetition_score = 0.7
        elif max_repetition <= 10:
            repetition_score = 0.3
        else:
            repetition_score = 0.0
        
        return {
            "repetition_score": repetition_score,
            "max_repetition": max_repetition
        }
    
    def _validate_velocity_parameters(self, tokens: np.ndarray) -> Dict[str, Any]:
        """
        Validate velocity parameter quality in sequence.
        
        Checks that velocity tokens produce audible MIDI velocities (not silent).
        """
        velocity_tokens = []
        silent_velocity_count = 0
        total_velocity_tokens = 0
        
        for token in tokens:
            event_type, value = self.vocab_config.token_to_event_info(int(token))
            
            if event_type == EventType.VELOCITY_CHANGE:
                total_velocity_tokens += 1
                
                # Convert to MIDI velocity using same logic as training loss
                velocity_bins = getattr(self.vocab_config, 'velocity_bins', 32)
                midi_velocity = int((value / (velocity_bins - 1)) * 126) + 1
                velocity_tokens.append(midi_velocity)
                
                # Count silent/near-silent velocities (unusable)
                if midi_velocity <= 20:
                    silent_velocity_count += 1
        
        # Calculate quality score
        if total_velocity_tokens == 0:
            velocity_quality_score = 0.5  # No velocity tokens is suboptimal
        else:
            usable_velocity_ratio = (total_velocity_tokens - silent_velocity_count) / total_velocity_tokens
            velocity_quality_score = usable_velocity_ratio
        
        return {
            "velocity_quality_score": velocity_quality_score,
            "velocity_tokens": velocity_tokens,
            "silent_velocity_count": silent_velocity_count,
            "total_velocity_tokens": total_velocity_tokens
        }
    
    def _validate_timing_parameters(self, tokens: np.ndarray) -> Dict[str, Any]:
        """
        Validate timing parameter quality in sequence.
        
        Checks that timing tokens produce visible note durations (not zero).
        """
        timing_tokens = []
        zero_duration_count = 0
        total_timing_tokens = 0
        
        for token in tokens:
            event_type, value = self.vocab_config.token_to_event_info(int(token))
            
            if event_type == EventType.TIME_SHIFT:
                total_timing_tokens += 1
                
                # Convert to duration using same logic as training loss
                time_shift_ms = getattr(self.vocab_config, 'time_shift_ms', 15.625)
                duration_seconds = value * time_shift_ms / 1000.0
                timing_tokens.append(duration_seconds)
                
                # Count zero/near-zero durations (invisible)
                if duration_seconds <= 0.01:  # Less than 10ms
                    zero_duration_count += 1
        
        # Calculate quality score
        if total_timing_tokens == 0:
            timing_quality_score = 0.5  # No timing tokens is suboptimal
        else:
            usable_timing_ratio = (total_timing_tokens - zero_duration_count) / total_timing_tokens
            timing_quality_score = usable_timing_ratio
        
        return {
            "timing_quality_score": timing_quality_score,
            "timing_tokens": timing_tokens,
            "zero_duration_count": zero_duration_count,
            "total_timing_tokens": total_timing_tokens
        }
    
    def should_stop_training(self) -> bool:
        """
        Check if training should be stopped due to grammar degradation.
        
        Returns:
            True if training should be stopped
        """
        if len(self.validation_history) < self.config.early_stop_patience:
            return False
        
        recent_scores = [result["grammar_score"] for result in self.validation_history[-self.config.early_stop_patience:]]
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        if avg_recent_score < self.config.grammar_score_threshold:
            logger.warning(f"Grammar score {avg_recent_score:.3f} below threshold {self.config.grammar_score_threshold}")
            return True
        
        return False


# TODO: Implement these in the next phase
def create_musical_grammar_training_config() -> MusicalGrammarConfig:
    """Create default configuration for musical grammar training."""
    return MusicalGrammarConfig()


def integrate_grammar_loss_into_training():
    """Integrate musical grammar loss into the main training loop."""
    # This will be implemented when integrating with the training pipeline
    pass


def create_training_data_validator():
    """Create validator for ensuring training data has proper musical grammar."""
    # This will be implemented for data pipeline integration
    pass


# New utility functions for Phase 5 training integration
def validate_generated_sequence(tokens: np.ndarray, vocab_config=None) -> Dict[str, Any]:
    """
    Quick validation function for generated sequences.
    
    Args:
        tokens: Generated token sequence
        vocab_config: Vocabulary configuration
        
    Returns:
        Validation results with parameter quality scores
    """
    validator = MusicalGrammarValidator()
    if vocab_config:
        validator.vocab_config = vocab_config
    
    return validator.validate_sequence(tokens)


def get_parameter_quality_report(tokens: np.ndarray, vocab_config=None) -> str:
    """
    Generate a human-readable report on parameter quality.
    
    Args:
        tokens: Token sequence to analyze
        vocab_config: Vocabulary configuration
        
    Returns:
        Formatted report string
    """
    results = validate_generated_sequence(tokens, vocab_config)
    
    report = f"""Musical Parameter Quality Report
=====================================
Overall Grammar Score: {results['grammar_score']:.3f}
Sequence Valid: {results['is_valid']}

Note Structure:
- Total Notes: {results['note_count']}
- Note Pairing Score: {results['note_pairing_score']:.3f}
- Note Pairing Errors: {results.get('note_pairing_errors', 0)}

Parameter Quality:
- Velocity Quality Score: {results['velocity_quality_score']:.3f}
- Silent Velocity Tokens: {results.get('silent_velocity_count', 0)}/{results.get('total_velocity_tokens', 0)}
- Timing Quality Score: {results['timing_quality_score']:.3f}
- Zero Duration Tokens: {results.get('zero_duration_count', 0)}/{results.get('total_timing_tokens', 0)}

Structural Quality:
- Timing Score: {results['timing_score']:.3f}
- Repetition Score: {results['repetition_score']:.3f}
- Max Repetition: {results.get('max_repetition', 1)}
"""
    
    if results['grammar_score'] < 0.8:
        report += "\n⚠️  WARNING: Low grammar score - consider retraining with grammar loss"
    if results.get('silent_velocity_count', 0) > 0:
        report += "\n🔇 WARNING: Contains silent velocity tokens (unusable in MIDI)"
    if results.get('zero_duration_count', 0) > 0:
        report += "\n⏱️  WARNING: Contains zero duration tokens (invisible in MIDI)"
    
    return report