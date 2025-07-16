"""
Token Sequence Validator and Corrector

This module ensures that generated token sequences follow musical grammar
and can be successfully converted to MIDI files with actual notes.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

from ..data.representation import VocabularyConfig, EventType
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationResult:
    """Result of token sequence validation."""
    is_valid: bool
    num_notes: int
    num_errors: int
    error_messages: List[str]
    corrected_tokens: Optional[np.ndarray] = None


class TokenValidator:
    """Validates and corrects token sequences to ensure valid MIDI conversion."""
    
    def __init__(self, vocab_config: Optional[VocabularyConfig] = None):
        self.vocab_config = vocab_config or VocabularyConfig()
        
        # Pre-compute token mappings for efficiency
        self._build_token_maps()
    
    def _build_token_maps(self):
        """Build mappings for different token types."""
        self.token_types = {}
        self.note_on_tokens = {}
        self.note_off_tokens = {}
        self.velocity_tokens = []
        self.time_shift_tokens = []
        
        for token in range(self.vocab_config.vocab_size):
            event_type, value = self.vocab_config.token_to_event_info(token)
            self.token_types[token] = (event_type, value)
            
            if event_type == EventType.NOTE_ON:
                self.note_on_tokens[value] = token
            elif event_type == EventType.NOTE_OFF:
                self.note_off_tokens[value] = token
            elif event_type == EventType.VELOCITY_CHANGE:
                self.velocity_tokens.append(token)
            elif event_type == EventType.TIME_SHIFT:
                self.time_shift_tokens.append(token)
        
        # Special tokens
        self.start_token = 0
        self.end_token = 1
        self.pad_token = 2
        self.unk_token = 3
    
    def validate_sequence(
        self, 
        tokens: Union[np.ndarray, torch.Tensor],
        fix_errors: bool = True
    ) -> ValidationResult:
        """
        Validate a token sequence and optionally fix errors.
        
        Args:
            tokens: Token sequence to validate
            fix_errors: Whether to attempt to fix errors
            
        Returns:
            ValidationResult with analysis and optionally corrected sequence
        """
        # Convert to numpy
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        
        if tokens.ndim > 1:
            tokens = tokens.flatten()
        
        errors = []
        active_notes = {}  # Track which notes are currently on
        has_velocity = False
        current_velocity = 64
        note_count = 0
        
        # Check for basic issues
        if len(tokens) == 0:
            errors.append("Empty token sequence")
            return ValidationResult(False, 0, len(errors), errors)
        
        # Analyze token sequence
        for i, token in enumerate(tokens):
            if token >= self.vocab_config.vocab_size:
                errors.append(f"Invalid token {token} at position {i}")
                continue
            
            event_type, value = self.token_types[token]
            
            if event_type == EventType.START_TOKEN:
                if i != 0:
                    errors.append(f"START_TOKEN at position {i} (should be at 0)")
            
            elif event_type == EventType.VELOCITY_CHANGE:
                has_velocity = True
                current_velocity = value
            
            elif event_type == EventType.NOTE_ON:
                if not has_velocity:
                    errors.append(f"NOTE_ON at position {i} without prior VELOCITY_CHANGE")
                active_notes[value] = i
            
            elif event_type == EventType.NOTE_OFF:
                if value not in active_notes:
                    errors.append(f"NOTE_OFF for pitch {value} at position {i} without matching NOTE_ON")
                else:
                    del active_notes[value]
                    note_count += 1
            
            elif event_type == EventType.END_TOKEN:
                # Check for unclosed notes
                if active_notes:
                    errors.append(f"END_TOKEN at position {i} with {len(active_notes)} unclosed notes")
                break
        
        # Check for unclosed notes at end
        if active_notes and tokens[-1] != self.end_token:
            errors.append(f"Sequence ends with {len(active_notes)} unclosed notes")
        
        is_valid = len(errors) == 0 and note_count > 0
        
        # Attempt to fix errors if requested
        corrected_tokens = None
        if fix_errors and not is_valid:
            corrected_tokens = self._fix_sequence(tokens, errors, active_notes)
            
            # Re-validate corrected sequence
            if corrected_tokens is not None:
                # Count notes in corrected sequence
                corrected_note_count = 0
                for token in corrected_tokens:
                    event_type, _ = self.token_types.get(token, (EventType.UNK_TOKEN, 0))
                    if event_type == EventType.NOTE_OFF:
                        corrected_note_count += 1
                
                if corrected_note_count > note_count:
                    note_count = corrected_note_count
        
        return ValidationResult(
            is_valid=is_valid,
            num_notes=note_count,
            num_errors=len(errors),
            error_messages=errors,
            corrected_tokens=corrected_tokens
        )
    
    def _fix_sequence(
        self, 
        tokens: np.ndarray, 
        errors: List[str],
        active_notes: dict
    ) -> Optional[np.ndarray]:
        """Attempt to fix common sequence errors."""
        fixed_tokens = []
        
        # Ensure sequence starts with START token
        if len(tokens) == 0 or tokens[0] != self.start_token:
            fixed_tokens.append(self.start_token)
        
        # Add default velocity if missing
        has_velocity = False
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token >= self.vocab_config.vocab_size:
                # Skip invalid tokens
                i += 1
                continue
            
            event_type, value = self.token_types[token]
            
            # Handle velocity
            if event_type == EventType.VELOCITY_CHANGE:
                has_velocity = True
                fixed_tokens.append(token)
            
            # Handle note on
            elif event_type == EventType.NOTE_ON:
                if not has_velocity:
                    # Add default velocity
                    if len(self.velocity_tokens) > 16:
                        fixed_tokens.append(self.velocity_tokens[16])  # Medium velocity
                    has_velocity = True
                fixed_tokens.append(token)
            
            # Handle other events
            elif event_type not in [EventType.PAD_TOKEN, EventType.UNK_TOKEN]:
                fixed_tokens.append(token)
            
            i += 1
        
        # Close any unclosed notes
        if active_notes:
            # Add a small time shift before closing notes
            if len(self.time_shift_tokens) > 16:
                fixed_tokens.append(self.time_shift_tokens[16])
            
            # Close all active notes
            for pitch in active_notes:
                if pitch in self.note_off_tokens:
                    fixed_tokens.append(self.note_off_tokens[pitch])
        
        # Ensure sequence ends with END token
        if not fixed_tokens or fixed_tokens[-1] != self.end_token:
            fixed_tokens.append(self.end_token)
        
        return np.array(fixed_tokens, dtype=np.int32)
    
    def create_valid_sequence(self, length: int = 50) -> np.ndarray:
        """
        Create a simple valid token sequence for testing.
        
        Args:
            length: Target length of sequence
            
        Returns:
            Valid token sequence that will produce MIDI notes
        """
        tokens = []
        
        # Start
        tokens.append(self.start_token)
        
        # Set velocity
        if self.velocity_tokens:
            tokens.append(self.velocity_tokens[len(self.velocity_tokens) // 2])
        
        # Generate simple melody
        pitches = [60, 62, 64, 65, 67, 65, 64, 62, 60]  # C major scale up and down
        
        for pitch in pitches:
            if len(tokens) >= length - 2:
                break
            
            # Note on
            if pitch in self.note_on_tokens:
                tokens.append(self.note_on_tokens[pitch])
                
                # Time shift
                if self.time_shift_tokens:
                    tokens.append(self.time_shift_tokens[32])  # ~500ms
                
                # Note off
                if pitch in self.note_off_tokens:
                    tokens.append(self.note_off_tokens[pitch])
                
                # Gap between notes
                if self.time_shift_tokens:
                    tokens.append(self.time_shift_tokens[8])  # ~125ms
        
        # End
        tokens.append(self.end_token)
        
        # Pad to length if needed
        while len(tokens) < length:
            tokens.append(self.pad_token)
        
        return np.array(tokens[:length], dtype=np.int32)


def validate_and_fix_tokens(
    tokens: Union[np.ndarray, torch.Tensor],
    vocab_config: Optional[VocabularyConfig] = None
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Convenience function to validate and fix token sequences.
    
    Args:
        tokens: Token sequence to validate
        vocab_config: Vocabulary configuration
        
    Returns:
        Tuple of (is_valid, corrected_tokens)
    """
    validator = TokenValidator(vocab_config)
    result = validator.validate_sequence(tokens, fix_errors=True)
    
    if result.is_valid:
        return True, None
    else:
        logger.warning(f"Token sequence validation failed with {result.num_errors} errors")
        if result.corrected_tokens is not None:
            logger.info(f"Corrected sequence has {result.num_notes} notes")
        return False, result.corrected_tokens