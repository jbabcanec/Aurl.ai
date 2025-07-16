"""
Musical Constraints Module

Implements musical theory-based constraints for generation including:
- Harmonic progression rules
- Voice leading constraints
- Rhythmic pattern validation
- Dynamic consistency checks
- Style-specific constraints

These constraints ensure generated music follows musical principles.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import music21

from ..utils.constants import SPECIAL_TOKENS
from ..data.representation import EventType

# Token constants based on our representation
NOTE_ON_TOKEN = 13  # Start of NOTE_ON tokens in our vocab
NOTE_OFF_TOKEN = 141  # Start of NOTE_OFF tokens in our vocab  
TIME_SHIFT_TOKEN = 269  # Start of TIME_SHIFT tokens in our vocab
VELOCITY_TOKEN = 369  # Start of VELOCITY tokens in our vocab
MIDI_VOCAB_SIZE = 774
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ConstraintType(Enum):
    """Types of musical constraints."""
    HARMONIC = "harmonic"
    MELODIC = "melodic"
    RHYTHMIC = "rhythmic"
    DYNAMIC = "dynamic"
    STRUCTURAL = "structural"
    STYLE = "style"


@dataclass
class ConstraintConfig:
    """Configuration for musical constraints."""
    # Enable/disable constraint types
    use_harmonic_constraints: bool = True
    use_melodic_constraints: bool = True
    use_rhythmic_constraints: bool = True
    use_dynamic_constraints: bool = True
    use_structural_constraints: bool = True
    
    # Harmonic constraints
    allowed_chord_progressions: Optional[List[List[str]]] = None
    dissonance_threshold: float = 0.3
    voice_leading_strictness: float = 0.7
    
    # Melodic constraints
    max_leap_interval: int = 12  # semitones
    prefer_stepwise_motion: bool = True
    melodic_contour_similarity: float = 0.8
    
    # Rhythmic constraints
    allowed_time_signatures: List[Tuple[int, int]] = None
    syncopation_level: float = 0.2
    rhythmic_complexity: float = 0.5
    
    # Dynamic constraints
    dynamic_range: Tuple[int, int] = (20, 100)  # MIDI velocity range
    dynamic_smoothness: float = 0.8
    
    # Structural constraints
    phrase_length_range: Tuple[int, int] = (16, 64)  # in tokens
    repetition_threshold: float = 0.3
    
    # Style-specific constraints
    style: Optional[str] = None  # "classical", "jazz", "pop", etc.
    
    def __post_init__(self):
        """Initialize default values."""
        if self.allowed_time_signatures is None:
            self.allowed_time_signatures = [(4, 4), (3, 4), (6, 8)]


class MusicalConstraintEngine:
    """
    Engine for applying musical constraints during generation.
    
    This class implements various musical theory rules and constraints
    to ensure generated music is musically coherent and stylistically
    appropriate.
    """
    
    def __init__(self, config: ConstraintConfig = None):
        """
        Initialize the constraint engine.
        
        Args:
            config: Constraint configuration
        """
        self.config = config or ConstraintConfig()
        
        # Constraint functions registry
        self.constraint_functions = {
            ConstraintType.HARMONIC: self._apply_harmonic_constraints,
            ConstraintType.MELODIC: self._apply_melodic_constraints,
            ConstraintType.RHYTHMIC: self._apply_rhythmic_constraints,
            ConstraintType.DYNAMIC: self._apply_dynamic_constraints,
            ConstraintType.STRUCTURAL: self._apply_structural_constraints,
        }
        
        # Music theory utilities
        self._init_music_theory()
        
        # Statistics tracking
        self.constraint_stats = {
            "total_applied": 0,
            "violations_corrected": 0,
            "constraint_counts": {ct.value: 0 for ct in ConstraintType}
        }
    
    def _init_music_theory(self):
        """Initialize music theory utilities and mappings."""
        # Common chord progressions in different keys
        self.common_progressions = {
            "major": [
                ["I", "IV", "V", "I"],
                ["I", "vi", "IV", "V"],
                ["I", "V", "vi", "IV"],
                ["I", "IV", "I", "V"],
                ["I", "ii", "V", "I"],
            ],
            "minor": [
                ["i", "iv", "V", "i"],
                ["i", "VI", "III", "VII"],
                ["i", "iv", "VII", "i"],
                ["i", "ii°", "V", "i"],
            ]
        }
        
        # Voice leading rules
        self.voice_leading_rules = {
            "avoid_parallel_fifths": True,
            "avoid_parallel_octaves": True,
            "resolve_leading_tone": True,
            "smooth_voice_leading": True,
        }
        
        # Scale patterns for melodic validation
        self.scale_patterns = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        }
    
    def apply_constraints(
        self,
        logits: torch.Tensor,
        generated_sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Apply musical constraints to generation logits.
        
        Args:
            logits: Model output logits for next token
            generated_sequence: Sequence generated so far
            step: Current generation step
            context: Additional context (key, time signature, etc.)
            
        Returns:
            Modified logits with constraints applied
        """
        # Track statistics
        self.constraint_stats["total_applied"] += 1
        
        # Apply each enabled constraint type
        if self.config.use_harmonic_constraints:
            logits = self._apply_harmonic_constraints(
                logits, generated_sequence, step, context
            )
            self.constraint_stats["constraint_counts"]["harmonic"] += 1
        
        if self.config.use_melodic_constraints:
            logits = self._apply_melodic_constraints(
                logits, generated_sequence, step, context
            )
            self.constraint_stats["constraint_counts"]["melodic"] += 1
        
        if self.config.use_rhythmic_constraints:
            logits = self._apply_rhythmic_constraints(
                logits, generated_sequence, step, context
            )
            self.constraint_stats["constraint_counts"]["rhythmic"] += 1
        
        if self.config.use_dynamic_constraints:
            logits = self._apply_dynamic_constraints(
                logits, generated_sequence, step, context
            )
            self.constraint_stats["constraint_counts"]["dynamic"] += 1
        
        if self.config.use_structural_constraints:
            logits = self._apply_structural_constraints(
                logits, generated_sequence, step, context
            )
            self.constraint_stats["constraint_counts"]["structural"] += 1
        
        return logits
    
    def _apply_harmonic_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply harmonic progression and voice leading constraints."""
        # Extract current harmonic context
        current_notes = self._extract_active_notes(sequence)
        
        if len(current_notes) > 0:
            # Check for dissonance
            dissonance_level = self._calculate_dissonance(current_notes)
            
            if dissonance_level > self.config.dissonance_threshold:
                # Reduce probability of adding more dissonant notes
                logits = self._reduce_dissonant_note_probability(
                    logits, current_notes
                )
            
            # Apply voice leading rules
            if self.config.voice_leading_strictness > 0:
                logits = self._apply_voice_leading_rules(
                    logits, sequence, current_notes
                )
        
        return logits
    
    def _apply_melodic_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply melodic contour and interval constraints."""
        # Get recent melodic line
        recent_melody = self._extract_melody_line(sequence, lookback=16)
        
        if len(recent_melody) > 1:
            last_pitch = recent_melody[-1]
            
            # Constrain large leaps
            for token_id in range(logits.size(-1)):
                if self._is_note_on_token(token_id):
                    pitch = self._token_to_pitch(token_id)
                    interval = abs(pitch - last_pitch)
                    
                    if interval > self.config.max_leap_interval:
                        # Reduce probability of large leaps
                        logits[..., token_id] -= 2.0
                    
                    elif self.config.prefer_stepwise_motion and interval <= 2:
                        # Increase probability of stepwise motion
                        logits[..., token_id] += 0.5
        
        return logits
    
    def _apply_rhythmic_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply rhythmic pattern and timing constraints."""
        # Extract rhythmic pattern
        rhythm_pattern = self._extract_rhythm_pattern(sequence)
        
        # Check time signature alignment
        if context and "time_signature" in context:
            time_sig = context["time_signature"]
            beat_position = self._get_beat_position(sequence, time_sig)
            
            # Encourage notes on strong beats
            if self._is_strong_beat(beat_position, time_sig):
                # Increase probability of note events
                note_tokens = self._get_note_on_tokens()
                logits[..., note_tokens] += 0.3
        
        # Control syncopation level
        current_syncopation = self._calculate_syncopation(rhythm_pattern)
        if current_syncopation > self.config.syncopation_level:
            # Reduce off-beat note probability
            logits = self._reduce_syncopation(logits, sequence)
        
        return logits
    
    def _apply_dynamic_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply dynamic (velocity) constraints."""
        # Get recent dynamics
        recent_velocities = self._extract_velocities(sequence, lookback=32)
        
        if len(recent_velocities) > 0:
            avg_velocity = np.mean(recent_velocities)
            
            # Constrain velocity tokens to configured range
            velocity_tokens = self._get_velocity_tokens()
            for token_id in velocity_tokens:
                velocity = self._token_to_velocity(token_id)
                
                # Check if within allowed range
                if not (self.config.dynamic_range[0] <= velocity <= self.config.dynamic_range[1]):
                    logits[..., token_id] = -float('inf')
                
                # Encourage smooth dynamics
                elif self.config.dynamic_smoothness > 0:
                    velocity_diff = abs(velocity - avg_velocity)
                    if velocity_diff > 20:  # Large dynamic change
                        logits[..., token_id] -= (
                            velocity_diff * self.config.dynamic_smoothness / 20
                        )
        
        return logits
    
    def _apply_structural_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Apply structural and phrase-level constraints."""
        # Check phrase boundaries
        phrase_length = self._get_current_phrase_length(sequence)
        
        # Encourage phrase endings at appropriate lengths
        if phrase_length >= self.config.phrase_length_range[0]:
            if phrase_length >= self.config.phrase_length_range[1]:
                # Force phrase ending
                rest_tokens = self._get_rest_tokens()
                logits[..., rest_tokens] += 2.0
            else:
                # Gentle encouragement for phrase ending
                rest_tokens = self._get_rest_tokens()
                logits[..., rest_tokens] += 0.5
        
        # Check for excessive repetition
        repetition_score = self._calculate_repetition_score(sequence)
        if repetition_score > self.config.repetition_threshold:
            # Reduce probability of repeating recent patterns
            logits = self._reduce_repetition_probability(logits, sequence)
        
        return logits
    
    # Utility methods for constraint application
    
    def _extract_active_notes(self, sequence: torch.Tensor) -> List[int]:
        """Extract currently active (sounding) notes from sequence."""
        active_notes = []
        note_states = {}
        
        for token in sequence.cpu().numpy().flatten():
            if self._is_note_on_token(token):
                pitch = self._token_to_pitch(token)
                note_states[pitch] = True
            elif self._is_note_off_token(token):
                pitch = self._token_to_pitch(token)
                note_states[pitch] = False
        
        # Return currently active notes
        return [pitch for pitch, active in note_states.items() if active]
    
    def _extract_melody_line(
        self, 
        sequence: torch.Tensor, 
        lookback: int = 16
    ) -> List[int]:
        """Extract the melodic line (highest notes) from sequence."""
        melody = []
        window = sequence[-lookback:] if len(sequence) > lookback else sequence
        
        for token in window.cpu().numpy().flatten():
            if self._is_note_on_token(token):
                pitch = self._token_to_pitch(token)
                melody.append(pitch)
        
        return melody
    
    def _extract_rhythm_pattern(
        self, 
        sequence: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """Extract rhythm pattern as (time, duration) pairs."""
        pattern = []
        current_time = 0
        
        for token in sequence.cpu().numpy().flatten():
            if self._is_time_shift_token(token):
                current_time += self._token_to_time_shift(token)
            elif self._is_note_on_token(token):
                # Simplified: assume fixed duration for now
                pattern.append((current_time, 1))
        
        return pattern
    
    def _extract_velocities(
        self, 
        sequence: torch.Tensor, 
        lookback: int = 32
    ) -> List[int]:
        """Extract recent velocity values from sequence."""
        velocities = []
        window = sequence[-lookback:] if len(sequence) > lookback else sequence
        
        for token in window.cpu().numpy().flatten():
            if self._is_velocity_token(token):
                velocity = self._token_to_velocity(token)
                velocities.append(velocity)
        
        return velocities
    
    def _calculate_dissonance(self, notes: List[int]) -> float:
        """Calculate dissonance level of current note set."""
        if len(notes) < 2:
            return 0.0
        
        dissonance = 0.0
        count = 0
        
        # Check intervals between all note pairs
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                interval = abs(notes[i] - notes[j]) % 12
                
                # Dissonant intervals (minor 2nd, tritone, major 7th)
                if interval in [1, 6, 11]:
                    dissonance += 1.0
                # Somewhat dissonant (major 2nd, minor 7th)
                elif interval in [2, 10]:
                    dissonance += 0.5
                
                count += 1
        
        return dissonance / count if count > 0 else 0.0
    
    def _calculate_syncopation(self, rhythm_pattern: List[Tuple[int, int]]) -> float:
        """Calculate syncopation level of rhythm pattern."""
        if len(rhythm_pattern) < 2:
            return 0.0
        
        syncopation = 0.0
        # Simplified: count off-beat notes
        for time, _ in rhythm_pattern:
            if time % 4 not in [0, 2]:  # Not on strong beats
                syncopation += 1.0
        
        return syncopation / len(rhythm_pattern)
    
    def _calculate_repetition_score(self, sequence: torch.Tensor) -> float:
        """Calculate how repetitive the sequence is."""
        if len(sequence) < 8:
            return 0.0
        
        # Look for repeated patterns
        seq_array = sequence.cpu().numpy()
        pattern_counts = {}
        
        # Check various pattern lengths
        for pattern_len in [4, 8, 16]:
            for i in range(len(seq_array) - pattern_len):
                pattern = tuple(seq_array[i:i + pattern_len])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate repetition score
        max_count = max(pattern_counts.values()) if pattern_counts else 1
        return (max_count - 1) / len(seq_array)
    
    # Token type checking utilities
    
    def _is_note_on_token(self, token: int) -> bool:
        """Check if token is a note-on event."""
        return NOTE_ON_TOKEN <= token < NOTE_ON_TOKEN + 128
    
    def _is_note_off_token(self, token: int) -> bool:
        """Check if token is a note-off event."""
        return NOTE_OFF_TOKEN <= token < NOTE_OFF_TOKEN + 128
    
    def _is_time_shift_token(self, token: int) -> bool:
        """Check if token is a time shift event."""
        return TIME_SHIFT_TOKEN <= token < TIME_SHIFT_TOKEN + 100
    
    def _is_velocity_token(self, token: int) -> bool:
        """Check if token is a velocity event."""
        return VELOCITY_TOKEN <= token < VELOCITY_TOKEN + 128
    
    # Token conversion utilities
    
    def _token_to_pitch(self, token: int) -> int:
        """Convert note token to MIDI pitch."""
        if self._is_note_on_token(token):
            return token - NOTE_ON_TOKEN
        elif self._is_note_off_token(token):
            return token - NOTE_OFF_TOKEN
        return 0
    
    def _token_to_velocity(self, token: int) -> int:
        """Convert velocity token to MIDI velocity."""
        if self._is_velocity_token(token):
            return token - VELOCITY_TOKEN
        return 64  # Default velocity
    
    def _token_to_time_shift(self, token: int) -> int:
        """Convert time shift token to time units."""
        if self._is_time_shift_token(token):
            return token - TIME_SHIFT_TOKEN
        return 0
    
    # Token set utilities
    
    def _get_note_on_tokens(self) -> torch.Tensor:
        """Get all note-on token indices."""
        return torch.arange(NOTE_ON_TOKEN, NOTE_ON_TOKEN + 128)
    
    def _get_velocity_tokens(self) -> torch.Tensor:
        """Get all velocity token indices."""
        return torch.arange(VELOCITY_TOKEN, VELOCITY_TOKEN + 128)
    
    def _get_rest_tokens(self) -> torch.Tensor:
        """Get tokens that represent rests or phrase endings."""
        # Time shifts can act as rests
        return torch.arange(TIME_SHIFT_TOKEN, TIME_SHIFT_TOKEN + 100)
    
    # Additional utility methods
    
    def _get_beat_position(
        self, 
        sequence: torch.Tensor, 
        time_signature: Tuple[int, int]
    ) -> int:
        """Get current position within measure."""
        total_time = 0
        for token in sequence.cpu().numpy().flatten():
            if self._is_time_shift_token(token):
                total_time += self._token_to_time_shift(token)
        
        beats_per_measure = time_signature[0]
        beat_unit = time_signature[1]
        
        # Simplified: assume quarter note units
        measure_length = beats_per_measure * 4
        return total_time % measure_length
    
    def _is_strong_beat(
        self, 
        position: int, 
        time_signature: Tuple[int, int]
    ) -> bool:
        """Check if position is on a strong beat."""
        if time_signature == (4, 4):
            return position % 4 == 0
        elif time_signature == (3, 4):
            return position % 3 == 0
        elif time_signature == (6, 8):
            return position % 6 in [0, 3]
        return position == 0
    
    def _get_current_phrase_length(self, sequence: torch.Tensor) -> int:
        """Get length of current musical phrase."""
        # Flatten sequence to 1D for analysis
        seq_flat = sequence.flatten()
        
        # Look for last significant rest or pause
        phrase_start = 0
        
        for i in range(len(seq_flat) - 1, -1, -1):
            token = seq_flat[i].item()
            if self._is_time_shift_token(token):
                shift = self._token_to_time_shift(token)
                if shift > 8:  # Significant pause
                    phrase_start = i + 1
                    break
        
        return len(seq_flat) - phrase_start
    
    def _reduce_dissonant_note_probability(
        self,
        logits: torch.Tensor,
        current_notes: List[int]
    ) -> torch.Tensor:
        """Reduce probability of notes that would create dissonance."""
        for note_token in range(NOTE_ON_TOKEN, NOTE_ON_TOKEN + 128):
            pitch = note_token - NOTE_ON_TOKEN
            
            # Check dissonance with current notes
            for active_pitch in current_notes:
                interval = abs(pitch - active_pitch) % 12
                if interval in [1, 6, 11]:  # Dissonant intervals
                    logits[..., note_token] -= 1.0
        
        return logits
    
    def _apply_voice_leading_rules(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        current_notes: List[int]
    ) -> torch.Tensor:
        """Apply voice leading rules to logits."""
        # This is a simplified implementation
        # Full implementation would track individual voices
        
        # Encourage smooth voice leading (small intervals)
        if len(current_notes) > 0:
            for note_token in range(NOTE_ON_TOKEN, NOTE_ON_TOKEN + 128):
                pitch = note_token - NOTE_ON_TOKEN
                
                # Find smallest interval to any current note
                min_interval = min(
                    abs(pitch - note) for note in current_notes
                )
                
                # Encourage smaller intervals
                if min_interval <= 2:  # Step motion
                    logits[..., note_token] += 0.5
                elif min_interval > 7:  # Large leap
                    logits[..., note_token] -= 0.5
        
        return logits
    
    def _reduce_syncopation(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """Reduce probability of syncopated rhythms."""
        # Increase probability of time shifts (rests)
        time_shift_tokens = self._get_rest_tokens()
        logits[..., time_shift_tokens] += 0.3
        
        return logits
    
    def _reduce_repetition_probability(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor
    ) -> torch.Tensor:
        """Reduce probability of repeating recent patterns."""
        # Get recent tokens
        recent_tokens = sequence[-16:] if len(sequence) > 16 else sequence
        
        # Reduce probability of recently used tokens
        for token in recent_tokens.unique():
            if token < logits.size(-1):
                logits[..., token] -= 0.2
        
        return logits
    
    def get_stats(self) -> Dict[str, Any]:
        """Get constraint application statistics."""
        return self.constraint_stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.constraint_stats = {
            "total_applied": 0,
            "violations_corrected": 0,
            "constraint_counts": {ct.value: 0 for ct in ConstraintType}
        }


class StyleSpecificConstraints:
    """
    Style-specific constraint implementations.
    
    This class provides specialized constraints for different musical styles
    like classical, jazz, pop, etc.
    """
    
    def __init__(self):
        """Initialize style-specific constraints."""
        self.style_constraints = {
            "classical": self._classical_constraints,
            "jazz": self._jazz_constraints,
            "pop": self._pop_constraints,
            "blues": self._blues_constraints,
        }
    
    def get_style_constraints(
        self, 
        style: str
    ) -> Optional[Callable]:
        """Get constraint function for specific style."""
        return self.style_constraints.get(style.lower())
    
    def _classical_constraints(
        self, 
        logits: torch.Tensor,
        sequence: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply classical music specific constraints."""
        # Strict voice leading
        # Avoid parallel fifths and octaves
        # Proper cadences
        # Traditional harmonic progressions
        return logits
    
    def _jazz_constraints(
        self, 
        logits: torch.Tensor,
        sequence: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply jazz specific constraints."""
        # Extended chords (7ths, 9ths, etc.)
        # Swing rhythm
        # Blue notes
        # ii-V-I progressions
        return logits
    
    def _pop_constraints(
        self, 
        logits: torch.Tensor,
        sequence: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply pop music specific constraints."""
        # Simple chord progressions
        # Verse-chorus structure
        # Repetitive hooks
        # 4/4 time preference
        return logits
    
    def _blues_constraints(
        self, 
        logits: torch.Tensor,
        sequence: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply blues specific constraints."""
        # 12-bar blues structure
        # Blues scale
        # Shuffle rhythm
        # I-IV-V progressions
        return logits