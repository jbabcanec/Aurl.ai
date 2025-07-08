"""
Velocity normalization with style preservation for Aurl.ai.

Normalizes MIDI velocities while preserving musical dynamics and phrasing.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from .midi_parser import MidiData, MidiNote, MidiInstrument


@dataclass
class VelocityNormalizationConfig:
    """Configuration for velocity normalization."""
    target_mean: float = 64.0  # Target mean velocity (0-127)
    target_std: float = 20.0   # Target standard deviation
    preserve_dynamics: bool = True  # Preserve relative dynamics
    min_velocity: int = 20     # Minimum allowed velocity
    max_velocity: int = 110    # Maximum allowed velocity
    smooth_window: int = 8     # Window size for smoothing
    curve_type: str = "logarithmic"  # linear, logarithmic, exponential


class VelocityNormalizer:
    """
    Normalizes MIDI velocities while preserving musical expression.
    
    Features:
    - Global normalization across entire piece
    - Piece-relative normalization
    - Style-preserving normalization that maintains phrasing
    - Dynamic curve adjustments
    - Smoothing to prevent abrupt changes
    """
    
    def __init__(self, config: VelocityNormalizationConfig = None):
        self.config = config or VelocityNormalizationConfig()
        self.logger = logging.getLogger(__name__)
    
    def normalize_midi_data(self, midi_data: MidiData, mode: str = "style_preserving") -> MidiData:
        """
        Normalize velocities in MIDI data.
        
        Args:
            midi_data: Original MIDI data
            mode: Normalization mode - "global", "piece_relative", "style_preserving"
            
        Returns:
            MIDI data with normalized velocities
        """
        # Create a copy to avoid modifying original
        normalized_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=midi_data.filename
        )
        
        if mode == "global":
            normalized_data = self._global_normalization(midi_data)
        elif mode == "piece_relative":
            normalized_data = self._piece_relative_normalization(midi_data)
        elif mode == "style_preserving":
            normalized_data = self._style_preserving_normalization(midi_data)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        
        return normalized_data
    
    def _global_normalization(self, midi_data: MidiData) -> MidiData:
        """
        Normalize all velocities to a fixed range.
        
        Simple linear scaling to target range.
        """
        normalized_data = self._copy_midi_structure(midi_data)
        
        for instrument in midi_data.instruments:
            normalized_notes = []
            
            for note in instrument.notes:
                # Linear scaling to target range
                normalized_velocity = int(np.clip(
                    self.config.target_mean,
                    self.config.min_velocity,
                    self.config.max_velocity
                ))
                
                normalized_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=normalized_velocity,
                    start=note.start,
                    end=note.end
                ))
            
            normalized_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=normalized_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return normalized_data
    
    def _piece_relative_normalization(self, midi_data: MidiData) -> MidiData:
        """
        Normalize velocities relative to the piece's own dynamics.
        
        Scales the piece's velocity range to a standard range.
        """
        # Collect all velocities
        all_velocities = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_velocities.append(note.velocity)
        
        if not all_velocities:
            return self._copy_midi_structure(midi_data)
        
        # Calculate piece statistics
        piece_min = min(all_velocities)
        piece_max = max(all_velocities)
        piece_range = piece_max - piece_min
        
        if piece_range == 0:
            return self._copy_midi_structure(midi_data)
        
        # Create normalized data
        normalized_data = self._copy_midi_structure(midi_data)
        
        for instrument in midi_data.instruments:
            normalized_notes = []
            
            for note in instrument.notes:
                # Scale to target range
                normalized_ratio = (note.velocity - piece_min) / piece_range
                normalized_velocity = int(
                    self.config.min_velocity + 
                    normalized_ratio * (self.config.max_velocity - self.config.min_velocity)
                )
                
                normalized_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=normalized_velocity,
                    start=note.start,
                    end=note.end
                ))
            
            normalized_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=normalized_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return normalized_data
    
    def _style_preserving_normalization(self, midi_data: MidiData) -> MidiData:
        """
        Normalize while preserving musical phrasing and dynamics.
        
        This is the most sophisticated method that:
        1. Preserves relative dynamics within phrases
        2. Applies dynamic curves (logarithmic, exponential)
        3. Smooths transitions between sections
        4. Maintains musical expression
        """
        normalized_data = self._copy_midi_structure(midi_data)
        
        for instrument in midi_data.instruments:
            if not instrument.notes:
                normalized_data.instruments.append(MidiInstrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name,
                    notes=[],
                    pitch_bends=instrument.pitch_bends.copy(),
                    control_changes=instrument.control_changes.copy()
                ))
                continue
            
            # Sort notes by time for phrase analysis
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
            
            # Extract velocity contour
            velocities = np.array([n.velocity for n in sorted_notes])
            times = np.array([n.start for n in sorted_notes])
            
            # Analyze local dynamics
            normalized_velocities = self._normalize_with_context(velocities, times)
            
            # Apply smoothing
            if self.config.smooth_window > 1 and len(normalized_velocities) > self.config.smooth_window:
                normalized_velocities = self._smooth_velocities(normalized_velocities)
            
            # Apply dynamic curve
            normalized_velocities = self._apply_dynamic_curve(normalized_velocities)
            
            # Create normalized notes
            normalized_notes = []
            for i, note in enumerate(sorted_notes):
                normalized_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=int(np.clip(normalized_velocities[i], 
                                       self.config.min_velocity, 
                                       self.config.max_velocity)),
                    start=note.start,
                    end=note.end
                ))
            
            normalized_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=normalized_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return normalized_data
    
    def _normalize_with_context(self, velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Normalize velocities considering temporal context.
        
        Preserves relative dynamics within musical phrases.
        """
        if len(velocities) == 0:
            return velocities
        
        # Identify phrases based on time gaps
        time_diffs = np.diff(times)
        phrase_boundaries = np.where(time_diffs > 1.0)[0] + 1  # 1 second gap = new phrase
        phrase_boundaries = np.concatenate([[0], phrase_boundaries, [len(velocities)]])
        
        normalized = np.zeros_like(velocities, dtype=float)
        
        # Normalize each phrase independently
        for i in range(len(phrase_boundaries) - 1):
            start_idx = phrase_boundaries[i]
            end_idx = phrase_boundaries[i + 1]
            
            phrase_velocities = velocities[start_idx:end_idx]
            
            if len(phrase_velocities) == 0:
                continue
            
            # Calculate phrase statistics
            phrase_mean = np.mean(phrase_velocities)
            phrase_std = np.std(phrase_velocities)
            
            if phrase_std == 0:
                # No variation in phrase, just shift to target mean
                normalized[start_idx:end_idx] = self.config.target_mean
            else:
                # Standardize and rescale
                standardized = (phrase_velocities - phrase_mean) / phrase_std
                rescaled = standardized * self.config.target_std + self.config.target_mean
                normalized[start_idx:end_idx] = rescaled
        
        return normalized
    
    def _smooth_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Apply smoothing to prevent abrupt velocity changes."""
        if len(velocities) <= self.config.smooth_window:
            return velocities
        
        # Use convolution for smoothing
        kernel = np.ones(self.config.smooth_window) / self.config.smooth_window
        
        # Pad to handle edges
        pad_width = self.config.smooth_window // 2
        padded = np.pad(velocities, pad_width, mode='edge')
        
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        # Ensure smoothed has same length as input
        if len(smoothed) > len(velocities):
            smoothed = smoothed[:len(velocities)]
        elif len(smoothed) < len(velocities):
            # This shouldn't happen with proper padding, but just in case
            smoothed = np.pad(smoothed, (0, len(velocities) - len(smoothed)), mode='edge')
        
        # Preserve original dynamic range
        if self.config.preserve_dynamics:
            # Blend smoothed with original based on local variance
            local_variance = np.array([
                np.var(velocities[max(0, i-2):min(len(velocities), i+3)])
                for i in range(len(velocities))
            ])
            
            # High variance = less smoothing (preserve dynamics)
            mean_var = np.mean(local_variance)
            if mean_var > 0:
                blend_weights = 1.0 / (1.0 + local_variance / mean_var)
                smoothed = smoothed * blend_weights + velocities * (1 - blend_weights)
        
        return smoothed
    
    def _apply_dynamic_curve(self, velocities: np.ndarray) -> np.ndarray:
        """Apply dynamic curve transformation."""
        if self.config.curve_type == "linear":
            return velocities
        
        # Normalize to 0-1 range for curve application
        v_min, v_max = velocities.min(), velocities.max()
        if v_max == v_min:
            return velocities
        
        normalized = (velocities - v_min) / (v_max - v_min)
        
        if self.config.curve_type == "logarithmic":
            # Emphasize quiet passages
            curved = np.log1p(normalized * 9) / np.log(10)
        elif self.config.curve_type == "exponential":
            # Emphasize loud passages
            curved = (np.exp(normalized) - 1) / (np.e - 1)
        else:
            curved = normalized
        
        # Scale back to velocity range
        return v_min + curved * (v_max - v_min)
    
    def _copy_midi_structure(self, midi_data: MidiData) -> MidiData:
        """Create empty copy of MIDI structure."""
        return MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=midi_data.filename
        )
    
    def analyze_dynamics(self, midi_data: MidiData) -> Dict:
        """
        Analyze dynamic patterns in MIDI data.
        
        Useful for understanding the piece's dynamic character.
        """
        all_velocities = []
        velocity_changes = []
        
        for instrument in midi_data.instruments:
            if not instrument.notes:
                continue
                
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
            inst_velocities = [n.velocity for n in sorted_notes]
            all_velocities.extend(inst_velocities)
            
            # Calculate velocity changes
            if len(inst_velocities) > 1:
                changes = np.diff(inst_velocities)
                velocity_changes.extend(changes)
        
        if not all_velocities:
            return {}
        
        velocities_array = np.array(all_velocities)
        
        # Dynamic range analysis
        dynamic_ranges = {
            "ppp": sum(velocities_array < 30),
            "pp": sum((velocities_array >= 30) & (velocities_array < 45)),
            "p": sum((velocities_array >= 45) & (velocities_array < 60)),
            "mp": sum((velocities_array >= 60) & (velocities_array < 75)),
            "mf": sum((velocities_array >= 75) & (velocities_array < 90)),
            "f": sum((velocities_array >= 90) & (velocities_array < 105)),
            "ff": sum((velocities_array >= 105) & (velocities_array < 120)),
            "fff": sum(velocities_array >= 120)
        }
        
        return {
            "num_notes": len(all_velocities),
            "velocity_range": [int(velocities_array.min()), int(velocities_array.max())],
            "mean_velocity": float(velocities_array.mean()),
            "std_velocity": float(velocities_array.std()),
            "dynamic_distribution": dynamic_ranges,
            "most_common_dynamic": max(dynamic_ranges.items(), key=lambda x: x[1])[0],
            "average_velocity_change": float(np.mean(np.abs(velocity_changes))) if velocity_changes else 0,
            "dynamic_contrast": float(velocities_array.max() - velocities_array.min())
        }