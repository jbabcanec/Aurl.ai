"""
Musical quantization implementation for Aurl.ai.

Provides intelligent quantization that preserves musical feel while
aligning notes to a grid for cleaner training data.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import logging

from .midi_parser import MidiData, MidiNote, MidiInstrument


class GridResolution(Enum):
    """Standard musical grid resolutions."""
    WHOLE = 4.0          # Whole note
    HALF = 2.0           # Half note
    QUARTER = 1.0        # Quarter note
    EIGHTH = 0.5         # Eighth note
    SIXTEENTH = 0.25     # 16th note
    THIRTY_SECOND = 0.125  # 32nd note
    SIXTY_FOURTH = 0.0625  # 64th note (for extreme virtuosity)
    TRIPLET_QUARTER = 0.667  # Quarter note triplet
    TRIPLET_EIGHTH = 0.333  # Eighth note triplet
    TRIPLET_SIXTEENTH = 0.167  # 16th note triplet
    TRIPLET_THIRTY_SECOND = 0.083  # 32nd note triplet


@dataclass
class QuantizationConfig:
    """Configuration for musical quantization."""
    resolution: GridResolution = GridResolution.THIRTY_SECOND  # Default to 32nd note precision
    adaptive_resolution: bool = True  # Auto-detect needed resolution
    strength: float = 1.0  # 0.0 = no quantization, 1.0 = full snap
    swing_ratio: float = 0.0  # 0.0 = straight, 0.67 = full swing
    preserve_micro_timing: bool = True  # Keep small timing variations
    humanize_amount: float = 0.0  # Add slight randomness after quantization


class MusicalQuantizer:
    """
    Implements musical quantization with groove preservation.
    
    Features:
    - Multiple grid resolutions (16th notes, triplets, etc.)
    - Partial quantization to preserve feel
    - Swing quantization
    - Micro-timing preservation for humanization
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.logger = logging.getLogger(__name__)
        
    def quantize_midi_data(self, midi_data: MidiData) -> MidiData:
        """
        Apply musical quantization to entire MIDI data.
        
        Args:
            midi_data: Original MIDI data
            
        Returns:
            Quantized MIDI data
        """
        # Create a copy to avoid modifying original
        quantized_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=midi_data.filename
        )
        
        # Quantize each instrument
        for instrument in midi_data.instruments:
            quantized_instrument = self._quantize_instrument(instrument)
            quantized_data.instruments.append(quantized_instrument)
        
        # Update end time based on quantized notes
        if quantized_data.instruments:
            max_end_time = max(
                note.end for inst in quantized_data.instruments 
                for note in inst.notes
            )
            quantized_data.end_time = max_end_time
        
        return quantized_data
    
    def _quantize_instrument(self, instrument: MidiInstrument) -> MidiInstrument:
        """Quantize all notes in an instrument."""
        quantized_notes = []
        
        for note in instrument.notes:
            quantized_note = self._quantize_note(note)
            quantized_notes.append(quantized_note)
        
        return MidiInstrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
            notes=quantized_notes,
            pitch_bends=instrument.pitch_bends.copy(),
            control_changes=instrument.control_changes.copy()
        )
    
    def _quantize_note(self, note: MidiNote) -> MidiNote:
        """
        Quantize a single note with configurable strength.
        
        Args:
            note: Original note
            
        Returns:
            Quantized note
        """
        # Get grid size in seconds (assuming 120 BPM as base)
        grid_size = self._get_grid_size_seconds()
        
        # Quantize start time
        quantized_start = self._quantize_time(note.start, grid_size)
        
        # Apply swing if configured
        if self.config.swing_ratio > 0:
            quantized_start = self._apply_swing(quantized_start, grid_size)
        
        # Blend original and quantized based on strength
        final_start = self._blend_times(note.start, quantized_start, self.config.strength)
        
        # Preserve micro-timing if enabled
        if self.config.preserve_micro_timing:
            micro_timing = self._extract_micro_timing(note.start, grid_size)
            final_start += micro_timing * 0.1  # Scale down micro variations
        
        # Add humanization
        if self.config.humanize_amount > 0:
            final_start += np.random.normal(0, self.config.humanize_amount * grid_size * 0.1)
        
        # Quantize duration (optional - often better to preserve original)
        duration = note.end - note.start
        # Keep original duration for now to preserve musical phrasing
        
        return MidiNote(
            pitch=note.pitch,
            velocity=note.velocity,
            start=max(0, final_start),  # Ensure non-negative
            end=max(0, final_start + duration)
        )
    
    def _get_grid_size_seconds(self, bpm: float = 120.0) -> float:
        """Convert grid resolution to seconds."""
        # Quarter note duration at given BPM
        quarter_duration = 60.0 / bpm
        return self.config.resolution.value * quarter_duration
    
    def _quantize_time(self, time: float, grid_size: float) -> float:
        """Quantize time to nearest grid position."""
        return round(time / grid_size) * grid_size
    
    def _apply_swing(self, time: float, grid_size: float) -> float:
        """
        Apply swing timing to off-beats.
        
        Swing delays every other grid position to create shuffle feel.
        """
        grid_position = int(round(time / grid_size))
        
        # Apply swing to off-beats (odd positions)
        if grid_position % 2 == 1:
            swing_delay = grid_size * self.config.swing_ratio * 0.2
            return time + swing_delay
        
        return time
    
    def _blend_times(self, original: float, quantized: float, strength: float) -> float:
        """Blend original and quantized times based on strength."""
        return original * (1 - strength) + quantized * strength
    
    def _extract_micro_timing(self, time: float, grid_size: float) -> float:
        """Extract micro-timing deviation from grid."""
        quantized = round(time / grid_size) * grid_size
        return time - quantized
    
    def analyze_timing(self, midi_data: MidiData) -> Dict:
        """
        Analyze timing patterns in MIDI data.
        
        Useful for determining best quantization settings.
        """
        all_starts = []
        all_durations = []
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_starts.append(note.start)
                all_durations.append(note.end - note.start)
        
        if not all_starts:
            return {}
        
        # Analyze note onset intervals
        all_starts.sort()
        intervals = np.diff(all_starts)
        
        # Detect common subdivisions
        grid_size = self._get_grid_size_seconds()
        
        # Calculate how well notes align to different grids
        alignments = {}
        for resolution in GridResolution:
            test_grid = resolution.value * (60.0 / 120.0)  # Assume 120 BPM
            deviations = []
            
            for start in all_starts:
                quantized = round(start / test_grid) * test_grid
                deviation = abs(start - quantized)
                deviations.append(deviation)
            
            alignments[resolution.name] = {
                "mean_deviation": np.mean(deviations),
                "max_deviation": np.max(deviations),
                "alignment_score": 1.0 - (np.mean(deviations) / test_grid)
            }
        
        return {
            "num_notes": len(all_starts),
            "duration": midi_data.end_time,
            "mean_interval": np.mean(intervals) if len(intervals) > 0 else 0,
            "min_interval": np.min(intervals) if len(intervals) > 0 else 0,
            "grid_alignments": alignments,
            "recommended_resolution": max(alignments.items(), 
                                         key=lambda x: x[1]["alignment_score"])[0]
        }


def create_adaptive_quantizer(midi_data: MidiData) -> MusicalQuantizer:
    """
    Create a quantizer with settings adapted to the input MIDI.
    
    Args:
        midi_data: MIDI data to analyze
        
    Returns:
        Configured MusicalQuantizer
    """
    quantizer = MusicalQuantizer()
    analysis = quantizer.analyze_timing(midi_data)
    
    if analysis:
        # Set resolution based on analysis
        recommended = analysis.get("recommended_resolution", "SIXTEENTH")
        try:
            resolution = GridResolution[recommended]
            config = QuantizationConfig(
                resolution=resolution,
                strength=0.8,  # Not full strength to preserve feel
                preserve_micro_timing=True
            )
            return MusicalQuantizer(config)
        except KeyError:
            pass
    
    # Return default quantizer if analysis fails
    return quantizer