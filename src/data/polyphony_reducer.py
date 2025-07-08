"""
Polyphony reduction for Aurl.ai.

Intelligently reduces polyphony while preserving musical essence.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from .midi_parser import MidiData, MidiNote, MidiInstrument


@dataclass
class PolyphonyReductionConfig:
    """Configuration for polyphony reduction."""
    max_polyphony: int = 6  # Maximum simultaneous notes
    priority_mode: str = "musical"  # "musical", "velocity", "pitch", "newest"
    preserve_bass: bool = True  # Keep lowest notes
    preserve_melody: bool = True  # Keep highest/loudest notes
    voice_leading_weight: float = 0.5  # Consider voice leading in decisions


class PolyphonyReducer:
    """
    Reduces polyphony in MIDI data while preserving musicality.
    
    Features:
    - Multiple prioritization strategies
    - Voice leading preservation
    - Bass and melody preservation
    - Intelligent note selection
    """
    
    def __init__(self, config: PolyphonyReductionConfig = None):
        self.config = config or PolyphonyReductionConfig()
        self.logger = logging.getLogger(__name__)
    
    def reduce_polyphony(self, midi_data: MidiData) -> MidiData:
        """
        Reduce polyphony in MIDI data.
        
        Args:
            midi_data: Original MIDI data
            
        Returns:
            MIDI data with reduced polyphony
        """
        reduced_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=midi_data.filename
        )
        
        for instrument in midi_data.instruments:
            reduced_instrument = self._reduce_instrument_polyphony(instrument)
            reduced_data.instruments.append(reduced_instrument)
        
        return reduced_data
    
    def _reduce_instrument_polyphony(self, instrument: MidiInstrument) -> MidiInstrument:
        """Reduce polyphony for a single instrument."""
        if not instrument.notes:
            return MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=[],
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            )
        
        # Sort notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
        
        # Process notes in time order
        reduced_notes = []
        active_notes: List[MidiNote] = []
        kept_note_indices = set()  # Track indices of notes we're keeping
        
        # Create index mapping
        note_to_index = {id(note): i for i, note in enumerate(sorted_notes)}
        
        for i, note in enumerate(sorted_notes):
            # Remove notes that have ended
            active_notes = [n for n in active_notes if n.end > note.start]
            
            # Add current note
            active_notes.append(note)
            
            # If we exceed max polyphony, choose which notes to keep
            if len(active_notes) > self.config.max_polyphony:
                selected_notes = self._select_notes_to_keep(active_notes, note.start)
                # Update kept indices based on selection
                for n in active_notes:
                    idx = note_to_index.get(id(n))
                    if idx is not None:
                        if n in selected_notes:
                            kept_note_indices.add(idx)
                        else:
                            kept_note_indices.discard(idx)
                active_notes = selected_notes
            else:
                # All active notes are kept
                for n in active_notes:
                    idx = note_to_index.get(id(n))
                    if idx is not None:
                        kept_note_indices.add(idx)
        
        # Build final reduced notes list maintaining original order
        for i, note in enumerate(sorted_notes):
            if i in kept_note_indices:
                reduced_notes.append(note)
        
        return MidiInstrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
            notes=reduced_notes,
            pitch_bends=instrument.pitch_bends.copy(),
            control_changes=instrument.control_changes.copy()
        )
    
    def _select_notes_to_keep(self, notes: List[MidiNote], current_time: float) -> List[MidiNote]:
        """
        Select which notes to keep when polyphony exceeds limit.
        
        Args:
            notes: Currently active notes
            current_time: Current time position
            
        Returns:
            Selected notes to keep (max_polyphony count)
        """
        if self.config.priority_mode == "musical":
            return self._musical_selection(notes, current_time)
        elif self.config.priority_mode == "velocity":
            return self._velocity_selection(notes)
        elif self.config.priority_mode == "pitch":
            return self._pitch_selection(notes)
        elif self.config.priority_mode == "newest":
            return self._newest_selection(notes)
        else:
            return notes[:self.config.max_polyphony]
    
    def _musical_selection(self, notes: List[MidiNote], current_time: float) -> List[MidiNote]:
        """
        Select notes based on musical importance.
        
        Considers:
        - Bass notes (lowest)
        - Melody notes (highest/loudest)
        - Voice leading
        - Velocity
        - Duration
        """
        # Calculate importance scores
        scores = []
        
        for note in notes:
            score = 0.0
            
            # Velocity importance (louder = more important)
            score += note.velocity / 127.0 * 0.3
            
            # Duration importance (longer = more important)
            remaining_duration = note.end - current_time
            score += min(remaining_duration / 2.0, 1.0) * 0.2
            
            # Pitch extremes (bass and soprano lines)
            all_pitches = [n.pitch for n in notes]
            if self.config.preserve_bass and note.pitch == min(all_pitches):
                score += 0.5
            if self.config.preserve_melody and note.pitch == max(all_pitches):
                score += 0.4
            
            # Recent onset (newer notes might be melodically important)
            time_since_onset = current_time - note.start
            if time_since_onset < 0.5:  # Recent note
                score += 0.2
            
            scores.append((score, note))
        
        # Sort by score and select top notes
        scores.sort(key=lambda x: x[0], reverse=True)
        return [note for _, note in scores[:self.config.max_polyphony]]
    
    def _velocity_selection(self, notes: List[MidiNote]) -> List[MidiNote]:
        """Select notes with highest velocities."""
        sorted_notes = sorted(notes, key=lambda n: n.velocity, reverse=True)
        return sorted_notes[:self.config.max_polyphony]
    
    def _pitch_selection(self, notes: List[MidiNote]) -> List[MidiNote]:
        """Select notes based on pitch priority."""
        selected = []
        
        # Always keep bass if configured
        if self.config.preserve_bass and notes:
            bass_note = min(notes, key=lambda n: n.pitch)
            selected.append(bass_note)
            notes = [n for n in notes if n != bass_note]
        
        # Always keep highest if configured
        if self.config.preserve_melody and notes:
            melody_note = max(notes, key=lambda n: n.pitch)
            if melody_note not in selected:
                selected.append(melody_note)
                notes = [n for n in notes if n != melody_note]
        
        # Fill remaining slots with middle voices by velocity
        remaining_slots = self.config.max_polyphony - len(selected)
        if remaining_slots > 0 and notes:
            sorted_remaining = sorted(notes, key=lambda n: n.velocity, reverse=True)
            selected.extend(sorted_remaining[:remaining_slots])
        
        return selected
    
    def _newest_selection(self, notes: List[MidiNote]) -> List[MidiNote]:
        """Select most recently started notes."""
        sorted_notes = sorted(notes, key=lambda n: n.start, reverse=True)
        return sorted_notes[:self.config.max_polyphony]
    
    def analyze_polyphony(self, midi_data: MidiData) -> Dict:
        """
        Analyze polyphony patterns in MIDI data.
        
        Returns statistics about polyphony usage.
        """
        # Time-based polyphony analysis
        time_resolution = 0.05  # 50ms resolution
        max_time = midi_data.end_time
        time_steps = int(max_time / time_resolution) + 1
        
        polyphony_over_time = defaultdict(lambda: defaultdict(int))
        
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                start_step = int(note.start / time_resolution)
                end_step = int(note.end / time_resolution)
                
                for step in range(start_step, min(end_step + 1, time_steps)):
                    polyphony_over_time[inst_idx][step] += 1
        
        # Calculate statistics
        all_polyphony_values = []
        max_polyphony_per_instrument = {}
        
        for inst_idx, time_data in polyphony_over_time.items():
            if time_data:
                values = list(time_data.values())
                all_polyphony_values.extend(values)
                max_polyphony_per_instrument[inst_idx] = max(values)
        
        if not all_polyphony_values:
            return {
                "max_polyphony": 0,
                "mean_polyphony": 0,
                "polyphony_distribution": {},
                "reduction_needed": False
            }
        
        # Polyphony distribution
        polyphony_counts = defaultdict(int)
        for value in all_polyphony_values:
            polyphony_counts[value] += 1
        
        # Convert to percentages
        total_samples = len(all_polyphony_values)
        polyphony_distribution = {
            str(k): v / total_samples * 100 
            for k, v in sorted(polyphony_counts.items())
        }
        
        max_polyphony = max(all_polyphony_values)
        
        return {
            "max_polyphony": max_polyphony,
            "mean_polyphony": np.mean(all_polyphony_values),
            "std_polyphony": np.std(all_polyphony_values),
            "polyphony_distribution": polyphony_distribution,
            "max_per_instrument": max_polyphony_per_instrument,
            "reduction_needed": max_polyphony > self.config.max_polyphony,
            "time_above_threshold": sum(1 for v in all_polyphony_values 
                                       if v > self.config.max_polyphony) / total_samples * 100
        }