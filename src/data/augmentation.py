"""
Real-time data augmentation system for Aurl.ai.

Provides on-the-fly musical data augmentation to increase training variety
without storing millions of preprocessed files.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .midi_parser import MidiData, MidiNote, MidiInstrument


class AugmentationType(Enum):
    """Types of musical augmentation."""
    PITCH_TRANSPOSE = "pitch_transpose"
    TIME_STRETCH = "time_stretch"
    VELOCITY_SCALE = "velocity_scale"
    INSTRUMENT_SUBSTITUTE = "instrument_substitute"
    RHYTHMIC_VARIATION = "rhythmic_variation"
    HUMANIZATION = "humanization"


@dataclass
class AugmentationConfig:
    """Configuration for musical augmentation."""
    # Pitch transposition
    transpose_range: Tuple[int, int] = (-12, 12)  # Semitones
    transpose_probability: float = 0.7
    
    # Time stretching
    time_stretch_range: Tuple[float, float] = (0.85, 1.15)
    time_stretch_probability: float = 0.5
    preserve_rhythm_patterns: bool = True
    
    # Velocity scaling
    velocity_scale_range: Tuple[float, float] = (0.7, 1.3)
    velocity_scale_probability: float = 0.6
    preserve_velocity_curves: bool = True
    
    # Instrument substitution
    instrument_substitution_probability: float = 0.3
    allowed_instruments: List[int] = None  # MIDI program numbers
    
    # Rhythmic variations
    swing_probability: float = 0.2
    swing_ratio_range: Tuple[float, float] = (0.55, 0.67)
    humanization_probability: float = 0.4
    humanization_amount: float = 0.02  # Timing variation in seconds
    
    # Global settings
    max_simultaneous_augmentations: int = 3
    seed: Optional[int] = None


@dataclass
class AugmentationSchedule:
    """Scheduling parameters for augmentation during training."""
    epoch_schedule: Dict[int, float] = None  # epoch -> probability multiplier
    warmup_epochs: int = 5  # Start with reduced augmentation
    max_probability: float = 1.0
    min_probability: float = 0.1
    decay_factor: float = 0.95  # Reduce augmentation over time


class PitchTransposer:
    """Handles pitch transposition augmentation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def transpose(self, midi_data: MidiData, semitones: int) -> MidiData:
        """
        Transpose all notes by the specified number of semitones.
        
        Args:
            midi_data: Original MIDI data
            semitones: Number of semitones to transpose (-12 to +12)
            
        Returns:
            Transposed MIDI data
        """
        if semitones == 0:
            return midi_data
        
        # Create transposed copy
        transposed_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=[(t, (key + semitones) % 12) for t, key in midi_data.key_signature_changes],
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=f"{midi_data.filename}_transpose{semitones:+d}"
        )
        
        for instrument in midi_data.instruments:
            transposed_notes = []
            
            for note in instrument.notes:
                new_pitch = note.pitch + semitones
                
                # Keep within MIDI range (0-127)
                if 0 <= new_pitch <= 127:
                    transposed_notes.append(MidiNote(
                        pitch=new_pitch,
                        velocity=note.velocity,
                        start=note.start,
                        end=note.end
                    ))
                # Skip notes that go out of range
            
            if transposed_notes:  # Only add instrument if it has valid notes
                transposed_data.instruments.append(MidiInstrument(
                    program=instrument.program,
                    is_drum=instrument.is_drum,
                    name=instrument.name,
                    notes=transposed_notes,
                    pitch_bends=instrument.pitch_bends.copy(),
                    control_changes=instrument.control_changes.copy()
                ))
        
        return transposed_data


class TimeStretcher:
    """Handles time stretching with rhythm preservation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def stretch(self, midi_data: MidiData, factor: float) -> MidiData:
        """
        Stretch time by the specified factor.
        
        Args:
            midi_data: Original MIDI data
            factor: Time stretch factor (0.5 = half speed, 2.0 = double speed)
            
        Returns:
            Time-stretched MIDI data
        """
        if factor == 1.0:
            return midi_data
        
        # Ensure factor is a numeric type
        factor = float(factor)
        
        # Create stretched copy with error handling
        try:
            tempo_changes = []
            for t, tempo in midi_data.tempo_changes:
                tempo_changes.append((float(t) * factor, float(tempo) / factor))
            
            time_signature_changes = []
            for t, num, den in midi_data.time_signature_changes:
                time_signature_changes.append((float(t) * factor, num, den))
            
            key_signature_changes = []
            for t, key in midi_data.key_signature_changes:
                key_signature_changes.append((float(t) * factor, key))
            
            stretched_data = MidiData(
                instruments=[],
                tempo_changes=tempo_changes,
                time_signature_changes=time_signature_changes,
                key_signature_changes=key_signature_changes,
                resolution=midi_data.resolution,
                end_time=float(midi_data.end_time) * factor,
                filename=f"{midi_data.filename}_stretch{factor:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error in time stretch data preparation: {e}")
            raise
        
        for instrument in midi_data.instruments:
            stretched_notes = []
            
            for note in instrument.notes:
                try:
                    stretched_notes.append(MidiNote(
                        pitch=note.pitch,
                        velocity=note.velocity,
                        start=float(note.start) * factor,
                        end=float(note.end) * factor
                    ))
                except Exception as e:
                    self.logger.warning(f"Error stretching note: {e}, skipping")
                    continue
            
            # Handle pitch bends with error checking
            stretched_pitch_bends = []
            for t, value in instrument.pitch_bends:
                try:
                    stretched_pitch_bends.append((float(t) * factor, value))
                except Exception as e:
                    self.logger.warning(f"Error stretching pitch bend: {e}, skipping")
                    continue
            
            # Handle control changes with error checking
            stretched_control_changes = []
            for t, cc, value in instrument.control_changes:
                try:
                    stretched_control_changes.append((float(t) * factor, cc, value))
                except Exception as e:
                    self.logger.warning(f"Error stretching control change: {e}, skipping")
                    continue
            
            stretched_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=stretched_notes,
                pitch_bends=stretched_pitch_bends,
                control_changes=stretched_control_changes
            ))
        
        return stretched_data


class VelocityScaler:
    """Handles velocity scaling with dynamics preservation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def scale(self, midi_data: MidiData, factor: float) -> MidiData:
        """
        Scale velocities by the specified factor.
        
        Args:
            midi_data: Original MIDI data
            factor: Velocity scaling factor
            
        Returns:
            Velocity-scaled MIDI data
        """
        if factor == 1.0:
            return midi_data
        
        # Create scaled copy
        scaled_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=f"{midi_data.filename}_vel{factor:.2f}"
        )
        
        for instrument in midi_data.instruments:
            scaled_notes = []
            
            for note in instrument.notes:
                if self.config.preserve_velocity_curves:
                    # Scale relative to the piece's dynamic range
                    scaled_velocity = self._scale_preserving_curves(note.velocity, factor, midi_data)
                else:
                    # Simple linear scaling
                    scaled_velocity = int(note.velocity * factor)
                
                # Clamp to valid MIDI range
                scaled_velocity = max(1, min(127, scaled_velocity))
                
                scaled_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=scaled_velocity,
                    start=note.start,
                    end=note.end
                ))
            
            scaled_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=scaled_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return scaled_data
    
    def _scale_preserving_curves(self, velocity: int, factor: float, midi_data: MidiData) -> int:
        """Scale velocity while preserving dynamic curves."""
        # Get piece's velocity range
        all_velocities = [note.velocity for inst in midi_data.instruments for note in inst.notes]
        if not all_velocities:
            return velocity
        
        min_vel, max_vel = min(all_velocities), max(all_velocities)
        vel_range = max_vel - min_vel
        
        if vel_range == 0:
            return int(velocity * factor)
        
        # Normalize to 0-1, scale, then map back
        normalized = (velocity - min_vel) / vel_range
        scaled_normalized = normalized * factor
        
        # Map back to MIDI range with some headroom
        target_min = max(1, int(min_vel * factor))
        target_max = min(127, int(max_vel * factor))
        
        return int(target_min + scaled_normalized * (target_max - target_min))


class InstrumentSubstituter:
    """Handles instrument substitution for timbral variety."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Default instrument groups for substitution
        self.instrument_groups = {
            "piano": [0, 1, 2, 3, 4, 5, 6, 7],  # Piano family
            "chromatic": [8, 9, 10, 11, 12, 13, 14, 15],  # Chromatic percussion
            "organ": [16, 17, 18, 19, 20, 21, 22, 23],  # Organ family
            "guitar": [24, 25, 26, 27, 28, 29, 30, 31],  # Guitar family
            "bass": [32, 33, 34, 35, 36, 37, 38, 39],  # Bass family
            "strings": [40, 41, 42, 43, 44, 45, 46, 47],  # String section
            "ensemble": [48, 49, 50, 51, 52, 53, 54, 55],  # Ensemble
            "brass": [56, 57, 58, 59, 60, 61, 62, 63],  # Brass section
            "reed": [64, 65, 66, 67, 68, 69, 70, 71],  # Reed instruments
            "pipe": [72, 73, 74, 75, 76, 77, 78, 79],  # Pipe instruments
            "synth_lead": [80, 81, 82, 83, 84, 85, 86, 87],  # Synth lead
            "synth_pad": [88, 89, 90, 91, 92, 93, 94, 95],  # Synth pad
        }
    
    def substitute(self, midi_data: MidiData) -> MidiData:
        """
        Substitute instruments while maintaining musical character.
        
        Args:
            midi_data: Original MIDI data
            
        Returns:
            MIDI data with substituted instruments
        """
        # Create substituted copy
        substituted_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=f"{midi_data.filename}_instruments"
        )
        
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                # Don't substitute drum instruments
                substituted_data.instruments.append(instrument)
                continue
            
            # Find appropriate substitute
            new_program = self._find_substitute_instrument(instrument.program)
            
            substituted_data.instruments.append(MidiInstrument(
                program=new_program,
                is_drum=instrument.is_drum,
                name=f"Substituted {instrument.name}",
                notes=instrument.notes.copy(),
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return substituted_data
    
    def _find_substitute_instrument(self, original_program: int) -> int:
        """Find an appropriate substitute instrument."""
        if self.config.allowed_instruments:
            return random.choice(self.config.allowed_instruments)
        
        # Find the instrument group
        for group_name, programs in self.instrument_groups.items():
            if original_program in programs:
                # Choose a different instrument from the same group
                candidates = [p for p in programs if p != original_program]
                if candidates:
                    return random.choice(candidates)
        
        # If not in any group, choose a similar instrument
        return min(127, max(0, original_program + random.randint(-8, 8)))


class RhythmicVariator:
    """Handles rhythmic variations like swing and humanization."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def add_swing(self, midi_data: MidiData, swing_ratio: float) -> MidiData:
        """
        Add swing feel to the music.
        
        Args:
            midi_data: Original MIDI data
            swing_ratio: Swing ratio (0.5 = straight, 0.67 = strong swing)
            
        Returns:
            MIDI data with swing applied
        """
        if swing_ratio == 0.5:  # No swing
            return midi_data
        
        # Create swung copy
        swung_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=f"{midi_data.filename}_swing{swing_ratio:.2f}"
        )
        
        # Assume 4/4 time and quarter note = 1 beat
        beat_duration = 1.0  # seconds (at 60 BPM baseline)
        eighth_note = beat_duration / 2
        
        for instrument in midi_data.instruments:
            swung_notes = []
            
            for note in instrument.notes:
                new_start = self._apply_swing_timing(note.start, swing_ratio, eighth_note)
                new_end = self._apply_swing_timing(note.end, swing_ratio, eighth_note)
                
                swung_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start=new_start,
                    end=new_end
                ))
            
            swung_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=swung_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return swung_data
    
    def _apply_swing_timing(self, time: float, swing_ratio: float, eighth_note: float) -> float:
        """Apply swing timing to a specific time point."""
        # Find which eighth note position this is
        eighth_position = time / eighth_note
        eighth_index = int(eighth_position)
        
        # Apply swing to off-beats (odd eighth note positions)
        if eighth_index % 2 == 1:  # Off-beat
            # Calculate swing adjustment
            pair_start = (eighth_index // 2) * 2 * eighth_note
            swing_delay = eighth_note * (swing_ratio - 0.5) * 2
            return time + swing_delay
        
        return time
    
    def humanize(self, midi_data: MidiData, amount: float) -> MidiData:
        """
        Add subtle timing and velocity variations to humanize the performance.
        
        Args:
            midi_data: Original MIDI data
            amount: Humanization amount (seconds of timing variation)
            
        Returns:
            Humanized MIDI data
        """
        if amount == 0:
            return midi_data
        
        # Create humanized copy
        humanized_data = MidiData(
            instruments=[],
            tempo_changes=midi_data.tempo_changes.copy(),
            time_signature_changes=midi_data.time_signature_changes.copy(),
            key_signature_changes=midi_data.key_signature_changes.copy(),
            resolution=midi_data.resolution,
            end_time=midi_data.end_time,
            filename=f"{midi_data.filename}_humanized"
        )
        
        for instrument in midi_data.instruments:
            humanized_notes = []
            
            for note in instrument.notes:
                # Add subtle timing variations
                timing_variation = np.random.normal(0, amount / 3)  # 3-sigma within amount
                velocity_variation = int(np.random.normal(0, 3))  # ±3 velocity variation
                
                new_start = max(0, note.start + timing_variation)
                new_end = max(new_start + 0.01, note.end + timing_variation)  # Ensure minimum duration
                new_velocity = max(1, min(127, note.velocity + velocity_variation))
                
                humanized_notes.append(MidiNote(
                    pitch=note.pitch,
                    velocity=new_velocity,
                    start=new_start,
                    end=new_end
                ))
            
            humanized_data.instruments.append(MidiInstrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name,
                notes=humanized_notes,
                pitch_bends=instrument.pitch_bends.copy(),
                control_changes=instrument.control_changes.copy()
            ))
        
        return humanized_data


class MusicAugmenter:
    """
    Main augmentation system that coordinates all augmentation types.
    
    Features:
    - Multiple simultaneous augmentations
    - Probability-based selection
    - Training schedule support
    - Reproducible with seeds
    """
    
    def __init__(self, config: AugmentationConfig = None, schedule: AugmentationSchedule = None):
        self.config = config or AugmentationConfig()
        self.schedule = schedule or AugmentationSchedule()
        self.logger = logging.getLogger(__name__)
        
        # Initialize augmentation components
        self.pitch_transposer = PitchTransposer(self.config)
        self.time_stretcher = TimeStretcher(self.config)
        self.velocity_scaler = VelocityScaler(self.config)
        self.instrument_substituter = InstrumentSubstituter(self.config)
        self.rhythmic_variator = RhythmicVariator(self.config)
        
        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
    
    def augment(self, midi_data: MidiData, epoch: int = 0) -> Tuple[MidiData, Dict[str, Any]]:
        """
        Apply augmentations to MIDI data based on configuration and schedule.
        
        Args:
            midi_data: Original MIDI data
            epoch: Current training epoch (for scheduling)
            
        Returns:
            Tuple of (augmented MIDI data, augmentation info dict)
        """
        # Calculate probability multiplier based on epoch
        prob_multiplier = self._get_probability_multiplier(epoch)
        
        # Select which augmentations to apply
        selected_augmentations = self._select_augmentations(prob_multiplier)
        
        if not selected_augmentations:
            return midi_data, {}
        
        # Apply selected augmentations in sequence
        augmented_data = midi_data
        applied_augmentations = []
        
        for aug_type, params in selected_augmentations:
            try:
                if aug_type == AugmentationType.PITCH_TRANSPOSE:
                    augmented_data = self.pitch_transposer.transpose(augmented_data, params['semitones'])
                    applied_augmentations.append(f"transpose{params['semitones']:+d}")
                
                elif aug_type == AugmentationType.TIME_STRETCH:
                    augmented_data = self.time_stretcher.stretch(augmented_data, params['factor'])
                    applied_augmentations.append(f"stretch{params['factor']:.2f}")
                
                elif aug_type == AugmentationType.VELOCITY_SCALE:
                    augmented_data = self.velocity_scaler.scale(augmented_data, params['factor'])
                    applied_augmentations.append(f"vel{params['factor']:.2f}")
                
                elif aug_type == AugmentationType.INSTRUMENT_SUBSTITUTE:
                    augmented_data = self.instrument_substituter.substitute(augmented_data)
                    applied_augmentations.append("instruments")
                
                elif aug_type == AugmentationType.RHYTHMIC_VARIATION:
                    augmented_data = self.rhythmic_variator.add_swing(augmented_data, params['swing_ratio'])
                    applied_augmentations.append(f"swing{params['swing_ratio']:.2f}")
                
                elif aug_type == AugmentationType.HUMANIZATION:
                    augmented_data = self.rhythmic_variator.humanize(augmented_data, params['amount'])
                    applied_augmentations.append("humanized")
                
            except Exception as e:
                self.logger.warning(f"Failed to apply {aug_type.value}: {e}")
                continue
        
        # Update filename to reflect augmentations
        if applied_augmentations:
            augmented_data.filename = f"{midi_data.filename}_aug_{'_'.join(applied_augmentations)}"
        
        # Return augmented data and info about what was applied
        augmentation_info = {
            "applied_augmentations": applied_augmentations,
            "epoch": epoch,
            "probability_multiplier": prob_multiplier
        }
        
        return augmented_data, augmentation_info
    
    def _get_probability_multiplier(self, epoch: int) -> float:
        """Calculate probability multiplier based on training schedule."""
        if self.schedule.epoch_schedule and epoch in self.schedule.epoch_schedule:
            return self.schedule.epoch_schedule[epoch]
        
        # Warmup period
        if epoch < self.schedule.warmup_epochs:
            return self.schedule.min_probability + (self.schedule.max_probability - self.schedule.min_probability) * (epoch / self.schedule.warmup_epochs)
        
        # Gradual decay
        epochs_after_warmup = epoch - self.schedule.warmup_epochs
        return max(self.schedule.min_probability, 
                  self.schedule.max_probability * (self.schedule.decay_factor ** epochs_after_warmup))
    
    def _select_augmentations(self, prob_multiplier: float) -> List[Tuple[AugmentationType, Dict]]:
        """Select which augmentations to apply based on probabilities."""
        selected = []
        
        # Pitch transposition
        if random.random() < self.config.transpose_probability * prob_multiplier:
            semitones = random.randint(*self.config.transpose_range)
            if semitones != 0:  # Don't apply no-op transposition
                selected.append((AugmentationType.PITCH_TRANSPOSE, {'semitones': semitones}))
        
        # Time stretching
        if random.random() < self.config.time_stretch_probability * prob_multiplier:
            factor = random.uniform(*self.config.time_stretch_range)
            selected.append((AugmentationType.TIME_STRETCH, {'factor': factor}))
        
        # Velocity scaling
        if random.random() < self.config.velocity_scale_probability * prob_multiplier:
            factor = random.uniform(*self.config.velocity_scale_range)
            selected.append((AugmentationType.VELOCITY_SCALE, {'factor': factor}))
        
        # Instrument substitution
        if random.random() < self.config.instrument_substitution_probability * prob_multiplier:
            selected.append((AugmentationType.INSTRUMENT_SUBSTITUTE, {}))
        
        # Swing
        if random.random() < self.config.swing_probability * prob_multiplier:
            swing_ratio = random.uniform(*self.config.swing_ratio_range)
            selected.append((AugmentationType.RHYTHMIC_VARIATION, {'swing_ratio': swing_ratio}))
        
        # Humanization
        if random.random() < self.config.humanization_probability * prob_multiplier:
            selected.append((AugmentationType.HUMANIZATION, {'amount': self.config.humanization_amount}))
        
        # Limit simultaneous augmentations
        if len(selected) > self.config.max_simultaneous_augmentations:
            selected = random.sample(selected, self.config.max_simultaneous_augmentations)
        
        return selected
    
    def get_augmentation_stats(self) -> Dict:
        """Get statistics about augmentation usage."""
        return {
            "config": {
                "transpose_probability": self.config.transpose_probability,
                "time_stretch_probability": self.config.time_stretch_probability,
                "velocity_scale_probability": self.config.velocity_scale_probability,
                "instrument_substitution_probability": self.config.instrument_substitution_probability,
                "swing_probability": self.config.swing_probability,
                "humanization_probability": self.config.humanization_probability,
                "max_simultaneous": self.config.max_simultaneous_augmentations
            },
            "schedule": {
                "warmup_epochs": self.schedule.warmup_epochs,
                "max_probability": self.schedule.max_probability,
                "min_probability": self.schedule.min_probability,
                "decay_factor": self.schedule.decay_factor
            }
        }