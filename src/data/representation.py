"""
Data representation system for Aurl.ai.

This module implements a hybrid representation that combines:
1. Event-based sequences (note_on, note_off, time_shift, velocity_change)
2. Piano roll features (polyphonic representation)
3. Efficient tokenization for neural network consumption

The system preserves musical meaning while enabling scalable processing.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import IntEnum
import logging

from src.data.midi_parser import MidiData, MidiNote, MidiInstrument
from src.utils.base_logger import setup_logger
from src.utils.constants import (
    MIDI_NOTE_MIN, MIDI_NOTE_MAX, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX
)

logger = setup_logger(__name__)


class EventType(IntEnum):
    """Musical event types for sequence representation."""
    # Note events
    NOTE_ON = 0
    NOTE_OFF = 1
    
    # Timing events
    TIME_SHIFT = 2
    
    # Velocity events
    VELOCITY_CHANGE = 3
    
    # Control events
    SUSTAIN_ON = 4
    SUSTAIN_OFF = 5
    
    # Meta events
    TEMPO_CHANGE = 6
    TIME_SIGNATURE = 7
    KEY_SIGNATURE = 8
    
    # Instrument events
    INSTRUMENT_CHANGE = 9
    
    # Special tokens
    START_TOKEN = 10
    END_TOKEN = 11
    PAD_TOKEN = 12
    UNK_TOKEN = 13


@dataclass
class MusicEvent:
    """Represents a single musical event in the sequence."""
    event_type: EventType
    value: int = 0        # Primary value (pitch, velocity, time, etc.)
    channel: int = 0      # MIDI channel (0-15)
    time: float = 0.0     # Absolute time in seconds
    
    def to_token(self, vocab_config: 'VocabularyConfig') -> int:
        """Convert event to vocabulary token."""
        return vocab_config.event_to_token(self)
    
    def __str__(self) -> str:
        return f"{self.event_type.name}(val={self.value}, ch={self.channel}, t={self.time:.3f})"


@dataclass
class VocabularyConfig:
    """Configuration for the musical vocabulary and tokenization."""
    
    # Event ranges
    max_pitch: int = MIDI_NOTE_MAX
    min_pitch: int = MIDI_NOTE_MIN
    max_velocity: int = MIDI_VELOCITY_MAX
    min_velocity: int = MIDI_VELOCITY_MIN
    
    # Time quantization (32nd note precision)
    time_shift_bins: int = 512      # Up to ~8 seconds at 15.625ms resolution
    time_shift_ms: float = 15.625   # 15.625ms per time shift (32nd note at 120 BPM)
    adaptive_resolution: bool = True # Use coarser resolution for simple pieces
    
    # Velocity quantization
    velocity_bins: int = 32         # Quantize 128 velocities to 32 bins
    
    # Tempo range (BPM)
    min_tempo: int = 60
    max_tempo: int = 200
    tempo_bins: int = 32
    
    # Maximum instruments
    max_instruments: int = 16
    
    def __post_init__(self):
        """Calculate vocabulary size and token mappings."""
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build the complete vocabulary mapping."""
        self.token_to_event = {}
        self.event_to_token_map = {}
        
        current_token = 0
        
        # Special tokens
        for event_type in [EventType.START_TOKEN, EventType.END_TOKEN, 
                          EventType.PAD_TOKEN, EventType.UNK_TOKEN]:
            self.token_to_event[current_token] = (event_type, 0)
            self.event_to_token_map[(event_type, 0)] = current_token
            current_token += 1
        
        # Note events (NOTE_ON, NOTE_OFF)
        for event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
            for pitch in range(self.min_pitch, self.max_pitch + 1):
                self.token_to_event[current_token] = (event_type, pitch)
                self.event_to_token_map[(event_type, pitch)] = current_token
                current_token += 1
        
        # Time shift events
        for time_bin in range(self.time_shift_bins):
            self.token_to_event[current_token] = (EventType.TIME_SHIFT, time_bin)
            self.event_to_token_map[(EventType.TIME_SHIFT, time_bin)] = current_token
            current_token += 1
        
        # Velocity change events
        for vel_bin in range(self.velocity_bins):
            self.token_to_event[current_token] = (EventType.VELOCITY_CHANGE, vel_bin)
            self.event_to_token_map[(EventType.VELOCITY_CHANGE, vel_bin)] = current_token
            current_token += 1
        
        # Control events
        for event_type in [EventType.SUSTAIN_ON, EventType.SUSTAIN_OFF]:
            self.token_to_event[current_token] = (event_type, 0)
            self.event_to_token_map[(event_type, 0)] = current_token
            current_token += 1
        
        # Tempo events
        for tempo_bin in range(self.tempo_bins):
            self.token_to_event[current_token] = (EventType.TEMPO_CHANGE, tempo_bin)
            self.event_to_token_map[(EventType.TEMPO_CHANGE, tempo_bin)] = current_token
            current_token += 1
        
        # Instrument change events
        for instrument in range(self.max_instruments):
            self.token_to_event[current_token] = (EventType.INSTRUMENT_CHANGE, instrument)
            self.event_to_token_map[(EventType.INSTRUMENT_CHANGE, instrument)] = current_token
            current_token += 1
        
        self.vocab_size = current_token
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
    
    def event_to_token(self, event: MusicEvent) -> int:
        """Convert a MusicEvent to a vocabulary token."""
        key = (event.event_type, event.value)
        return self.event_to_token_map.get(key, self.event_to_token_map[(EventType.UNK_TOKEN, 0)])
    
    def token_to_event_info(self, token: int) -> Tuple[EventType, int]:
        """Convert a token back to event type and value."""
        return self.token_to_event.get(token, (EventType.UNK_TOKEN, 0))
    
    def quantize_velocity(self, velocity: int) -> int:
        """Quantize MIDI velocity to vocabulary bins."""
        if velocity == 0:
            return 0  # Note off
        # Map 1-127 to 1-(velocity_bins-1)
        return max(1, min(self.velocity_bins - 1, 
                         int((velocity - 1) * (self.velocity_bins - 1) / 126) + 1))
    
    def quantize_time_shift(self, time_delta: float, piece_complexity: str = "complex") -> int:
        """
        Quantize time delta to time shift bins with adaptive resolution.
        
        Args:
            time_delta: Time difference in seconds
            piece_complexity: "simple" for 16th note res, "complex" for 32nd note res
        """
        # Use adaptive resolution based on piece complexity
        if self.adaptive_resolution and piece_complexity == "simple":
            # Use 4x coarser resolution for simple pieces (125ms = 16th note precision)
            effective_resolution = self.time_shift_ms * 4
        else:
            # Use full 32nd note precision for complex pieces
            effective_resolution = self.time_shift_ms
        
        time_ms = time_delta * 1000
        bins = int(time_ms / effective_resolution)
        return min(self.time_shift_bins - 1, max(0, bins))
    
    def detect_piece_complexity(self, midi_data) -> str:
        """
        Detect if piece needs fine timing resolution based on musical content.
        
        Returns:
            "simple" if piece can use 16th note resolution
            "complex" if piece needs 32nd note resolution
        """
        if not hasattr(midi_data, 'instruments'):
            return "complex"  # Default to high precision
            
        # Analyze note timing to detect fine subdivisions
        all_note_times = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_note_times.extend([note.start, note.end])
        
        if not all_note_times:
            return "simple"
        
        # Check for timing patterns that suggest complex rhythms
        all_note_times = sorted(set(all_note_times))
        
        # Look for very short note durations (< 100ms = likely 32nd notes or faster)
        min_gap = float('inf')
        for i in range(1, len(all_note_times)):
            gap = all_note_times[i] - all_note_times[i-1]
            if gap > 0.001:  # Ignore tiny floating point differences
                min_gap = min(min_gap, gap)
        
        # If we have gaps smaller than 100ms, we likely need fine resolution
        if min_gap < 0.1:  # 100ms threshold
            return "complex"
        
        # Check for triplet-like timing (times that don't align to 16th note grid)
        sixteenth_note_at_120bpm = 0.125  # 125ms
        
        for time_val in all_note_times[:20]:  # Check first 20 times
            # See if time aligns to 16th note grid
            grid_position = time_val / sixteenth_note_at_120bpm
            if abs(grid_position - round(grid_position)) > 0.1:  # Not close to grid
                return "complex"
        
        return "simple"
    
    def quantize_tempo(self, tempo: float) -> int:
        """Quantize tempo (BPM) to tempo bins."""
        clamped_tempo = max(self.min_tempo, min(self.max_tempo, tempo))
        normalized = (clamped_tempo - self.min_tempo) / (self.max_tempo - self.min_tempo)
        return int(normalized * (self.tempo_bins - 1))


@dataclass
class PianoRollConfig:
    """Configuration for piano roll representation."""
    
    # Pitch range
    min_pitch: int = MIDI_NOTE_MIN
    max_pitch: int = MIDI_NOTE_MAX
    
    # Time resolution (32nd note precision)
    time_resolution: float = 0.015625  # 15.625ms per step (32nd note at 120 BPM)
    
    # Feature dimensions
    velocity_feature: bool = True
    onset_feature: bool = True      # Mark note onsets
    offset_feature: bool = True     # Mark note offsets
    
    @property
    def pitch_bins(self) -> int:
        return self.max_pitch - self.min_pitch + 1
    
    def time_to_step(self, time: float) -> int:
        """Convert time in seconds to time step."""
        return int(time / self.time_resolution)
    
    def step_to_time(self, step: int) -> float:
        """Convert time step to time in seconds."""
        return step * self.time_resolution


@dataclass
class MusicalMetadata:
    """Metadata associated with musical content."""
    
    # Basic information
    title: Optional[str] = None
    composer: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    
    # Musical characteristics
    key: Optional[str] = None          # "C major", "A minor", etc.
    mode: Optional[str] = None         # "major", "minor", "dorian", etc.
    time_signature: str = "4/4"
    tempo_marking: Optional[str] = None # "Allegro", "Andante", etc.
    
    # Style and analysis
    style_tags: List[str] = field(default_factory=list)  # ["classical", "baroque", "jazz"]
    difficulty_level: Optional[int] = None  # 1-10 scale
    emotion_tags: List[str] = field(default_factory=list)  # ["happy", "melancholy", "energetic"]
    
    # Technical metadata
    source_file: Optional[str] = None
    processing_date: Optional[str] = None
    processing_version: Optional[str] = None
    
    # Augmentation metadata
    augmented: bool = False
    augmentation_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "composer": self.composer,
            "genre": self.genre,
            "year": self.year,
            "key": self.key,
            "mode": self.mode,
            "time_signature": self.time_signature,
            "tempo_marking": self.tempo_marking,
            "style_tags": self.style_tags,
            "difficulty_level": self.difficulty_level,
            "emotion_tags": self.emotion_tags,
            "source_file": self.source_file,
            "processing_date": self.processing_date,
            "processing_version": self.processing_version,
            "augmented": self.augmented,
            "augmentation_info": self.augmentation_info
        }


@dataclass 
class MusicalRepresentation:
    """Combined representation of musical data."""
    
    # Event sequence representation
    events: List[MusicEvent] = field(default_factory=list)
    tokens: Optional[np.ndarray] = None
    
    # Piano roll representation
    piano_roll: Optional[np.ndarray] = None      # Shape: (time_steps, pitch_bins)
    velocity_roll: Optional[np.ndarray] = None   # Shape: (time_steps, pitch_bins)
    onset_roll: Optional[np.ndarray] = None      # Shape: (time_steps, pitch_bins)
    offset_roll: Optional[np.ndarray] = None     # Shape: (time_steps, pitch_bins)
    
    # Core musical data
    duration: float = 0.0
    num_instruments: int = 0
    tempo_changes: List[Tuple[float, float]] = field(default_factory=list)
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: int = 0  # C major
    
    # Statistics
    pitch_range: Tuple[int, int] = (60, 72)  # C4 to C5
    velocity_range: Tuple[int, int] = (64, 96)
    note_density: float = 0.0  # Notes per second
    
    # Extended metadata
    metadata: Optional[MusicalMetadata] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a human-readable summary of the representation."""
        summary = {
            "duration": self.duration,
            "num_events": len(self.events),
            "num_tokens": len(self.tokens) if self.tokens is not None else 0,
            "piano_roll_shape": self.piano_roll.shape if self.piano_roll is not None else None,
            "pitch_range": self.pitch_range,
            "velocity_range": self.velocity_range,
            "note_density": self.note_density,
            "num_instruments": self.num_instruments,
            "tempo_changes": len(self.tempo_changes),
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}"
        }
        
        if self.metadata:
            summary["metadata"] = self.metadata.to_dict()
        
        return summary


class MusicRepresentationConverter:
    """Converts MIDI data to various musical representations."""
    
    def __init__(self, vocab_config: VocabularyConfig = None, 
                 piano_roll_config: PianoRollConfig = None):
        self.vocab_config = vocab_config or VocabularyConfig()
        self.piano_roll_config = piano_roll_config or PianoRollConfig()
        self.logger = logger
    
    def midi_to_representation(self, midi_data: MidiData) -> MusicalRepresentation:
        """Convert MIDI data to complete musical representation."""
        
        # Create event sequence
        events = self._create_event_sequence(midi_data)
        
        # Create piano roll
        piano_roll, velocity_roll, onset_roll, offset_roll = self._create_piano_roll(midi_data)
        
        # Tokenize events
        tokens = self._tokenize_events(events)
        
        # Calculate statistics
        stats = self._calculate_statistics(midi_data, events)
        
        representation = MusicalRepresentation(
            events=events,
            tokens=tokens,
            piano_roll=piano_roll,
            velocity_roll=velocity_roll,
            onset_roll=onset_roll,
            offset_roll=offset_roll,
            **stats
        )
        
        self.logger.info(f"Created representation: {representation.get_summary()}")
        return representation
    
    def _create_event_sequence(self, midi_data: MidiData) -> List[MusicEvent]:
        """Create a chronological sequence of musical events."""
        events = []
        
        # Add start token
        events.append(MusicEvent(EventType.START_TOKEN, 0, 0, 0.0))
        
        # Collect all note events with timing
        note_events = []
        
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                # Note on event
                note_events.append((note.start, MusicEvent(
                    EventType.NOTE_ON, note.pitch, inst_idx, note.start
                )))
                
                # Note off event
                note_events.append((note.end, MusicEvent(
                    EventType.NOTE_OFF, note.pitch, inst_idx, note.end
                )))
        
        # Sort events by time
        note_events.sort(key=lambda x: x[0])
        
        # Add tempo changes
        for tempo_time, tempo in midi_data.tempo_changes:
            tempo_bin = self.vocab_config.quantize_tempo(tempo)
            note_events.append((tempo_time, MusicEvent(
                EventType.TEMPO_CHANGE, tempo_bin, 0, tempo_time
            )))
        
        # Sort again after adding tempo events
        note_events.sort(key=lambda x: x[0])
        
        # Convert to event sequence with time shifts
        current_time = 0.0
        current_velocity = 64  # Default velocity
        current_instrument = 0
        
        for event_time, event in note_events:
            # Add time shift if needed
            time_delta = event_time - current_time
            if time_delta > 0:
                time_shift_bins = self.vocab_config.quantize_time_shift(time_delta)
                if time_shift_bins > 0:
                    events.append(MusicEvent(
                        EventType.TIME_SHIFT, time_shift_bins, 0, event_time
                    ))
            
            # Add instrument change if needed
            if event.channel != current_instrument:
                events.append(MusicEvent(
                    EventType.INSTRUMENT_CHANGE, event.channel, 0, event_time
                ))
                current_instrument = event.channel
            
            # For note on events, add velocity change if needed
            if event.event_type == EventType.NOTE_ON:
                # Find the note to get velocity
                note_velocity = self._find_note_velocity(midi_data, event)
                quantized_velocity = self.vocab_config.quantize_velocity(note_velocity)
                
                if quantized_velocity != current_velocity:
                    events.append(MusicEvent(
                        EventType.VELOCITY_CHANGE, quantized_velocity, 
                        event.channel, event_time
                    ))
                    current_velocity = quantized_velocity
            
            # Add the main event
            events.append(event)
            current_time = event_time
        
        # Add end token
        events.append(MusicEvent(EventType.END_TOKEN, 0, 0, current_time))
        
        return events
    
    def _find_note_velocity(self, midi_data: MidiData, note_event: MusicEvent) -> int:
        """Find the velocity for a note on event."""
        instrument = midi_data.instruments[note_event.channel]
        for note in instrument.notes:
            if (abs(note.start - note_event.time) < 0.001 and 
                note.pitch == note_event.value):
                return note.velocity
        return 64  # Default velocity
    
    def _create_piano_roll(self, midi_data: MidiData) -> Tuple[np.ndarray, ...]:
        """Create piano roll representations."""
        
        # Calculate dimensions
        max_time = midi_data.end_time
        time_steps = self.piano_roll_config.time_to_step(max_time) + 1
        pitch_bins = self.piano_roll_config.pitch_bins
        
        # Initialize arrays
        piano_roll = np.zeros((time_steps, pitch_bins), dtype=np.float32)
        velocity_roll = np.zeros((time_steps, pitch_bins), dtype=np.float32)
        onset_roll = np.zeros((time_steps, pitch_bins), dtype=np.bool_)
        offset_roll = np.zeros((time_steps, pitch_bins), dtype=np.bool_)
        
        # Fill piano roll
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Convert to piano roll coordinates
                start_step = self.piano_roll_config.time_to_step(note.start)
                end_step = self.piano_roll_config.time_to_step(note.end)
                pitch_idx = note.pitch - self.piano_roll_config.min_pitch
                
                # Check bounds
                if (pitch_idx < 0 or pitch_idx >= pitch_bins or 
                    start_step >= time_steps):
                    continue
                
                end_step = min(end_step, time_steps - 1)
                
                # Fill piano roll (note is active)
                piano_roll[start_step:end_step + 1, pitch_idx] = 1.0
                
                # Fill velocity (normalized)
                velocity_roll[start_step:end_step + 1, pitch_idx] = note.velocity / 127.0
                
                # Mark onset and offset
                if start_step < time_steps:
                    onset_roll[start_step, pitch_idx] = True
                if end_step < time_steps:
                    offset_roll[end_step, pitch_idx] = True
        
        return piano_roll, velocity_roll, onset_roll, offset_roll
    
    def _tokenize_events(self, events: List[MusicEvent]) -> np.ndarray:
        """Convert events to token sequence."""
        tokens = []
        for event in events:
            token = self.vocab_config.event_to_token(event)
            tokens.append(token)
        
        return np.array(tokens, dtype=np.int32)
    
    def _calculate_statistics(self, midi_data: MidiData, events: List[MusicEvent]) -> Dict[str, Any]:
        """Calculate musical statistics."""
        
        # Get pitch and velocity ranges
        all_pitches = []
        all_velocities = []
        total_notes = 0
        
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_pitches.append(note.pitch)
                all_velocities.append(note.velocity)
                total_notes += 1
        
        pitch_range = (min(all_pitches), max(all_pitches)) if all_pitches else (60, 60)
        velocity_range = (min(all_velocities), max(all_velocities)) if all_velocities else (64, 64)
        
        # Calculate note density
        note_density = total_notes / midi_data.end_time if midi_data.end_time > 0 else 0
        
        return {
            "duration": midi_data.end_time,
            "num_instruments": len(midi_data.instruments),
            "tempo_changes": midi_data.tempo_changes,
            "time_signature": (4, 4),  # Default, could extract from MIDI
            "key_signature": 0,        # Default C major
            "pitch_range": pitch_range,
            "velocity_range": velocity_range,
            "note_density": note_density
        }
    
    def representation_to_midi(self, representation: MusicalRepresentation) -> MidiData:
        """Convert representation back to MIDI data (reversible encoding)."""
        return self._events_to_midi(representation.events)
    
    def tokens_to_midi(self, tokens: np.ndarray) -> MidiData:
        """Convert token sequence back to MIDI data."""
        events = self._tokens_to_events(tokens)
        return self._events_to_midi(events)
    
    def _tokens_to_events(self, tokens: np.ndarray) -> List[MusicEvent]:
        """Convert token sequence back to events."""
        events = []
        current_time = 0.0
        
        for token in tokens:
            event_type, value = self.vocab_config.token_to_event_info(int(token))
            
            # Handle time accumulation for time shift events
            if event_type == EventType.TIME_SHIFT:
                time_delta = value * self.vocab_config.time_shift_ms / 1000.0
                current_time += time_delta
                # Don't add time shift to events list - it's implicit in timing
            else:
                event = MusicEvent(event_type, value, 0, current_time)
                events.append(event)
        
        return events
    
    def _events_to_midi(self, events: List[MusicEvent]) -> MidiData:
        """Convert event sequence back to MIDI data."""
        from collections import defaultdict
        
        # Track active notes and instruments
        active_notes = defaultdict(dict)  # {channel: {pitch: (start_time, velocity)}}
        instruments_data = defaultdict(list)  # {channel: [notes]}
        
        current_time = 0.0
        current_velocity = 64
        current_channel = 0
        tempo_changes = []
        
        for event in events:
            # Update current time from event
            if event.time > current_time:
                current_time = event.time
            
            if event.event_type == EventType.NOTE_ON:
                # Start a note
                active_notes[current_channel][event.value] = (current_time, current_velocity)
            
            elif event.event_type == EventType.NOTE_OFF:
                # End a note
                if (current_channel in active_notes and 
                    event.value in active_notes[current_channel]):
                    
                    start_time, velocity = active_notes[current_channel][event.value]
                    
                    # Create MIDI note
                    note = MidiNote(
                        pitch=event.value,
                        velocity=velocity,
                        start=start_time,
                        end=current_time,
                        channel=current_channel
                    )
                    
                    instruments_data[current_channel].append(note)
                    del active_notes[current_channel][event.value]
            
            elif event.event_type == EventType.VELOCITY_CHANGE:
                # Update current velocity (dequantize)
                current_velocity = int((event.value / (self.vocab_config.velocity_bins - 1)) * 126) + 1
                current_velocity = max(1, min(127, current_velocity))
            
            elif event.event_type == EventType.INSTRUMENT_CHANGE:
                current_channel = event.value
            
            elif event.event_type == EventType.TEMPO_CHANGE:
                # Dequantize tempo
                tempo_ratio = event.value / (self.vocab_config.tempo_bins - 1)
                tempo = (self.vocab_config.min_tempo + 
                        tempo_ratio * (self.vocab_config.max_tempo - self.vocab_config.min_tempo))
                tempo_changes.append((current_time, tempo))
        
        # Create MIDI instruments
        midi_instruments = []
        for channel, notes in instruments_data.items():
            if notes:  # Only add instruments with notes
                instrument = MidiInstrument(
                    program=channel,
                    is_drum=(channel == 9),  # Channel 9 is typically drums
                    name=f"Instrument {channel}",
                    notes=notes
                )
                midi_instruments.append(instrument)
        
        # Create MIDI data
        midi_data = MidiData(
            instruments=midi_instruments,
            tempo_changes=tempo_changes,
            end_time=current_time
        )
        
        return midi_data
    
    def save_representation(self, representation: MusicalRepresentation, 
                          output_path: Union[str, Path]) -> None:
        """Save representation to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokens as NPZ
        if representation.tokens is not None:
            np.savez_compressed(
                output_path / "tokens.npz",
                tokens=representation.tokens
            )
        
        # Save piano roll as NPZ
        if representation.piano_roll is not None:
            np.savez_compressed(
                output_path / "piano_roll.npz",
                piano_roll=representation.piano_roll,
                velocity_roll=representation.velocity_roll,
                onset_roll=representation.onset_roll,
                offset_roll=representation.offset_roll
            )
        
        # Save human-readable information
        human_readable = {
            "summary": representation.get_summary(),
            "vocab_config": {
                "vocab_size": self.vocab_config.vocab_size,
                "time_shift_ms": self.vocab_config.time_shift_ms,
                "velocity_bins": self.vocab_config.velocity_bins
            },
            "first_20_events": [str(event) for event in representation.events[:20]],
            "first_20_tokens": representation.tokens[:20].tolist() if representation.tokens is not None else [],
            "token_meanings": [
                f"Token {i}: {self.vocab_config.token_to_event_info(token)}" 
                for i, token in enumerate(representation.tokens[:20]) 
                if representation.tokens is not None
            ]
        }
        
        with open(output_path / "analysis.json", 'w') as f:
            json.dump(human_readable, f, indent=2)
        
        self.logger.info(f"Saved representation to {output_path}")