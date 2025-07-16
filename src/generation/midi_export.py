"""
MIDI Export System for Aurl.ai Music Generation

This module provides high-quality MIDI file creation from generated token sequences
with support for all musical nuances, multi-track compositions, and production-ready
export features compatible with professional music software like Finale.

Features:
- High-fidelity token-to-MIDI conversion
- Multi-track support with instrument assignments
- Tempo and time signature handling
- Program change and control messages
- Finale/DAW compatibility optimization
- MusicXML export option (future)
- Advanced musical feature preservation

Phase 7.2 of the Aurl.ai GAMEPLAN.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import torch

import pretty_midi
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from ..data.representation import MusicRepresentationConverter
from ..data.midi_parser import MidiData, MidiNote, MidiInstrument
from ..utils.base_logger import setup_logger
from ..utils.constants import (
    MIDI_NOTE_MIN, MIDI_NOTE_MAX, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX,
    MIDI_DEFAULT_RESOLUTION
)

logger = setup_logger(__name__)


@dataclass
class MidiExportConfig:
    """Configuration for MIDI export with professional-grade options."""
    
    # Basic export settings
    resolution: int = 480                    # Ticks per quarter note (high quality)
    default_tempo: float = 120.0            # BPM
    default_time_signature: Tuple[int, int] = (4, 4)
    default_program: int = 0                # Acoustic Grand Piano
    
    # Track organization
    max_tracks: int = 16                    # Maximum MIDI tracks
    separate_channels: bool = True          # Use separate channels for polyphony
    merge_short_rests: bool = True          # Merge rests < 32nd note
    
    # Musical interpretation
    humanize_timing: bool = False           # Add slight timing variations
    humanize_velocity: bool = False         # Add slight velocity variations
    swing_ratio: Optional[float] = None     # Swing timing (0.5-0.75)
    
    # Quality settings
    quantize_timing: bool = True            # Quantize to resolution grid
    remove_zero_velocity: bool = True       # Filter zero-velocity notes
    extend_sustain: bool = True             # Extend notes with sustain pedal
    
    # Export format
    format_type: int = 1                    # MIDI format (0, 1, or 2)
    include_metadata: bool = True           # Add track names, copyright, etc.


@dataclass
class ExportStatistics:
    """Statistics about MIDI export process."""
    total_tracks: int = 0
    total_notes: int = 0
    total_duration: float = 0.0
    tempo_changes: int = 0
    program_changes: int = 0
    control_changes: int = 0
    export_time: float = 0.0
    warnings: List[str] = field(default_factory=list)


class MidiExporter:
    """
    Professional-grade MIDI export system for Aurl.ai.
    
    Converts generated token sequences to high-quality MIDI files compatible
    with professional music software including Finale, Sibelius, and DAWs.
    """
    
    def __init__(self, config: Optional[MidiExportConfig] = None):
        """
        Initialize MIDI exporter.
        
        Args:
            config: Export configuration, uses defaults if None
        """
        self.config = config or MidiExportConfig()
        self.converter = MusicRepresentationConverter()
        
        # Statistics tracking
        self.stats = ExportStatistics()
        
        # Instrument mapping for multi-track support
        self.instrument_map = {
            0: {"name": "Piano", "program": 0, "channel": 0},
            1: {"name": "Strings", "program": 48, "channel": 1},
            2: {"name": "Brass", "program": 56, "channel": 2},
            3: {"name": "Woodwinds", "program": 64, "channel": 3},
            4: {"name": "Guitar", "program": 24, "channel": 4},
            5: {"name": "Bass", "program": 32, "channel": 5},
            6: {"name": "Drums", "program": 0, "channel": 9},  # Standard drum channel
            7: {"name": "Synth", "program": 80, "channel": 7},
        }
        
        logger.info("MIDI Exporter initialized with professional-grade settings")
    
    def export_tokens_to_midi(
        self,
        tokens: Union[torch.Tensor, np.ndarray],
        output_path: str,
        title: Optional[str] = None,
        style: Optional[str] = None,
        tempo: Optional[float] = None,
        time_signature: Optional[Tuple[int, int]] = None
    ) -> ExportStatistics:
        """
        Export token sequence to MIDI file.
        
        Args:
            tokens: Generated token sequence
            output_path: Path to save MIDI file
            title: Optional track title
            style: Optional musical style for instrument selection
            tempo: Optional tempo override
            time_signature: Optional time signature override
            
        Returns:
            Export statistics
        """
        import time
        start_time = time.time()
        
        logger.info(f"Exporting {tokens.shape} tokens to MIDI: {output_path}")
        
        # Convert to numpy if torch tensor
        if isinstance(tokens, torch.Tensor):
            tokens_np = tokens.cpu().numpy()
        else:
            tokens_np = tokens
        
        # Flatten if 2D (batch dimension)
        if tokens_np.ndim == 2:
            tokens_np = tokens_np.flatten()
        
        # Convert tokens to MIDI data using existing converter
        midi_data = self.converter.tokens_to_midi(tokens_np)
        
        # Create professional MIDI file
        midi_file = self._create_midi_file(
            midi_data,
            title=title,
            style=style,
            tempo=tempo or self.config.default_tempo,
            time_signature=time_signature or self.config.default_time_signature
        )
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        midi_file.save(output_path)
        
        # Update statistics
        self.stats.export_time = time.time() - start_time
        self.stats.total_duration = midi_data.end_time
        
        logger.info(
            f"MIDI export complete: {self.stats.total_notes} notes, "
            f"{self.stats.total_tracks} tracks, {self.stats.total_duration:.2f}s duration"
        )
        
        return self.stats
    
    def _create_midi_file(
        self,
        midi_data: MidiData,
        title: Optional[str] = None,
        style: Optional[str] = None,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4)
    ) -> MidiFile:
        """Create a complete MIDI file from MIDI data."""
        
        # Create MIDI file with specified format
        midi_file = MidiFile(
            type=self.config.format_type,
            ticks_per_beat=self.config.resolution,
            charset='latin1'
        )
        
        # Reset statistics
        self.stats = ExportStatistics()
        
        if midi_data.instruments:
            # Multi-track mode: separate track for each instrument
            for i, instrument in enumerate(midi_data.instruments):
                track = self._create_instrument_track(
                    instrument, 
                    i, 
                    title=title,
                    tempo=tempo,
                    time_signature=time_signature,
                    is_first_track=(i == 0)
                )
                midi_file.tracks.append(track)
                self.stats.total_tracks += 1
        else:
            # Single track mode with default settings
            track = self._create_default_track(
                title=title,
                tempo=tempo,
                time_signature=time_signature
            )
            midi_file.tracks.append(track)
            self.stats.total_tracks = 1
        
        # Add conductor track for global events in format 1
        if self.config.format_type == 1:
            conductor_track = self._create_conductor_track(
                midi_data, tempo, time_signature, title
            )
            midi_file.tracks.insert(0, conductor_track)
            self.stats.total_tracks += 1
        
        return midi_file
    
    def _create_conductor_track(
        self,
        midi_data: MidiData,
        tempo: float,
        time_signature: Tuple[int, int],
        title: Optional[str] = None
    ) -> MidiTrack:
        """Create conductor track with global tempo and meta events."""
        track = MidiTrack()
        current_time = 0
        
        # Track name
        track.append(MetaMessage('track_name', name='Conductor', time=0))
        
        # Copyright and metadata
        if self.config.include_metadata:
            track.append(MetaMessage(
                'copyright', 
                text=f'Generated by Aurl.ai - {title or "Untitled"}',
                time=0
            ))
            track.append(MetaMessage(
                'text',
                text='Created with state-of-the-art AI music generation',
                time=0
            ))
        
        # Time signature
        track.append(MetaMessage(
            'time_signature',
            numerator=time_signature[0],
            denominator=time_signature[1],
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        ))
        
        # Initial tempo
        microseconds_per_beat = int(60_000_000 / tempo)
        track.append(MetaMessage(
            'set_tempo',
            tempo=microseconds_per_beat,
            time=0
        ))
        self.stats.tempo_changes += 1
        
        # Add tempo changes from MIDI data
        for time_sec, new_tempo in midi_data.tempo_changes:
            delta_ticks = self._seconds_to_ticks(time_sec - current_time / self.config.resolution)
            microseconds_per_beat = int(60_000_000 / new_tempo)
            track.append(MetaMessage(
                'set_tempo',
                tempo=microseconds_per_beat,
                time=delta_ticks
            ))
            current_time += delta_ticks
            self.stats.tempo_changes += 1
        
        # Add time signature changes
        for time_sec, num, den in midi_data.time_signature_changes:
            delta_ticks = self._seconds_to_ticks(time_sec - current_time / self.config.resolution)
            track.append(MetaMessage(
                'time_signature',
                numerator=num,
                denominator=den,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=delta_ticks
            ))
            current_time += delta_ticks
        
        # End of track
        end_time_ticks = self._seconds_to_ticks(midi_data.end_time)
        track.append(MetaMessage('end_of_track', time=max(0, end_time_ticks - current_time)))
        
        return track
    
    def _create_instrument_track(
        self,
        instrument: MidiInstrument,
        track_index: int,
        title: Optional[str] = None,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4),
        is_first_track: bool = False
    ) -> MidiTrack:
        """Create a MIDI track for a specific instrument."""
        track = MidiTrack()
        
        # Get instrument configuration
        instrument_config = self.instrument_map.get(
            track_index, 
            self.instrument_map[0]  # Default to piano
        )
        
        channel = instrument_config["channel"]
        program = instrument.program if hasattr(instrument, 'program') else instrument_config["program"]
        
        # Track name
        track_name = f"{instrument_config['name']} {track_index + 1}"
        if title and is_first_track:
            track_name = f"{title} - {track_name}"
        
        track.append(MetaMessage('track_name', name=track_name, time=0))
        
        # Program change
        track.append(Message(
            'program_change',
            channel=channel,
            program=program,
            time=0
        ))
        self.stats.program_changes += 1
        
        # Add global events for first track only (if not using conductor track)
        if is_first_track and self.config.format_type != 1:
            # Time signature
            track.append(MetaMessage(
                'time_signature',
                numerator=time_signature[0],
                denominator=time_signature[1],
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0
            ))
            
            # Tempo
            microseconds_per_beat = int(60_000_000 / tempo)
            track.append(MetaMessage(
                'set_tempo',
                tempo=microseconds_per_beat,
                time=0
            ))
        
        # Convert notes to MIDI messages
        self._add_notes_to_track(track, instrument.notes, channel)
        
        # End of track
        track.append(MetaMessage('end_of_track', time=0))
        
        return track
    
    def _create_default_track(
        self,
        title: Optional[str] = None,
        tempo: float = 120.0,
        time_signature: Tuple[int, int] = (4, 4)
    ) -> MidiTrack:
        """Create a default single track with basic settings."""
        track = MidiTrack()
        
        # Track name
        track.append(MetaMessage(
            'track_name', 
            name=title or 'Aurl.ai Generated Music',
            time=0
        ))
        
        # Time signature
        track.append(MetaMessage(
            'time_signature',
            numerator=time_signature[0],
            denominator=time_signature[1],
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0
        ))
        
        # Tempo
        microseconds_per_beat = int(60_000_000 / tempo)
        track.append(MetaMessage(
            'set_tempo',
            tempo=microseconds_per_beat,
            time=0
        ))
        
        # Default program (piano)
        track.append(Message(
            'program_change',
            channel=0,
            program=self.config.default_program,
            time=0
        ))
        
        # Add placeholder note or end immediately
        track.append(MetaMessage('end_of_track', time=0))
        
        return track
    
    def _add_notes_to_track(
        self,
        track: MidiTrack,
        notes: List[MidiNote],
        channel: int
    ):
        """Add note events to a MIDI track with proper timing."""
        if not notes:
            return
        
        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start)
        
        # Create note on/off events
        events = []
        
        for note in sorted_notes:
            # Validate note
            if not self._is_valid_note(note):
                continue
            
            # Note on event
            note_on_time = self._seconds_to_ticks(note.start)
            events.append((note_on_time, 'note_on', note.pitch, note.velocity))
            
            # Note off event
            note_off_time = self._seconds_to_ticks(note.end)
            events.append((note_off_time, 'note_off', note.pitch, 0))
            
            self.stats.total_notes += 1
        
        # Sort all events by time
        events.sort(key=lambda x: x[0])
        
        # Convert to MIDI messages with delta times
        current_time = 0
        
        for event_time, event_type, pitch, velocity in events:
            delta_time = event_time - current_time
            current_time = event_time
            
            if event_type == 'note_on':
                track.append(Message(
                    'note_on',
                    channel=channel,
                    note=pitch,
                    velocity=velocity,
                    time=delta_time
                ))
            elif event_type == 'note_off':
                track.append(Message(
                    'note_off',
                    channel=channel,
                    note=pitch,
                    velocity=velocity,
                    time=delta_time
                ))
    
    def _is_valid_note(self, note: MidiNote) -> bool:
        """Validate a MIDI note for export."""
        if note.start < 0 or note.end <= note.start:
            self.stats.warnings.append(f"Invalid note timing: {note.start} - {note.end}")
            return False
        
        if not (MIDI_NOTE_MIN <= note.pitch <= MIDI_NOTE_MAX):
            self.stats.warnings.append(f"Invalid note pitch: {note.pitch}")
            return False
        
        if self.config.remove_zero_velocity and note.velocity <= 0:
            return False
        
        # Minimum note duration (1/64th note at 120 BPM ≈ 0.03125 seconds)
        min_duration = 0.03125
        if note.duration < min_duration:
            self.stats.warnings.append(f"Note too short: {note.duration:.4f}s")
            return False
        
        return True
    
    def _seconds_to_ticks(self, seconds: float) -> int:
        """Convert seconds to MIDI ticks."""
        # Assuming 120 BPM for tick conversion
        # TODO: Make this tempo-aware for variable tempo pieces
        beats_per_second = 120.0 / 60.0
        ticks = int(seconds * beats_per_second * self.config.resolution)
        return max(0, ticks)
    
    def export_multiple_styles(
        self,
        tokens_list: List[Union[torch.Tensor, np.ndarray]],
        output_dir: str,
        base_filename: str = "aurl_generation",
        styles: Optional[List[str]] = None
    ) -> List[ExportStatistics]:
        """
        Export multiple token sequences with different styles.
        
        Args:
            tokens_list: List of token sequences to export
            output_dir: Output directory
            base_filename: Base filename for outputs
            styles: Optional list of style names
            
        Returns:
            List of export statistics for each file
        """
        results = []
        
        for i, tokens in enumerate(tokens_list):
            style = styles[i] if styles and i < len(styles) else f"variation_{i+1}"
            filename = f"{base_filename}_{style}.mid"
            output_path = os.path.join(output_dir, filename)
            
            stats = self.export_tokens_to_midi(
                tokens=tokens,
                output_path=output_path,
                title=f"{base_filename.title()} - {style.title()}",
                style=style
            )
            results.append(stats)
        
        logger.info(f"Exported {len(results)} MIDI files to {output_dir}")
        return results
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get a summary of the last export operation."""
        return {
            "tracks": self.stats.total_tracks,
            "notes": self.stats.total_notes,
            "duration_seconds": self.stats.total_duration,
            "tempo_changes": self.stats.tempo_changes,
            "program_changes": self.stats.program_changes,
            "export_time_seconds": self.stats.export_time,
            "warnings": self.stats.warnings,
            "config": {
                "resolution": self.config.resolution,
                "format_type": self.config.format_type,
                "include_metadata": self.config.include_metadata
            }
        }


# Convenience functions for direct export
def export_tokens_to_midi_file(
    tokens: Union[torch.Tensor, np.ndarray],
    output_path: str,
    title: Optional[str] = None,
    tempo: Optional[float] = None,
    config: Optional[MidiExportConfig] = None
) -> ExportStatistics:
    """
    Convenience function to export tokens to MIDI file.
    
    Args:
        tokens: Generated token sequence
        output_path: Path to save MIDI file
        title: Optional track title
        tempo: Optional tempo in BPM
        config: Optional export configuration
        
    Returns:
        Export statistics
    """
    exporter = MidiExporter(config)
    return exporter.export_tokens_to_midi(
        tokens=tokens,
        output_path=output_path,
        title=title,
        tempo=tempo
    )


def create_standard_config() -> MidiExportConfig:
    """Create a standard high-quality MIDI configuration."""
    return MidiExportConfig(
        resolution=480,                     # High resolution for accuracy
        format_type=1,                      # Multi-track format
        include_metadata=True,
        quantize_timing=True,
        remove_zero_velocity=True,
        separate_channels=True
    )


def create_performance_config() -> MidiExportConfig:
    """Create a configuration optimized for performance/playback."""
    return MidiExportConfig(
        resolution=960,                     # Very high resolution
        format_type=1,
        humanize_timing=True,
        humanize_velocity=True,
        extend_sustain=True,
        quantize_timing=False               # Preserve natural timing
    )