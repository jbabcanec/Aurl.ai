"""
MIDI Parser for Aurl.ai Music Generation System.

This module provides fault-tolerant MIDI file parsing with support for
multiple tracks, instruments, and comprehensive metadata extraction.
Based on research findings, it uses a two-layer approach:
- pretty_midi for high-level musical representation
- mido for low-level MIDI message access and validation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
import numpy as np

import pretty_midi
import mido
from mido import MidiFile, MidiTrack

from src.utils.base_logger import setup_logger
from src.utils.constants import (
    MIDI_NOTE_MIN, MIDI_NOTE_MAX, MIDI_VELOCITY_MIN, MIDI_VELOCITY_MAX,
    is_valid_midi_note, is_valid_velocity, clamp_midi_note, clamp_velocity
)

logger = setup_logger(__name__)


@dataclass
class MidiNote:
    """Represents a single MIDI note with all musical attributes."""
    pitch: int              # MIDI note number (21-108)
    velocity: int           # Note velocity (1-127)
    start: float           # Start time in seconds
    end: float             # End time in seconds  
    channel: int = 0       # MIDI channel (0-15)
    
    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end - self.start
    
    def __post_init__(self):
        """Validate note parameters."""
        self.pitch = clamp_midi_note(self.pitch)
        self.velocity = clamp_velocity(self.velocity)
        if self.start < 0:
            self.start = 0.0
        if self.end <= self.start:
            self.end = self.start + 0.1  # Minimum duration


@dataclass
class MidiInstrument:
    """Represents a MIDI instrument/track with all its musical content."""
    program: int                                    # MIDI program number (0-127)
    is_drum: bool = False                          # Whether this is a drum track
    name: str = ""                                 # Instrument name
    notes: List[MidiNote] = field(default_factory=list)
    control_changes: List[Dict[str, Any]] = field(default_factory=list)
    pitch_bends: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Sort notes by start time for consistent processing."""
        self.notes.sort(key=lambda note: note.start)


@dataclass
class MidiData:
    """Complete MIDI file representation preserving all musical information."""
    instruments: List[MidiInstrument] = field(default_factory=list)
    tempo_changes: List[Tuple[float, float]] = field(default_factory=list)  # (time, tempo)
    time_signature_changes: List[Tuple[float, int, int]] = field(default_factory=list)  # (time, num, den)
    key_signature_changes: List[Tuple[float, int]] = field(default_factory=list)  # (time, key)
    resolution: int = 220                          # Ticks per beat
    end_time: float = 0.0                         # Total duration in seconds
    filename: str = ""                            # Source filename
    
    def __post_init__(self):
        """Calculate end time if not provided."""
        if self.end_time == 0.0:
            max_end = 0.0
            for instrument in self.instruments:
                for note in instrument.notes:
                    max_end = max(max_end, note.end)
            self.end_time = max_end


class MidiParsingError(Exception):
    """Exception raised when MIDI parsing fails."""
    pass


class MidiParser:
    """
    Fault-tolerant MIDI parser with validation and repair capabilities.
    
    Uses pretty_midi for high-level parsing and mido for low-level validation.
    Designed to handle real-world MIDI files with various issues.
    """
    
    def __init__(self, repair_mode: bool = True, strict_validation: bool = False):
        """
        Initialize MIDI parser.
        
        Args:
            repair_mode: Whether to attempt repairs on corrupted files
            strict_validation: Whether to raise errors on validation failures
        """
        self.repair_mode = repair_mode
        self.strict_validation = strict_validation
        self.logger = logger
    
    def parse(self, file_path: Union[str, Path]) -> MidiData:
        """
        Parse a MIDI file into structured MidiData.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            MidiData object with all musical information
            
        Raises:
            MidiParsingError: If parsing fails and cannot be repaired
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise MidiParsingError(f"MIDI file not found: {file_path}")
        
        if not self._is_midi_file(file_path):
            raise MidiParsingError(f"File is not a valid MIDI file: {file_path}")
        
        try:
            # First, validate with mido for low-level issues
            self._validate_with_mido(file_path)
            
            # Parse with pretty_midi for musical content
            midi_data = self._parse_with_pretty_midi(file_path)
            midi_data.filename = str(file_path)
            
            # Validate and repair if needed
            if self.repair_mode:
                midi_data = self._repair_midi_data(midi_data)
            
            self._validate_midi_data(midi_data)
            
            self.logger.info(f"Successfully parsed MIDI: {file_path}")
            self.logger.info(f"  Duration: {midi_data.end_time:.2f}s, "
                           f"Instruments: {len(midi_data.instruments)}, "
                           f"Notes: {sum(len(inst.notes) for inst in midi_data.instruments)}")
            
            return midi_data
            
        except Exception as e:
            error_msg = f"Failed to parse MIDI file {file_path}: {e}"
            self.logger.error(error_msg)
            
            if self.strict_validation:
                raise MidiParsingError(error_msg)
            else:
                # Return empty MidiData if repair fails
                self.logger.warning("Returning empty MIDI data due to parsing failure")
                return MidiData(filename=str(file_path))
    
    def _is_midi_file(self, file_path: Path) -> bool:
        """Check if file is a valid MIDI file."""
        try:
            # Check file extension
            if file_path.suffix.lower() not in ['.mid', '.midi', '.kar']:
                return False
            
            # Check MIDI header
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'MThd'
                
        except Exception:
            return False
    
    def _validate_with_mido(self, file_path: Path) -> None:
        """Validate MIDI file using mido for low-level issues."""
        try:
            midi_file = MidiFile(file_path)
            
            # Basic validation
            if len(midi_file.tracks) == 0:
                raise MidiParsingError("MIDI file has no tracks")
            
            # Check for common issues
            total_messages = sum(len(track) for track in midi_file.tracks)
            if total_messages == 0:
                raise MidiParsingError("MIDI file has no messages")
            
            self.logger.debug(f"Mido validation passed: {len(midi_file.tracks)} tracks, "
                            f"{total_messages} messages")
                            
        except Exception as e:
            raise MidiParsingError(f"Low-level MIDI validation failed: {e}")
    
    def _parse_with_pretty_midi(self, file_path: Path) -> MidiData:
        """Parse MIDI file using pretty_midi for musical content."""
        try:
            pm = pretty_midi.PrettyMIDI(str(file_path))
            
            # Extract basic info
            midi_data = MidiData(
                resolution=pm.resolution,
                end_time=pm.get_end_time()
            )
            
            # Extract tempo changes
            tempo_times, tempos = pm.get_tempo_changes()
            midi_data.tempo_changes = list(zip(tempo_times, tempos))
            
            # Extract time signature changes
            for ts in pm.time_signature_changes:
                midi_data.time_signature_changes.append(
                    (ts.time, ts.numerator, ts.denominator)
                )
            
            # Extract key signature changes
            for ks in pm.key_signature_changes:
                midi_data.key_signature_changes.append(
                    (ks.time, ks.key_number)
                )
            
            # Extract instruments and their content
            for pm_instrument in pm.instruments:
                instrument = self._convert_instrument(pm_instrument)
                if instrument.notes:  # Only add instruments with notes
                    midi_data.instruments.append(instrument)
            
            return midi_data
            
        except Exception as e:
            raise MidiParsingError(f"Pretty_midi parsing failed: {e}")
    
    def _convert_instrument(self, pm_instrument) -> MidiInstrument:
        """Convert pretty_midi instrument to our MidiInstrument format."""
        instrument = MidiInstrument(
            program=pm_instrument.program,
            is_drum=pm_instrument.is_drum,
            name=pm_instrument.name or f"Program {pm_instrument.program}"
        )
        
        # Convert notes with validation during conversion
        for pm_note in pm_instrument.notes:
            # Apply repairs during conversion to handle pretty_midi limitations
            pitch = clamp_midi_note(pm_note.pitch)
            velocity = clamp_velocity(pm_note.velocity)
            start = max(0.0, pm_note.start)
            end = max(start + 0.01, pm_note.end)  # Ensure minimum duration
            
            note = MidiNote(
                pitch=pitch,
                velocity=velocity,
                start=start,
                end=end
            )
            
            # Only add notes with reasonable duration
            if note.duration >= 0.01:  # Minimum 10ms
                instrument.notes.append(note)
            else:
                self.logger.debug(f"Skipped extremely short note: {note.duration:.3f}s")
        
        # Convert control changes
        for cc in pm_instrument.control_changes:
            instrument.control_changes.append({
                'number': cc.number,
                'value': cc.value,
                'time': cc.time
            })
        
        # Convert pitch bends
        for pb in pm_instrument.pitch_bends:
            instrument.pitch_bends.append({
                'pitch': pb.pitch,
                'time': pb.time
            })
        
        return instrument
    
    def _repair_midi_data(self, midi_data: MidiData) -> MidiData:
        """Repair common issues in MIDI data."""
        self.logger.debug("Applying MIDI repairs...")
        
        repaired_instruments = []
        
        for instrument in midi_data.instruments:
            repaired_notes = []
            
            for note in instrument.notes:
                # Fix invalid pitches
                original_pitch = note.pitch
                note.pitch = clamp_midi_note(note.pitch)
                if note.pitch != original_pitch:
                    self.logger.debug(f"Clamped pitch {original_pitch} to {note.pitch}")
                
                # Fix invalid velocities
                original_velocity = note.velocity
                note.velocity = clamp_velocity(note.velocity)
                if note.velocity != original_velocity:
                    self.logger.debug(f"Clamped velocity {original_velocity} to {note.velocity}")
                
                # Fix timing issues
                if note.start < 0:
                    note.start = 0.0
                    self.logger.debug("Fixed negative start time")
                
                if note.end <= note.start:
                    note.end = note.start + 0.1  # Minimum duration
                    self.logger.debug("Fixed invalid note duration")
                
                # Remove extremely short notes (likely errors)
                if note.duration >= 0.01:  # Minimum 10ms duration
                    repaired_notes.append(note)
                else:
                    self.logger.debug(f"Removed extremely short note: {note.duration:.3f}s")
            
            # Only keep instruments with valid notes
            if repaired_notes:
                instrument.notes = repaired_notes
                repaired_instruments.append(instrument)
            else:
                self.logger.debug(f"Removed empty instrument: {instrument.name}")
        
        midi_data.instruments = repaired_instruments
        
        # Recalculate end time after repairs
        max_end = 0.0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                max_end = max(max_end, note.end)
        midi_data.end_time = max_end
        
        return midi_data
    
    def _validate_midi_data(self, midi_data: MidiData) -> None:
        """Validate parsed MIDI data for correctness."""
        if not midi_data.instruments:
            raise MidiParsingError("No valid instruments found in MIDI file")
        
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        if total_notes == 0:
            raise MidiParsingError("No valid notes found in MIDI file")
        
        if midi_data.end_time <= 0:
            raise MidiParsingError("Invalid MIDI duration")
        
        # Validate individual notes
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if not is_valid_midi_note(note.pitch):
                    raise MidiParsingError(f"Invalid note pitch: {note.pitch}")
                
                if not is_valid_velocity(note.velocity):
                    raise MidiParsingError(f"Invalid note velocity: {note.velocity}")
                
                if note.start < 0 or note.end <= note.start:
                    raise MidiParsingError(f"Invalid note timing: start={note.start}, end={note.end}")
        
        self.logger.debug("MIDI data validation passed")
    
    def get_statistics(self, midi_data: MidiData) -> Dict[str, Any]:
        """Generate statistics about the parsed MIDI data."""
        stats = {
            'filename': midi_data.filename,
            'duration': midi_data.end_time,
            'resolution': midi_data.resolution,
            'num_instruments': len(midi_data.instruments),
            'num_tempo_changes': len(midi_data.tempo_changes),
            'num_time_signature_changes': len(midi_data.time_signature_changes),
            'num_key_signature_changes': len(midi_data.key_signature_changes),
            'instruments': [],
            'pitch_range': [127, 0],  # [min, max]
            'velocity_range': [127, 0],  # [min, max]
            'total_notes': 0
        }
        
        for instrument in midi_data.instruments:
            inst_stats = {
                'program': instrument.program,
                'is_drum': instrument.is_drum,
                'name': instrument.name,
                'num_notes': len(instrument.notes),
                'num_control_changes': len(instrument.control_changes),
                'num_pitch_bends': len(instrument.pitch_bends),
                'pitch_range': [127, 0] if instrument.notes else [0, 0],
                'velocity_range': [127, 0] if instrument.notes else [0, 0]
            }
            
            for note in instrument.notes:
                # Update global stats
                stats['pitch_range'][0] = min(stats['pitch_range'][0], note.pitch)
                stats['pitch_range'][1] = max(stats['pitch_range'][1], note.pitch)
                stats['velocity_range'][0] = min(stats['velocity_range'][0], note.velocity)
                stats['velocity_range'][1] = max(stats['velocity_range'][1], note.velocity)
                stats['total_notes'] += 1
                
                # Update instrument stats
                inst_stats['pitch_range'][0] = min(inst_stats['pitch_range'][0], note.pitch)
                inst_stats['pitch_range'][1] = max(inst_stats['pitch_range'][1], note.pitch)
                inst_stats['velocity_range'][0] = min(inst_stats['velocity_range'][0], note.velocity)
                inst_stats['velocity_range'][1] = max(inst_stats['velocity_range'][1], note.velocity)
            
            stats['instruments'].append(inst_stats)
        
        return stats


def load_midi_file(file_path: Union[str, Path], 
                   repair_mode: bool = True,
                   strict_validation: bool = False) -> MidiData:
    """
    Convenience function to load a MIDI file.
    
    Args:
        file_path: Path to MIDI file
        repair_mode: Whether to attempt repairs on corrupted files
        strict_validation: Whether to raise errors on validation failures
        
    Returns:
        MidiData object with all musical information
    """
    parser = MidiParser(repair_mode=repair_mode, strict_validation=strict_validation)
    return parser.parse(file_path)


def batch_parse_midi_files(file_paths: List[Union[str, Path]],
                          repair_mode: bool = True,
                          strict_validation: bool = False) -> List[MidiData]:
    """
    Parse multiple MIDI files in batch.
    
    Args:
        file_paths: List of paths to MIDI files
        repair_mode: Whether to attempt repairs on corrupted files
        strict_validation: Whether to raise errors on validation failures
        
    Returns:
        List of MidiData objects (empty MidiData for failed files)
    """
    parser = MidiParser(repair_mode=repair_mode, strict_validation=strict_validation)
    results = []
    
    for file_path in file_paths:
        try:
            midi_data = parser.parse(file_path)
            results.append(midi_data)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            results.append(MidiData(filename=str(file_path)))
    
    return results


class StreamingMidiParser:
    """
    Memory-efficient streaming MIDI parser for large files and datasets.
    
    Processes MIDI files in chunks to avoid loading entire files into memory.
    Useful for very large MIDI files or processing thousands of files.
    """
    
    def __init__(self, chunk_size: int = 1024, repair_mode: bool = True):
        """
        Initialize streaming parser.
        
        Args:
            chunk_size: Number of notes to process in each chunk
            repair_mode: Whether to attempt repairs on corrupted data
        """
        self.chunk_size = chunk_size
        self.repair_mode = repair_mode
        self.logger = logger
        self.parser = MidiParser(repair_mode=repair_mode, strict_validation=False)
    
    def parse_stream(self, file_path: Union[str, Path]) -> Iterator[List[MidiNote]]:
        """
        Parse MIDI file in streaming fashion, yielding chunks of notes.
        
        Args:
            file_path: Path to MIDI file
            
        Yields:
            Chunks of MidiNote objects
        """
        try:
            # First parse the file normally to get structure
            midi_data = self.parser.parse(file_path)
            
            # Collect all notes from all instruments
            all_notes = []
            for instrument in midi_data.instruments:
                all_notes.extend(instrument.notes)
            
            # Sort by start time for chronological processing
            all_notes.sort(key=lambda note: note.start)
            
            # Yield chunks of notes
            for i in range(0, len(all_notes), self.chunk_size):
                chunk = all_notes[i:i + self.chunk_size]
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Failed to stream parse {file_path}: {e}")
            yield []
    
    def parse_batch_stream(self, file_paths: List[Union[str, Path]]) -> Iterator[Tuple[str, List[MidiNote]]]:
        """
        Parse multiple files in streaming fashion.
        
        Args:
            file_paths: List of paths to MIDI files
            
        Yields:
            Tuples of (filename, note_chunk) for each chunk
        """
        for file_path in file_paths:
            file_path = Path(file_path)
            try:
                for chunk in self.parse_stream(file_path):
                    if chunk:  # Only yield non-empty chunks
                        yield (str(file_path), chunk)
            except Exception as e:
                self.logger.error(f"Failed to stream parse {file_path}: {e}")
                continue
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic file information without full parsing.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found", "filename": str(file_path)}
        
        try:
            # Quick validation with mido
            midi_file = mido.MidiFile(file_path)
            
            # Count basic elements
            num_tracks = len(midi_file.tracks)
            num_messages = sum(len(track) for track in midi_file.tracks)
            
            # Calculate approximate duration
            total_time = 0
            for track in midi_file.tracks:
                track_time = 0
                for msg in track:
                    track_time += msg.time
                total_time = max(total_time, track_time)
            
            # Convert ticks to seconds (approximate)
            ticks_per_beat = midi_file.ticks_per_beat
            # Default tempo: 120 BPM = 0.5 seconds per beat
            seconds_per_tick = 0.5 / ticks_per_beat
            duration_seconds = total_time * seconds_per_tick
            
            return {
                "filename": str(file_path),
                "file_size": file_path.stat().st_size,
                "num_tracks": num_tracks,
                "num_messages": num_messages,
                "ticks_per_beat": ticks_per_beat,
                "estimated_duration": duration_seconds,
                "format": midi_file.type
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "filename": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            }


def stream_parse_midi_files(file_paths: List[Union[str, Path]],
                           chunk_size: int = 1024,
                           repair_mode: bool = True) -> Iterator[Tuple[str, List[MidiNote]]]:
    """
    Convenience function for streaming MIDI file parsing.
    
    Args:
        file_paths: List of paths to MIDI files
        chunk_size: Number of notes per chunk
        repair_mode: Whether to attempt repairs on corrupted files
        
    Yields:
        Tuples of (filename, note_chunk) for each chunk
    """
    parser = StreamingMidiParser(chunk_size=chunk_size, repair_mode=repair_mode)
    yield from parser.parse_batch_stream(file_paths)