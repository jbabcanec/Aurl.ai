"""
Musical constants and configuration values for MidiFly.

This module contains all the musical constants, MIDI specifications,
and default configuration values used throughout the system.
"""

from typing import Dict, List, Tuple
import math

# ============================================================================
# MIDI CONSTANTS
# ============================================================================

# MIDI note range
MIDI_NOTE_MIN = 21      # A0
MIDI_NOTE_MAX = 108     # C8
MIDI_NOTE_RANGE = MIDI_NOTE_MAX - MIDI_NOTE_MIN + 1

# MIDI velocity range
MIDI_VELOCITY_MIN = 1
MIDI_VELOCITY_MAX = 127
MIDI_VELOCITY_RANGE = MIDI_VELOCITY_MAX - MIDI_VELOCITY_MIN + 1

# MIDI channel range
MIDI_CHANNEL_MIN = 0
MIDI_CHANNEL_MAX = 15
MIDI_CHANNEL_COUNT = MIDI_CHANNEL_MAX - MIDI_CHANNEL_MIN + 1

# MIDI program (instrument) range
MIDI_PROGRAM_MIN = 0
MIDI_PROGRAM_MAX = 127
MIDI_PROGRAM_COUNT = MIDI_PROGRAM_MAX - MIDI_PROGRAM_MIN + 1

# MIDI control change range
MIDI_CC_MIN = 0
MIDI_CC_MAX = 127
MIDI_CC_COUNT = MIDI_CC_MAX - MIDI_CC_MIN + 1

# Standard MIDI file resolution (ticks per quarter note)
MIDI_DEFAULT_RESOLUTION = 480
MIDI_MIN_RESOLUTION = 96
MIDI_MAX_RESOLUTION = 960

# ============================================================================
# MUSICAL CONSTANTS
# ============================================================================

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Number of semitones in an octave
SEMITONES_PER_OCTAVE = 12

# Reference note for frequency calculations
A4_MIDI = 69
A4_FREQUENCY = 440.0

# Time signatures (numerator, denominator)
COMMON_TIME_SIGNATURES = [
    (4, 4), (3, 4), (2, 4), (6, 8), (9, 8), (12, 8),
    (2, 2), (3, 2), (5, 4), (7, 4), (5, 8), (7, 8)
]

# Common tempos (BPM)
TEMPO_RANGES = {
    "largo": (40, 60),
    "andante": (76, 108),
    "moderato": (108, 120),
    "allegro": (120, 168),
    "presto": (168, 200),
}

# Key signatures (number of sharps/flats)
KEY_SIGNATURES = {
    "C_major": 0, "G_major": 1, "D_major": 2, "A_major": 3,
    "E_major": 4, "B_major": 5, "F#_major": 6, "C#_major": 7,
    "F_major": -1, "Bb_major": -2, "Eb_major": -3, "Ab_major": -4,
    "Db_major": -5, "Gb_major": -6, "Cb_major": -7,
    "A_minor": 0, "E_minor": 1, "B_minor": 2, "F#_minor": 3,
    "C#_minor": 4, "G#_minor": 5, "D#_minor": 6, "A#_minor": 7,
    "D_minor": -1, "G_minor": -2, "C_minor": -3, "F_minor": -4,
    "Bb_minor": -5, "Eb_minor": -6, "Ab_minor": -7,
}

# Circle of fifths
CIRCLE_OF_FIFTHS = [
    "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab"
]

# ============================================================================
# CHORD DEFINITIONS
# ============================================================================

# Chord intervals (semitones from root)
CHORD_TYPES = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
    "diminished7": [0, 3, 6, 9],
    "major9": [0, 4, 7, 11, 14],
    "minor9": [0, 3, 7, 10, 14],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

# Common chord progressions
CHORD_PROGRESSIONS = {
    "I-V-vi-IV": [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],
    "ii-V-I": [(2, "minor"), (7, "major"), (0, "major")],
    "I-vi-ii-V": [(0, "major"), (9, "minor"), (2, "minor"), (7, "major")],
    "vi-IV-I-V": [(9, "minor"), (5, "major"), (0, "major"), (7, "major")],
}

# ============================================================================
# SCALE DEFINITIONS
# ============================================================================

# Scale intervals (semitones from root)
SCALE_TYPES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}

# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================

# Sequence processing
DEFAULT_SEQUENCE_LENGTH = 2048
MIN_SEQUENCE_LENGTH = 64
MAX_SEQUENCE_LENGTH = 8192

# Quantization levels (16th note = 1)
QUANTIZATION_LEVELS = {
    "32nd": 0.5,
    "16th": 1.0,
    "8th": 2.0,
    "quarter": 4.0,
    "half": 8.0,
    "whole": 16.0,
}

# Velocity quantization levels
VELOCITY_LEVELS = {
    "ppp": 16,
    "pp": 32,
    "p": 48,
    "mp": 64,
    "mf": 80,
    "f": 96,
    "ff": 112,
    "fff": 127,
}

# Data augmentation ranges
AUGMENTATION_RANGES = {
    "pitch_shift_semitones": (-12, 12),
    "time_stretch_factor": (0.8, 1.2),
    "velocity_scale_factor": (0.7, 1.3),
    "tempo_scale_factor": (0.8, 1.2),
}

# ============================================================================
# MODEL CONSTANTS
# ============================================================================

# Token vocabulary
SPECIAL_TOKENS = {
    "PAD": 0,
    "START": 1,
    "END": 2,
    "UNKNOWN": 3,
}

# Event types for event-based representation
EVENT_TYPES = {
    "NOTE_ON": 0,
    "NOTE_OFF": 1,
    "TIME_SHIFT": 2,
    "VELOCITY_CHANGE": 3,
    "PROGRAM_CHANGE": 4,
    "CONTROL_CHANGE": 5,
    "TEMPO_CHANGE": 6,
    "KEY_CHANGE": 7,
    "TIME_SIGNATURE": 8,
}

# Model architecture defaults
DEFAULT_MODEL_CONFIG = {
    "vocab_size": 512,
    "hidden_dim": 512,
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.1,
    "max_sequence_length": 2048,
}

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "warmup_steps": 1000,
    "gradient_clip_norm": 1.0,
    "weight_decay": 1e-5,
}

# ============================================================================
# AUDIO CONSTANTS
# ============================================================================

# Audio processing
SAMPLE_RATE = 44100
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128

# Audio file formats
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".aac"]

# ============================================================================
# FILE FORMATS
# ============================================================================

# MIDI file formats
SUPPORTED_MIDI_FORMATS = [".mid", ".midi", ".kar"]

# Data file formats
SUPPORTED_DATA_FORMATS = [".npz", ".h5", ".hdf5", ".pkl", ".json"]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def midi_note_to_frequency(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return A4_FREQUENCY * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def frequency_to_midi_note(frequency: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    return int(round(A4_MIDI + 12.0 * math.log2(frequency / A4_FREQUENCY)))


def note_name_to_midi(note_name: str, octave: int) -> int:
    """Convert note name and octave to MIDI note number."""
    if note_name not in NOTE_NAMES:
        raise ValueError(f"Invalid note name: {note_name}")
    
    semitone = NOTE_NAMES.index(note_name)
    return (octave + 1) * 12 + semitone


def midi_to_note_name(midi_note: int) -> Tuple[str, int]:
    """Convert MIDI note number to note name and octave."""
    octave = (midi_note // 12) - 1
    semitone = midi_note % 12
    note_name = NOTE_NAMES[semitone]
    return note_name, octave


def get_chord_notes(root_midi: int, chord_type: str) -> List[int]:
    """Get MIDI note numbers for a chord."""
    if chord_type not in CHORD_TYPES:
        raise ValueError(f"Unknown chord type: {chord_type}")
    
    intervals = CHORD_TYPES[chord_type]
    return [root_midi + interval for interval in intervals]


def get_scale_notes(root_midi: int, scale_type: str) -> List[int]:
    """Get MIDI note numbers for a scale."""
    if scale_type not in SCALE_TYPES:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    intervals = SCALE_TYPES[scale_type]
    return [root_midi + interval for interval in intervals]


def is_valid_midi_note(note: int) -> bool:
    """Check if a note is within valid MIDI range."""
    return MIDI_NOTE_MIN <= note <= MIDI_NOTE_MAX


def is_valid_velocity(velocity: int) -> bool:
    """Check if a velocity is within valid MIDI range."""
    return MIDI_VELOCITY_MIN <= velocity <= MIDI_VELOCITY_MAX


def clamp_midi_note(note: int) -> int:
    """Clamp note to valid MIDI range."""
    return max(MIDI_NOTE_MIN, min(MIDI_NOTE_MAX, note))


def clamp_velocity(velocity: int) -> int:
    """Clamp velocity to valid MIDI range."""
    return max(MIDI_VELOCITY_MIN, min(MIDI_VELOCITY_MAX, velocity))