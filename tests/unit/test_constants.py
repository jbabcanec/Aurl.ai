"""
Unit tests for musical constants and utility functions.
"""

import pytest
import math
from src.utils.constants import (
    MIDI_NOTE_MIN, MIDI_NOTE_MAX, NOTE_NAMES, CHORD_TYPES, SCALE_TYPES,
    midi_note_to_frequency, frequency_to_midi_note, note_name_to_midi,
    midi_to_note_name, get_chord_notes, get_scale_notes, is_valid_midi_note,
    is_valid_velocity, clamp_midi_note, clamp_velocity, A4_MIDI, A4_FREQUENCY
)


class TestMidiConstants:
    """Test MIDI constant definitions."""
    
    def test_midi_note_range(self):
        """Test MIDI note range constants."""
        assert MIDI_NOTE_MIN == 21  # A0
        assert MIDI_NOTE_MAX == 108  # C8
        assert MIDI_NOTE_MAX > MIDI_NOTE_MIN
    
    def test_note_names(self):
        """Test note name definitions."""
        assert len(NOTE_NAMES) == 12
        assert NOTE_NAMES[0] == 'C'
        assert NOTE_NAMES[6] == 'F#'
        assert 'C' in NOTE_NAMES
        assert 'B' in NOTE_NAMES


class TestFrequencyConversion:
    """Test frequency conversion functions."""
    
    def test_a4_frequency(self):
        """Test A4 frequency conversion."""
        freq = midi_note_to_frequency(A4_MIDI)
        assert abs(freq - A4_FREQUENCY) < 0.001
    
    def test_frequency_roundtrip(self):
        """Test frequency to MIDI and back."""
        original_midi = 60  # Middle C
        freq = midi_note_to_frequency(original_midi)
        converted_midi = frequency_to_midi_note(freq)
        assert converted_midi == original_midi
    
    def test_octave_doubling(self):
        """Test that octaves double frequency."""
        c4_freq = midi_note_to_frequency(60)  # C4
        c5_freq = midi_note_to_frequency(72)  # C5
        assert abs(c5_freq / c4_freq - 2.0) < 0.001


class TestNoteNameConversion:
    """Test note name conversion functions."""
    
    def test_note_name_to_midi(self):
        """Test note name to MIDI conversion."""
        # Middle C
        assert note_name_to_midi('C', 4) == 60
        # A4 (concert pitch)
        assert note_name_to_midi('A', 4) == 69
    
    def test_midi_to_note_name(self):
        """Test MIDI to note name conversion."""
        note, octave = midi_to_note_name(60)
        assert note == 'C'
        assert octave == 4
        
        note, octave = midi_to_note_name(69)
        assert note == 'A'
        assert octave == 4
    
    def test_note_name_roundtrip(self):
        """Test note name conversion roundtrip."""
        original_note = 'F#'
        original_octave = 3
        midi = note_name_to_midi(original_note, original_octave)
        converted_note, converted_octave = midi_to_note_name(midi)
        assert converted_note == original_note
        assert converted_octave == original_octave
    
    def test_invalid_note_name(self):
        """Test invalid note name handling."""
        with pytest.raises(ValueError):
            note_name_to_midi('H', 4)  # Invalid note name


class TestChordFunctions:
    """Test chord-related functions."""
    
    def test_major_chord(self):
        """Test major chord construction."""
        # C major: C-E-G
        notes = get_chord_notes(60, 'major')  # C4 major
        expected = [60, 64, 67]  # C4, E4, G4
        assert notes == expected
    
    def test_minor_chord(self):
        """Test minor chord construction."""
        # A minor: A-C-E
        notes = get_chord_notes(57, 'minor')  # A3 minor
        expected = [57, 60, 64]  # A3, C4, E4
        assert notes == expected
    
    def test_seventh_chord(self):
        """Test seventh chord construction."""
        notes = get_chord_notes(60, 'major7')  # C major 7
        expected = [60, 64, 67, 71]  # C, E, G, B
        assert notes == expected
    
    def test_invalid_chord_type(self):
        """Test invalid chord type handling."""
        with pytest.raises(ValueError):
            get_chord_notes(60, 'invalid_chord')


class TestScaleFunctions:
    """Test scale-related functions."""
    
    def test_major_scale(self):
        """Test major scale construction."""
        # C major scale
        notes = get_scale_notes(60, 'major')
        expected = [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B
        assert notes == expected
    
    def test_minor_scale(self):
        """Test minor scale construction."""
        # A minor scale (natural minor)
        notes = get_scale_notes(57, 'minor')
        expected = [57, 59, 60, 62, 64, 65, 67]  # A B C D E F G
        assert notes == expected
    
    def test_pentatonic_scale(self):
        """Test pentatonic scale construction."""
        notes = get_scale_notes(60, 'pentatonic_major')
        expected = [60, 62, 64, 67, 69]  # C D E G A
        assert notes == expected
        assert len(notes) == 5
    
    def test_chromatic_scale(self):
        """Test chromatic scale construction."""
        notes = get_scale_notes(60, 'chromatic')
        assert len(notes) == 12
        assert notes[0] == 60
        assert notes[-1] == 71
    
    def test_invalid_scale_type(self):
        """Test invalid scale type handling."""
        with pytest.raises(ValueError):
            get_scale_notes(60, 'invalid_scale')


class TestValidationFunctions:
    """Test validation functions."""
    
    def test_valid_midi_note(self):
        """Test MIDI note validation."""
        assert is_valid_midi_note(60)  # Middle C
        assert is_valid_midi_note(MIDI_NOTE_MIN)  # Minimum
        assert is_valid_midi_note(MIDI_NOTE_MAX)  # Maximum
        assert not is_valid_midi_note(MIDI_NOTE_MIN - 1)  # Below range
        assert not is_valid_midi_note(MIDI_NOTE_MAX + 1)  # Above range
    
    def test_valid_velocity(self):
        """Test velocity validation."""
        assert is_valid_velocity(64)  # Mezzo forte
        assert is_valid_velocity(1)   # Minimum
        assert is_valid_velocity(127) # Maximum
        assert not is_valid_velocity(0)   # Below range
        assert not is_valid_velocity(128) # Above range
    
    def test_clamp_midi_note(self):
        """Test MIDI note clamping."""
        assert clamp_midi_note(60) == 60  # Within range
        assert clamp_midi_note(MIDI_NOTE_MIN - 10) == MIDI_NOTE_MIN
        assert clamp_midi_note(MIDI_NOTE_MAX + 10) == MIDI_NOTE_MAX
    
    def test_clamp_velocity(self):
        """Test velocity clamping."""
        assert clamp_velocity(64) == 64  # Within range
        assert clamp_velocity(0) == 1    # Below range
        assert clamp_velocity(200) == 127  # Above range


class TestMusicalTheory:
    """Test musical theory constants."""
    
    def test_chord_types_exist(self):
        """Test that all expected chord types are defined."""
        expected_chords = ['major', 'minor', 'diminished', 'augmented', 'major7']
        for chord in expected_chords:
            assert chord in CHORD_TYPES
    
    def test_scale_types_exist(self):
        """Test that all expected scale types are defined."""
        expected_scales = ['major', 'minor', 'pentatonic_major', 'blues']
        for scale in expected_scales:
            assert scale in SCALE_TYPES
    
    def test_chord_intervals(self):
        """Test specific chord interval definitions."""
        # Major triad: root, major third, perfect fifth
        assert CHORD_TYPES['major'] == [0, 4, 7]
        # Minor triad: root, minor third, perfect fifth
        assert CHORD_TYPES['minor'] == [0, 3, 7]
        # Diminished triad: root, minor third, diminished fifth
        assert CHORD_TYPES['diminished'] == [0, 3, 6]
    
    def test_scale_intervals(self):
        """Test specific scale interval definitions."""
        # Major scale: W-W-H-W-W-W-H pattern
        assert SCALE_TYPES['major'] == [0, 2, 4, 5, 7, 9, 11]
        # Minor scale: W-H-W-W-H-W-W pattern
        assert SCALE_TYPES['minor'] == [0, 2, 3, 5, 7, 8, 10]


if __name__ == '__main__':
    pytest.main([__file__])