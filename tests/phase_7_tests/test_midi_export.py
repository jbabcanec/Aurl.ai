"""
Comprehensive tests for Phase 7.2 MIDI Export System

Tests all MIDI export functionality including:
- Token sequence to MIDI conversion
- Professional-grade MIDI file creation
- Multi-track support
- Tempo and time signature handling
- Program change and control messages
- Finale compatibility optimization
- Error handling and edge cases

Phase 7.2 of the Aurl.ai GAMEPLAN.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

import mido
from mido import MidiFile

# Import MIDI export components
from src.generation import (
    MidiExporter, MidiExportConfig, ExportStatistics,
    export_tokens_to_midi_file, create_standard_config,
    create_performance_config
)
from src.data.representation import MusicRepresentationConverter, EventType


class TestMidiExportConfig:
    """Test MIDI export configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MidiExportConfig()
        
        assert config.resolution == 480
        assert config.default_tempo == 120.0
        assert config.default_time_signature == (4, 4)
        assert config.max_tracks == 16
        assert config.include_metadata is True
    
    def test_standard_config(self):
        """Test standard high-quality configuration."""
        config = create_standard_config()
        
        assert config.resolution == 480
        assert config.format_type == 1
        assert config.include_metadata is True
        assert config.quantize_timing is True
    
    def test_performance_config(self):
        """Test performance-optimized configuration."""
        config = create_performance_config()
        
        assert config.resolution == 960
        assert config.humanize_timing is True
        assert config.humanize_velocity is True
        assert config.quantize_timing is False


class TestMidiExporter:
    """Test MIDI exporter functionality."""
    
    @pytest.fixture
    def exporter(self):
        """Create a MIDI exporter for testing."""
        config = MidiExportConfig()
        return MidiExporter(config)
    
    @pytest.fixture
    def sample_tokens(self):
        """Create a sample token sequence for testing."""
        # Create a simple musical sequence: START, NOTE_ON C4, TIME_SHIFT, NOTE_OFF C4, END
        converter = MusicRepresentationConverter()
        
        # Get token values from vocabulary
        start_token = 1  # START token
        end_token = 2    # END token
        
        # NOTE_ON C4 (MIDI note 60, token offset 4 for special tokens + 60-21 = 43)
        note_on_c4 = 4 + (60 - 21)  # 43
        
        # NOTE_OFF C4 (after NOTE_ON range)
        note_off_c4 = 4 + 128 + (60 - 21)  # 171
        
        # TIME_SHIFT (after NOTE_ON and NOTE_OFF ranges)
        time_shift = 4 + 128 + 128 + 10  # Small time shift
        
        tokens = torch.tensor([
            start_token,
            note_on_c4,
            time_shift,
            note_off_c4,
            end_token
        ])
        
        return tokens.unsqueeze(0)  # Add batch dimension
    
    def test_exporter_initialization(self, exporter):
        """Test MIDI exporter initialization."""
        assert isinstance(exporter.config, MidiExportConfig)
        assert isinstance(exporter.converter, MusicRepresentationConverter)
        assert isinstance(exporter.stats, ExportStatistics)
        assert len(exporter.instrument_map) > 0
    
    def test_export_tokens_to_midi(self, exporter, sample_tokens):
        """Test exporting tokens to MIDI file."""
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = exporter.export_tokens_to_midi(
                tokens=sample_tokens,
                output_path=output_path,
                title="Test Song",
                style="classical",
                tempo=120.0
            )
            
            # Check file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Check statistics
            assert isinstance(stats, ExportStatistics)
            assert stats.total_tracks >= 1
            assert stats.export_time > 0
            
            # Load and verify MIDI file
            midi_file = MidiFile(output_path)
            assert len(midi_file.tracks) >= 1
            assert midi_file.ticks_per_beat == exporter.config.resolution
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_export_with_different_formats(self, exporter, sample_tokens):
        """Test export with different MIDI format types."""
        for format_type in [0, 1]:
            exporter.config.format_type = format_type
            
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                output_path = f.name
            
            try:
                stats = exporter.export_tokens_to_midi(
                    tokens=sample_tokens,
                    output_path=output_path,
                    title=f"Format {format_type} Test"
                )
                
                assert os.path.exists(output_path)
                
                midi_file = MidiFile(output_path)
                assert midi_file.type == format_type
                
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
    
    def test_export_multiple_styles(self, exporter):
        """Test exporting multiple token sequences with different styles."""
        # Create multiple sample sequences
        tokens_list = [
            torch.randint(0, 100, (1, 20)),  # Random tokens for testing
            torch.randint(0, 100, (1, 25)),
            torch.randint(0, 100, (1, 30))
        ]
        styles = ["classical", "jazz", "pop"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = exporter.export_multiple_styles(
                tokens_list=tokens_list,
                output_dir=temp_dir,
                base_filename="test_music",
                styles=styles
            )
            
            assert len(results) == 3
            
            # Check files were created
            for i, style in enumerate(styles):
                filename = f"test_music_{style}.mid"
                file_path = os.path.join(temp_dir, filename)
                assert os.path.exists(file_path)
                
                # Check statistics
                assert isinstance(results[i], ExportStatistics)
    
    def test_invalid_tokens(self, exporter):
        """Test handling of invalid token sequences."""
        # Empty token sequence
        empty_tokens = torch.tensor([]).unsqueeze(0)
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = exporter.export_tokens_to_midi(
                tokens=empty_tokens,
                output_path=output_path
            )
            
            # Should handle gracefully
            assert os.path.exists(output_path)
            assert stats.total_notes == 0
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_seconds_to_ticks_conversion(self, exporter):
        """Test time conversion functionality."""
        # Test basic conversion
        ticks = exporter._seconds_to_ticks(1.0)  # 1 second
        expected_ticks = int(120.0 / 60.0 * exporter.config.resolution)  # 120 BPM
        assert ticks == expected_ticks
        
        # Test zero and negative values
        assert exporter._seconds_to_ticks(0.0) == 0
        assert exporter._seconds_to_ticks(-1.0) == 0
    
    def test_export_summary(self, exporter, sample_tokens):
        """Test export summary generation."""
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            exporter.export_tokens_to_midi(
                tokens=sample_tokens,
                output_path=output_path
            )
            
            summary = exporter.get_export_summary()
            
            assert 'tracks' in summary
            assert 'notes' in summary
            assert 'duration_seconds' in summary
            assert 'export_time_seconds' in summary
            assert 'config' in summary
            
            # Check config in summary
            assert summary['config']['resolution'] == exporter.config.resolution
            assert summary['config']['format_type'] == exporter.config.format_type
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMidiFileStructure:
    """Test MIDI file structure and content."""
    
    def test_midi_file_headers(self):
        """Test MIDI file header creation."""
        exporter = MidiExporter(create_standard_config())
        
        # Create minimal MIDI data
        from src.data.midi_parser import MidiData, MidiInstrument, MidiNote
        
        # Create a simple note
        note = MidiNote(pitch=60, velocity=80, start=0.0, end=1.0)
        instrument = MidiInstrument(program=0, is_drum=False, name="Piano")
        instrument.notes = [note]
        
        midi_data = MidiData(
            instruments=[instrument],
            end_time=1.0,
            tempo_changes=[(0.0, 120.0)],
            time_signature_changes=[(0.0, 4, 4)]
        )
        
        # Create MIDI file
        midi_file = exporter._create_midi_file(
            midi_data,
            title="Test Header",
            tempo=120.0,
            time_signature=(4, 4)
        )
        
        assert midi_file.type == 1  # Multi-track format
        assert midi_file.ticks_per_beat == 480
        assert len(midi_file.tracks) >= 1
        
        # Check for conductor track (should be first in format 1)
        conductor_track = midi_file.tracks[0]
        
        # Look for meta messages
        meta_messages = [msg for msg in conductor_track if hasattr(msg, 'type') and hasattr(msg, 'text')]
        tempo_messages = [msg for msg in conductor_track if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'set_tempo']
        time_sig_messages = [msg for msg in conductor_track if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'time_signature']
        
        # For format 1 files, tempo/time sig may be in first track instead of conductor
        if len(midi_file.tracks) > 1:
            first_track = midi_file.tracks[1] if len(midi_file.tracks) > 1 else midi_file.tracks[0]
            tempo_messages.extend([msg for msg in first_track if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'set_tempo'])
            time_sig_messages.extend([msg for msg in first_track if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'time_signature'])
        
        assert len(tempo_messages) >= 1
        assert len(time_sig_messages) >= 1
    
    def test_track_creation(self):
        """Test individual track creation."""
        exporter = MidiExporter()
        
        from src.data.midi_parser import MidiInstrument, MidiNote
        
        # Create test instrument with notes
        notes = [
            MidiNote(pitch=60, velocity=80, start=0.0, end=0.5),
            MidiNote(pitch=64, velocity=75, start=0.5, end=1.0),
            MidiNote(pitch=67, velocity=70, start=1.0, end=1.5)
        ]
        
        instrument = MidiInstrument(program=0, is_drum=False, name="Piano")
        instrument.notes = notes
        
        track = exporter._create_instrument_track(
            instrument=instrument,
            track_index=0,
            title="Test Track",
            is_first_track=True
        )
        
        # Check track structure
        assert len(track) > 0
        
        # Look for program change
        program_changes = [msg for msg in track if hasattr(msg, 'type') and msg.type == 'program_change']
        assert len(program_changes) >= 1
        
        # Look for note events
        note_ons = [msg for msg in track if hasattr(msg, 'type') and msg.type == 'note_on']
        note_offs = [msg for msg in track if hasattr(msg, 'type') and msg.type == 'note_off']
        
        assert len(note_ons) == len(notes)
        assert len(note_offs) == len(notes)
        
        # Check end of track
        eot_messages = [msg for msg in track if hasattr(msg, 'type') and msg.type == 'end_of_track']
        assert len(eot_messages) == 1


class TestConvenienceFunctions:
    """Test convenience functions for MIDI export."""
    
    def test_export_tokens_to_midi_file(self):
        """Test the convenience function for direct export."""
        tokens = torch.randint(0, 100, (1, 10))
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = export_tokens_to_midi_file(
                tokens=tokens,
                output_path=output_path,
                title="Convenience Test",
                tempo=140.0
            )
            
            assert os.path.exists(output_path)
            assert isinstance(stats, ExportStatistics)
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_config_factories(self):
        """Test configuration factory functions."""
        standard_config = create_standard_config()
        performance_config = create_performance_config()
        
        # Standard config should prioritize quality
        assert standard_config.quantize_timing is True
        assert standard_config.include_metadata is True
        
        # Performance config should prioritize realism
        assert performance_config.humanize_timing is True
        assert performance_config.humanize_velocity is True
        assert performance_config.resolution == 960  # Higher resolution


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_output_path(self):
        """Test handling of invalid output paths."""
        exporter = MidiExporter()
        tokens = torch.randint(0, 100, (1, 10))
        
        # Try to write to invalid path
        invalid_path = "/invalid/path/test.mid"
        
        with pytest.raises(Exception):
            exporter.export_tokens_to_midi(
                tokens=tokens,
                output_path=invalid_path
            )
    
    def test_malformed_tokens(self):
        """Test handling of malformed token sequences."""
        exporter = MidiExporter()
        
        # Test various malformed inputs
        test_cases = [
            torch.tensor([]),  # Empty
            torch.tensor([[-1, -2]]),  # Negative values
            torch.tensor([[1000, 2000]]),  # Out of range values
        ]
        
        for tokens in test_cases:
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
                output_path = f.name
            
            try:
                # Should handle gracefully without crashing
                stats = exporter.export_tokens_to_midi(
                    tokens=tokens,
                    output_path=output_path
                )
                
                assert os.path.exists(output_path)
                assert isinstance(stats, ExportStatistics)
                
            except Exception as e:
                # Acceptable if it raises a clear error
                assert isinstance(e, (ValueError, RuntimeError))
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large token sequences."""
        exporter = MidiExporter()
        
        # Create a reasonably large sequence
        large_tokens = torch.randint(0, 100, (1, 1000))
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = exporter.export_tokens_to_midi(
                tokens=large_tokens,
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            assert stats.export_time < 10.0  # Should complete in reasonable time
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestStandardCompatibility:
    """Test standard MIDI compatibility features."""
    
    def test_standard_midi_export(self):
        """Test export with standard high-quality settings."""
        config = create_standard_config()
        exporter = MidiExporter(config)
        
        tokens = torch.randint(0, 100, (1, 20))
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = exporter.export_tokens_to_midi(
                tokens=tokens,
                output_path=output_path,
                title="Standard MIDI Test"
            )
            
            # Load and verify MIDI file structure
            midi_file = MidiFile(output_path)
            
            # Should be multi-track format (Type 1)
            assert midi_file.type == 1
            
            # Should have high resolution
            assert midi_file.ticks_per_beat == 480
            
            # Check for metadata
            conductor_track = midi_file.tracks[0]
            track_names = [msg for msg in conductor_track if hasattr(msg, 'type') and msg.type == 'track_name']
            assert len(track_names) >= 1
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_metadata_inclusion(self):
        """Test inclusion of standard metadata."""
        config = MidiExportConfig(include_metadata=True)
        exporter = MidiExporter(config)
        
        tokens = torch.randint(0, 100, (1, 10))
        
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            output_path = f.name
        
        try:
            stats = exporter.export_tokens_to_midi(
                tokens=tokens,
                output_path=output_path,
                title="Metadata Test",
                style="classical"
            )
            
            # Load and check for metadata messages
            midi_file = MidiFile(output_path)
            all_messages = []
            for track in midi_file.tracks:
                all_messages.extend(track)
            
            # Look for copyright and text messages
            copyright_msgs = [msg for msg in all_messages if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'copyright']
            text_msgs = [msg for msg in all_messages if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'text']
            track_name_msgs = [msg for msg in all_messages if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'track_name']
            
            # Should have some metadata (at minimum track names)
            assert len(copyright_msgs) + len(text_msgs) + len(track_name_msgs) > 0
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])