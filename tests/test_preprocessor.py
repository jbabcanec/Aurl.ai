"""
Test suite for streaming preprocessor functionality.

Tests preprocessing pipeline with various options and validates
that the structure is set up correctly for future musical intelligence features.
"""

import sys
import tempfile
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import (
    StreamingPreprocessor, 
    PreprocessingOptions, 
    QuantizationMode,
    VelocityNormalizationMode,
    create_preprocessor
)
from src.utils.config import MidiFlyConfig


def test_preprocessor_initialization():
    """Test that preprocessor initializes correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        # Test default options
        processor = StreamingPreprocessor(config)
        assert processor.options.quantization_mode == QuantizationMode.GROOVE_PRESERVING
        assert processor.options.velocity_normalization == VelocityNormalizationMode.STYLE_PRESERVING
        assert processor.options.enable_chord_detection == False
        
        # Test custom options
        options = PreprocessingOptions(
            quantization_mode=QuantizationMode.STRICT,
            velocity_normalization=VelocityNormalizationMode.GLOBAL,
            enable_chord_detection=True
        )
        processor = StreamingPreprocessor(config, options)
        assert processor.options.quantization_mode == QuantizationMode.STRICT
        assert processor.options.enable_chord_detection == True


def test_factory_function():
    """Test the factory function for creating preprocessors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        processor = create_preprocessor(
            config,
            quantization_mode=QuantizationMode.ADAPTIVE,
            enable_chord_detection=True
        )
        
        assert processor.options.quantization_mode == QuantizationMode.ADAPTIVE
        assert processor.options.enable_chord_detection == True


def test_chord_detection_placeholder():
    """Test that chord detection placeholder structure is set up correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        processor = StreamingPreprocessor(config)
        
        # Test chord detection placeholder
        from src.data.midi_parser import MidiData
        
        # Create a simple MIDI data object
        midi_data = MidiData(
            filename="test.mid"
        )
        
        # Test chord detection returns proper structure
        chord_result = processor.chord_detector.detect_chords(midi_data)
        
        expected_keys = ["chords", "key_signature", "harmonic_rhythm", "chord_changes", "tonal_center", "note"]
        for key in expected_keys:
            assert key in chord_result
        
        assert "Phase 6" in chord_result["note"]


def test_preprocessing_options_validation():
    """Test that preprocessing options are validated correctly."""
    # Test valid options
    options = PreprocessingOptions(
        quantization_mode=QuantizationMode.GROOVE_PRESERVING,
        velocity_normalization=VelocityNormalizationMode.STYLE_PRESERVING,
        quantization_resolution=125,
        max_sequence_length=2048
    )
    
    assert options.quantization_mode == QuantizationMode.GROOVE_PRESERVING
    assert options.quantization_resolution == 125
    assert options.max_sequence_length == 2048


def test_cache_key_generation():
    """Test cache key generation for different configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        # Create a dummy file
        test_file = Path(temp_dir) / "test.mid"
        test_file.write_text("dummy content")
        
        processor = StreamingPreprocessor(config)
        
        # Generate cache key
        cache_key = processor._generate_cache_key(test_file)
        
        # Should be a valid hash string
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length
        
        # Same file should generate same key
        cache_key2 = processor._generate_cache_key(test_file)
        assert cache_key == cache_key2


def test_statistics_tracking():
    """Test that statistics are tracked correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        processor = StreamingPreprocessor(config)
        
        # Initial stats
        stats = processor.get_statistics()
        assert stats["files_processed"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_hit_rate"] == 0
        
        # Check that all expected statistics are present
        expected_stats = [
            "files_processed", "cache_hits", "processing_time", 
            "quantization_applied", "velocity_normalized",
            "cache_hit_rate", "avg_processing_time"
        ]
        
        for stat in expected_stats:
            assert stat in stats


def test_metadata_extraction():
    """Test metadata extraction from files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        processor = StreamingPreprocessor(config)
        
        # Create a dummy file
        test_file = Path(temp_dir) / "test.mid"
        test_file.write_text("dummy content")
        
        # Create simple MIDI data
        from src.data.midi_parser import MidiData
        
        midi_data = MidiData(
            filename=str(test_file)
        )
        
        # Extract metadata
        metadata = processor._extract_metadata(test_file, midi_data)
        
        # Check required fields
        required_fields = [
            "file_path", "file_size", "duration", "num_instruments",
            "total_notes", "preprocessing_options"
        ]
        
        for field in required_fields:
            assert field in metadata
        
        # Check preprocessing options in metadata
        assert "quantization_mode" in metadata["preprocessing_options"]
        assert "velocity_normalization" in metadata["preprocessing_options"]
        assert "chord_detection_enabled" in metadata["preprocessing_options"]


def test_stream_processing_structure():
    """Test that streaming processing structure is set up correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        processor = StreamingPreprocessor(config)
        
        # Test that stream_process_directory method exists and is callable
        assert hasattr(processor, 'stream_process_directory')
        assert callable(processor.stream_process_directory)
        
        # Test with empty directory
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        
        results = list(processor.stream_process_directory(empty_dir))
        assert len(results) == 0


def test_quantization_modes():
    """Test different quantization modes are available."""
    modes = [
        QuantizationMode.NONE,
        QuantizationMode.STRICT,
        QuantizationMode.GROOVE_PRESERVING,
        QuantizationMode.ADAPTIVE
    ]
    
    for mode in modes:
        assert isinstance(mode, QuantizationMode)
        assert isinstance(mode.value, str)


def test_velocity_normalization_modes():
    """Test different velocity normalization modes are available."""
    modes = [
        VelocityNormalizationMode.NONE,
        VelocityNormalizationMode.GLOBAL,
        VelocityNormalizationMode.PIECE_RELATIVE,
        VelocityNormalizationMode.STYLE_PRESERVING
    ]
    
    for mode in modes:
        assert isinstance(mode, VelocityNormalizationMode)
        assert isinstance(mode.value, str)


if __name__ == "__main__":
    # Run basic tests
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Running Preprocessor Tests")
    print("=" * 50)
    
    test_preprocessor_initialization()
    print("✅ Preprocessor initialization test passed")
    
    test_factory_function()
    print("✅ Factory function test passed")
    
    test_chord_detection_placeholder()
    print("✅ Chord detection placeholder test passed")
    
    test_preprocessing_options_validation()
    print("✅ Preprocessing options validation test passed")
    
    test_cache_key_generation()
    print("✅ Cache key generation test passed")
    
    test_statistics_tracking()
    print("✅ Statistics tracking test passed")
    
    test_metadata_extraction()
    print("✅ Metadata extraction test passed")
    
    test_stream_processing_structure()
    print("✅ Stream processing structure test passed")
    
    test_quantization_modes()
    print("✅ Quantization modes test passed")
    
    test_velocity_normalization_modes()
    print("✅ Velocity normalization modes test passed")
    
    print("\n🎉 All preprocessor tests passed!")
    print("📋 Phase 2.3 preprocessing structure is ready")
    print("💡 Chord detection placeholder set up for Phase 6")