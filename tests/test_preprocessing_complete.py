"""
Comprehensive test suite for Phase 2.3 preprocessing pipeline.

Tests musical quantization, velocity normalization, polyphony reduction,
and the integrated streaming preprocessor.
"""

import sys
import tempfile
from pathlib import Path
import logging
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.midi_parser import MidiData, MidiNote, MidiInstrument
from src.data.musical_quantizer import (
    MusicalQuantizer, QuantizationConfig, GridResolution, create_adaptive_quantizer
)
from src.data.velocity_normalizer import (
    VelocityNormalizer, VelocityNormalizationConfig
)
from src.data.polyphony_reducer import (
    PolyphonyReducer, PolyphonyReductionConfig
)
from src.data.preprocessor import (
    StreamingPreprocessor, PreprocessingOptions, QuantizationMode, VelocityNormalizationMode
)
from src.utils.config import MidiFlyConfig


def create_test_midi_data():
    """Create test MIDI data with known properties."""
    notes = [
        # Some notes with imperfect timing (for quantization testing)
        MidiNote(pitch=60, velocity=64, start=0.03, end=0.53),  # Slightly off-grid
        MidiNote(pitch=64, velocity=80, start=0.51, end=1.01),  # Slightly late
        MidiNote(pitch=67, velocity=100, start=1.02, end=1.52), # Slightly late
        # Chord for polyphony testing
        MidiNote(pitch=60, velocity=70, start=2.0, end=3.0),
        MidiNote(pitch=64, velocity=75, start=2.0, end=3.0),
        MidiNote(pitch=67, velocity=80, start=2.0, end=3.0),
        MidiNote(pitch=72, velocity=85, start=2.0, end=3.0),
        # More notes for complex polyphony
        MidiNote(pitch=55, velocity=60, start=2.5, end=3.5),
        MidiNote(pitch=59, velocity=65, start=2.5, end=3.5),
        MidiNote(pitch=62, velocity=70, start=2.5, end=3.5),
    ]
    
    instrument = MidiInstrument(
        program=0,
        is_drum=False,
        name="Piano",
        notes=notes,
        pitch_bends=[],
        control_changes=[]
    )
    
    return MidiData(
        instruments=[instrument],
        tempo_changes=[(0.0, 120.0)],
        time_signature_changes=[(0.0, 4, 4)],
        key_signature_changes=[],
        resolution=220,
        end_time=4.0,
        filename="test.mid"
    )


def test_musical_quantization():
    """Test musical quantization functionality."""
    print("\n🎵 Testing Musical Quantization")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    
    # Test strict quantization
    config = QuantizationConfig(
        resolution=GridResolution.SIXTEENTH,
        strength=1.0,
        preserve_micro_timing=False
    )
    quantizer = MusicalQuantizer(config)
    
    # Analyze timing before quantization
    analysis_before = quantizer.analyze_timing(midi_data)
    print(f"Before quantization - Recommended resolution: {analysis_before.get('recommended_resolution', 'N/A')}")
    
    # Apply quantization
    quantized = quantizer.quantize_midi_data(midi_data)
    
    # Check that notes are quantized
    grid_size = 0.125  # 16th note at 120 BPM
    for note in quantized.instruments[0].notes[:3]:  # Check first 3 notes
        # Should be close to grid positions
        grid_position = round(note.start / grid_size) * grid_size
        assert abs(note.start - grid_position) < 0.01, f"Note not properly quantized: {note.start}"
    
    print("✅ Strict quantization test passed")
    
    # Test groove-preserving quantization
    config = QuantizationConfig(
        resolution=GridResolution.SIXTEENTH,
        strength=0.5,  # Partial quantization
        preserve_micro_timing=True
    )
    quantizer = MusicalQuantizer(config)
    quantized = quantizer.quantize_midi_data(midi_data)
    
    # Notes should be partially quantized (between original and grid)
    original_note = midi_data.instruments[0].notes[0]
    quantized_note = quantized.instruments[0].notes[0]
    assert original_note.start != quantized_note.start, "Note should be modified"
    assert quantized_note.start != 0.0, "Note should not be fully quantized"
    
    print("✅ Groove-preserving quantization test passed")
    
    # Test adaptive quantization
    adaptive_quantizer = create_adaptive_quantizer(midi_data)
    adaptively_quantized = adaptive_quantizer.quantize_midi_data(midi_data)
    
    print("✅ Adaptive quantization test passed")


def test_velocity_normalization():
    """Test velocity normalization functionality."""
    print("\n🎹 Testing Velocity Normalization")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    
    # Test global normalization
    config = VelocityNormalizationConfig(
        target_mean=64.0,
        target_std=20.0,
        min_velocity=20,
        max_velocity=110
    )
    normalizer = VelocityNormalizer(config)
    
    # Analyze dynamics before
    analysis_before = normalizer.analyze_dynamics(midi_data)
    print(f"Before normalization - Velocity range: {analysis_before['velocity_range']}")
    
    # Apply global normalization
    normalized = normalizer.normalize_midi_data(midi_data, mode="global")
    
    # All velocities should be the target mean
    for note in normalized.instruments[0].notes:
        assert note.velocity == 64, f"Global normalization failed: {note.velocity}"
    
    print("✅ Global normalization test passed")
    
    # Test piece-relative normalization
    normalized = normalizer.normalize_midi_data(midi_data, mode="piece_relative")
    
    # Check that velocities are scaled to the target range
    velocities = [n.velocity for n in normalized.instruments[0].notes]
    assert min(velocities) >= config.min_velocity
    assert max(velocities) <= config.max_velocity
    
    print("✅ Piece-relative normalization test passed")
    
    # Test style-preserving normalization
    normalized = normalizer.normalize_midi_data(midi_data, mode="style_preserving")
    
    # Analyze dynamics after
    analysis_after = normalizer.analyze_dynamics(normalized)
    
    # Should preserve relative dynamics
    original_velocities = [n.velocity for n in midi_data.instruments[0].notes]
    normalized_velocities = [n.velocity for n in normalized.instruments[0].notes]
    
    # Check that ordering is preserved (louder notes stay louder)
    for i in range(len(original_velocities) - 1):
        if original_velocities[i] < original_velocities[i + 1]:
            assert normalized_velocities[i] <= normalized_velocities[i + 1], \
                "Style preservation failed: relative dynamics not preserved"
    
    print("✅ Style-preserving normalization test passed")
    print(f"After normalization - Mean velocity: {analysis_after['mean_velocity']:.1f}")


def test_polyphony_reduction():
    """Test polyphony reduction functionality."""
    print("\n🎼 Testing Polyphony Reduction")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    
    # Analyze polyphony before reduction
    config = PolyphonyReductionConfig(max_polyphony=4)
    reducer = PolyphonyReducer(config)
    
    analysis_before = reducer.analyze_polyphony(midi_data)
    print(f"Before reduction - Max polyphony: {analysis_before['max_polyphony']}")
    print(f"Reduction needed: {analysis_before['reduction_needed']}")
    
    # Apply polyphony reduction
    reduced = reducer.reduce_polyphony(midi_data)
    
    # Analyze after reduction
    analysis_after = reducer.analyze_polyphony(reduced)
    print(f"After reduction - Max polyphony: {analysis_after['max_polyphony']}")
    
    # Verify polyphony is reduced
    assert analysis_after['max_polyphony'] <= config.max_polyphony, \
        f"Polyphony not reduced properly: {analysis_after['max_polyphony']}"
    
    # Count notes at the chord position (t=2.0)
    notes_at_chord = [n for n in reduced.instruments[0].notes 
                      if n.start <= 2.0 <= n.end]
    assert len(notes_at_chord) <= config.max_polyphony, \
        f"Too many simultaneous notes: {len(notes_at_chord)}"
    
    print("✅ Polyphony reduction test passed")
    
    # Test different priority modes
    for mode in ["musical", "velocity", "pitch", "newest"]:
        config = PolyphonyReductionConfig(max_polyphony=3, priority_mode=mode)
        reducer = PolyphonyReducer(config)
        reduced = reducer.reduce_polyphony(midi_data)
        
        # Should always respect max polyphony
        analysis = reducer.analyze_polyphony(reduced)
        assert analysis['max_polyphony'] <= 3, f"Mode {mode} failed"
        
    print("✅ All priority modes test passed")


def test_integrated_preprocessing():
    """Test the complete preprocessing pipeline."""
    print("\n🎭 Testing Integrated Preprocessing Pipeline")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config and preprocessor
        config = MidiFlyConfig()
        config.system.cache_dir = str(temp_dir)
        
        options = PreprocessingOptions(
            quantization_mode=QuantizationMode.GROOVE_PRESERVING,
            quantization_strength=0.8,
            velocity_normalization=VelocityNormalizationMode.STYLE_PRESERVING,
            max_polyphony=6,
            reduce_polyphony=True,
            cache_processed=False  # Disable for testing
        )
        
        preprocessor = StreamingPreprocessor(config, options)
        
        # Create test MIDI file
        test_file = Path(temp_dir) / "test.mid"
        test_file.write_text("dummy")  # Just for file existence
        
        # Create test data
        midi_data = create_test_midi_data()
        
        # Apply preprocessing
        processed = preprocessor._apply_preprocessing(midi_data)
        
        # Verify all preprocessing was applied
        assert preprocessor.stats["quantization_applied"] > 0
        assert preprocessor.stats["velocity_normalized"] > 0
        
        # Check that data is actually modified
        original_note = midi_data.instruments[0].notes[0]
        processed_note = processed.instruments[0].notes[0]
        
        # Timing should be quantized
        assert original_note.start != processed_note.start, \
            "Quantization not applied"
        
        # Velocity should be normalized
        original_velocities = [n.velocity for n in midi_data.instruments[0].notes]
        processed_velocities = [n.velocity for n in processed.instruments[0].notes]
        assert original_velocities != processed_velocities, \
            "Velocity normalization not applied"
        
        print("✅ Integrated preprocessing test passed")
        
        # Test statistics
        stats = preprocessor.get_statistics()
        print(f"\nPreprocessing Statistics:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Quantization applied: {stats['quantization_applied']}")
        print(f"  Velocity normalized: {stats['velocity_normalized']}")


def test_preprocessing_analysis():
    """Test preprocessing analysis functions."""
    print("\n📊 Testing Preprocessing Analysis")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    
    # Test quantization analysis
    quantizer = MusicalQuantizer()
    timing_analysis = quantizer.analyze_timing(midi_data)
    
    print("Timing Analysis:")
    print(f"  Recommended resolution: {timing_analysis['recommended_resolution']}")
    print(f"  Grid alignments: {list(timing_analysis['grid_alignments'].keys())[:3]}...")
    
    # Test velocity analysis
    normalizer = VelocityNormalizer()
    dynamics_analysis = normalizer.analyze_dynamics(midi_data)
    
    print("\nDynamics Analysis:")
    print(f"  Velocity range: {dynamics_analysis['velocity_range']}")
    print(f"  Mean velocity: {dynamics_analysis['mean_velocity']:.1f}")
    print(f"  Most common dynamic: {dynamics_analysis['most_common_dynamic']}")
    
    # Test polyphony analysis
    reducer = PolyphonyReducer()
    polyphony_analysis = reducer.analyze_polyphony(midi_data)
    
    print("\nPolyphony Analysis:")
    print(f"  Max polyphony: {polyphony_analysis['max_polyphony']}")
    print(f"  Mean polyphony: {polyphony_analysis['mean_polyphony']:.2f}")
    
    print("\n✅ All analysis functions working correctly")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Running Complete Phase 2.3 Preprocessing Tests")
    print("=" * 60)
    
    try:
        test_musical_quantization()
        test_velocity_normalization()
        test_polyphony_reduction()
        test_integrated_preprocessing()
        test_preprocessing_analysis()
        
        print("\n🎉 All Phase 2.3 preprocessing tests passed!")
        print("✅ Musical quantization: COMPLETE")
        print("✅ Velocity normalization: COMPLETE")
        print("✅ Polyphony reduction: COMPLETE")
        print("✅ Integrated pipeline: COMPLETE")
        print("✅ Analysis tools: COMPLETE")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise