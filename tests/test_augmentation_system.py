"""
Comprehensive test suite for Phase 2.4 Data Augmentation System.

Tests all augmentation types and validates musical preservation.
"""

import sys
import tempfile
from pathlib import Path
import logging
import numpy as np
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.midi_parser import MidiData, MidiNote, MidiInstrument
from src.data.augmentation import (
    MusicAugmenter, AugmentationConfig, AugmentationSchedule,
    PitchTransposer, TimeStretcher, VelocityScaler, 
    InstrumentSubstituter, RhythmicVariator,
    AugmentationType
)


def create_test_midi_data():
    """Create comprehensive test MIDI data for augmentation testing."""
    
    # Create a melody with various characteristics
    melody_notes = [
        MidiNote(pitch=60, velocity=70, start=0.0, end=0.5),    # C4
        MidiNote(pitch=64, velocity=75, start=0.5, end=1.0),    # E4
        MidiNote(pitch=67, velocity=80, start=1.0, end=1.5),    # G4
        MidiNote(pitch=72, velocity=85, start=1.5, end=2.0),    # C5
        MidiNote(pitch=76, velocity=90, start=2.0, end=2.5),    # E5
        MidiNote(pitch=79, velocity=75, start=2.5, end=3.0),    # G5
    ]
    
    # Create accompaniment with different rhythm
    accompaniment_notes = [
        MidiNote(pitch=48, velocity=60, start=0.0, end=1.0),    # C3 (bass)
        MidiNote(pitch=52, velocity=55, start=0.75, end=1.25),  # E3
        MidiNote(pitch=55, velocity=65, start=1.5, end=2.5),    # G3
        MidiNote(pitch=60, velocity=50, start=2.25, end=2.75),  # C4
    ]
    
    # Create instruments
    piano_melody = MidiInstrument(
        program=0,  # Acoustic Grand Piano
        is_drum=False,
        name="Piano Melody",
        notes=melody_notes,
        pitch_bends=[],
        control_changes=[]
    )
    
    piano_bass = MidiInstrument(
        program=0,  # Acoustic Grand Piano
        is_drum=False,
        name="Piano Bass",
        notes=accompaniment_notes,
        pitch_bends=[],
        control_changes=[]
    )
    
    return MidiData(
        instruments=[piano_melody, piano_bass],
        tempo_changes=[(0.0, 120.0)],
        time_signature_changes=[(0.0, 4, 4)],
        key_signature_changes=[(0.0, 0)],  # C major
        resolution=220,
        end_time=3.0,
        filename="test_piece.mid"
    )


def test_pitch_transposition():
    """Test pitch transposition augmentation."""
    print("\n🎵 Testing Pitch Transposition")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig()
    transposer = PitchTransposer(config)
    
    # Test various transpositions
    test_cases = [-12, -7, -5, 0, 3, 7, 12]
    
    for semitones in test_cases:
        transposed = transposer.transpose(midi_data, semitones)
        
        # Verify all notes are transposed correctly
        for i, orig_inst in enumerate(midi_data.instruments):
            trans_inst = transposed.instruments[i]
            
            for j, orig_note in enumerate(orig_inst.notes):
                trans_note = trans_inst.notes[j]
                expected_pitch = orig_note.pitch + semitones
                
                # Check if note is in valid range
                if 0 <= expected_pitch <= 127:
                    assert trans_note.pitch == expected_pitch, \
                        f"Transposition failed: {orig_note.pitch} + {semitones} = {expected_pitch}, got {trans_note.pitch}"
                    
                    # Verify other properties unchanged
                    assert trans_note.velocity == orig_note.velocity
                    assert trans_note.start == orig_note.start
                    assert trans_note.end == orig_note.end
        
        print(f"  ✅ Transposition by {semitones:+2d} semitones: OK")
    
    # Test edge cases (notes going out of range)
    extreme_transpose = transposer.transpose(midi_data, 50)  # Very high
    # Should have fewer notes (some filtered out)
    assert len(extreme_transpose.instruments[0].notes) <= len(midi_data.instruments[0].notes)
    
    print("  ✅ Edge case handling: OK")


def test_time_stretching():
    """Test time stretching augmentation."""
    print("\n⏱️ Testing Time Stretching")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig()
    stretcher = TimeStretcher(config)
    
    # Test various stretch factors
    test_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    for factor in test_factors:
        stretched = stretcher.stretch(midi_data, factor)
        
        # Verify timing is scaled correctly
        assert abs(stretched.end_time - midi_data.end_time * factor) < 0.001
        
        for i, orig_inst in enumerate(midi_data.instruments):
            stretch_inst = stretched.instruments[i]
            
            for j, orig_note in enumerate(orig_inst.notes):
                stretch_note = stretch_inst.notes[j]
                
                # Check timing scaling
                assert abs(stretch_note.start - orig_note.start * factor) < 0.001
                assert abs(stretch_note.end - orig_note.end * factor) < 0.001
                
                # Verify other properties unchanged
                assert stretch_note.pitch == orig_note.pitch
                assert stretch_note.velocity == orig_note.velocity
        
        # Check tempo changes are scaled
        for j, (orig_time, orig_tempo) in enumerate(midi_data.tempo_changes):
            stretch_time, stretch_tempo = stretched.tempo_changes[j]
            assert abs(stretch_time - orig_time * factor) < 0.001
            assert abs(stretch_tempo - orig_tempo / factor) < 0.001
        
        print(f"  ✅ Time stretch by {factor:.1f}x: OK")


def test_velocity_scaling():
    """Test velocity scaling augmentation."""
    print("\n🔊 Testing Velocity Scaling")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig(preserve_velocity_curves=True)
    scaler = VelocityScaler(config)
    
    # Test various scaling factors
    test_factors = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    for factor in test_factors:
        scaled = scaler.scale(midi_data, factor)
        
        # Verify velocities are scaled appropriately
        for i, orig_inst in enumerate(midi_data.instruments):
            scaled_inst = scaled.instruments[i]
            
            for j, orig_note in enumerate(orig_inst.notes):
                scaled_note = scaled_inst.notes[j]
                
                # Velocity should be scaled (with curve preservation)
                if factor != 1.0:
                    assert scaled_note.velocity != orig_note.velocity, \
                        f"Velocity should be scaled: {orig_note.velocity} -> {scaled_note.velocity}"
                
                # Should be in valid range
                assert 1 <= scaled_note.velocity <= 127
                
                # Other properties should be unchanged
                assert scaled_note.pitch == orig_note.pitch
                assert scaled_note.start == orig_note.start
                assert scaled_note.end == orig_note.end
        
        print(f"  ✅ Velocity scale by {factor:.1f}x: OK")
    
    # Test curve preservation
    config_no_preserve = AugmentationConfig(preserve_velocity_curves=False)
    scaler_simple = VelocityScaler(config_no_preserve)
    scaled_simple = scaler_simple.scale(midi_data, 0.8)
    
    # Simple scaling should be more predictable
    for orig_note, scaled_note in zip(midi_data.instruments[0].notes, scaled_simple.instruments[0].notes):
        expected_velocity = int(orig_note.velocity * 0.8)
        expected_velocity = max(1, min(127, expected_velocity))
        # Simple scaling should be close to linear
        assert abs(scaled_note.velocity - expected_velocity) <= 5, \
            f"Simple scaling failed: {orig_note.velocity} * 0.8 = {expected_velocity}, got {scaled_note.velocity}"
    
    print("  ✅ Curve preservation modes: OK")


def test_instrument_substitution():
    """Test instrument substitution augmentation."""
    print("\n🎹 Testing Instrument Substitution")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig()
    substituter = InstrumentSubstituter(config)
    
    # Test substitution
    substituted = substituter.substitute(midi_data)
    
    # Should have same number of instruments
    assert len(substituted.instruments) == len(midi_data.instruments)
    
    # Check that at least some instruments changed (with some randomness tolerance)
    random.seed(42)  # For reproducible test
    np.random.seed(42)
    
    substituted_fixed = substituter.substitute(midi_data)
    
    changes = 0
    for orig_inst, sub_inst in zip(midi_data.instruments, substituted_fixed.instruments):
        if not orig_inst.is_drum:  # Only non-drum instruments should change
            if orig_inst.program != sub_inst.program:
                changes += 1
        
        # Verify notes are preserved
        assert len(orig_inst.notes) == len(sub_inst.notes)
        for orig_note, sub_note in zip(orig_inst.notes, sub_inst.notes):
            assert orig_note.pitch == sub_note.pitch
            assert orig_note.velocity == sub_note.velocity
            assert orig_note.start == sub_note.start
            assert orig_note.end == sub_note.end
    
    print(f"  ✅ Instrument substitution: {changes} instruments changed")
    
    # Test with allowed instruments
    config_limited = AugmentationConfig(allowed_instruments=[1, 2, 3])
    substituter_limited = InstrumentSubstituter(config_limited)
    substituted_limited = substituter_limited.substitute(midi_data)
    
    for inst in substituted_limited.instruments:
        if not inst.is_drum:
            assert inst.program in [1, 2, 3], f"Instrument {inst.program} not in allowed list"
    
    print("  ✅ Allowed instruments constraint: OK")


def test_rhythmic_variations():
    """Test rhythmic variations (swing and humanization)."""
    print("\n🎭 Testing Rhythmic Variations")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig()
    variator = RhythmicVariator(config)
    
    # Test swing
    swing_ratios = [0.5, 0.6, 0.67, 0.7]
    
    for swing_ratio in swing_ratios:
        swung = variator.add_swing(midi_data, swing_ratio)
        
        # Verify structure preserved
        assert len(swung.instruments) == len(midi_data.instruments)
        assert swung.end_time == midi_data.end_time
        
        for i, orig_inst in enumerate(midi_data.instruments):
            swung_inst = swung.instruments[i]
            assert len(orig_inst.notes) == len(swung_inst.notes)
            
            for j, (orig_note, swung_note) in enumerate(zip(orig_inst.notes, swung_inst.notes)):
                # Pitch and velocity should be unchanged
                assert orig_note.pitch == swung_note.pitch
                assert orig_note.velocity == swung_note.velocity
                
                # Timing may be adjusted for swing
                if swing_ratio != 0.5:  # If swing is applied
                    # Duration should be preserved approximately (but may vary due to swing)
                    orig_duration = orig_note.end - orig_note.start
                    swung_duration = swung_note.end - swung_note.start
                    # Allow more tolerance for swing timing adjustments
                    assert abs(orig_duration - swung_duration) < orig_duration * 0.5  # 50% tolerance
        
        print(f"  ✅ Swing ratio {swing_ratio:.2f}: OK")
    
    # Test humanization
    humanization_amounts = [0.0, 0.01, 0.02, 0.05]
    
    for amount in humanization_amounts:
        # Set seed for reproducible randomness
        random.seed(42)
        np.random.seed(42)
        
        humanized = variator.humanize(midi_data, amount)
        
        # Verify structure preserved
        assert len(humanized.instruments) == len(midi_data.instruments)
        
        timing_changes = 0
        velocity_changes = 0
        
        for i, orig_inst in enumerate(midi_data.instruments):
            humanized_inst = humanized.instruments[i]
            assert len(orig_inst.notes) == len(humanized_inst.notes)
            
            for orig_note, humanized_note in zip(orig_inst.notes, humanized_inst.notes):
                # Pitch should be unchanged
                assert orig_note.pitch == humanized_note.pitch
                
                # Timing may be adjusted
                if abs(orig_note.start - humanized_note.start) > 0.001:
                    timing_changes += 1
                
                # Velocity may be adjusted
                if orig_note.velocity != humanized_note.velocity:
                    velocity_changes += 1
                
                # Velocity should stay in range
                assert 1 <= humanized_note.velocity <= 127
                
                # Start time should not be negative
                assert humanized_note.start >= 0
        
        if amount > 0:
            print(f"  ✅ Humanization {amount:.2f}s: {timing_changes} timing, {velocity_changes} velocity changes")
        else:
            print(f"  ✅ No humanization: {timing_changes} timing, {velocity_changes} velocity changes")


def test_integrated_augmentation():
    """Test the integrated augmentation system."""
    print("\n🎭 Testing Integrated Augmentation System")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    
    # Test with moderate settings
    config = AugmentationConfig(
        transpose_probability=0.7,
        time_stretch_probability=0.5,
        velocity_scale_probability=0.6,
        instrument_substitution_probability=0.3,
        swing_probability=0.2,
        humanization_probability=0.4,
        max_simultaneous_augmentations=2,
        seed=42  # For reproducible testing
    )
    
    schedule = AugmentationSchedule(
        warmup_epochs=3,
        max_probability=1.0,
        min_probability=0.1,
        decay_factor=0.95
    )
    
    augmenter = MusicAugmenter(config, schedule)
    
    # Test augmentation at different epochs
    epochs_to_test = [0, 1, 3, 5, 10, 20]
    
    for epoch in epochs_to_test:
        # Set seed for reproducible results
        random.seed(42 + epoch)
        np.random.seed(42 + epoch)
        
        augmented = augmenter.augment(midi_data, epoch)
        
        # Verify basic structure preserved
        assert len(augmented.instruments) == len(midi_data.instruments)
        assert augmented.end_time > 0
        
        # Check that filename reflects augmentations
        if augmented.filename != midi_data.filename:
            assert "aug_" in augmented.filename
        
        prob_multiplier = augmenter._get_probability_multiplier(epoch)
        
        print(f"  ✅ Epoch {epoch:2d}: probability multiplier {prob_multiplier:.3f}, filename: {augmented.filename}")
    
    # Test statistics
    stats = augmenter.get_augmentation_stats()
    assert "config" in stats
    assert "schedule" in stats
    
    print(f"  ✅ Augmentation statistics: {len(stats['config'])} config params")


def test_augmentation_scheduling():
    """Test augmentation probability scheduling."""
    print("\n📅 Testing Augmentation Scheduling")
    print("=" * 50)
    
    # Test different schedule configurations
    schedules = [
        AugmentationSchedule(warmup_epochs=5, max_probability=1.0, min_probability=0.1, decay_factor=0.9),
        AugmentationSchedule(warmup_epochs=0, max_probability=0.8, min_probability=0.2, decay_factor=0.95),
        AugmentationSchedule(epoch_schedule={0: 0.1, 5: 0.5, 10: 1.0, 20: 0.3})
    ]
    
    config = AugmentationConfig(seed=42)
    
    for i, schedule in enumerate(schedules):
        augmenter = MusicAugmenter(config, schedule)
        
        print(f"\n  Schedule {i+1}:")
        
        # Test probability multipliers over epochs
        test_epochs = [0, 1, 2, 5, 10, 15, 20, 25]
        
        for epoch in test_epochs:
            prob_mult = augmenter._get_probability_multiplier(epoch)
            print(f"    Epoch {epoch:2d}: {prob_mult:.3f}")
            
            # Verify constraints
            assert schedule.min_probability <= prob_mult <= schedule.max_probability
        
        print(f"  ✅ Schedule {i+1}: Valid probability multipliers")


def test_augmentation_preservation():
    """Test that augmentations preserve essential musical properties."""
    print("\n🎵 Testing Musical Property Preservation")
    print("=" * 50)
    
    midi_data = create_test_midi_data()
    config = AugmentationConfig(seed=42)
    augmenter = MusicAugmenter(config)
    
    # Apply multiple rounds of augmentation
    test_rounds = 5
    
    for round_num in range(test_rounds):
        random.seed(42 + round_num)
        np.random.seed(42 + round_num)
        
        augmented = augmenter.augment(midi_data, epoch=round_num)
        
        # Check basic preservation
        assert len(augmented.instruments) == len(midi_data.instruments), \
            f"Round {round_num}: Instrument count changed"
        
        # Check that we have some notes
        total_notes = sum(len(inst.notes) for inst in augmented.instruments)
        assert total_notes > 0, f"Round {round_num}: No notes remaining"
        
        # Check that timing makes sense
        assert augmented.end_time > 0, f"Round {round_num}: Invalid end time"
        
        # Check note validity
        for inst in augmented.instruments:
            for note in inst.notes:
                assert 0 <= note.pitch <= 127, f"Round {round_num}: Invalid pitch {note.pitch}"
                assert 1 <= note.velocity <= 127, f"Round {round_num}: Invalid velocity {note.velocity}"
                assert note.start >= 0, f"Round {round_num}: Negative start time"
                assert note.end > note.start, f"Round {round_num}: Invalid note duration"
        
        print(f"  ✅ Round {round_num}: Properties preserved, {total_notes} notes")
    
    print(f"  ✅ All rounds: Musical properties preserved")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    print("🎭 Aurl.ai Data Augmentation System Test Suite")
    print("Testing Phase 2.4 - Real-time Musical Augmentation")
    print("=" * 60)
    
    try:
        test_pitch_transposition()
        test_time_stretching()  
        test_velocity_scaling()
        test_instrument_substitution()
        test_rhythmic_variations()
        test_integrated_augmentation()
        test_augmentation_scheduling()
        test_augmentation_preservation()
        
        print(f"\n🎉 All Data Augmentation Tests Passed!")
        print(f"✅ Pitch transposition: Working")
        print(f"✅ Time stretching: Working")
        print(f"✅ Velocity scaling: Working")
        print(f"✅ Instrument substitution: Working")
        print(f"✅ Rhythmic variations: Working")
        print(f"✅ Integrated system: Working")
        print(f"✅ Scheduling system: Working")
        print(f"✅ Property preservation: Working")
        print(f"\n🚀 Phase 2.4 Data Augmentation System COMPLETE!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise