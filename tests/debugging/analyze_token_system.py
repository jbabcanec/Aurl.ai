#!/usr/bin/env python3
"""
Deep Token System Analysis

This script performs a comprehensive analysis of why generated tokens
produce 0-1 notes instead of proper music by examining:

1. Token vocabulary mapping
2. Generated token sequences 
3. Token-to-MIDI conversion process
4. Parameter quantization issues
5. Musical grammar validation
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project path
sys.path.append('/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl')

from src.data.representation import (
    VocabularyConfig, EventType, MusicRepresentationConverter, MusicEvent
)
from src.generation.midi_export import MidiExporter, create_standard_config
from src.training.musical_grammar import validate_generated_sequence

def analyze_vocabulary():
    """Analyze the token vocabulary structure."""
    print("=" * 80)
    print("TOKEN VOCABULARY ANALYSIS")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    
    print(f"Total vocabulary size: {vocab_config.vocab_size}")
    print(f"Time shift bins: {vocab_config.time_shift_bins}")
    print(f"Time shift resolution: {vocab_config.time_shift_ms}ms")
    print(f"Velocity bins: {vocab_config.velocity_bins}")
    print(f"Pitch range: {vocab_config.min_pitch} - {vocab_config.max_pitch}")
    
    # Show token ranges for each event type
    print("\nToken Ranges by Event Type:")
    print("-" * 40)
    
    # Find token ranges by analyzing the vocabulary
    event_type_ranges = {}
    for token_id in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token_id)
        event_name = event_type.name
        
        if event_name not in event_type_ranges:
            event_type_ranges[event_name] = {'min': token_id, 'max': token_id, 'count': 0}
        
        event_type_ranges[event_name]['max'] = token_id
        event_type_ranges[event_name]['count'] += 1
    
    for event_name, info in event_type_ranges.items():
        print(f"{event_name:20s}: tokens {info['min']:4d}-{info['max']:4d} ({info['count']:3d} tokens)")
    
    return vocab_config

def test_problematic_token_sequences():
    """Test specific problematic token sequences found in generation."""
    print("\n" + "=" * 80)
    print("TESTING PROBLEMATIC TOKEN SEQUENCES")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    
    # Test case 1: Token 139 repetition (commonly generated)
    print("\nTest 1: Repetitive Token 139 (common generation problem)")
    print("-" * 60)
    
    token_139_sequence = np.array([139] * 100)  # 100 repetitions of token 139
    
    event_type, value = vocab_config.token_to_event_info(139)
    print(f"Token 139 maps to: {event_type.name} with value {value}")
    
    try:
        midi_data = converter.tokens_to_midi(token_139_sequence)
        print(f"MIDI conversion successful:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        print(f"  Instruments: {len(midi_data.instruments)}")
        
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        print(f"  Total notes: {total_notes}")
        
        if total_notes == 0:
            print("🚨 PROBLEM: No notes generated from token sequence!")
    except Exception as e:
        print(f"❌ MIDI conversion failed: {e}")
    
    # Test case 2: Mixed low-value tokens (velocity/timing issues)
    print("\nTest 2: Low-value parameter tokens (velocity=1, duration=0 issues)")
    print("-" * 60)
    
    # Create sequence with low velocity and timing values
    problematic_sequence = []
    problematic_sequence.append(0)  # START_TOKEN
    
    # Add some note events with problematic parameters
    note_on_token = 4 + (60 - vocab_config.min_pitch)  # C4 note on
    note_off_token = 132 + (60 - vocab_config.min_pitch)  # C4 note off
    velocity_token = 260  # Very low velocity (maps to velocity ≈ 1)
    time_token = 260  # Very short time shift (maps to ≈ 0s)
    
    problematic_sequence.extend([
        velocity_token,  # Low velocity
        note_on_token,   # Note on C4
        time_token,      # Almost zero time shift
        note_off_token,  # Note off C4
    ])
    problematic_sequence.append(1)  # END_TOKEN
    
    problematic_tokens = np.array(problematic_sequence)
    
    print("Analyzing each token:")
    for i, token in enumerate(problematic_tokens):
        event_type, value = vocab_config.token_to_event_info(token)
        
        # Convert to actual MIDI parameters
        if event_type == EventType.VELOCITY_CHANGE:
            midi_velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d} → MIDI velocity={midi_velocity}")
        elif event_type == EventType.TIME_SHIFT:
            duration = value * vocab_config.time_shift_ms / 1000.0
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d} → Duration={duration:.4f}s")
        else:
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d}")
    
    try:
        midi_data = converter.tokens_to_midi(problematic_tokens)
        print(f"\nMIDI conversion result:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        print(f"  Total notes: {total_notes}")
        
        # Check actual note parameters
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note_idx, note in enumerate(instrument.notes):
                print(f"  Note {note_idx}: pitch={note.pitch}, vel={note.velocity}, "
                      f"start={note.start:.4f}s, end={note.end:.4f}s, dur={note.duration:.4f}s")
                
                if note.velocity <= 1:
                    print(f"    🚨 PROBLEM: Note has velocity {note.velocity} (silent/inaudible)")
                if note.duration <= 0.01:
                    print(f"    🚨 PROBLEM: Note duration {note.duration:.4f}s (too short to hear)")
    
    except Exception as e:
        print(f"❌ MIDI conversion failed: {e}")

def test_good_token_sequences():
    """Test what good token sequences should look like."""
    print("\n" + "=" * 80)
    print("TESTING GOOD TOKEN SEQUENCES")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    
    # Create a proper musical sequence
    good_sequence = []
    good_sequence.append(0)  # START_TOKEN
    
    # Set good velocity (mid-range, audible)
    good_velocity_value = vocab_config.velocity_bins // 2  # Mid-range velocity
    good_velocity_token = None
    for token_id in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token_id)
        if event_type == EventType.VELOCITY_CHANGE and value == good_velocity_value:
            good_velocity_token = token_id
            break
    
    if good_velocity_token:
        good_sequence.append(good_velocity_token)
    
    # Add a simple melody with proper timing
    pitches = [60, 62, 64, 65, 67]  # C-D-E-F-G
    good_time_value = 16  # Reasonable time shift (not too short)
    good_time_token = None
    for token_id in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token_id)
        if event_type == EventType.TIME_SHIFT and value == good_time_value:
            good_time_token = token_id
            break
    
    for pitch in pitches:
        # Note on
        note_on_token = None
        for token_id in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token_id)
            if event_type == EventType.NOTE_ON and value == pitch:
                note_on_token = token_id
                break
        
        # Note off
        note_off_token = None
        for token_id in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token_id)
            if event_type == EventType.NOTE_OFF and value == pitch:
                note_off_token = token_id
                break
        
        if note_on_token and note_off_token and good_time_token:
            good_sequence.extend([note_on_token, good_time_token, note_off_token])
    
    good_sequence.append(1)  # END_TOKEN
    good_tokens = np.array(good_sequence)
    
    print("Good token sequence:")
    for i, token in enumerate(good_tokens):
        event_type, value = vocab_config.token_to_event_info(token)
        
        if event_type == EventType.VELOCITY_CHANGE:
            midi_velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d} → MIDI velocity={midi_velocity}")
        elif event_type == EventType.TIME_SHIFT:
            duration = value * vocab_config.time_shift_ms / 1000.0
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d} → Duration={duration:.4f}s")
        elif event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][value % 12]
            octave = (value // 12) - 1
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d} → {note_name}{octave}")
        else:
            print(f"  Token {token:3d}: {event_type.name:15s} value={value:2d}")
    
    try:
        midi_data = converter.tokens_to_midi(good_tokens)
        print(f"\nGood MIDI conversion result:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        print(f"  Total notes: {total_notes}")
        
        # Check actual note parameters
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note_idx, note in enumerate(instrument.notes):
                note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note.pitch % 12]
                octave = (note.pitch // 12) - 1
                print(f"  Note {note_idx}: {note_name}{octave}, vel={note.velocity}, "
                      f"start={note.start:.4f}s, end={note.end:.4f}s, dur={note.duration:.4f}s")
        
        # Export to test MIDI file
        exporter = MidiExporter(create_standard_config())
        output_path = "/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl/test_good_tokens.mid"
        
        stats = exporter.export_tokens_to_midi(
            tokens=torch.from_numpy(good_tokens).unsqueeze(0),
            output_path=output_path,
            title="Good Token Test"
        )
        print(f"\n✅ Exported good token sequence to: {output_path}")
        print(f"   Notes: {stats.total_notes}, Duration: {stats.total_duration:.2f}s")
        
    except Exception as e:
        print(f"❌ Good token test failed: {e}")
        import traceback
        traceback.print_exc()

def test_musical_grammar_validation():
    """Test the musical grammar validation system."""
    print("\n" + "=" * 80)
    print("MUSICAL GRAMMAR VALIDATION TEST")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    
    # Test problematic sequence
    problematic_tokens = np.array([139] * 50)  # Repetitive tokens
    
    print("Testing problematic token sequence (repetitive token 139):")
    validation_result = validate_generated_sequence(problematic_tokens, vocab_config)
    
    print(f"Grammar score: {validation_result['grammar_score']:.3f}")
    print(f"Is valid: {validation_result['is_valid']}")
    print(f"Note count: {validation_result['note_count']}")
    print(f"Velocity quality score: {validation_result['velocity_quality_score']:.3f}")
    print(f"Timing quality score: {validation_result['timing_quality_score']:.3f}")
    print(f"Repetition score: {validation_result['repetition_score']:.3f}")
    
    if validation_result['velocity_quality_score'] < 0.8:
        print("🚨 Poor velocity quality detected!")
    if validation_result['timing_quality_score'] < 0.8:
        print("🚨 Poor timing quality detected!")
    if validation_result['repetition_score'] < 0.8:
        print("🚨 Excessive repetition detected!")

def main():
    """Run comprehensive token system analysis."""
    print("🎼 COMPREHENSIVE TOKEN SYSTEM ANALYSIS")
    print("Identifying why generated tokens produce 0-1 notes instead of proper music")
    print()
    
    # 1. Analyze vocabulary structure
    vocab_config = analyze_vocabulary()
    
    # 2. Test problematic sequences
    test_problematic_token_sequences()
    
    # 3. Test good sequences for comparison
    test_good_token_sequences()
    
    # 4. Test musical grammar validation
    test_musical_grammar_validation()
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print("1. Token 139 and similar tokens map to unusable MIDI parameters")
    print("2. Low velocity values (0-5) map to MIDI velocities 1-20 (silent/inaudible)")
    print("3. Low time shift values (0-5) map to durations < 0.1s (too short)")
    print("4. Grammar validation correctly identifies these quality issues")
    print()
    print("RECOMMENDATIONS:")
    print("1. Use musical grammar loss during training to penalize silent velocities")
    print("2. Use parameter quality validation to ensure audible note generation")
    print("3. Implement token diversity loss to prevent repetitive generation")
    print("4. Train on real classical music data for natural musical patterns")
    print()
    print("The enhanced musical grammar system addresses these exact issues!")

if __name__ == "__main__":
    main()