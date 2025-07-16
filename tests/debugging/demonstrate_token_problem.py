#!/usr/bin/env python3
"""
Token Problem Demonstration

This script creates side-by-side examples of:
1. BAD token sequences (what the model currently generates)
2. GOOD token sequences (what it should generate)
3. The exact MIDI conversion showing why bad tokens fail
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project path
sys.path.append('/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl')

from src.data.representation import VocabularyConfig, EventType, MusicRepresentationConverter
from src.generation.midi_export import MidiExporter, create_standard_config
from src.training.musical_grammar import validate_generated_sequence

def demonstrate_exact_problem():
    """Show the exact tokens the model generates and why they fail."""
    
    print("🎼 EXACT TOKEN PROBLEM DEMONSTRATION")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    exporter = MidiExporter(create_standard_config())
    
    # ========================================================================
    # BAD EXAMPLE 1: Token 139 Repetition (Model Collapse)
    # ========================================================================
    print("\n🚨 BAD EXAMPLE 1: Model Collapse (Real Generation Issue)")
    print("-" * 60)
    
    bad_tokens_1 = np.array([139] * 50)  # Actual problematic generation
    
    print("Generated token sequence:")
    print(f"  Tokens: [139, 139, 139, ...] (repeated 50 times)")
    
    # Show what token 139 means
    event_type, value = vocab_config.token_to_event_info(139)
    print(f"  Token 139 = {event_type.name} with value {value}")
    
    if event_type == EventType.NOTE_OFF:
        note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][value % 12]
        octave = (value // 12) - 1
        print(f"  Meaning: NOTE_OFF for {note_name}{octave} (MIDI note {value})")
    
    # Convert to MIDI
    try:
        midi_data = converter.tokens_to_midi(bad_tokens_1)
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        
        print(f"\nMIDI Conversion Result:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        print(f"  Total notes: {total_notes}")
        print(f"  Problem: {total_notes} notes from 50 tokens!")
        print(f"  Reason: All NOTE_OFF events, no NOTE_ON events to create notes")
        
    except Exception as e:
        print(f"  Conversion failed: {e}")
    
    # ========================================================================
    # BAD EXAMPLE 2: Silent Velocity + Poor Timing (Parameter Quality Issue)
    # ========================================================================
    print("\n🚨 BAD EXAMPLE 2: Silent Velocity + Poor Timing")
    print("-" * 60)
    
    # Create sequence with silent velocity
    bad_tokens_2 = []
    bad_tokens_2.append(0)   # START_TOKEN
    bad_tokens_2.append(692) # VELOCITY_CHANGE value=0 (maps to MIDI velocity=1)
    bad_tokens_2.append(43)  # NOTE_ON C4 (value=60)
    bad_tokens_2.append(180) # TIME_SHIFT value=0 (maps to 0.0s duration)  
    bad_tokens_2.append(131) # NOTE_OFF C4 (value=60)
    bad_tokens_2.append(1)   # END_TOKEN
    bad_tokens_2 = np.array(bad_tokens_2)
    
    print("Generated token sequence:")
    for i, token in enumerate(bad_tokens_2):
        event_type, value = vocab_config.token_to_event_info(token)
        
        if event_type == EventType.VELOCITY_CHANGE:
            midi_velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            print(f"  Token {token:3d}: {event_type.name} value={value} → MIDI velocity={midi_velocity} 🚨 SILENT!")
        elif event_type == EventType.TIME_SHIFT:
            duration = value * vocab_config.time_shift_ms / 1000.0
            print(f"  Token {token:3d}: {event_type.name} value={value} → Duration={duration:.4f}s 🚨 ZERO!")
        elif event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][value % 12]
            octave = (value // 12) - 1
            print(f"  Token {token:3d}: {event_type.name} value={value} → {note_name}{octave}")
        else:
            print(f"  Token {token:3d}: {event_type.name}")
    
    # Convert to MIDI
    try:
        midi_data = converter.tokens_to_midi(bad_tokens_2)
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        
        print(f"\nMIDI Conversion Result:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        print(f"  Total notes: {total_notes}")
        
        # Check note properties
        for inst in midi_data.instruments:
            for note in inst.notes:
                print(f"  Note: pitch={note.pitch}, velocity={note.velocity}, duration={note.duration:.4f}s")
                print(f"  Problem: velocity={note.velocity} (inaudible), duration={note.duration:.4f}s (invisible)")
        
    except Exception as e:
        print(f"  Conversion failed: {e}")
    
    # ========================================================================
    # GOOD EXAMPLE: Proper Token Sequence
    # ========================================================================
    print("\n✅ GOOD EXAMPLE: Proper Musical Sequence")
    print("-" * 60)
    
    # Create proper sequence
    good_tokens = []
    good_tokens.append(0)   # START_TOKEN
    good_tokens.append(708) # VELOCITY_CHANGE value=16 (maps to MIDI velocity=66)
    
    # Simple melody: C-D-E-F-G
    pitches = [60, 62, 64, 65, 67]
    
    for pitch in pitches:
        # Find correct tokens for this pitch
        note_on_token = 4 + (pitch - 21)   # NOTE_ON token
        note_off_token = 92 + (pitch - 21) # NOTE_OFF token
        time_shift_token = 196  # TIME_SHIFT value=16 (maps to 0.25s)
        
        good_tokens.extend([note_on_token, time_shift_token, note_off_token])
    
    good_tokens.append(1)   # END_TOKEN
    good_tokens = np.array(good_tokens)
    
    print("Generated token sequence:")
    for i, token in enumerate(good_tokens):
        event_type, value = vocab_config.token_to_event_info(token)
        
        if event_type == EventType.VELOCITY_CHANGE:
            midi_velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            print(f"  Token {token:3d}: {event_type.name} value={value} → MIDI velocity={midi_velocity} ✅ AUDIBLE")
        elif event_type == EventType.TIME_SHIFT:
            duration = value * vocab_config.time_shift_ms / 1000.0
            print(f"  Token {token:3d}: {event_type.name} value={value} → Duration={duration:.4f}s ✅ VISIBLE")
        elif event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
            note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][value % 12]
            octave = (value // 12) - 1
            print(f"  Token {token:3d}: {event_type.name} value={value} → {note_name}{octave}")
        else:
            print(f"  Token {token:3d}: {event_type.name}")
    
    # Convert to MIDI
    try:
        midi_data = converter.tokens_to_midi(good_tokens)
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        
        print(f"\nMIDI Conversion Result:")
        print(f"  Duration: {midi_data.end_time:.4f}s")
        print(f"  Total notes: {total_notes} ✅ PROPER MUSIC!")
        
        # Check note properties
        for inst in midi_data.instruments:
            for i, note in enumerate(inst.notes):
                note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][note.pitch % 12]
                octave = (note.pitch // 12) - 1
                print(f"  Note {i+1}: {note_name}{octave}, vel={note.velocity}, dur={note.duration:.3f}s ✅")
        
        # Export to demonstrate
        output_path = "/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl/good_example.mid"
        stats = exporter.export_tokens_to_midi(
            tokens=torch.from_numpy(good_tokens).unsqueeze(0),
            output_path=output_path,
            title="Good Token Example"
        )
        print(f"\n  ✅ Exported to: {output_path}")
        
    except Exception as e:
        print(f"  Conversion failed: {e}")

def show_grammar_validation_comparison():
    """Show how grammar validation scores the different sequences."""
    
    print("\n" + "=" * 80)
    print("GRAMMAR VALIDATION COMPARISON")
    print("=" * 80)
    
    vocab_config = VocabularyConfig()
    
    # Test sequences
    sequences = {
        "Model Collapse (Token 139)": np.array([139] * 50),
        "Silent Velocity": np.array([0, 692, 43, 180, 131, 1]),
        "Good Melody": np.array([0, 708, 43, 196, 131, 45, 196, 133, 47, 196, 135, 48, 196, 136, 50, 196, 138, 1])
    }
    
    for name, tokens in sequences.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        validation = validate_generated_sequence(tokens, vocab_config)
        
        print(f"  Overall Grammar Score: {validation['grammar_score']:.3f}")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Note Count: {validation['note_count']}")
        print(f"  Note Pairing Score: {validation['note_pairing_score']:.3f}")
        print(f"  Velocity Quality: {validation['velocity_quality_score']:.3f}")
        print(f"  Timing Quality: {validation['timing_quality_score']:.3f}")
        print(f"  Repetition Score: {validation['repetition_score']:.3f}")
        
        # Identify specific problems
        if validation['velocity_quality_score'] < 0.8:
            print(f"  🚨 Poor velocity quality!")
        if validation['timing_quality_score'] < 0.8:
            print(f"  🚨 Poor timing quality!")
        if validation['repetition_score'] < 0.8:
            print(f"  🚨 Excessive repetition!")
        if validation['note_count'] == 0:
            print(f"  🚨 No notes generated!")

def main():
    """Demonstrate the exact token problems and solutions."""
    
    demonstrate_exact_problem()
    show_grammar_validation_comparison()
    
    print("\n" + "=" * 80)
    print("SUMMARY: WHY GENERATED TOKENS PRODUCE 0-1 NOTES")
    print("=" * 80)
    print()
    print("🔍 ROOT CAUSES IDENTIFIED:")
    print()
    print("1. MODEL COLLAPSE:")
    print("   - Generates repetitive tokens (e.g., token 139 = NOTE_OFF)")
    print("   - No matching NOTE_ON events → 0 notes")
    print("   - Grammar score: ~0.27 (correctly identifies as bad)")
    print()
    print("2. SILENT VELOCITY SELECTION:")
    print("   - Tokens 692-696 map to MIDI velocities 1-20 (inaudible)")
    print("   - Notes exist but can't be heard")
    print("   - Grammar score: ~0.50 (detects velocity problems)")
    print()
    print("3. ZERO DURATION SELECTION:")
    print("   - Token 180 maps to 0.0s duration (invisible)")
    print("   - Notes exist but disappear instantly")
    print("   - Grammar score: ~0.50 (detects timing problems)")
    print()
    print("🎯 THE FIX:")
    print()
    print("Enhanced Musical Grammar Loss Functions (already implemented):")
    print("  ✅ Heavily penalize silent velocity tokens (vel ≤ 20)")
    print("  ✅ Heavily penalize invisible timing tokens (dur ≤ 10ms)")
    print("  ✅ Prevent note pairing mismatches")
    print("  ✅ Encourage token diversity")
    print()
    print("Target: Grammar scores > 0.85 for usable music")
    print("Current: Grammar scores 0.65-0.70 still produce unusable output")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Retrain model with enhanced grammar loss")
    print("2. Validate grammar scores > 0.85 during generation")
    print("3. Use rejection sampling to filter bad sequences")

if __name__ == "__main__":
    main()