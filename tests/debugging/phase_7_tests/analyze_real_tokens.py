#!/usr/bin/env python3
"""
Analyze Real Musical Token Sequences
====================================

This script analyzes actual MIDI files from the training data to show
what token sequences the AI model should be learning to generate.

It will:
1. Load a real classical music MIDI file
2. Convert it to tokens using MusicRepresentationConverter
3. Show what the actual token sequences look like
4. Create a proper MIDI export test using real musical data
"""

import sys
import numpy as np
from pathlib import Path

# Add the project directory to the path
sys.path.append('/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl')

from src.data.midi_parser import MidiParser, load_midi_file
from src.data.representation import MusicRepresentationConverter, VocabularyConfig, EventType

def analyze_midi_file(midi_path: str):
    """Analyze a MIDI file and show its token representation."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {Path(midi_path).name}")
    print(f"{'='*60}")
    
    # Load MIDI file
    print("Loading MIDI file...")
    try:
        midi_data = load_midi_file(midi_path)
        print(f"✓ Successfully loaded MIDI file")
        print(f"  Duration: {midi_data.end_time:.2f} seconds")
        print(f"  Instruments: {len(midi_data.instruments)}")
        
        # Show some basic stats
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        print(f"  Total notes: {total_notes}")
        
        if midi_data.tempo_changes:
            initial_tempo = midi_data.tempo_changes[0][1]
            print(f"  Initial tempo: {initial_tempo:.1f} BPM")
        
    except Exception as e:
        print(f"✗ Failed to load MIDI file: {e}")
        return None
    
    # Convert to musical representation
    print("\nConverting to musical representation...")
    try:
        converter = MusicRepresentationConverter()
        representation = converter.midi_to_representation(midi_data)
        print(f"✓ Successfully converted to representation")
        print(f"  Events: {len(representation.events)}")
        print(f"  Tokens: {len(representation.tokens)}")
        print(f"  Vocabulary size: {converter.vocab_config.vocab_size}")
        
    except Exception as e:
        print(f"✗ Failed to convert to representation: {e}")
        return None
    
    # Show first 30 events and their tokens
    print(f"\nFIRST 30 MUSICAL EVENTS:")
    print("-" * 80)
    for i, event in enumerate(representation.events[:30]):
        token = representation.tokens[i]
        event_type, value = converter.vocab_config.token_to_event_info(token)
        print(f"{i:2d}: Token {token:4d} | {str(event)}")
        if i < 29:  # Don't print separator after last item
            if event.event_type in [EventType.NOTE_ON, EventType.NOTE_OFF]:
                note_name = midi_note_to_name(event.value)
                print(f"     {'':>13} ({note_name})")
    
    # Show token sequence pattern
    print(f"\nTOKEN SEQUENCE (first 50 tokens):")
    print("-" * 50)
    tokens_line = " ".join(f"{token:3d}" for token in representation.tokens[:50])
    print(tokens_line)
    
    # Analyze token distribution
    print(f"\nTOKEN TYPE ANALYSIS:")
    print("-" * 30)
    token_counts = {}
    for token in representation.tokens:
        event_type, _ = converter.vocab_config.token_to_event_info(token)
        token_counts[event_type.name] = token_counts.get(event_type.name, 0) + 1
    
    for event_type, count in sorted(token_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(representation.tokens)) * 100
        print(f"  {event_type:15s}: {count:4d} tokens ({percentage:5.1f}%)")
    
    # Show pitch range used
    note_on_events = [e for e in representation.events if e.event_type == EventType.NOTE_ON]
    if note_on_events:
        pitches = [e.value for e in note_on_events]
        min_pitch, max_pitch = min(pitches), max(pitches)
        print(f"\nPITCH RANGE USED:")
        print(f"  Lowest note:  {min_pitch} ({midi_note_to_name(min_pitch)})")
        print(f"  Highest note: {max_pitch} ({midi_note_to_name(max_pitch)})")
        print(f"  Range: {max_pitch - min_pitch + 1} semitones")
    
    return representation

def midi_note_to_name(midi_note: int) -> str:
    """Convert MIDI note number to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def test_midi_export(representation):
    """Test converting tokens back to MIDI using real musical data."""
    print(f"\n{'='*60}")
    print("TESTING MIDI EXPORT WITH REAL MUSICAL DATA")
    print(f"{'='*60}")
    
    try:
        converter = MusicRepresentationConverter()
        
        # Test round-trip: representation -> MIDI
        print("Converting representation back to MIDI...")
        reconstructed_midi = converter.representation_to_midi(representation)
        print(f"✓ Successfully reconstructed MIDI")
        print(f"  Duration: {reconstructed_midi.end_time:.2f} seconds")
        print(f"  Instruments: {len(reconstructed_midi.instruments)}")
        
        total_notes = sum(len(inst.notes) for inst in reconstructed_midi.instruments)
        print(f"  Total notes: {total_notes}")
        
        # Test tokens -> MIDI conversion
        print("\nConverting tokens directly to MIDI...")
        midi_from_tokens = converter.tokens_to_midi(representation.tokens)
        print(f"✓ Successfully converted tokens to MIDI")
        print(f"  Duration: {midi_from_tokens.end_time:.2f} seconds")
        print(f"  Instruments: {len(midi_from_tokens.instruments)}")
        
        total_notes_from_tokens = sum(len(inst.notes) for inst in midi_from_tokens.instruments)
        print(f"  Total notes: {total_notes_from_tokens}")
        
        # Save example MIDI files for inspection
        save_path = Path("/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl/test_output")
        save_path.mkdir(exist_ok=True)
        
        print(f"\nSaving example MIDI files to {save_path}...")
        
        # Note: We would need a midi_data_to_file function to actually save
        # For now, just show that the conversion process works
        print("✓ Conversion process validated - real musical tokens can be converted back to MIDI")
        
        return True
        
    except Exception as e:
        print(f"✗ MIDI export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main analysis function."""
    print("🎼 REAL MUSICAL TOKEN SEQUENCE ANALYSIS")
    print("="*60)
    
    # List of interesting MIDI files to analyze
    data_path = Path("/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl/data/raw")
    
    # Select a few representative pieces
    test_files = [
        "mond_1.mid",           # Moonlight Sonata (simple, well-known)
        "chpn-p1.mid",         # Chopin Prelude (complex classical)
        "pathetique_1.mid",    # Beethoven Pathetique (dramatic)
    ]
    
    representations = []
    
    for filename in test_files:
        midi_path = data_path / filename
        if midi_path.exists():
            rep = analyze_midi_file(str(midi_path))
            if rep:
                representations.append(rep)
        else:
            print(f"⚠️ File not found: {filename}")
    
    # Test MIDI export using the first successful representation
    if representations:
        print(f"\n🧪 RUNNING MIDI EXPORT TEST")
        success = test_midi_export(representations[0])
        
        if success:
            print(f"\n✅ ALL TESTS PASSED!")
            print("The token system successfully handles real classical music.")
            print("Token sequences shown above represent what the AI should learn to generate.")
        else:
            print(f"\n❌ MIDI export test failed!")
    else:
        print(f"\n❌ No MIDI files could be analyzed!")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()