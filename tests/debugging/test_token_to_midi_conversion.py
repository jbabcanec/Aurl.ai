"""
Test script to debug token-to-MIDI conversion issues.
This script will:
1. Load a real MIDI file and convert it to tokens
2. Show the token sequence with meanings
3. Convert tokens back to MIDI
4. Compare the original and reconstructed MIDI
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.data.midi_parser import load_midi_file
from src.data.representation import (
    MusicRepresentationConverter, VocabularyConfig, EventType
)


def analyze_token_vocabulary():
    """Analyze the token vocabulary structure."""
    print("=== Token Vocabulary Analysis ===")
    vocab_config = VocabularyConfig()
    print(f"Total vocabulary size: {vocab_config.vocab_size}")
    
    # Count tokens by type
    token_counts = {}
    for token in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token)
        event_name = event_type.name
        if event_name not in token_counts:
            token_counts[event_name] = 0
        token_counts[event_name] += 1
    
    print("\nTokens by event type:")
    for event_name, count in sorted(token_counts.items()):
        print(f"  {event_name:20s}: {count:5d} tokens")
    
    # Show some example tokens
    print("\nExample tokens:")
    examples = [
        ("START_TOKEN", EventType.START_TOKEN, 0),
        ("NOTE_ON C4", EventType.NOTE_ON, 60),
        ("NOTE_OFF C4", EventType.NOTE_OFF, 60),
        ("TIME_SHIFT 15.625ms", EventType.TIME_SHIFT, 1),
        ("VELOCITY_CHANGE mid", EventType.VELOCITY_CHANGE, 16),
    ]
    
    for desc, event_type, value in examples:
        # Find the token for this event
        for token in range(vocab_config.vocab_size):
            token_event_type, token_value = vocab_config.token_to_event_info(token)
            if token_event_type == event_type and token_value == value:
                print(f"  Token {token:4d} = {desc}")
                break


def test_midi_to_tokens_and_back():
    """Test converting MIDI to tokens and back."""
    print("\n=== MIDI to Tokens to MIDI Test ===")
    
    # Load a simple MIDI file
    midi_file = Path("data/raw/chpn-p18.mid")  # Chopin prelude - relatively simple
    if not midi_file.exists():
        print(f"Error: {midi_file} not found")
        return
    
    print(f"Loading MIDI file: {midi_file}")
    original_midi = load_midi_file(midi_file)
    
    # Show original MIDI stats
    print(f"\nOriginal MIDI:")
    print(f"  Duration: {original_midi.end_time:.2f}s")
    print(f"  Instruments: {len(original_midi.instruments)}")
    total_notes = sum(len(inst.notes) for inst in original_midi.instruments)
    print(f"  Total notes: {total_notes}")
    
    # Convert to representation
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    representation = converter.midi_to_representation(original_midi)
    
    print(f"\nRepresentation:")
    print(f"  Events: {len(representation.events)}")
    print(f"  Tokens: {len(representation.tokens)}")
    
    # Show first 30 events
    print("\nFirst 30 events:")
    for i, event in enumerate(representation.events[:30]):
        print(f"  {i:3d}: {event}")
    
    # Show first 30 tokens with meanings
    print("\nFirst 30 tokens:")
    for i in range(min(30, len(representation.tokens))):
        token = int(representation.tokens[i])
        event_type, value = vocab_config.token_to_event_info(token)
        
        # Create human-readable description
        if event_type == EventType.NOTE_ON or event_type == EventType.NOTE_OFF:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = (value // 12) - 1
            note = note_names[value % 12]
            desc = f"{note}{octave}"
        elif event_type == EventType.TIME_SHIFT:
            ms = value * vocab_config.time_shift_ms
            desc = f"{ms:.1f}ms"
        elif event_type == EventType.VELOCITY_CHANGE:
            velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            desc = f"vel={velocity}"
        else:
            desc = f"val={value}"
        
        print(f"  {i:3d}: Token {token:4d} = {event_type.name:18s} {desc}")
    
    # Convert back to MIDI
    print("\nConverting tokens back to MIDI...")
    reconstructed_midi = converter.tokens_to_midi(representation.tokens)
    
    print(f"\nReconstructed MIDI:")
    print(f"  Duration: {reconstructed_midi.end_time:.2f}s")
    print(f"  Instruments: {len(reconstructed_midi.instruments)}")
    total_notes_recon = sum(len(inst.notes) for inst in reconstructed_midi.instruments)
    print(f"  Total notes: {total_notes_recon}")
    
    # Save reconstructed MIDI for inspection
    output_file = Path("tests/debugging/reconstructed_test.mid")
    output_file.parent.mkdir(exist_ok=True)
    # save_midi_file(reconstructed_midi, output_file)
    print(f"\n(MIDI save function not available, skipping file save)")
    
    # Compare notes
    if total_notes_recon == 0:
        print("\n❌ ERROR: No notes in reconstructed MIDI!")
        # Debug the conversion process
        debug_token_to_midi_conversion(representation.tokens, vocab_config, converter)
    else:
        print(f"\n✅ Successfully reconstructed {total_notes_recon}/{total_notes} notes")


def debug_token_to_midi_conversion(tokens, vocab_config, converter):
    """Debug why tokens aren't producing notes."""
    print("\n=== Debugging Token to MIDI Conversion ===")
    
    # Track what's happening during conversion
    print("\nStep-by-step token processing:")
    
    current_time = 0.0
    current_velocity = 64
    current_channel = 0
    active_notes = {}
    notes_created = []
    
    for i, token in enumerate(tokens[:50]):  # First 50 tokens
        event_type, value = vocab_config.token_to_event_info(int(token))
        
        print(f"\nToken {i}: {token} -> {event_type.name}(value={value})")
        
        if event_type == EventType.TIME_SHIFT:
            time_delta = value * vocab_config.time_shift_ms / 1000.0
            current_time += time_delta
            print(f"  Time advanced by {time_delta:.3f}s to {current_time:.3f}s")
            
        elif event_type == EventType.NOTE_ON:
            print(f"  NOTE ON: pitch={value}, time={current_time:.3f}, velocity={current_velocity}")
            active_notes[value] = (current_time, current_velocity)
            
        elif event_type == EventType.NOTE_OFF:
            if value in active_notes:
                start_time, velocity = active_notes[value]
                duration = current_time - start_time
                print(f"  NOTE OFF: pitch={value}, duration={duration:.3f}s")
                notes_created.append({
                    'pitch': value,
                    'start': start_time,
                    'end': current_time,
                    'velocity': velocity
                })
                del active_notes[value]
            else:
                print(f"  NOTE OFF: pitch={value} - WARNING: No matching NOTE_ON!")
                
        elif event_type == EventType.VELOCITY_CHANGE:
            current_velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            print(f"  Velocity changed to {current_velocity}")
            
        elif event_type == EventType.INSTRUMENT_CHANGE:
            current_channel = value
            print(f"  Instrument changed to {current_channel}")
    
    print(f"\n\nTotal notes created: {len(notes_created)}")
    print(f"Active notes not closed: {len(active_notes)}")
    
    if notes_created:
        print("\nFirst few notes created:")
        for note in notes_created[:5]:
            print(f"  {note}")


def create_simple_token_sequence():
    """Create a simple, meaningful token sequence for testing."""
    print("\n=== Creating Simple Test Sequence ===")
    
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    
    # Create a simple sequence: C major scale
    tokens = []
    
    # Start token
    tokens.append(0)  # START_TOKEN is typically token 0
    
    # Set velocity
    velocity_token = None
    for token in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token)
        if event_type == EventType.VELOCITY_CHANGE and value == 16:  # Mid velocity
            velocity_token = token
            break
    if velocity_token:
        tokens.append(velocity_token)
    
    # Play C major scale
    c_major_pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    
    for pitch in c_major_pitches:
        # Find NOTE_ON token for this pitch
        note_on_token = None
        note_off_token = None
        time_shift_token = None
        
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.NOTE_ON and value == pitch:
                note_on_token = token
            elif event_type == EventType.NOTE_OFF and value == pitch:
                note_off_token = token
            elif event_type == EventType.TIME_SHIFT and value == 32:  # ~500ms
                time_shift_token = token
        
        if note_on_token and note_off_token and time_shift_token:
            # Note on
            tokens.append(note_on_token)
            # Wait
            tokens.append(time_shift_token)
            # Note off
            tokens.append(note_off_token)
            # Small gap between notes
            tokens.append(time_shift_token)
    
    # End token
    tokens.append(1)  # END_TOKEN is typically token 1
    
    # Convert to numpy array
    token_array = np.array(tokens, dtype=np.int32)
    
    print(f"Created token sequence with {len(tokens)} tokens")
    print("Token sequence:", tokens[:20], "...")
    
    # Convert to MIDI
    print("\nConverting to MIDI...")
    midi_data = converter.tokens_to_midi(token_array)
    
    print(f"Result: {len(midi_data.instruments)} instruments")
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"Total notes: {total_notes}")
    
    if total_notes > 0:
        print("\nFirst instrument notes:")
        for note in midi_data.instruments[0].notes[:10]:
            print(f"  Pitch {note.pitch}, start={note.start:.3f}, end={note.end:.3f}")
    
    # Save test MIDI
    output_file = Path("tests/debugging/test_c_major_scale.mid")
    # save_midi_file(midi_data, output_file)
    print(f"\n(MIDI save function not available, skipping file save)")


if __name__ == "__main__":
    # Run all tests
    analyze_token_vocabulary()
    print("\n" + "="*60 + "\n")
    
    test_midi_to_tokens_and_back()
    print("\n" + "="*60 + "\n")
    
    create_simple_token_sequence()