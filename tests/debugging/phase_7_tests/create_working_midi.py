#!/usr/bin/env python3
"""
Create a working MIDI file by constructing proper token sequences
"""

import torch
import numpy as np
import os
from src.data.representation import MusicRepresentationConverter, VocabularyConfig, EventType
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Creating Working MIDI File ===\n")

converter = MusicRepresentationConverter()
vocab_config = VocabularyConfig()

# Find the correct token mappings for a simple melody
print("1. Finding Correct Token Mappings:")

# We need to find tokens that create proper note on/off pairs
# Let's find NOTE_ON and NOTE_OFF tokens for middle C (MIDI note 60)

note_on_60 = None
note_off_60 = None
velocity_token = None
time_shift_token = None

# Search through the vocabulary
for token in range(vocab_config.vocab_size):
    event_type, value = vocab_config.token_to_event_info(token)
    
    if event_type == EventType.NOTE_ON and value == 60:
        note_on_60 = token
    elif event_type == EventType.NOTE_OFF and value == 60:
        note_off_60 = token
    elif event_type == EventType.VELOCITY_CHANGE and velocity_token is None:
        velocity_token = token  # Take first velocity token
    elif event_type == EventType.TIME_SHIFT and time_shift_token is None:
        time_shift_token = token  # Take first time shift token

print(f"NOTE_ON for MIDI 60: token {note_on_60}")
print(f"NOTE_OFF for MIDI 60: token {note_off_60}")
print(f"Velocity token: {velocity_token}")
print(f"Time shift token: {time_shift_token}")

if note_on_60 is None or note_off_60 is None:
    print("❌ Cannot find matching note on/off tokens!")
    exit(1)

print("\n2. Creating Proper Token Sequence:")
# Create a simple melody with correct note on/off pairs
working_sequence = [
    1,  # START
    velocity_token,  # Set velocity
    note_on_60,      # Note ON middle C
    time_shift_token,  # Short time shift
    note_off_60,     # Note OFF middle C (matching pitch!)
    2   # END
]

print(f"Working sequence: {working_sequence}")

# Verify the sequence
print("\n3. Verifying Token Sequence:")
for i, token in enumerate(working_sequence):
    event_type, value = vocab_config.token_to_event_info(token)
    print(f"  Token {token}: {event_type.name} = {value}")

print("\n4. Converting to MIDI:")
os.makedirs("working_output", exist_ok=True)

try:
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    tokens_tensor = torch.from_numpy(np.array(working_sequence)).unsqueeze(0)
    
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path="working_output/corrected_sequence.mid",
        title="Corrected Sequence",
        tempo=120.0
    )
    
    print(f"Result: {stats.total_notes} notes, {stats.total_duration:.2f}s")
    
    if stats.total_notes > 0:
        print("✅ SUCCESS! Created working MIDI file!")
        print("📁 File: working_output/corrected_sequence.mid")
        print("🎵 This file should open properly in Finale!")
    else:
        print("❌ Still failed - deeper issue in conversion")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Creating a Simple Melody:")
# Let's create a more interesting sequence with multiple notes
try:
    # Find tokens for a simple scale: C, D, E, F
    notes = [60, 62, 64, 65]  # C, D, E, F
    note_tokens = {}
    
    for note in notes:
        note_on = None
        note_off = None
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.NOTE_ON and value == note:
                note_on = token
            elif event_type == EventType.NOTE_OFF and value == note:
                note_off = token
        
        if note_on and note_off:
            note_tokens[note] = (note_on, note_off)
    
    if len(note_tokens) >= 4:
        # Create a simple melody
        melody_sequence = [1, velocity_token]  # START + velocity
        
        for note in notes:
            if note in note_tokens:
                note_on, note_off = note_tokens[note]
                melody_sequence.extend([
                    note_on,         # Note ON
                    time_shift_token, # Time shift
                    note_off         # Note OFF
                ])
        
        melody_sequence.append(2)  # END
        
        print(f"Melody sequence: {len(melody_sequence)} tokens")
        
        melody_tensor = torch.from_numpy(np.array(melody_sequence)).unsqueeze(0)
        
        stats = exporter.export_tokens_to_midi(
            tokens=melody_tensor,
            output_path="working_output/simple_melody.mid",
            title="Simple Melody - C D E F",
            tempo=120.0
        )
        
        print(f"Melody result: {stats.total_notes} notes, {stats.total_duration:.2f}s")
        
        if stats.total_notes > 0:
            print("✅ Created working melody!")
            print("📁 File: working_output/simple_melody.mid")
        
except Exception as e:
    print(f"Melody creation error: {e}")

print(f"\n=== Summary ===")
print("The problem was that generated tokens don't form proper note on/off pairs.")
print("The model generates random tokens that don't respect musical grammar.")
print()
print("Files created:")
print("- working_output/corrected_sequence.mid (single note)")
print("- working_output/simple_melody.mid (4-note melody)")
print()
print("These should open properly in Finale and play actual notes!")