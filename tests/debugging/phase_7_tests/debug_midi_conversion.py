#!/usr/bin/env python3
"""
Debug the token-to-MIDI conversion process to find why MIDI files are empty
"""

import torch
import numpy as np
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.generation.midi_export import MidiExporter, create_standard_config
from src.generation.token_validator import validate_and_fix_tokens

print("=== MIDI Conversion Debug ===\n")

# Load the generated tokens
tokens = torch.load('natural_piano_piece/aurl_natural_generation.tokens')
if tokens.dim() > 1:
    tokens = tokens.flatten()

tokens_np = tokens.numpy()
print(f"Original tokens: {tokens_np[:20]}...")
print(f"Total tokens: {len(tokens_np)}")

# Test 1: Check vocabulary mapping
print("\n1. Testing Vocabulary Mapping:")
vocab_config = VocabularyConfig()
converter = MusicRepresentationConverter()

print(f"Vocab size: {vocab_config.vocab_size}")

# Check specific token meanings
test_tokens = [1, 43, 77, 112, 215, 273, 738]
for token in test_tokens:
    if token < vocab_config.vocab_size:
        try:
            event_type, value = vocab_config.token_to_event_info(token)
            print(f"  Token {token}: {event_type} = {value}")
        except Exception as e:
            print(f"  Token {token}: ERROR - {e}")
    else:
        print(f"  Token {token}: OUT OF VOCAB RANGE!")

# Test 2: Token validation
print("\n2. Testing Token Validation:")
try:
    is_valid, corrected_tokens = validate_and_fix_tokens(tokens_np)
    print(f"  Original valid: {is_valid}")
    print(f"  Corrected tokens shape: {corrected_tokens.shape if corrected_tokens is not None else 'None'}")
    
    if corrected_tokens is not None:
        print(f"  Corrected tokens: {corrected_tokens[:20]}...")
        unique_corrected = len(np.unique(corrected_tokens))
        print(f"  Unique corrected tokens: {unique_corrected}")
    
except Exception as e:
    print(f"  Validation ERROR: {e}")
    corrected_tokens = tokens_np

# Test 3: Check if tokens represent actual musical events
print("\n3. Analyzing Musical Content:")
tokens_to_check = corrected_tokens if corrected_tokens is not None else tokens_np

# Count different event types
note_events = []
time_events = []
velocity_events = []

for token in tokens_to_check:
    if 13 <= token <= 140:  # Note ON range
        note_events.append(('NOTE_ON', token - 13))
    elif 141 <= token <= 268:  # Note OFF range  
        note_events.append(('NOTE_OFF', token - 141))
    elif 269 <= token <= 368:  # Time shift range
        time_events.append(token - 269)
    elif 369 <= token <= 496:  # Velocity range
        velocity_events.append(token - 369)

print(f"  Note events found: {len(note_events)}")
print(f"  Time events found: {len(time_events)}")
print(f"  Velocity events found: {len(velocity_events)}")

if note_events:
    print(f"  First few note events: {note_events[:10]}")
    
    # Check for note on/off pairs
    note_states = {}
    valid_notes = 0
    for event_type, pitch in note_events:
        if event_type == 'NOTE_ON':
            note_states[pitch] = True
        elif event_type == 'NOTE_OFF' and pitch in note_states:
            valid_notes += 1
            
    print(f"  Valid note on/off pairs: {valid_notes}")

# Test 4: Direct MIDI conversion attempt
print("\n4. Testing Direct MIDI Conversion:")
try:
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    # Try converting the tokens
    tokens_tensor = torch.from_numpy(tokens_to_check).unsqueeze(0)
    
    # Let's trace what happens inside the MIDI export
    print("  Converting tokens to events...")
    
    # This should reveal where the conversion fails
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path="debug_output.mid",
        title="Debug Test",
        tempo=120.0
    )
    
    print(f"  Export results: {stats.total_notes} notes, {stats.total_duration:.2f}s")
    if stats.warnings:
        print(f"  Warnings: {stats.warnings}")
        
except Exception as e:
    print(f"  MIDI Export ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Create a simple test sequence manually
print("\n5. Testing with Manual Simple Sequence:")
try:
    # Create a simple known-good sequence
    simple_sequence = np.array([
        1,    # START
        369,  # Set velocity (369 = velocity 0)
        13,   # Note ON C (13 = note 0 = C)
        269,  # Time shift (269 = 0 time units - minimum)  
        141,  # Note OFF C (141 = note 0 = C)
        2     # END
    ])
    
    print(f"  Manual sequence: {simple_sequence}")
    
    simple_tensor = torch.from_numpy(simple_sequence).unsqueeze(0)
    stats = exporter.export_tokens_to_midi(
        tokens=simple_tensor,
        output_path="simple_test.mid", 
        title="Simple Test",
        tempo=120.0
    )
    
    print(f"  Simple test results: {stats.total_notes} notes")
    
except Exception as e:
    print(f"  Simple test ERROR: {e}")

print(f"\n=== Diagnosis ===")
print("If the manual simple sequence also produces 0 notes,")
print("then the issue is in the token-to-event conversion logic.")
print("If it works, then the generated tokens don't form valid musical sequences.")