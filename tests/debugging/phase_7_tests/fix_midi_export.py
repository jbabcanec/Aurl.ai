#!/usr/bin/env python3
"""
Fix the MIDI export issues: velocity too low, timing problems
"""

import torch
import numpy as np
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Fixing MIDI Export Issues ===\n")

# Let's fix the problems we identified:
# 1. Velocity = 1 (too quiet) should be 64+
# 2. NOTE_OFF at same time as NOTE_ON (zero duration)

converter = MusicRepresentationConverter()
vocab_config = VocabularyConfig()

print("1. Finding better tokens for proper velocity and timing:")

# Find a velocity token that gives reasonable velocity (not 1)
velocity_token = None
for token in range(vocab_config.vocab_size):
    event_type, value = vocab_config.token_to_event_info(token)
    if event_type.name == 'VELOCITY_CHANGE':
        # Check what velocity this produces
        # From the code: velocity = int((value / (velocity_bins - 1)) * 126) + 1
        velocity_bins = 32  # From VocabularyConfig
        actual_velocity = int((value / (velocity_bins - 1)) * 126) + 1
        if 64 <= actual_velocity <= 100:  # Good range
            velocity_token = token
            print(f"  Found good velocity token {token}: value={value} -> velocity={actual_velocity}")
            break

# Find a time shift token that gives reasonable duration
time_token = None
for token in range(vocab_config.vocab_size):
    event_type, value = vocab_config.token_to_event_info(token)
    if event_type.name == 'TIME_SHIFT':
        # Check what time this produces
        # From the code: time_delta = value * time_shift_ms / 1000.0
        time_shift_ms = 15.625  # From VocabularyConfig
        time_delta = value * time_shift_ms / 1000.0
        if 0.2 <= time_delta <= 1.0:  # Good duration
            time_token = token
            print(f"  Found good time token {token}: value={value} -> duration={time_delta:.3f}s")
            break

if velocity_token is None or time_token is None:
    print("❌ Could not find good velocity/time tokens")
    # Use defaults but they might not work well
    velocity_token = 692
    time_token = 180

print(f"\n2. Creating corrected token sequence:")
corrected_tokens = [
    1,              # START
    velocity_token, # Good velocity
    43,             # NOTE_ON MIDI 60
    time_token,     # Good time shift
    131,            # NOTE_OFF MIDI 60
    2               # END
]

print(f"Corrected tokens: {corrected_tokens}")

# Test the corrected sequence
print(f"\n3. Testing corrected sequence:")
try:
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    tokens_tensor = torch.from_numpy(np.array(corrected_tokens)).unsqueeze(0)
    
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path="finale_test/aurl_corrected.mid",
        title="Aurl Corrected",
        tempo=120.0
    )
    
    print(f"✅ Created: finale_test/aurl_corrected.mid")
    print(f"   Notes: {stats.total_notes}")
    print(f"   Duration: {stats.total_duration:.2f}s")
    
    # Analyze what we created
    import pretty_midi
    pm = pretty_midi.PrettyMIDI("finale_test/aurl_corrected.mid")
    if pm.instruments and pm.instruments[0].notes:
        note = pm.instruments[0].notes[0]
        print(f"   Note details: pitch={note.pitch}, start={note.start:.3f}, end={note.end:.3f}, vel={note.velocity}")
        
        if note.velocity >= 64:
            print("   ✅ Velocity is good")
        else:
            print("   ❌ Velocity still too low")
            
        if (note.end - note.start) >= 0.2:
            print("   ✅ Duration is good")
        else:
            print("   ❌ Duration still too short")
    
except Exception as e:
    print(f"❌ Corrected test failed: {e}")
    import traceback
    traceback.print_exc()

# Let's also create a sequence with multiple notes
print(f"\n4. Creating multi-note sequence:")
try:
    # Find tokens for different pitches
    notes = [60, 62, 64, 65]  # C, D, E, F
    note_tokens = {}
    
    for note in notes:
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type.name == 'NOTE_ON' and value == note:
                # Find corresponding NOTE_OFF
                for off_token in range(vocab_config.vocab_size):
                    off_event_type, off_value = vocab_config.token_to_event_info(off_token)
                    if off_event_type.name == 'NOTE_OFF' and off_value == note:
                        note_tokens[note] = (token, off_token)
                        break
                break
    
    if len(note_tokens) >= 4:
        melody_tokens = [1, velocity_token]  # START + velocity
        
        for note in notes:
            if note in note_tokens:
                note_on, note_off = note_tokens[note]
                melody_tokens.extend([note_on, time_token, note_off])
        
        melody_tokens.append(2)  # END
        
        print(f"Melody tokens: {len(melody_tokens)} tokens")
        
        melody_tensor = torch.from_numpy(np.array(melody_tokens)).unsqueeze(0)
        
        stats = exporter.export_tokens_to_midi(
            tokens=melody_tensor,
            output_path="finale_test/aurl_melody.mid",
            title="Aurl Melody",
            tempo=120.0
        )
        
        print(f"✅ Created: finale_test/aurl_melody.mid")
        print(f"   Notes: {stats.total_notes}")
        print(f"   Duration: {stats.total_duration:.2f}s")
        
except Exception as e:
    print(f"❌ Melody creation failed: {e}")

print(f"\n=== Test Results ===")
print("Created new test files:")
print("- finale_test/aurl_corrected.mid (single note with better velocity/timing)")
print("- finale_test/aurl_melody.mid (4-note melody)")
print()
print("Please test these in Finale. If they work, we've fixed the export issues!")
print("If they still don't work, the problem is deeper in the MIDI format.")

# Show what we learned about the tokens
print(f"\n=== Token Analysis ===")
print("The original problem tokens:")
print("- Token 692 (velocity): produces velocity=1 (too quiet)")
print("- Token 180 (time): produces 0.0s duration (too short)")
print()
print("We need to find/use tokens that produce:")
print("- Velocity 64-100 (audible)")
print("- Time duration 0.2-1.0s (visible notes)")
print()
print("This explains why generated music has no audible/visible notes!")