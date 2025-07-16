#!/usr/bin/env python3
"""
Test natural classical music generation - no forced chord progressions!

This simulates what should happen when your AI model is properly trained on 
classical music data (Bach, Mozart, etc.). Classical music is naturally melodic
and contrapuntal, not chord-based.
"""

import numpy as np
import torch
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Natural Classical Music Generation Test ===\n")

# Initialize the converter to understand token structure
converter = MusicRepresentationConverter()
vocab_config = VocabularyConfig()

# IMPORTANT: In real generation, these tokens would come from your trained model
# which has learned patterns from Bach, Mozart, etc. - NOT predefined progressions!

# Let's simulate what a model trained on classical music might generate:
# A simple melodic line like you'd find in Bach or Mozart

# First, let's understand the token structure
print("Token vocabulary structure:")
print(f"Total vocabulary size: {vocab_config.vocab_size}")

# Create a natural melodic sequence (simulating trained model output)
# This is what your model should learn to generate from classical training data
melodic_sequence = []

# START token
melodic_sequence.append(0)

# Set a moderate velocity (classical music dynamics)
velocity_token = 708  # This would be learned from training data

# Create a simple melodic line (ascending and descending)
# This simulates what the model learns from Bach/Mozart patterns
note_pitches = [
    60,  # C4
    62,  # D4
    64,  # E4 
    65,  # F4
    67,  # G4
    65,  # F4
    64,  # E4
    62,  # D4
    60,  # C4
    # Add some variation
    59,  # B3
    60,  # C4
    64,  # E4
    67,  # G4
    72,  # C5
]

# Build the sequence with proper musical grammar
# (The model should learn this pattern from training data)
melodic_sequence.append(velocity_token)

for i, pitch in enumerate(note_pitches):
    # NOTE_ON token (calculated based on vocabulary structure)
    note_on = 4 + (pitch - 21)  # Offset for special tokens + pitch mapping
    melodic_sequence.append(note_on)
    
    # TIME_SHIFT - varying durations for musical interest
    # Short notes for running passages, longer for phrase endings
    if i in [4, 8, 13]:  # Phrase endings
        time_shift = 270  # Longer duration
    else:
        time_shift = 266  # Shorter duration (eighth notes)
    melodic_sequence.append(time_shift)
    
    # NOTE_OFF
    note_off = 132 + (pitch - 21)  # Offset for NOTE_OFF range
    melodic_sequence.append(note_off)

# END token
melodic_sequence.append(1)

# Convert to numpy array
tokens = np.array(melodic_sequence)
print(f"\nGenerated melodic sequence: {len(tokens)} tokens")
print("This represents a natural melodic line, not forced chords!")

# Export to MIDI
config = create_standard_config()
exporter = MidiExporter(config)

output_path = "outputs/generated/natural_classical_melody.mid"
tokens_tensor = torch.from_numpy(tokens).unsqueeze(0)

try:
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path=output_path,
        title="Natural Classical Melody",
        style="classical",
        tempo=120.0
    )
    
    print(f"\nMIDI Export Results:")
    print(f"  File: {output_path}")
    print(f"  Notes: {stats.total_notes}")
    print(f"  Duration: {stats.total_duration:.2f}s")
    
    print("\n✅ SUCCESS: Created a natural melodic line!")
    print("This is what your AI should generate after training on classical music.")
    
except Exception as e:
    print(f"Error: {e}")

print("\n=== Key Points for Natural Music Generation ===")
print()
print("1. NO FORCED CHORD PROGRESSIONS!")
print("   - Classical music is melodic and contrapuntal")
print("   - Let the AI learn patterns from Bach, Mozart, etc.")
print()
print("2. The AI will naturally learn:")
print("   - Melodic phrases and sequences")
print("   - Voice leading and counterpoint")
print("   - Classical phrasing and structure")
print()
print("3. Training on classical MIDI files will teach:")
print("   - Natural note sequences (scales, arpeggios)")
print("   - Rhythmic patterns")
print("   - Dynamic expression")
print()
print("4. No need to program music theory - it learns from data!")
print()
print("The current 'clump of chords' issue happens because:")
print("- We're manually creating test data with chords")
print("- The model hasn't been trained on real classical music yet")
print("- Once trained, it will generate natural melodic lines")