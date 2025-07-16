#!/usr/bin/env python3
"""
Simple MIDI debug - fix the path issue and test basic conversion
"""

import torch
import numpy as np
import os
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Simple MIDI Debug ===\n")

# Create output directory
os.makedirs("debug_output", exist_ok=True)

# Test with a very simple, known-good sequence
print("1. Testing Simple Note Sequence:")
simple_sequence = np.array([
    1,    # START token
    708,  # Velocity token (708 seems to be what the validator adds)
    43,   # Note ON (token 43 = note 30 according to debug)
    273,  # Time shift 
    171,  # Note OFF (43 + 128 = 171 for same pitch)
    2     # END token
])

print(f"Simple sequence: {simple_sequence}")

# Test conversion
try:
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    tokens_tensor = torch.from_numpy(simple_sequence).unsqueeze(0)
    
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path="debug_output/simple_test.mid",
        title="Simple Test",
        tempo=120.0
    )
    
    print(f"Simple test: {stats.total_notes} notes, {stats.total_duration:.2f}s")
    
    if stats.total_notes > 0:
        print("✅ SUCCESS: Basic conversion works!")
    else:
        print("❌ FAILED: Even simple sequence produces no notes")
        
except Exception as e:
    print(f"Simple test error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Load and analyze the corrected tokens from generation
print("\n2. Testing Generated Tokens (corrected):")
try:
    # Load original tokens
    tokens = torch.load('natural_piano_piece/aurl_natural_generation.tokens')
    if tokens.dim() > 1:
        tokens = tokens.flatten()
    
    # Get the corrected version (what the system actually tried to convert)
    from src.generation.token_validator import validate_and_fix_tokens
    is_valid, corrected_tokens = validate_and_fix_tokens(tokens.numpy())
    
    if corrected_tokens is not None:
        print(f"Using corrected tokens: {len(corrected_tokens)} tokens")
        print(f"First 20: {corrected_tokens[:20]}")
        
        corrected_tensor = torch.from_numpy(corrected_tokens).unsqueeze(0)
        
        stats = exporter.export_tokens_to_midi(
            tokens=corrected_tensor,
            output_path="debug_output/corrected_test.mid",
            title="Corrected Test", 
            tempo=120.0
        )
        
        print(f"Corrected tokens: {stats.total_notes} notes, {stats.total_duration:.2f}s")
        
    else:
        print("No corrected tokens available")
        
except Exception as e:
    print(f"Corrected test error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Manual construction of a known working sequence
print("\n3. Testing Manually Built Musical Sequence:")
try:
    # Build a sequence that should definitely work
    # Based on the token mapping we saw: token 43 = note 30 (MIDI note 51)
    manual_sequence = [
        1,    # START
        708,  # Velocity (this was added by validator)
        43,   # Note ON - MIDI note 30  
        273,  # Time shift
        171,  # Note OFF - same note (43 + 128)
        44,   # Note ON - MIDI note 31
        273,  # Time shift  
        172,  # Note OFF - note 31 (44 + 128)
        2     # END
    ]
    
    manual_array = np.array(manual_sequence)
    manual_tensor = torch.from_numpy(manual_array).unsqueeze(0)
    
    print(f"Manual sequence: {manual_sequence}")
    
    stats = exporter.export_tokens_to_midi(
        tokens=manual_tensor,
        output_path="debug_output/manual_test.mid",
        title="Manual Test",
        tempo=120.0
    )
    
    print(f"Manual test: {stats.total_notes} notes, {stats.total_duration:.2f}s")
    
    if stats.total_notes > 0:
        print("✅ Manual sequence works - issue is in generated token structure")
    else:
        print("❌ Manual sequence fails - issue is in conversion logic")
        
except Exception as e:
    print(f"Manual test error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Files Created ===")
print("Check these MIDI files in Finale:")
print("- debug_output/simple_test.mid")
print("- debug_output/corrected_test.mid") 
print("- debug_output/manual_test.mid")
print()
print("If any of these work in Finale, we've isolated the problem!")