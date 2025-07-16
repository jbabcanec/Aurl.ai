#!/usr/bin/env python3
"""
Generate Working MIDI Using Diagnosis Results

Uses the exact token combinations that we confirmed work in Finale
from our previous MIDI export diagnosis.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.midi_export import MidiExporter

def create_working_token_sequence():
    """Create a token sequence using the tokens we confirmed work."""
    
    # From diagnosis: Token 708 (velocity=66), Token 193 (duration=0.203s)
    # Create a simple C-D-E-F melody with proper note on/off pairing
    
    working_tokens = []
    
    # Set velocity first (token 708 = velocity 66)
    working_tokens.append(708)
    
    # C major scale melody with proper note on/off pairs
    notes = [60, 62, 64, 65, 67, 65, 64, 62, 60]  # C-D-E-F-G-F-E-D-C
    
    for pitch in notes:
        # NOTE_ON for this pitch
        working_tokens.append(pitch)  # Assuming 60 = NOTE_ON C4
        
        # Time duration (token 193 = 0.203s duration)
        working_tokens.append(193)
        
        # NOTE_OFF for this pitch  
        working_tokens.append(pitch + 128)  # Assuming NOTE_OFF = NOTE_ON + 128
        
        # Short pause between notes
        working_tokens.append(193)
    
    return torch.tensor(working_tokens, dtype=torch.long)

def export_working_midi():
    """Export MIDI using the working token sequence."""
    
    print("Creating working token sequence...")
    tokens = create_working_token_sequence()
    print(f"Token sequence: {tokens}")
    
    # Export to MIDI
    exporter = MidiExporter()
    output_path = "outputs/working_test.mid"
    
    print(f"Exporting to: {output_path}")
    
    try:
        stats = exporter.export_tokens_to_midi(
            tokens=tokens,
            output_path=output_path,
            title="Working Test Generation"
        )
        
        print(f"Export stats: {stats}")
        print(f"✅ MIDI exported to: {output_path}")
        
        # Also create the corrected sequence from diagnosis
        create_corrected_sequence()
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

def create_corrected_sequence():
    """Create the exact corrected sequence that worked in our diagnosis."""
    
    # From MIDI_EXPORT_DIAGNOSIS.md - the corrected sequence that worked
    corrected_tokens = torch.tensor([
        708,  # Velocity token (velocity=66)
        60,   # NOTE_ON C4
        193,  # Time shift (duration=0.203s)
        188,  # NOTE_OFF C4 (60 + 128 = 188)
        193,  # Time shift (pause)
        62,   # NOTE_ON D4
        193,  # Time shift
        190,  # NOTE_OFF D4 (62 + 128 = 190)
        193,  # Time shift
        64,   # NOTE_ON E4
        193,  # Time shift
        192,  # NOTE_OFF E4 (64 + 128 = 192)
    ], dtype=torch.long)
    
    exporter = MidiExporter()
    output_path = "outputs/corrected_test.mid"
    
    print(f"Creating corrected sequence from diagnosis...")
    print(f"Corrected tokens: {corrected_tokens}")
    
    try:
        stats = exporter.export_tokens_to_midi(
            tokens=corrected_tokens,
            output_path=output_path,
            title="Corrected Test from Diagnosis"
        )
        
        print(f"✅ Corrected MIDI exported to: {output_path}")
        print(f"Notes generated: {stats.total_notes}")
        
    except Exception as e:
        print(f"❌ Corrected export failed: {e}")

if __name__ == "__main__":
    print("🎵 Generating Working MIDI using Diagnosis Results")
    print("=" * 50)
    export_working_midi()
    print("\nFiles created in outputs/ directory")