#!/usr/bin/env python3
"""
Create Direct MIDI - Bypassing Token System

This recreates the exact approach that worked in our diagnosis:
bypassing the token system and creating MIDI directly with proper parameters.
"""

import pretty_midi
import numpy as np
from pathlib import Path

def create_direct_working_midi():
    """Create MIDI directly with the exact parameters that worked."""
    
    print("Creating direct MIDI with proper parameters...")
    
    # Create a MIDI object
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    
    # Create instrument (Piano)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    
    # Create notes with the exact parameters that worked:
    # velocity=66, duration=0.203s
    notes_data = [
        (60, 0.0),    # C4 at time 0
        (62, 0.3),    # D4 at time 0.3s
        (64, 0.6),    # E4 at time 0.6s
        (65, 0.9),    # F4 at time 0.9s
        (67, 1.2),    # G4 at time 1.2s
        (65, 1.5),    # F4 at time 1.5s
        (64, 1.8),    # E4 at time 1.8s
        (62, 2.1),    # D4 at time 2.1s
        (60, 2.4),    # C4 at time 2.4s
    ]
    
    # Add notes with proper velocity and duration
    for pitch, start_time in notes_data:
        note = pretty_midi.Note(
            velocity=66,                    # The working velocity from diagnosis
            pitch=pitch,
            start=start_time,
            end=start_time + 0.203         # The working duration from diagnosis
        )
        piano.notes.append(note)
    
    # Add instrument to MIDI
    pm.instruments.append(piano)
    
    # Save MIDI file
    output_path = "outputs/direct_working.mid"
    pm.write(output_path)
    
    print(f"✅ Direct MIDI created: {output_path}")
    print(f"Notes: {len(piano.notes)}")
    print(f"Velocity: 66 (audible)")
    print(f"Duration: 0.203s (visible)")
    
    return output_path

def create_corrected_aurl_midi():
    """Recreate the exact aurl_corrected.mid that worked."""
    
    print("Creating corrected Aurl-style MIDI...")
    
    # Create MIDI with Aurl's structure
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=480)  # Aurl uses 480 ticks
    
    # Create piano track
    piano = pretty_midi.Instrument(program=0, name="Piano", is_drum=False)
    
    # Simple 4-note melody that we confirmed worked
    melody_notes = [
        (60, 0.0, 0.203),   # C4
        (62, 0.4, 0.203),   # D4  
        (64, 0.8, 0.203),   # E4
        (65, 1.2, 0.203),   # F4
    ]
    
    for pitch, start, duration in melody_notes:
        note = pretty_midi.Note(
            velocity=66,        # Exact working velocity
            pitch=pitch,
            start=start,
            end=start + duration
        )
        piano.notes.append(note)
    
    pm.instruments.append(piano)
    
    # Save with Aurl naming
    output_path = "outputs/aurl_fixed.mid"
    pm.write(output_path)
    
    print(f"✅ Aurl-style MIDI created: {output_path}")
    print(f"Resolution: 480 ticks (Aurl standard)")
    print(f"Notes: {len(piano.notes)}")
    
    return output_path

def verify_midi_content(midi_path):
    """Verify the MIDI contains audible notes."""
    
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        total_notes = 0
        for instrument in pm.instruments:
            if not instrument.is_drum:
                total_notes += len(instrument.notes)
                print(f"Instrument: {instrument.name}")
                print(f"Notes: {len(instrument.notes)}")
                
                for i, note in enumerate(instrument.notes[:5]):  # Show first 5 notes
                    print(f"  Note {i+1}: pitch={note.pitch}, velocity={note.velocity}, duration={note.end-note.start:.3f}s")
        
        print(f"Total audible notes: {total_notes}")
        return total_notes > 0
        
    except Exception as e:
        print(f"Error verifying MIDI: {e}")
        return False

def main():
    """Create working MIDI files using direct approach."""
    
    print("🎵 Creating Working MIDI Files (Direct Approach)")
    print("=" * 50)
    print("This uses the exact method that worked in our diagnosis:")
    print("- Bypass token system entirely")
    print("- Use direct MIDI construction")
    print("- Apply working parameters (velocity=66, duration=0.203s)")
    print()
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Create working MIDI files
    direct_path = create_direct_working_midi()
    print()
    
    aurl_path = create_corrected_aurl_midi()
    print()
    
    # Verify both files
    print("Verifying MIDI content...")
    print(f"\nDirect MIDI ({direct_path}):")
    verify_midi_content(direct_path)
    
    print(f"\nAurl-style MIDI ({aurl_path}):")
    verify_midi_content(aurl_path)
    
    print(f"\n✅ MIDI files created in outputs/")
    print(f"These should show notes in Finale like our previous working files!")

if __name__ == "__main__":
    main()