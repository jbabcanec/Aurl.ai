#!/usr/bin/env python3
"""
Create a basic MIDI file using direct pretty_midi to verify Finale compatibility
"""

import pretty_midi
import numpy as np
import os

print("=== Creating Basic MIDI File Test ===\n")

# Create output directory
os.makedirs("finale_test", exist_ok=True)

# Method 1: Direct pretty_midi creation (should definitely work)
print("1. Creating MIDI with direct pretty_midi...")
try:
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    
    # Create an instrument (piano)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    
    # Create some simple notes (C major scale)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    
    for i, pitch in enumerate(notes):
        note = pretty_midi.Note(
            velocity=64,
            pitch=pitch,
            start=i * 0.5,      # Half second apart
            end=(i + 1) * 0.5   # Half second duration
        )
        piano.notes.append(note)
    
    # Add the instrument to the MIDI file
    midi.instruments.append(piano)
    
    # Save it
    midi.write("finale_test/direct_pretty_midi.mid")
    print(f"✅ Created: finale_test/direct_pretty_midi.mid")
    print(f"   Notes: {len(piano.notes)}")
    print(f"   Duration: {midi.get_end_time():.2f}s")
    
except Exception as e:
    print(f"❌ Direct pretty_midi failed: {e}")

# Method 2: Using mido (alternative MIDI library)
print("\n2. Creating MIDI with mido...")
try:
    import mido
    
    # Create a new MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add some meta messages
    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    
    # Add notes (C major scale)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    ticks_per_beat = 480
    
    for i, pitch in enumerate(notes):
        # Note on
        track.append(mido.Message('note_on', channel=0, note=pitch, velocity=64, time=0))
        # Note off after quarter note
        track.append(mido.Message('note_off', channel=0, note=pitch, velocity=64, time=ticks_per_beat))
    
    # Save it
    mid.save("finale_test/mido_created.mid")
    print(f"✅ Created: finale_test/mido_created.mid")
    print(f"   Tracks: {len(mid.tracks)}")
    print(f"   Messages: {len(track)}")
    
except Exception as e:
    print(f"❌ Mido creation failed: {e}")

# Method 3: Test our Aurl system with the simplest possible input
print("\n3. Testing Aurl MIDI export with minimal input...")
try:
    import torch
    from src.generation.midi_export import MidiExporter, create_standard_config
    from src.data.representation import MusicRepresentationConverter
    
    # Use the working tokens we found earlier
    converter = MusicRepresentationConverter()
    
    # Find the correct tokens for a simple note
    working_tokens = [
        1,    # START
        692,  # Velocity (from our earlier test)
        43,   # Note ON MIDI 60
        180,  # Time shift
        131,  # Note OFF MIDI 60 (matching pitch!)
        2     # END
    ]
    
    print(f"Using working tokens: {working_tokens}")
    
    # Convert to tensor
    tokens_tensor = torch.from_numpy(np.array(working_tokens)).unsqueeze(0)
    
    # Create exporter
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    # Export
    stats = exporter.export_tokens_to_midi(
        tokens=tokens_tensor,
        output_path="finale_test/aurl_system.mid",
        title="Aurl System Test",
        tempo=120.0
    )
    
    print(f"✅ Aurl export: finale_test/aurl_system.mid")
    print(f"   Notes: {stats.total_notes}")
    print(f"   Duration: {stats.total_duration:.2f}s")
    
    if stats.total_notes == 0:
        print("❌ Aurl system still producing 0 notes - deeper issue")
    
except Exception as e:
    print(f"❌ Aurl system test failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n=== Test Results ===")
print("Files created in finale_test/ directory:")
print("1. direct_pretty_midi.mid - Should definitely work in Finale")
print("2. mido_created.mid - Alternative MIDI creation")
print("3. aurl_system.mid - Our system test")
print()
print("Try opening these files in Finale to see which ones work.")
print("If direct_pretty_midi.mid doesn't work, there may be a Finale configuration issue.")
print("If it does work, then our Aurl system has a conversion problem.")

# Also create a WAV file for audio verification
print(f"\n4. Creating WAV file for audio verification...")
try:
    # Load the pretty_midi file we just created
    pm = pretty_midi.PrettyMIDI("finale_test/direct_pretty_midi.mid")
    
    # Synthesize audio
    audio = pm.fluidsynth(fs=22050)
    
    # Save as WAV
    import scipy.io.wavfile
    scipy.io.wavfile.write("finale_test/audio_test.wav", 22050, audio)
    print(f"✅ Created: finale_test/audio_test.wav")
    print("You can play this WAV file to hear if the notes are correct")
    
except Exception as e:
    print(f"⚠️  WAV creation failed: {e}")
    print("(This is normal if fluidsynth is not installed)")

print(f"\n📁 Check the finale_test/ directory for all test files")
print("The direct_pretty_midi.mid should definitely work in Finale if the MIDI export is functioning correctly.")