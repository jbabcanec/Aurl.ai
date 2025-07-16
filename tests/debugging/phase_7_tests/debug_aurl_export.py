#!/usr/bin/env python3
"""
Debug why Aurl MIDI export creates files that don't show in Finale
"""

import pretty_midi
import torch
import numpy as np
from src.data.representation import MusicRepresentationConverter
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Debugging Aurl MIDI Export ===\n")

# First, let's see what's actually in the MIDI files
print("1. Analyzing MIDI file contents:")

# Check the working MIDI
print("\nWorking MIDI (direct_pretty_midi.mid):")
try:
    pm_working = pretty_midi.PrettyMIDI("finale_test/direct_pretty_midi.mid")
    print(f"  Instruments: {len(pm_working.instruments)}")
    for i, inst in enumerate(pm_working.instruments):
        print(f"  Instrument {i}: {len(inst.notes)} notes, program={inst.program}")
        for j, note in enumerate(inst.notes[:3]):  # First 3 notes
            print(f"    Note {j}: pitch={note.pitch}, start={note.start:.3f}, end={note.end:.3f}, vel={note.velocity}")
except Exception as e:
    print(f"  Error: {e}")

# Check the Aurl MIDI
print("\nAurl MIDI (aurl_system.mid):")
try:
    pm_aurl = pretty_midi.PrettyMIDI("finale_test/aurl_system.mid")
    print(f"  Instruments: {len(pm_aurl.instruments)}")
    for i, inst in enumerate(pm_aurl.instruments):
        print(f"  Instrument {i}: {len(inst.notes)} notes, program={inst.program}")
        for j, note in enumerate(inst.notes):  # All notes
            print(f"    Note {j}: pitch={note.pitch}, start={note.start:.3f}, end={note.end:.3f}, vel={note.velocity}")
except Exception as e:
    print(f"  Error: {e}")

# Now let's trace through the Aurl export process step by step
print("\n2. Tracing Aurl Export Process:")

converter = MusicRepresentationConverter()

# The tokens that supposedly created 1 note
test_tokens = [1, 692, 43, 180, 131, 2]
print(f"Test tokens: {test_tokens}")

# Convert tokens to events
print("\nConverting tokens to events:")
events = converter._tokens_to_events(np.array(test_tokens))
for i, event in enumerate(events):
    print(f"  Event {i}: {event.event_type.name} value={event.value} time={event.time:.3f}")

# Convert events to MIDI data
print("\nConverting events to MIDI data:")
midi_data = converter._events_to_midi(events)
print(f"  Instruments: {len(midi_data.instruments)}")
for i, inst in enumerate(midi_data.instruments):
    print(f"  Instrument {i}: {len(inst.notes)} notes")
    for j, note in enumerate(inst.notes):
        print(f"    Note: pitch={note.pitch}, start={note.start:.3f}, end={note.end:.3f}, vel={note.velocity}")

# Now let's create a MIDI using our system but bypass the token conversion
print("\n3. Creating MIDI with direct MidiData (bypassing tokens):")
try:
    from src.data.midi_parser import MidiData, MidiNote, MidiInstrument
    
    # Create MidiData directly
    notes = [
        MidiNote(pitch=60, velocity=64, start=0.0, end=0.5, channel=0),
        MidiNote(pitch=62, velocity=64, start=0.5, end=1.0, channel=0),
        MidiNote(pitch=64, velocity=64, start=1.0, end=1.5, channel=0),
    ]
    
    instrument = MidiInstrument(
        program=0,
        is_drum=False,
        name="Piano",
        notes=notes
    )
    
    midi_data_direct = MidiData(
        instruments=[instrument],
        tempo_changes=[(0.0, 120.0)],
        end_time=1.5
    )
    
    print(f"  Created MidiData with {len(notes)} notes")
    
    # Export using Aurl system
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    # Use the internal method directly
    midi_file = exporter._create_midi_file(
        midi_data_direct,
        title="Direct MidiData Test",
        tempo=120.0
    )
    
    # Save it
    midi_file.save("finale_test/aurl_direct_data.mid")
    
    print(f"✅ Created: finale_test/aurl_direct_data.mid")
    print(f"   Notes: {exporter.stats.total_notes}")
    
except Exception as e:
    print(f"❌ Direct MidiData test failed: {e}")
    import traceback
    traceback.print_exc()

# Let's also check the actual file structure
print("\n4. Checking MIDI file structure with mido:")
try:
    import mido
    
    print("\nWorking file structure (direct_pretty_midi.mid):")
    mid_working = mido.MidiFile("finale_test/direct_pretty_midi.mid")
    print(f"  Type: {mid_working.type}, Ticks/beat: {mid_working.ticks_per_beat}")
    print(f"  Tracks: {len(mid_working.tracks)}")
    for i, track in enumerate(mid_working.tracks[:2]):  # First 2 tracks
        print(f"  Track {i}: {len(track)} messages")
        for j, msg in enumerate(track[:5]):  # First 5 messages
            print(f"    {msg}")
    
    print("\nAurl file structure (aurl_system.mid):")
    mid_aurl = mido.MidiFile("finale_test/aurl_system.mid")
    print(f"  Type: {mid_aurl.type}, Ticks/beat: {mid_aurl.ticks_per_beat}")
    print(f"  Tracks: {len(mid_aurl.tracks)}")
    for i, track in enumerate(mid_aurl.tracks):  # All tracks
        print(f"  Track {i}: {len(track)} messages")
        for j, msg in enumerate(track):  # All messages
            print(f"    {msg}")
            
except Exception as e:
    print(f"Mido analysis failed: {e}")

print("\n=== Diagnosis ===")
print("The issue is likely in:")
print("1. How MidiData is converted to MIDI file format")
print("2. Missing metadata or track structure")
print("3. Timing/resolution issues")
print()
print("Check finale_test/aurl_direct_data.mid to see if bypassing")
print("the token system produces a working MIDI file.")