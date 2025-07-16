#!/usr/bin/env python3
"""
Trace the exact token-to-MIDI conversion process
"""

import torch
import numpy as np
from src.data.representation import MusicRepresentationConverter, VocabularyConfig, EventType

print("=== Token-to-MIDI Conversion Trace ===\n")

# Create converter and config
converter = MusicRepresentationConverter()
vocab_config = VocabularyConfig()

print(f"Vocab size: {vocab_config.vocab_size}")

# Test token sequence (the manual one that should work)
test_tokens = np.array([1, 708, 43, 273, 171, 2])
print(f"Test tokens: {test_tokens}")

print("\n1. Token-to-Event Conversion:")
events = converter._tokens_to_events(test_tokens)
print(f"Number of events: {len(events)}")

for i, event in enumerate(events):
    print(f"  Event {i}: {event.event_type.name} value={event.value} time={event.time}")

print("\n2. Event-to-MIDI Conversion:")
midi_data = converter._events_to_midi(events)
print(f"Number of instruments: {len(midi_data.instruments)}")
print(f"End time: {midi_data.end_time}")

total_notes = 0
for i, instrument in enumerate(midi_data.instruments):
    print(f"  Instrument {i}: {len(instrument.notes)} notes")
    total_notes += len(instrument.notes)
    for j, note in enumerate(instrument.notes[:5]):  # Show first 5 notes
        print(f"    Note {j}: pitch={note.pitch} start={note.start:.3f} end={note.end:.3f} vel={note.velocity}")

print(f"Total notes: {total_notes}")

print("\n3. Token Mapping Analysis:")
# Check what our test tokens mean
for token in test_tokens:
    if hasattr(vocab_config, 'token_to_event') and token in vocab_config.token_to_event:
        event_type, value = vocab_config.token_to_event[token]
        print(f"  Token {token}: {event_type.name} = {value}")
    else:
        # Try the function instead
        try:
            event_type, value = vocab_config.token_to_event_info(token)
            print(f"  Token {token}: {event_type.name} = {value}")
        except:
            print(f"  Token {token}: UNKNOWN")

print("\n4. Debugging Token 43 and 171:")
# These should be note on/off pair
token_43_info = vocab_config.token_to_event_info(43)
token_171_info = vocab_config.token_to_event_info(171)
print(f"Token 43: {token_43_info}")
print(f"Token 171: {token_171_info}")

# Check if 171 is supposed to be note off for same pitch as 43
if token_43_info[0] == EventType.NOTE_ON and token_171_info[0] == EventType.NOTE_OFF:
    if token_43_info[1] == token_171_info[1]:
        print("✅ Token 43 and 171 are matching note on/off pair")
    else:
        print(f"❌ Token 43 and 171 have different pitches: {token_43_info[1]} vs {token_171_info[1]}")
else:
    print(f"❌ Token 43/171 are not note on/off: {token_43_info[0]} / {token_171_info[0]}")

print("\n5. Manual Event Creation Test:")
# Create events manually and see if that works
from src.data.representation import MusicEvent
from src.data.midi_parser import MidiNote, MidiInstrument, MidiData

manual_events = [
    MusicEvent(EventType.START_TOKEN, 0, 0, 0.0),
    MusicEvent(EventType.VELOCITY_CHANGE, 64, 0, 0.0),  # Normal velocity
    MusicEvent(EventType.NOTE_ON, 60, 0, 0.0),  # Middle C
    MusicEvent(EventType.NOTE_OFF, 60, 0, 1.0),  # End after 1 second
    MusicEvent(EventType.END_TOKEN, 0, 0, 1.0)
]

print("Manual events:")
for event in manual_events:
    print(f"  {event}")

manual_midi = converter._events_to_midi(manual_events)
print(f"Manual MIDI: {len(manual_midi.instruments)} instruments")
for inst in manual_midi.instruments:
    print(f"  Instrument: {len(inst.notes)} notes")
    for note in inst.notes:
        print(f"    Note: pitch={note.pitch} start={note.start} end={note.end}")

if manual_midi.instruments and manual_midi.instruments[0].notes:
    print("✅ Manual event creation works!")
else:
    print("❌ Manual event creation also fails!")

print("\n=== Diagnosis ===")
if total_notes == 0:
    print("The token-to-event conversion is not producing note on/off pairs correctly.")
    print("Either:")
    print("1. Token mapping is wrong (tokens don't map to expected events)")
    print("2. Event timing logic is broken")
    print("3. Note on/off pairing logic is broken")
else:
    print("Token conversion works - issue is elsewhere in the pipeline.")