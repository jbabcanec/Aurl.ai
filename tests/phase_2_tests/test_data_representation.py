"""
Comprehensive test suite for data representation system.

This test analyzes the actual data flow from MIDI -> Events -> Tokens -> Piano Roll
and provides human-readable output to understand what we're working with.
"""

import sys
import json
import numpy as np
from pathlib import Path
import pretty_midi
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.midi_parser import load_midi_file
from src.data.representation import (
    MusicRepresentationConverter, VocabularyConfig, PianoRollConfig,
    MusicalMetadata, EventType
)

def create_comprehensive_test_midi():
    """Create a comprehensive test MIDI file with multiple musical elements."""
    pm = pretty_midi.PrettyMIDI()
    
    # Create a piano melody with clear musical structure
    piano = pretty_midi.Instrument(program=0, name="Piano")
    
    # Simple C major melody: C-D-E-F-G-A-B-C
    melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    
    current_time = 0.0
    for i, pitch in enumerate(melody_notes):
        # Vary velocities to show dynamics
        velocity = 60 + (i * 8)  # Crescendo from 60 to 116
        
        # Quarter notes (0.5 seconds at 120 BPM)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + 0.4  # Slightly shorter for articulation
        )
        piano.notes.append(note)
        current_time += 0.5
    
    # Add some chords for harmonic content
    chord_starts = [4.0, 5.0, 6.0, 7.0]
    chord_pitches = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # D minor
        [64, 67, 71],  # E minor
        [65, 69, 72]   # F major
    ]
    
    for start_time, pitches in zip(chord_starts, chord_pitches):
        for pitch in pitches:
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start_time,
                end=start_time + 0.8
            )
            piano.notes.append(note)
    
    pm.instruments.append(piano)
    
    # Add a simple bass line
    bass = pretty_midi.Instrument(program=32, name="Bass")  # Acoustic bass
    bass_notes = [48, 50, 52, 53]  # C3, D3, E3, F3
    
    for i, pitch in enumerate(bass_notes):
        start = i * 2.0  # Whole notes
        note = pretty_midi.Note(
            velocity=70,
            pitch=pitch,
            start=start,
            end=start + 1.8
        )
        bass.notes.append(note)
    
    pm.instruments.append(bass)
    
    # Add drums (channel 9)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    
    # Simple kick drum pattern
    for beat in range(8):  # 8 beats
        kick = pretty_midi.Note(
            velocity=100,
            pitch=36,  # Kick drum
            start=beat * 1.0,
            end=beat * 1.0 + 0.1
        )
        drums.notes.append(kick)
        
        # Hi-hat on off-beats
        if beat % 2 == 1:
            hihat = pretty_midi.Note(
                velocity=60,
                pitch=42,  # Hi-hat
                start=beat * 1.0,
                end=beat * 1.0 + 0.05
            )
            drums.notes.append(hihat)
    
    pm.instruments.append(drums)
    
    # Add tempo changes for testing (pretty_midi uses different syntax)
    # We'll let the default tempo be used since tempo changes need special handling
    
    # Save test file
    test_file = Path("test_comprehensive.mid")
    pm.write(str(test_file))
    return test_file

def analyze_vocabulary(vocab_config: VocabularyConfig):
    """Analyze the vocabulary configuration."""
    print("🔤 VOCABULARY ANALYSIS")
    print("=" * 50)
    print(f"Total vocabulary size: {vocab_config.vocab_size}")
    print(f"Note range: {vocab_config.min_pitch} to {vocab_config.max_pitch} ({vocab_config.max_pitch - vocab_config.min_pitch + 1} notes)")
    print(f"Time resolution: {vocab_config.time_shift_ms}ms per time shift")
    print(f"Max time shift: {vocab_config.time_shift_bins * vocab_config.time_shift_ms / 1000:.2f} seconds")
    print(f"Velocity quantization: {vocab_config.velocity_bins} bins")
    print(f"Tempo range: {vocab_config.min_tempo}-{vocab_config.max_tempo} BPM in {vocab_config.tempo_bins} bins")
    
    # Show some example token mappings
    print("\n📋 Example Token Mappings:")
    example_events = [
        (EventType.START_TOKEN, 0),
        (EventType.NOTE_ON, 60),  # C4
        (EventType.NOTE_ON, 72),  # C5
        (EventType.NOTE_OFF, 60),
        (EventType.TIME_SHIFT, 1),
        (EventType.VELOCITY_CHANGE, 16),
        (EventType.END_TOKEN, 0)
    ]
    
    for event_type, value in example_events:
        token = vocab_config.event_to_token_map.get((event_type, value), -1)
        print(f"  {event_type.name}({value}) -> Token {token}")
    
    print()

def analyze_events(representation):
    """Analyze the event sequence."""
    print("🎵 EVENT SEQUENCE ANALYSIS")
    print("=" * 50)
    print(f"Total events: {len(representation.events)}")
    
    # Count event types
    event_counts = {}
    for event in representation.events:
        event_type = event.event_type.name
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    print("\n📊 Event Type Distribution:")
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type}: {count} events")
    
    print("\n🎹 First 20 Events (showing musical flow):")
    for i, event in enumerate(representation.events[:20]):
        if event.event_type == EventType.NOTE_ON:
            note_name = get_note_name(event.value)
            print(f"  {i:2d}: {event.event_type.name:15} {note_name:4} (pitch={event.value:3d}) at t={event.time:.3f}s")
        elif event.event_type == EventType.TIME_SHIFT:
            print(f"  {i:2d}: {event.event_type.name:15} +{event.value * 125}ms -> t={event.time:.3f}s")
        else:
            print(f"  {i:2d}: {event.event_type.name:15} val={event.value:3d} at t={event.time:.3f}s")
    
    print()

def analyze_tokens(representation, vocab_config):
    """Analyze the token sequence."""
    print("🎯 TOKEN SEQUENCE ANALYSIS")
    print("=" * 50)
    print(f"Total tokens: {len(representation.tokens)}")
    print(f"Token range: {representation.tokens.min()} to {representation.tokens.max()}")
    print(f"Unique tokens used: {len(np.unique(representation.tokens))}/{vocab_config.vocab_size}")
    
    # Token frequency analysis
    unique_tokens, counts = np.unique(representation.tokens, return_counts=True)
    token_freq = list(zip(unique_tokens, counts))
    token_freq.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📈 Most Frequent Tokens:")
    for token, count in token_freq[:10]:
        event_type, value = vocab_config.token_to_event_info(token)
        print(f"  Token {token:3d}: {event_type.name:15} val={value:3d} -> {count:3d} times")
    
    print(f"\n🔢 First 20 Tokens:")
    for i, token in enumerate(representation.tokens[:20]):
        event_type, value = vocab_config.token_to_event_info(token)
        if event_type == EventType.NOTE_ON:
            note_name = get_note_name(value)
            print(f"  {i:2d}: Token {token:3d} -> {event_type.name:15} {note_name} (pitch={value})")
        else:
            print(f"  {i:2d}: Token {token:3d} -> {event_type.name:15} val={value}")
    
    print()

def analyze_piano_roll(representation):
    """Analyze the piano roll representation."""
    print("🎹 PIANO ROLL ANALYSIS")
    print("=" * 50)
    
    if representation.piano_roll is None:
        print("No piano roll data available")
        return
    
    piano_roll = representation.piano_roll
    print(f"Piano roll shape: {piano_roll.shape} (time_steps x pitch_bins)")
    print(f"Active notes: {np.sum(piano_roll > 0)} / {piano_roll.size} ({100 * np.sum(piano_roll > 0) / piano_roll.size:.1f}%)")
    
    # Find active pitch range
    active_pitches = np.where(np.sum(piano_roll, axis=0) > 0)[0]
    if len(active_pitches) > 0:
        min_pitch_idx = active_pitches.min()
        max_pitch_idx = active_pitches.max()
        min_pitch = min_pitch_idx + 21  # MIDI_NOTE_MIN
        max_pitch = max_pitch_idx + 21
        print(f"Active pitch range: {get_note_name(min_pitch)} to {get_note_name(max_pitch)} (MIDI {min_pitch}-{max_pitch})")
    
    # Analyze note density over time
    notes_per_timestep = np.sum(piano_roll, axis=1)
    print(f"Polyphony statistics:")
    print(f"  Max simultaneous notes: {int(notes_per_timestep.max())}")
    print(f"  Average simultaneous notes: {notes_per_timestep.mean():.2f}")
    print(f"  Silent timesteps: {np.sum(notes_per_timestep == 0)} / {len(notes_per_timestep)}")
    
    # Show a snippet of the piano roll (first few timesteps)
    print(f"\n🎼 Piano Roll Visualization (first 10 timesteps, pitches 60-72):")
    print("   Time:", end="")
    for t in range(min(10, piano_roll.shape[0])):
        print(f"{t:3d}", end="")
    print()
    
    for pitch_idx in range(40, 53):  # MIDI 60-72 (C4-C5)
        pitch = pitch_idx + 21
        note_name = get_note_name(pitch)
        print(f"{note_name:>7}:", end="")
        for t in range(min(10, piano_roll.shape[0])):
            value = piano_roll[t, pitch_idx]
            symbol = "█" if value > 0 else "·"
            print(f"  {symbol}", end="")
        print()
    
    print()

def analyze_reversibility(original_midi, reconstructed_midi):
    """Test the reversibility of the representation."""
    print("🔄 REVERSIBILITY TEST")
    print("=" * 50)
    
    # Compare basic statistics
    orig_notes = sum(len(inst.notes) for inst in original_midi.instruments)
    recon_notes = sum(len(inst.notes) for inst in reconstructed_midi.instruments)
    
    print(f"Original MIDI:")
    print(f"  Duration: {original_midi.end_time:.3f}s")
    print(f"  Instruments: {len(original_midi.instruments)}")
    print(f"  Total notes: {orig_notes}")
    
    print(f"\nReconstructed MIDI:")
    print(f"  Duration: {reconstructed_midi.end_time:.3f}s")
    print(f"  Instruments: {len(reconstructed_midi.instruments)}")
    print(f"  Total notes: {recon_notes}")
    
    print(f"\nReconstruction Accuracy:")
    note_preservation = (recon_notes / orig_notes * 100) if orig_notes > 0 else 0
    duration_error = abs(original_midi.end_time - reconstructed_midi.end_time)
    print(f"  Note preservation: {note_preservation:.1f}%")
    print(f"  Duration error: {duration_error:.3f}s")
    
    # Check if reconstruction is reasonably accurate
    if note_preservation > 80 and duration_error < 0.5:
        print("  ✅ Reconstruction quality: GOOD")
    elif note_preservation > 60 and duration_error < 1.0:
        print("  ⚠️  Reconstruction quality: FAIR")
    else:
        print("  ❌ Reconstruction quality: POOR")
    
    print()

def get_note_name(midi_pitch):
    """Convert MIDI pitch to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_pitch // 12) - 1
    note = note_names[midi_pitch % 12]
    return f"{note}{octave}"

def save_human_readable_analysis(representation, vocab_config, output_dir):
    """Save comprehensive human-readable analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detailed analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "summary": representation.get_summary(),
        "vocabulary_config": {
            "vocab_size": vocab_config.vocab_size,
            "pitch_range": f"{vocab_config.min_pitch}-{vocab_config.max_pitch}",
            "time_resolution_ms": vocab_config.time_shift_ms,
            "velocity_bins": vocab_config.velocity_bins,
            "tempo_range": f"{vocab_config.min_tempo}-{vocab_config.max_tempo} BPM"
        },
        "event_analysis": {
            "total_events": len(representation.events),
            "event_types": {},
            "first_20_events": []
        },
        "token_analysis": {
            "total_tokens": len(representation.tokens) if representation.tokens is not None else 0,
            "unique_tokens": len(np.unique(representation.tokens)) if representation.tokens is not None else 0,
            "first_20_tokens": representation.tokens[:20].tolist() if representation.tokens is not None else []
        },
        "piano_roll_analysis": {
            "shape": representation.piano_roll.shape if representation.piano_roll is not None else None,
            "active_notes_percentage": 0,
            "polyphony_stats": {}
        }
    }
    
    # Event type distribution
    for event in representation.events:
        event_type = event.event_type.name
        analysis["event_analysis"]["event_types"][event_type] = \
            analysis["event_analysis"]["event_types"].get(event_type, 0) + 1
    
    # First 20 events with details
    for i, event in enumerate(representation.events[:20]):
        event_info = {
            "index": i,
            "type": event.event_type.name,
            "value": event.value,
            "time": event.time,
            "description": str(event)
        }
        analysis["event_analysis"]["first_20_events"].append(event_info)
    
    # Piano roll analysis
    if representation.piano_roll is not None:
        piano_roll = representation.piano_roll
        active_percentage = np.sum(piano_roll > 0) / piano_roll.size * 100
        notes_per_timestep = np.sum(piano_roll, axis=1)
        
        analysis["piano_roll_analysis"]["active_notes_percentage"] = active_percentage
        analysis["piano_roll_analysis"]["polyphony_stats"] = {
            "max_simultaneous_notes": int(notes_per_timestep.max()),
            "avg_simultaneous_notes": float(notes_per_timestep.mean()),
            "silent_timesteps": int(np.sum(notes_per_timestep == 0))
        }
    
    # Save analysis
    with open(output_dir / "detailed_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save raw data samples for inspection
    samples = {
        "first_50_tokens": representation.tokens[:50].tolist() if representation.tokens is not None else [],
        "token_to_event_mapping": {
            int(token): vocab_config.token_to_event_info(int(token))
            for token in representation.tokens[:50] if representation.tokens is not None
        },
        "piano_roll_sample": representation.piano_roll[:20, 40:53].tolist() if representation.piano_roll is not None else []
    }
    
    with open(output_dir / "data_samples.json", 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"💾 Saved detailed analysis to {output_dir}")

def test_data_representation():
    """Main test function for data representation system."""
    print("🎼 MidiFly Data Representation Test Suite")
    print("=" * 60)
    print()
    
    # Create test MIDI file
    print("📝 Creating comprehensive test MIDI file...")
    test_file = create_comprehensive_test_midi()
    
    try:
        # Parse MIDI file
        print("📖 Parsing MIDI file...")
        midi_data = load_midi_file(test_file)
        print(f"   Loaded: {midi_data.end_time:.2f}s, {len(midi_data.instruments)} instruments")
        print()
        
        # Set up representation converter
        vocab_config = VocabularyConfig()
        piano_roll_config = PianoRollConfig()
        converter = MusicRepresentationConverter(vocab_config, piano_roll_config)
        
        # Analyze vocabulary
        analyze_vocabulary(vocab_config)
        
        # Convert to representation
        print("🔄 Converting to musical representation...")
        representation = converter.midi_to_representation(midi_data)
        print(f"   Created representation with {len(representation.events)} events")
        print()
        
        # Analyze different aspects
        analyze_events(representation)
        analyze_tokens(representation, vocab_config)
        analyze_piano_roll(representation)
        
        # Test reversibility
        print("🧪 Testing reversibility...")
        reconstructed_midi = converter.representation_to_midi(representation)
        analyze_reversibility(midi_data, reconstructed_midi)
        
        # Save representations and analysis
        output_dir = Path(__file__).parent.parent / "outputs" / "test_output_representation"
        print(f"💾 Saving representation data...")
        converter.save_representation(representation, output_dir)
        save_human_readable_analysis(representation, vocab_config, output_dir)
        
        # Create metadata example
        metadata = MusicalMetadata(
            title="Comprehensive Test Piece",
            composer="MidiFly Test Suite",
            genre="Test Music",
            key="C major",
            style_tags=["test", "educational"],
            source_file=str(test_file)
        )
        representation.metadata = metadata
        
        print("\n📋 Final Summary:")
        summary = representation.get_summary()
        for key, value in summary.items():
            if key != "metadata":
                print(f"   {key}: {value}")
        
        if "metadata" in summary:
            print(f"   metadata: {summary['metadata']['title']} by {summary['metadata']['composer']}")
        
        print("\n✅ All tests completed successfully!")
        print(f"📁 Check '{output_dir}' for detailed analysis files")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
            print(f"\n🧹 Cleaned up test file: {test_file}")

if __name__ == "__main__":
    test_data_representation()