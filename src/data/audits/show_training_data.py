"""
Show exactly what the training data looks like after transformation.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.midi_parser import load_midi_file
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.data.dataset import LazyMidiDataset, create_dataloader

def show_training_data_example():
    """Show a concrete example of what the neural network sees during training."""
    
    print("🎼 Aurl.ai Training Data Example")
    print("=" * 60)
    
    # Load a sample MIDI file
    sample_file = Path("data/raw/chpn-p18.mid")  # Chopin Prelude - shorter file
    
    if not sample_file.exists():
        print("Sample file not found. Using generated sample.")
        return
    
    # Parse and transform
    print(f"\n1️⃣ Loading MIDI file: {sample_file.name}")
    midi_data = load_midi_file(sample_file)
    print(f"   Duration: {midi_data.end_time:.1f}s")
    print(f"   Notes: {sum(len(inst.notes) for inst in midi_data.instruments)}")
    
    # Convert to representation
    print("\n2️⃣ Converting to neural network format...")
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    representation = converter.midi_to_representation(midi_data)
    
    print(f"   Events: {len(representation.events)}")
    print(f"   Tokens: {len(representation.tokens)}")
    print(f"   Piano roll shape: {representation.piano_roll.shape}")
    
    # Show token sequence
    print("\n3️⃣ Token Sequence (first 50 tokens):")
    print("   Position | Token | Event Type         | Details")
    print("   " + "-" * 50)
    
    for i in range(min(50, len(representation.tokens))):
        token = int(representation.tokens[i])
        event_type, value = vocab_config.token_to_event_info(token)
        
        # Add meaningful descriptions
        if event_type.name == "NOTE_ON" or event_type.name == "NOTE_OFF":
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = (value // 12) - 1
            note = note_names[value % 12]
            details = f"{note}{octave} (MIDI {value})"
        elif event_type.name == "VELOCITY_CHANGE":
            velocity = int((value / (vocab_config.velocity_bins - 1)) * 126) + 1
            details = f"Set velocity to ~{velocity}"
        elif event_type.name == "TIME_SHIFT":
            ms = value * vocab_config.time_shift_ms
            details = f"Wait {ms}ms"
        else:
            details = f"Value: {value}"
            
        print(f"   {i:8d} | {token:5d} | {event_type.name:18s} | {details}")
    
    # Show as training batch
    print("\n4️⃣ As PyTorch Training Batch:")
    dataset = LazyMidiDataset(
        data_dir=Path("data/raw"),
        sequence_length=128,  # Shorter for display
        max_files=1
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"   Batch shape: {sample['tokens'].shape}")
        print(f"   Data type: {sample['tokens'].dtype}")
        print(f"   Memory size: {sample['tokens'].nbytes} bytes")
        
        # Show actual tensor values
        print(f"\n   Raw tensor (first 20 values):")
        print(f"   {sample['tokens'][:20].detach().cpu().numpy()}")
        
        # Create a mini batch
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        
        print(f"\n5️⃣ Mini-batch for training:")
        print(f"   Batch tokens shape: {batch['tokens'].shape}")
        print(f"   Files in batch: {len(batch['file_paths'])}")
        
        if 'piano_roll' in batch:
            print(f"   Piano roll shape: {batch['piano_roll'].shape}")
    
    # Show piano roll visualization
    print("\n6️⃣ Piano Roll Visualization (first 20 time steps, octave C4-C5):")
    if representation.piano_roll is not None:
        # Show MIDI notes 60-72 (C4 to C5)
        start_pitch = 60 - 21  # Adjust for MIDI_NOTE_MIN offset
        end_pitch = 72 - 21
        
        print("   Time:", end="")
        for t in range(min(20, representation.piano_roll.shape[0])):
            print(f"{t:3d}", end="")
        print()
        
        for pitch_idx in range(start_pitch, end_pitch + 1):
            midi_pitch = pitch_idx + 21
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octave = (midi_pitch // 12) - 1
            note = note_names[midi_pitch % 12]
            
            print(f"   {note}{octave:>2}:", end="")
            for t in range(min(20, representation.piano_roll.shape[0])):
                value = representation.piano_roll[t, pitch_idx]
                symbol = "█" if value > 0 else "·"
                print(f"  {symbol}", end="")
            print()
    
    print("\n" + "=" * 60)
    print("This is exactly what the neural network processes during training!")


if __name__ == "__main__":
    show_training_data_example()