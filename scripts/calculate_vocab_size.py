"""
Calculate the actual vocabulary size from VocabularyConfig.

This helps us understand why we're getting 774 tokens instead of 387.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.representation import VocabularyConfig

def calculate_vocabulary_breakdown():
    """Calculate and breakdown the vocabulary size."""
    config = VocabularyConfig()
    
    print("🔢 Vocabulary Size Calculation")
    print("=" * 50)
    
    # Calculate each component
    special_tokens = 4  # START, END, PAD, UNK
    
    # Note events
    note_range = config.max_pitch - config.min_pitch + 1  # 0-127 = 128 values
    note_on_tokens = note_range
    note_off_tokens = note_range
    total_note_tokens = note_on_tokens + note_off_tokens
    
    # Time shift events
    time_shift_tokens = config.time_shift_bins  # 512
    
    # Velocity events
    velocity_tokens = config.velocity_bins  # 32
    
    # Control events
    control_tokens = 2  # SUSTAIN_ON, SUSTAIN_OFF
    
    # Tempo events
    tempo_tokens = config.tempo_bins  # 32
    
    # Instrument events
    instrument_tokens = config.max_instruments  # 16
    
    # Total
    total_calculated = (special_tokens + total_note_tokens + time_shift_tokens + 
                       velocity_tokens + control_tokens + tempo_tokens + instrument_tokens)
    
    print(f"Special tokens:     {special_tokens:4d}")
    print(f"Note ON tokens:     {note_on_tokens:4d} (pitches {config.min_pitch}-{config.max_pitch})")
    print(f"Note OFF tokens:    {note_off_tokens:4d} (pitches {config.min_pitch}-{config.max_pitch})")
    print(f"Time shift tokens:  {time_shift_tokens:4d} (bins)")
    print(f"Velocity tokens:    {velocity_tokens:4d} (bins)")
    print(f"Control tokens:     {control_tokens:4d}")
    print(f"Tempo tokens:       {tempo_tokens:4d} (bins)")
    print(f"Instrument tokens:  {instrument_tokens:4d}")
    print("-" * 50)
    print(f"TOTAL CALCULATED:   {total_calculated:4d}")
    print(f"ACTUAL vocab_size:  {config.vocab_size:4d}")
    print()
    
    if total_calculated != config.vocab_size:
        print("❌ WARNING: Calculated size doesn't match actual!")
    else:
        print("✅ Calculation matches actual vocabulary size")
    
    print("\n📊 Breakdown by category:")
    print(f"- Special:     {special_tokens/config.vocab_size*100:5.1f}%")
    print(f"- Note events: {total_note_tokens/config.vocab_size*100:5.1f}%")
    print(f"- Time shifts: {time_shift_tokens/config.vocab_size*100:5.1f}%")
    print(f"- Others:      {(velocity_tokens+control_tokens+tempo_tokens+instrument_tokens)/config.vocab_size*100:5.1f}%")
    
    return config.vocab_size

if __name__ == "__main__":
    vocab_size = calculate_vocabulary_breakdown()
    print(f"\n🎯 Use vocab_size={vocab_size} in all model configurations!")