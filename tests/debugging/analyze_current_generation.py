#!/usr/bin/env python3
"""
Analyze Current Generation Output

Examines the tokens and MIDI output from the current model generation
to demonstrate the exact issues we identified and validate our fixes.
"""

import torch
import numpy as np
from pathlib import Path

def load_and_analyze_tokens():
    """Load and analyze the generated tokens."""
    token_file = "tests/outputs/current_gen/current_test.tokens"
    
    if Path(token_file).exists():
        try:
            tokens = torch.load(token_file, map_location='cpu')
            print(f"Generated token sequence: {tokens}")
            print(f"Token sequence shape: {tokens.shape}")
            print(f"Unique tokens: {torch.unique(tokens)}")
            print(f"Token frequency:")
            unique, counts = torch.unique(tokens, return_counts=True)
            for token, count in zip(unique, counts):
                print(f"  Token {token}: {count} times")
            
            return tokens
        except Exception as e:
            print(f"Error loading tokens: {e}")
            return None
    else:
        print(f"Token file not found: {token_file}")
        return None

def analyze_token_repetition(tokens):
    """Analyze token repetition patterns."""
    if tokens is None:
        return
    
    tokens_np = tokens.numpy()
    
    # Find the most repeated token
    most_common_token = tokens_np[0]
    max_consecutive = 1
    current_consecutive = 1
    
    for i in range(1, len(tokens_np)):
        if tokens_np[i] == tokens_np[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    
    print(f"\nRepetition Analysis:")
    print(f"Most consecutive identical tokens: {max_consecutive}")
    print(f"First 20 tokens: {tokens_np[:20]}")
    print(f"Last 20 tokens: {tokens_np[-20:]}")
    
    # Check if this is the problematic token 139 we found before
    tokens_flat = tokens_np.flatten() if len(tokens_np.shape) > 1 else tokens_np
    unique_tokens = set(tokens_flat)
    
    if len(unique_tokens) == 1:
        print(f"🚨 MODEL COLLAPSE: Only generating token {list(unique_tokens)[0]} repeatedly!")
    elif len(unique_tokens) < 5:
        print(f"⚠️  Very limited diversity: Only {len(unique_tokens)} unique tokens")
    else:
        print(f"Token diversity: {len(unique_tokens)} unique tokens")

def analyze_midi_output():
    """Check if MIDI file contains actual notes."""
    midi_file = "tests/outputs/current_gen/current_test.mid"
    
    if Path(midi_file).exists():
        # Try to read with mido for basic analysis
        try:
            import mido
            mid = mido.MidiFile(midi_file)
            
            note_count = 0
            note_on_msgs = []
            note_off_msgs = []
            
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_count += 1
                        note_on_msgs.append(msg)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        note_off_msgs.append(msg)
            
            print(f"\nMIDI Analysis:")
            print(f"Note ON messages: {len(note_on_msgs)}")
            print(f"Note OFF messages: {len(note_off_msgs)}")
            print(f"Total audible notes: {note_count}")
            
            if note_count == 0:
                print("🚨 CONFIRMED: MIDI file contains no audible notes!")
                print("This demonstrates the exact issue we identified and fixed.")
            else:
                print(f"✅ MIDI contains {note_count} audible notes")
                
        except ImportError:
            print("mido not available for MIDI analysis")
        except Exception as e:
            print(f"Error analyzing MIDI: {e}")
    else:
        print(f"MIDI file not found: {midi_file}")

def demonstrate_issue():
    """Demonstrate the current model issues."""
    print("🎵 Current Model Generation Analysis")
    print("=" * 50)
    print("This analysis demonstrates the exact issues we identified:")
    print("1. Model collapse (repetitive token generation)")
    print("2. Tokens that don't translate to usable MIDI")
    print("3. Empty MIDI files despite 'successful' generation")
    print()
    
    tokens = load_and_analyze_tokens()
    analyze_token_repetition(tokens)
    analyze_midi_output()
    
    print("\n🔍 Summary:")
    print("This output confirms why we needed Phase 5.1 enhancements:")
    print("- Enhanced grammar loss functions to prevent token collapse")
    print("- Parameter quality validation for velocity and timing")
    print("- Musical structure validation for proper note sequences")
    print("\n✅ Our enhanced musical grammar system is designed to fix these exact issues!")

if __name__ == "__main__":
    demonstrate_issue()