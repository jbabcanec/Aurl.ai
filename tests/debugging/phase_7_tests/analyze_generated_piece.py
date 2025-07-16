#!/usr/bin/env python3
"""
Analyze the generated music piece
"""

import torch
import numpy as np

# Load and analyze the generated tokens
tokens = torch.load('natural_piano_piece/aurl_natural_generation.tokens')
print(f"=== Generated Music Analysis ===")
print(f"File: natural_piano_piece/aurl_natural_generation.mid")
print(f"Tokens file: natural_piano_piece/aurl_natural_generation.tokens")
print()

if tokens.dim() > 1:
    tokens = tokens.flatten()

tokens_list = tokens.tolist()
print(f"Total tokens: {len(tokens_list)}")
print(f"Token range: {min(tokens_list)} to {max(tokens_list)}")

# Count unique tokens
unique_tokens = len(set(tokens_list))
print(f"Unique tokens: {unique_tokens} ({unique_tokens/len(tokens_list)*100:.1f}% diversity)")

# Show first portion
print(f"First 30 tokens: {tokens_list[:30]}")

# Token type analysis
special_tokens = sum(1 for t in tokens_list if 0 <= t <= 12)
note_on_tokens = sum(1 for t in tokens_list if 13 <= t <= 140)
note_off_tokens = sum(1 for t in tokens_list if 141 <= t <= 268)
time_shift_tokens = sum(1 for t in tokens_list if 269 <= t <= 368)
velocity_tokens = sum(1 for t in tokens_list if 369 <= t <= 496)
other_tokens = sum(1 for t in tokens_list if t > 496)

print(f"\nToken Type Distribution:")
print(f"  Special tokens (0-12): {special_tokens}")
print(f"  Note ON (13-140): {note_on_tokens}")
print(f"  Note OFF (141-268): {note_off_tokens}")
print(f"  Time shifts (269-368): {time_shift_tokens}")
print(f"  Velocity (369-496): {velocity_tokens}")
print(f"  Other (>496): {other_tokens}")

# Check for repetition
def find_repetition(seq, max_len=20):
    for length in range(2, min(max_len, len(seq)//2)):
        for start in range(len(seq) - 2*length):
            if seq[start:start+length] == seq[start+length:start+2*length]:
                return length, start
    return None, None

rep_len, rep_start = find_repetition(tokens_list)
if rep_len:
    print(f"\nRepetition detected: {rep_len} tokens repeating at position {rep_start}")
    print(f"Repeating pattern: {tokens_list[rep_start:rep_start+rep_len]}")
else:
    print(f"\nNo immediate repetition detected (good!)")

print(f"\n=== Assessment ===")
if unique_tokens < 10:
    print("⚠️  Very low diversity - model still struggling")
elif unique_tokens < 20:
    print("🔶 Moderate diversity - some improvement")
else:
    print("✅ Good diversity - high temperature working")

if note_on_tokens > 0 and note_off_tokens > 0:
    print("✅ Contains note events (should produce sound)")
else:
    print("❌ Missing note events (may be silent)")

if time_shift_tokens > 0:
    print("✅ Contains timing information")
else:
    print("⚠️  No timing tokens")

print(f"\nFile Location: natural_piano_piece/aurl_natural_generation.mid")
print(f"You can play this MIDI file in any music software!")