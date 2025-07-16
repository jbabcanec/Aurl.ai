#!/usr/bin/env python3
"""
Analyze the generated tokens to understand what the model is producing
"""

import torch
import numpy as np

# Load the generated tokens
tokens = torch.load('natural_generation_test/generated_music_1752410816.tokens')
print(f"Generated token sequence shape: {tokens.shape}")
print(f"First 50 tokens: {tokens[:50].tolist()}")
print(f"Token range: {tokens.min().item()} - {tokens.max().item()}")

# Count token frequency to understand what's being generated
unique_tokens, counts = torch.unique(tokens, return_counts=True)
print(f"\nToken frequency analysis:")
print(f"Unique tokens: {len(unique_tokens)}")
print(f"Most frequent tokens:")
for i in range(min(10, len(unique_tokens))):
    idx = torch.argmax(counts)
    token = unique_tokens[idx].item()
    count = counts[idx].item()
    print(f"  Token {token}: {count} times ({count/len(tokens)*100:.1f}%)")
    counts[idx] = 0  # Remove for next iteration

# Check if we have note events
print(f"\nToken range analysis:")
print(f"Special tokens (0-12): {torch.sum((tokens >= 0) & (tokens <= 12)).item()}")
print(f"Note ON tokens (13-140): {torch.sum((tokens >= 13) & (tokens <= 140)).item()}")
print(f"Note OFF tokens (141-268): {torch.sum((tokens >= 141) & (tokens <= 268)).item()}")
print(f"Time shift tokens (269-368): {torch.sum((tokens >= 269) & (tokens <= 368)).item()}")
print(f"Velocity tokens (369-496): {torch.sum((tokens >= 369) & (tokens <= 496)).item()}")

# Check for repetitive patterns
print(f"\nSequence analysis:")
seq = tokens.tolist()
if len(seq) > 10:
    print(f"First 20 tokens: {seq[:20]}")
    print(f"Last 20 tokens: {seq[-20:]}")
    
    # Check for loops/repetition
    if seq[:10] == seq[10:20]:
        print("WARNING: Detected repetitive loop in first 20 tokens!")
    
    # Count how many times the sequence repeats itself
    max_repeat = 0
    for i in range(1, len(seq)//2):
        if seq[:i] == seq[i:2*i]:
            max_repeat = i
    
    if max_repeat > 0:
        print(f"Detected repeating pattern of length {max_repeat}")
        print(f"Pattern: {seq[:max_repeat]}")