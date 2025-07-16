#!/usr/bin/env python3
"""
Quick test with high temperature to generate diverse tokens
"""

import torch
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, GenerationConfig, SamplingStrategy
from src.generation.midi_export import MidiExporter, create_standard_config

print("=== Quick Diverse Generation Test ===\n")

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load('outputs/checkpoints/best_model.pt', map_location=device)

# Infer config
state_dict = checkpoint['model_state_dict']
vocab_size, d_model = state_dict['embedding.token_embedding.weight'].shape
n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if 'blocks.' in k]) + 1

model_config = {
    'vocab_size': vocab_size,
    'd_model': d_model,
    'n_heads': 8,
    'n_layers': n_layers,
    'max_sequence_length': 512,
    'mode': 'transformer'
}

# Create model
model = MusicTransformerVAEGAN(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Create sampler
sampler = MusicSampler(model, device)

# Generate with high temperature (known to work)
config = GenerationConfig(
    strategy=SamplingStrategy.TEMPERATURE,
    temperature=2.0,  # High temperature for diversity
    max_length=100,   # Moderate length
    use_musical_constraints=False  # No constraints
)

print("Generating with high temperature...")
output = sampler.generate(config=config)
tokens = output[0].cpu()

print(f"Generated {tokens.shape[1]} tokens")
print(f"Unique tokens: {len(torch.unique(tokens))}")
print(f"Sample tokens: {tokens[0, :30].tolist()}")

# Try to export to MIDI
try:
    config = create_standard_config()
    exporter = MidiExporter(config)
    
    stats = exporter.export_tokens_to_midi(
        tokens=tokens,
        output_path="outputs/generated/diverse_test.mid",
        title="High Temperature Test",
        tempo=120.0
    )
    
    print(f"MIDI export: {stats.total_notes} notes, {stats.total_duration:.2f}s")
    
    if stats.total_notes > 0:
        print("✅ SUCCESS: Generated actual notes!")
    else:
        print("⚠️  MIDI export created no notes")
        print("This indicates token-to-MIDI conversion issues")
        
except Exception as e:
    print(f"MIDI export error: {e}")

print(f"\n=== Analysis ===")
print("High temperature generation produces diverse tokens.")
print("The issue is likely in the token-to-MIDI conversion process.")
print("The model CAN generate varied sequences, but they may not be")
print("forming valid musical structures that convert to MIDI notes.")

# Token analysis
tokens_list = tokens[0].tolist()
print(f"\nToken distribution analysis:")
print(f"Range 0-12 (special): {sum(1 for t in tokens_list if 0 <= t <= 12)}")
print(f"Range 13-140 (note on): {sum(1 for t in tokens_list if 13 <= t <= 140)}")
print(f"Range 141-268 (note off): {sum(1 for t in tokens_list if 141 <= t <= 268)}")
print(f"Range 269-368 (time shift): {sum(1 for t in tokens_list if 269 <= t <= 368)}")
print(f"Range 369+ (velocity etc): {sum(1 for t in tokens_list if t >= 369)}")