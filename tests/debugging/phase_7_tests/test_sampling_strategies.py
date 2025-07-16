#!/usr/bin/env python3
"""
Test different sampling strategies to see if we can get better results
from the trained model.
"""

import torch
import numpy as np
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, GenerationConfig, SamplingStrategy

print("=== Testing Different Sampling Strategies ===\n")

# Load the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint = torch.load('outputs/checkpoints/best_model.pt', map_location=device)

# Infer model config
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

print(f"Model config: vocab_size={vocab_size}, d_model={d_model}, n_layers={n_layers}")

# Create model
model = MusicTransformerVAEGAN(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Create sampler
sampler = MusicSampler(model, device)

# Test different strategies
strategies = [
    ("Greedy", SamplingStrategy.GREEDY, {"temperature": 1.0}),
    ("High Temperature", SamplingStrategy.TEMPERATURE, {"temperature": 2.0}),
    ("Medium Temperature", SamplingStrategy.TEMPERATURE, {"temperature": 1.0}),
    ("Low Temperature", SamplingStrategy.TEMPERATURE, {"temperature": 0.5}),
    ("Top-k (k=10)", SamplingStrategy.TOP_K, {"temperature": 1.0, "top_k": 10}),
    ("Top-k (k=50)", SamplingStrategy.TOP_K, {"temperature": 1.0, "top_k": 50}),
    ("Top-p (p=0.9)", SamplingStrategy.TOP_P, {"temperature": 1.0, "top_p": 0.9}),
    ("Top-p (p=0.5)", SamplingStrategy.TOP_P, {"temperature": 1.0, "top_p": 0.5}),
]

for name, strategy, params in strategies:
    print(f"\nTesting {name}:")
    
    config = GenerationConfig(
        strategy=strategy,
        max_length=20,  # Short test
        use_musical_constraints=False,
        **params
    )
    
    try:
        # Generate short sequence
        output = sampler.generate(config=config)
        tokens = output[0].cpu().numpy()
        
        # Analyze output
        unique_tokens = np.unique(tokens)
        print(f"  Generated: {tokens.tolist()}")
        print(f"  Unique tokens: {len(unique_tokens)} ({unique_tokens.tolist()[:10]}...)")
        print(f"  Token range: {tokens.min()}-{tokens.max()}")
        
        # Check if stuck in loop
        if len(tokens) > 5 and len(unique_tokens) == 1:
            print(f"  ❌ STUCK: Repeating token {tokens[0]}")
        elif len(tokens) > 10 and len(unique_tokens) < 3:
            print(f"  ⚠️  LIMITED: Very few unique tokens")
        else:
            print(f"  ✅ DIVERSE: Good token variety")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

# Test if the model can produce basic logits without getting stuck
print(f"\n=== Direct Model Test ===")
with torch.no_grad():
    # Test with START token
    start_seq = torch.tensor([[1]], device=device)  # START token
    logits = model(start_seq)
    
    if hasattr(logits, 'logits'):
        logits = logits.logits
    
    print(f"Model output shape: {logits.shape}")
    print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
    
    # Check if logits are reasonable
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_tokens = torch.topk(probs, 10)
    
    print(f"Top 10 token probabilities:")
    for i in range(10):
        token_id = top_tokens.indices[i].item()
        prob = top_tokens.values[i].item()
        print(f"  Token {token_id}: {prob:.4f}")
        
    # Check if probability is too concentrated
    max_prob = probs.max().item()
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    print(f"Max probability: {max_prob:.4f}")
    print(f"Entropy: {entropy:.2f} (higher is more diverse)")
    
    if max_prob > 0.9:
        print("⚠️  Model is too confident - may cause repetition")
    elif entropy < 2.0:
        print("⚠️  Low entropy - limited diversity")
    else:
        print("✅ Reasonable probability distribution")

print(f"\n=== Diagnosis ===")
print("If the model keeps generating the same token, this indicates:")
print("1. Model collapse during training")
print("2. Need for better training data or longer training")
print("3. Possible gradient/optimization issues during training")
print("4. Model may need retraining with proper musical sequences")