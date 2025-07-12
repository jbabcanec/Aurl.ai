#!/usr/bin/env python3
"""Simple debug script to isolate the inplace operation issue."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN

def test_forward_pass():
    """Test a simple forward pass to isolate the issue."""
    print("Testing forward pass...")
    
    # Create model with minimal parameters
    model = MusicTransformerVAEGAN(
        vocab_size=774,
        d_model=256,
        n_layers=2,  # Minimal layers
        n_heads=4,
        latent_dim=33,  # Must be divisible by 3 for hierarchical mode
        max_sequence_length=64,  # Short sequences
        mode="vae_gan",
        discriminator_layers=2,
        beta=1.0,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1,
        attention_type="hierarchical",
        local_window_size=16,
        global_window_size=8,
        sliding_window_size=32
    )
    
    model.eval()
    
    # Create minimal dummy input
    batch_size = 4
    seq_len = 16
    tokens = torch.randint(0, 774, (batch_size, seq_len))
    
    print(f"Input shape: {tokens.shape}")
    print(f"Model: {model.__class__.__name__}")
    
    # Enable anomaly detection
    with torch.autograd.detect_anomaly():
        try:
            # Forward pass
            output = model(tokens)
            print("Forward pass successful!")
            print(f"Output keys: {list(output.keys()) if isinstance(output, dict) else 'Not a dict'}")
            
            # Try to compute a simple loss
            if isinstance(output, dict) and 'reconstruction' in output:
                reconstruction = output['reconstruction']
                print(f"Reconstruction shape: {reconstruction.shape}")
                
                # Simple cross-entropy loss
                loss = nn.CrossEntropyLoss()(
                    reconstruction.view(-1, 774),
                    tokens.view(-1)
                )
                print(f"Loss computed: {loss.item()}")
                
                # Try backward pass
                loss.backward()
                print("Backward pass successful!")
            else:
                print("No reconstruction output found")
                
        except Exception as e:
            print(f"Error during forward/backward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_forward_pass()