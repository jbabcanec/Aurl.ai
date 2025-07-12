#!/usr/bin/env python3
"""Debug script to isolate the inplace operation issue."""

import torch
import torch.nn as nn
import numpy as np
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.config import load_config

def test_forward_pass():
    """Test a simple forward pass to isolate the issue."""
    print("Testing forward pass...")
    
    # Load config
    config = load_config("configs/training_configs/quick_test.yaml")
    
    # Create model
    model = MusicTransformerVAEGAN(config)
    model.eval()
    
    # Create dummy input
    batch_size = 4  # Use smaller batch size
    seq_len = 32
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {tokens.shape}")
    
    # Enable anomaly detection to get better error messages
    with torch.autograd.detect_anomaly():
        try:
            # Forward pass
            output = model(tokens)
            print("Forward pass successful!")
            
            # Try to compute a simple loss
            if isinstance(output, dict):
                if 'reconstruction' in output:
                    loss = nn.CrossEntropyLoss()(
                        output['reconstruction'].view(-1, model.vocab_size),
                        tokens.view(-1)
                    )
                    print(f"Loss computed: {loss.item()}")
                    
                    # Try backward pass
                    loss.backward()
                    print("Backward pass successful!")
                    
        except Exception as e:
            print(f"Error during forward/backward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_forward_pass()