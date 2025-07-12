#!/usr/bin/env python3
"""Test without gradient computation to see if the issue is in the forward pass."""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN

def test_no_grad():
    """Test forward pass without gradient computation."""
    print("Testing forward pass without gradients...")
    
    # Load config
    config = load_config("configs/training_configs/quick_test.yaml")
    
    # Create model
    model_config = config["model"]
    model = MusicTransformerVAEGAN(
        vocab_size=model_config.get("vocab_size", 774),
        d_model=model_config.get("hidden_dim", 512),
        n_layers=model_config.get("num_layers", 8),
        n_heads=model_config.get("num_heads", 8),
        latent_dim=model_config.get("latent_dim", 128),
        max_sequence_length=model_config.get("max_sequence_length", 2048),
        mode=model_config.get("mode", "vae_gan"),
        encoder_layers=model_config.get("encoder_layers", 6),
        decoder_layers=model_config.get("decoder_layers", 6),
        discriminator_layers=model_config.get("discriminator_layers", 4),
        beta=model_config.get("beta", 1.0),
        dropout=model_config.get("dropout", 0.1),
        attention_type=model_config.get("attention_type", "hierarchical"),
        local_window_size=model_config.get("local_window_size", 256),
        global_window_size=model_config.get("global_window_size", 64),
        sliding_window_size=model_config.get("sliding_window_size", 512)
    )
    
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, 774, (batch_size, seq_len))
    
    print(f"Input shape: {tokens.shape}")
    print(f"Model: {model.__class__.__name__}")
    
    # Test without gradients
    with torch.no_grad():
        try:
            output = model(tokens)
            print("Forward pass successful!")
            print(f"Output type: {type(output)}")
            if isinstance(output, torch.Tensor):
                print(f"Output shape: {output.shape}")
            elif isinstance(output, dict):
                print(f"Output keys: {list(output.keys())}")
            
            # Test loss computation
            losses = model.compute_loss(tokens)
            print(f"Loss computation successful!")
            print(f"Loss keys: {list(losses.keys())}")
            for key, value in losses.items():
                print(f"  {key}: {value.item() if torch.is_tensor(value) else value}")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_no_grad()