#!/usr/bin/env python3
"""
Test script to verify that inplace operation errors are fixed.
"""

import torch
import torch.nn as nn
from src.training.core.losses import ComprehensiveLossFramework
from src.models.encoder import EnhancedMusicEncoder
from src.models.discriminator import MultiScaleDiscriminator
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN

def test_loss_computation():
    """Test that loss computation doesn't cause inplace operation errors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    seq_len = 512
    vocab_size = 774
    d_model = 512
    latent_dim = 128
    
    # Initialize loss framework
    loss_framework = ComprehensiveLossFramework(
        vocab_size=vocab_size,
        musical_weighting=True,
        use_automatic_balancing=True
    ).to(device)
    
    # Create dummy data
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    reconstruction_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True).to(device)
    
    # Create encoder output
    encoder_output = {
        'mu': torch.randn(batch_size, latent_dim, requires_grad=True).to(device),
        'logvar': torch.randn(batch_size, latent_dim, requires_grad=True).to(device),
        'z': torch.randn(batch_size, latent_dim, requires_grad=True).to(device),
        'kl_loss': torch.randn(batch_size, latent_dim, requires_grad=True).to(device),
        'latent_info': {
            'active_dims': 100
        }
    }
    
    # Create discriminator outputs
    real_discriminator_output = {
        'combined_logits': torch.randn(batch_size, 1, requires_grad=True).to(device),
        'features': {
            'local': torch.randn(batch_size, 256, requires_grad=True).to(device),
            'phrase': torch.randn(batch_size, 256, requires_grad=True).to(device),
            'global': torch.randn(batch_size, 256, requires_grad=True).to(device)
        }
    }
    
    fake_discriminator_output = {
        'combined_logits': torch.randn(batch_size, 1, requires_grad=True).to(device),
        'features': {
            'local': torch.randn(batch_size, 256, requires_grad=True).to(device),
            'phrase': torch.randn(batch_size, 256, requires_grad=True).to(device),
            'global': torch.randn(batch_size, 256, requires_grad=True).to(device)
        }
    }
    
    # Create dummy discriminator
    discriminator = nn.Linear(1, 1).to(device)  # Dummy model
    
    # Generated tokens
    generated_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Test forward pass
    try:
        losses = loss_framework(
            reconstruction_logits=reconstruction_logits,
            target_tokens=target_tokens,
            encoder_output=encoder_output,
            real_discriminator_output=real_discriminator_output,
            fake_discriminator_output=fake_discriminator_output,
            discriminator=discriminator,
            generated_tokens=generated_tokens,
            mask=None
        )
        
        # Test backward pass
        total_loss = losses['total_loss']
        total_loss.backward()
        
        print("✓ Loss computation successful!")
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Loss shape: {total_loss.shape}")
        
        # Test epoch step
        loss_framework.step_epoch(avg_kl=encoder_output['kl_loss'].mean().item())
        print("✓ Epoch step successful!")
        
        return True
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"✗ Inplace operation error still present: {e}")
            return False
        else:
            raise e

if __name__ == "__main__":
    print("Testing inplace operation fixes...")
    success = test_loss_computation()
    
    if success:
        print("\n✓ All tests passed! Inplace operations have been fixed.")
    else:
        print("\n✗ Tests failed. There are still inplace operation issues.")