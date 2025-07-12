#!/usr/bin/env python3
"""
Final targeted bug hunt based on research insights.
The issue is likely in gradient accumulation or loss framework.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework

def test_gradient_accumulation_scenario():
    """Test gradient accumulation which might cause the version mismatch."""
    print("🎯 TESTING GRADIENT ACCUMULATION SCENARIO")
    print("=" * 50)
    
    # Create model
    config = load_config("configs/training_configs/quick_test.yaml")
    model_config = config["model"]
    
    model = MusicTransformerVAEGAN(
        vocab_size=774, d_model=256, n_layers=4, n_heads=4,
        latent_dim=66, max_sequence_length=512, mode="vae_gan",
        encoder_layers=3, decoder_layers=3, discriminator_layers=3,
        beta=1.0, dropout=0.1, attention_type="hierarchical",
        local_window_size=64, global_window_size=32, sliding_window_size=128
    )
    model.train()
    
    # Create loss framework (this is what's different from synthetic test!)
    loss_framework = ComprehensiveLossFramework(
        vocab_size=774,
        musical_weighting=True,
        perceptual_emphasis=2.0,
        beta_start=0.0,
        beta_end=1.0,
        kl_schedule="linear",
        warmup_epochs=10,
        adversarial_weight=1.0,
        feature_matching_weight=10.0,
        enable_musical_constraints=True,
        constraint_weight=0.1,
        use_automatic_balancing=True
    )
    
    # Test gradient accumulation scenario
    # This simulates what happens in real training
    accumulation_steps = 16  # This would create 64 total samples (4 * 16)
    
    total_loss = None
    
    print(f"Testing {accumulation_steps} accumulation steps...")
    
    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        for step in range(accumulation_steps):
            print(f"Step {step + 1}/{accumulation_steps}")
            
            # Create mini-batch
            tokens = torch.randint(0, 774, (4, 512))  # Config batch_size = 4
            
            # Forward pass
            output = model(tokens, return_latent=True)
            
            # Compute losses using the framework
            losses = loss_framework(
                reconstruction_logits=output['logits'],
                target_tokens=tokens,
                encoder_output={
                    'mu': output['mu'],
                    'logvar': output['logvar'],
                    'z': output['z'],
                    'kl_loss': output['kl_loss']
                },
                real_discriminator_output={'logits': torch.randn(4, 1)},
                fake_discriminator_output={'logits': torch.randn(4, 1)},
                discriminator=model.discriminator,
                generated_tokens=tokens
            )
            
            step_loss = losses['total_loss'] / accumulation_steps
            
            if total_loss is None:
                total_loss = step_loss
            else:
                # This is where the issue might be - accumulating losses
                total_loss = total_loss + step_loss
            
            print(f"  Step loss: {step_loss.item():.4f}")
            
        print(f"Total accumulated loss: {total_loss.item():.4f}")
        
        # This is where the error should occur
        print("🔥 CRITICAL: Calling backward on accumulated loss...")
        total_loss.backward()
        
        print("✅ Gradient accumulation test PASSED!")
        
    except Exception as e:
        print(f"❌ GRADIENT ACCUMULATION TEST FAILED: {e}")
        
        # Check if this is our target error
        error_str = str(e)
        if "version 5; expected version 4" in error_str or "AsStridedBackward0" in error_str:
            print("🎯 FOUND THE EXACT ERROR!")
            print("🚨 The issue is in gradient accumulation or loss framework!")
        
        import traceback
        traceback.print_exc()
        
        return False
    
    finally:
        torch.autograd.set_detect_anomaly(False)
    
    return True

def test_loss_framework_directly():
    """Test the loss framework directly for inplace operations."""
    print("\n🔬 TESTING LOSS FRAMEWORK DIRECTLY")
    print("=" * 40)
    
    try:
        # Create loss framework
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            musical_weighting=True,
            perceptual_emphasis=2.0
        )
        
        # Create synthetic data
        batch_size = 64  # Use the exact batch size from the error
        seq_len = 512
        vocab_size = 774
        latent_dim = 66
        
        # Mock model outputs
        reconstruction_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        encoder_output = {
            'mu': torch.randn(batch_size, latent_dim, requires_grad=True),
            'logvar': torch.randn(batch_size, latent_dim, requires_grad=True),
            'z': torch.randn(batch_size, latent_dim, requires_grad=True),
            'kl_loss': torch.randn(1, requires_grad=True)
        }
        
        # Mock discriminator outputs
        real_discriminator_output = {'logits': torch.randn(batch_size, 1, requires_grad=True)}
        fake_discriminator_output = {'logits': torch.randn(batch_size, 1, requires_grad=True)}
        
        # Mock discriminator (simple)
        discriminator = nn.Linear(vocab_size, 1)
        generated_tokens = target_tokens
        
        print("Testing loss framework forward pass...")
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # This is the key test - the loss framework might have inplace operations
        losses = loss_framework(
            reconstruction_logits=reconstruction_logits,
            target_tokens=target_tokens,
            encoder_output=encoder_output,
            real_discriminator_output=real_discriminator_output,
            fake_discriminator_output=fake_discriminator_output,
            discriminator=discriminator,
            generated_tokens=generated_tokens
        )
        
        total_loss = losses['total_loss']
        print(f"Loss framework forward successful: {total_loss.item():.4f}")
        
        # Test backward pass
        print("🔥 Testing loss framework backward pass...")
        total_loss.backward()
        
        print("✅ Loss framework test PASSED!")
        
    except Exception as e:
        print(f"❌ LOSS FRAMEWORK TEST FAILED: {e}")
        
        # Check if this is our target error
        error_str = str(e)
        if "version" in error_str and "expected version" in error_str:
            print("🎯 FOUND THE ERROR IN LOSS FRAMEWORK!")
        
        import traceback
        traceback.print_exc()
        
        return False
    
    finally:
        torch.autograd.set_detect_anomaly(False)
    
    return True

def test_specific_tensor_operations():
    """Test specific operations that might create [64, 1] tensors."""
    print("\n🧪 TESTING SPECIFIC TENSOR OPERATIONS")
    print("=" * 40)
    
    # The error mentions [torch.FloatTensor [64, 1]]
    # Let's test operations that might create such tensors
    
    print("Testing operations that create [64, 1] tensors...")
    
    try:
        batch_size = 64
        
        # Test 1: Discriminator-like operations
        x = torch.randn(batch_size, 512, requires_grad=True)
        weight = torch.randn(512, 1, requires_grad=True)
        
        # This creates [64, 1] tensor
        output = torch.mm(x, weight)  # [64, 512] @ [512, 1] = [64, 1]
        print(f"Created tensor: {output.shape}")
        
        # Test inplace operations on this tensor
        print("Testing inplace operations on [64, 1] tensor...")
        
        # These might cause issues:
        output_copy = output.clone()
        loss1 = output_copy.sum()
        
        # Try modifying the original output (this might cause version mismatch)
        output = output + 0.1  # Non-inplace
        loss2 = output.sum()
        
        total_loss = loss1 + loss2
        
        print("Testing backward pass...")
        torch.autograd.set_detect_anomaly(True)
        total_loss.backward()
        
        print("✅ Tensor operations test PASSED!")
        
    except Exception as e:
        print(f"❌ TENSOR OPERATIONS TEST FAILED: {e}")
        
        if "[64, 1]" in str(e):
            print("🎯 FOUND THE [64, 1] TENSOR ISSUE!")
        
        import traceback
        traceback.print_exc()
        
        return False
    
    finally:
        torch.autograd.set_detect_anomaly(False)
    
    return True

if __name__ == "__main__":
    print("🚀 FINAL BUG HUNT")
    print("=" * 20)
    print("Based on research: The issue is likely in gradient accumulation")
    print("or the loss framework causing version mismatches.")
    print()
    
    # Test 1: Gradient accumulation scenario
    success1 = test_gradient_accumulation_scenario()
    
    if success1:
        # Test 2: Loss framework directly  
        success2 = test_loss_framework_directly()
        
        if success2:
            # Test 3: Specific tensor operations
            success3 = test_specific_tensor_operations()
    
    print("\n🎯 Final bug hunt completed!")
    print("If all tests passed, the issue might be in real data processing")
    print("or a combination of factors not captured in isolation.")