#!/usr/bin/env python3
"""
Test with synthetic data to isolate the exact training issue.
This bypasses data loading problems and tests the actual training step.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN

def test_training_with_synthetic_data():
    """Test training with synthetic data to isolate the issue."""
    print("🧪 TESTING WITH SYNTHETIC DATA")
    print("=" * 50)
    
    # Create the exact model configuration from the failing training
    config = load_config("configs/training_configs/quick_test.yaml")
    
    model_config = config["model"]
    model = MusicTransformerVAEGAN(
        vocab_size=model_config.get("vocab_size", 774),
        d_model=model_config.get("hidden_dim", 256),  # From config
        n_layers=model_config.get("num_layers", 4),   # From config
        n_heads=model_config.get("num_heads", 4),     # From config
        latent_dim=model_config.get("latent_dim", 66), # From config
        max_sequence_length=model_config.get("max_sequence_length", 512),
        mode=model_config.get("mode", "vae_gan"),
        encoder_layers=model_config.get("encoder_layers", 3),
        decoder_layers=model_config.get("decoder_layers", 3),
        discriminator_layers=model_config.get("discriminator_layers", 3),
        beta=model_config.get("beta", 1.0),
        dropout=model_config.get("dropout", 0.1),
        attention_type=model_config.get("attention_type", "hierarchical"),
        local_window_size=model_config.get("local_window_size", 64),
        global_window_size=model_config.get("global_window_size", 32),
        sliding_window_size=model_config.get("sliding_window_size", 128)
    )
    
    model.train()
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create synthetic data with EXACT batch size from the error message
    batch_size = 64  # This is the batch_size from the error "[64, 1]"
    seq_len = 512    # From config
    vocab_size = 774 # From config
    
    # Create synthetic tokens
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Synthetic data shape: {tokens.shape}")
    
    # Test different scenarios
    scenarios = [
        "model_forward_only",
        "model_loss_computation", 
        "model_backward_pass"
    ]
    
    for scenario in scenarios:
        print(f"\n🔬 Testing scenario: {scenario}")
        
        try:
            if scenario == "model_forward_only":
                # Just forward pass
                with torch.no_grad():
                    output = model(tokens, return_latent=True)
                    print(f"✅ Forward pass successful: {list(output.keys())}")
            
            elif scenario == "model_loss_computation":
                # Forward + loss computation (no gradients)
                with torch.no_grad():
                    losses = model.compute_loss(tokens)
                    print(f"✅ Loss computation successful: {losses['total_loss'].item():.4f}")
            
            elif scenario == "model_backward_pass":
                # Full forward + backward pass (THIS IS WHERE THE ERROR SHOULD OCCUR)
                print("🔥 CRITICAL: Testing full backward pass...")
                
                # Clear any previous gradients
                model.zero_grad()
                
                # Forward pass
                losses = model.compute_loss(tokens)
                total_loss = losses['total_loss']
                
                print(f"Total loss: {total_loss.item():.4f}")
                print(f"Loss shape: {total_loss.shape}")
                print(f"Loss requires_grad: {total_loss.requires_grad}")
                
                # This should trigger the error
                total_loss.backward()
                
                print("✅ Backward pass successful!")
                
                # Check gradients
                grad_count = sum(1 for p in model.parameters() if p.grad is not None)
                total_params = sum(1 for p in model.parameters())
                print(f"Gradients computed: {grad_count}/{total_params}")
        
        except Exception as e:
            print(f"❌ {scenario} FAILED: {e}")
            print(f"Error type: {type(e).__name__}")
            
            # Check if this is THE error we're looking for
            error_str = str(e)
            if "[64, 1]" in error_str and "AsStridedBackward0" in error_str:
                print("🎯 FOUND THE EXACT ERROR!")
                print("🔍 This is the same error as in the training pipeline!")
                
                # Enable anomaly detection for detailed traceback
                print("\n🔍 Re-running with anomaly detection...")
                torch.autograd.set_detect_anomaly(True)
                
                try:
                    model.zero_grad()
                    losses = model.compute_loss(tokens)
                    total_loss = losses['total_loss']
                    total_loss.backward()
                except Exception as e2:
                    print(f"🚨 DETAILED ERROR: {e2}")
                    import traceback
                    traceback.print_exc()
                finally:
                    torch.autograd.set_detect_anomaly(False)
                
                return False  # Found the error
            
            elif "version" in error_str and "expected version" in error_str:
                print("🎯 FOUND A VERSION MISMATCH ERROR!")
                return False
    
    print("\n✅ All scenarios passed!")
    return True

def test_step_by_step_debugging():
    """Step-by-step debugging to isolate the exact operation."""
    print("\n🔍 STEP-BY-STEP DEBUGGING")
    print("=" * 40)
    
    # Load config and create model
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
    
    # Synthetic data
    tokens = torch.randint(0, 774, (64, 512))
    
    print("🔬 Step 1: Test encoder only...")
    try:
        embeddings = model.embedding(tokens)
        encoder_output = model.encoder(embeddings)
        print(f"✅ Encoder successful: {list(encoder_output.keys())}")
        
        # Check for [64, 1] tensors in encoder output
        for key, value in encoder_output.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
                if len(value.shape) == 2 and value.shape[1] == 1:
                    print(f"  🎯 FOUND [64, 1] TENSOR: {key}")
    except Exception as e:
        print(f"❌ Encoder failed: {e}")
        return
    
    print("🔬 Step 2: Test decoder only...")
    try:
        z = encoder_output['z']
        decoder_output = model.decoder(z, embeddings)
        print(f"✅ Decoder successful: {decoder_output.shape}")
    except Exception as e:
        print(f"❌ Decoder failed: {e}")
        return
    
    print("🔬 Step 3: Test full forward pass...")
    try:
        full_output = model(tokens, return_latent=True)
        print(f"✅ Full forward successful: {list(full_output.keys())}")
    except Exception as e:
        print(f"❌ Full forward failed: {e}")
        return
    
    print("🔬 Step 4: Test loss computation...")
    try:
        losses = model.compute_loss(tokens)
        total_loss = losses['total_loss']
        print(f"✅ Loss computation successful: {total_loss.item():.4f}")
        
        # Check loss tensor properties
        print(f"Loss tensor details:")
        print(f"  Shape: {total_loss.shape}")
        print(f"  Requires grad: {total_loss.requires_grad}")
        print(f"  Grad fn: {total_loss.grad_fn}")
        print(f"  Is leaf: {total_loss.is_leaf}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return
    
    print("🔬 Step 5: Test backward pass (THE CRITICAL STEP)...")
    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        model.zero_grad()
        losses = model.compute_loss(tokens)
        total_loss = losses['total_loss']
        
        print("🔥 Calling backward()...")
        total_loss.backward()
        
        print("✅ Backward pass successful!")
        
    except Exception as e:
        print(f"❌ BACKWARD PASS FAILED: {e}")
        print("🚨 THIS IS THE ERROR WE'VE BEEN HUNTING!")
        
        import traceback
        traceback.print_exc()
        
    finally:
        torch.autograd.set_detect_anomaly(False)

if __name__ == "__main__":
    print("🚀 SYNTHETIC DATA TRAINING TEST")
    print("=" * 35)
    
    # Test with synthetic data
    success = test_training_with_synthetic_data()
    
    if not success:
        # Do step-by-step debugging
        test_step_by_step_debugging()
    
    print("\n🎯 Testing completed!")