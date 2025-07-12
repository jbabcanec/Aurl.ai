#!/usr/bin/env python3
"""
Minimal reproduction of the inplace operation bug.
Reproduces the exact issue from the training pipeline.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset

def create_exact_training_setup():
    """Create the exact same setup as the real training pipeline."""
    # Load config
    config = load_config("configs/training_configs/quick_test.yaml")
    
    # Create model with exact same parameters as training pipeline
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
        local_window_size=model_config.get("local_window_size", 64),
        global_window_size=model_config.get("global_window_size", 32),
        sliding_window_size=model_config.get("sliding_window_size", 128)
    )
    
    # Create dataset with minimal parameters
    dataset = LazyMidiDataset(
        data_dir="data/raw",
        sequence_length=512,
        overlap=64,
        cache_dir="data/cache"
    )
    
    return model, dataset, config

def test_exact_training_step():
    """Test the exact training step that fails."""
    print("🎯 REPRODUCING EXACT TRAINING FAILURE")
    print("=" * 50)
    
    # DON'T enable anomaly detection initially to match real training
    # torch.autograd.set_detect_anomaly(True)
    
    try:
        print("Creating model and dataset...")
        model, dataset, config = create_exact_training_setup()
        model.train()
        
        print(f"Model created: {model.__class__.__name__}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dataloader with exact same settings as training
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config["training"]["batch_size"],  # Use config batch size
            shuffle=False,
            num_workers=0
        )
        
        print("Getting first batch...")
        batch = next(iter(dataloader))
        tokens = batch['tokens']
        print(f"Batch shape: {tokens.shape}")
        print(f"Batch device: {tokens.device}")
        print(f"Batch dtype: {tokens.dtype}")
        
        # This is the exact sequence from the trainer
        print("Forward pass...")
        output = model(tokens, return_latent=True)
        print(f"Forward pass successful! Output keys: {list(output.keys())}")
        
        print("Loss computation...")
        losses = model.compute_loss(tokens)
        total_loss = losses['total_loss']
        print(f"Loss computation successful! Total loss: {total_loss.item()}")
        
        print(f"Loss tensor: {total_loss}")
        print(f"Loss requires_grad: {total_loss.requires_grad}")
        print(f"Loss grad_fn: {total_loss.grad_fn}")
        
        # This is where it fails in the real training
        print("🔥 CRITICAL: Calling loss.backward()...")
        total_loss.backward()
        
        print("✅ SUCCESS: Backward pass completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR REPRODUCED: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Now enable anomaly detection to get more details
        print("\n🔍 Re-running with anomaly detection...")
        torch.autograd.set_detect_anomaly(True)
        
        try:
            # Quick re-run to get detailed traceback
            model, dataset, config = create_exact_training_setup()
            model.train()
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
            batch = next(iter(dataloader))
            tokens = batch['tokens']
            
            output = model(tokens, return_latent=True)
            losses = model.compute_loss(tokens)
            total_loss = losses['total_loss']
            
            # This should give us the detailed traceback
            total_loss.backward()
            
        except Exception as e2:
            print(f"❌ Detailed error: {e2}")
            import traceback
            traceback.print_exc()
        
        finally:
            torch.autograd.set_detect_anomaly(False)
        
        return False

def test_with_different_batch_sizes():
    """Test with different batch sizes to see if it's batch-size related."""
    print("\n🔬 TESTING DIFFERENT BATCH SIZES")
    print("=" * 40)
    
    for batch_size in [1, 2, 4, 8]:
        print(f"\nTesting batch_size={batch_size}...")
        
        try:
            model, dataset, config = create_exact_training_setup()
            model.train()
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            batch = next(iter(dataloader))
            tokens = batch['tokens']
            
            output = model(tokens, return_latent=True)
            losses = model.compute_loss(tokens)
            total_loss = losses['total_loss']
            
            total_loss.backward()
            
            print(f"✅ batch_size={batch_size} SUCCESS")
            
        except Exception as e:
            print(f"❌ batch_size={batch_size} FAILED: {e}")
            
            # Check if it's specifically the [64, 1] tensor issue
            if "[64, 1]" in str(e) or "64, 1" in str(e):
                print(f"🎯 FOUND THE PATTERN! Error mentions [64, 1] with batch_size={batch_size}")
            
            if batch_size == 4:  # Same as training config
                print("🚨 This is the same batch size as the training config!")

if __name__ == "__main__":
    print("🚀 MINIMAL BUG REPRODUCTION")
    print("=" * 30)
    
    # Test exact training step
    success = test_exact_training_step()
    
    if not success:
        # Test with different batch sizes
        test_with_different_batch_sizes()
    
    print("\n🎯 Bug reproduction completed!")