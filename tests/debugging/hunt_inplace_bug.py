#!/usr/bin/env python3
"""
Systematic hunt for the inplace operation bug.
Based on research: we need to find [64, 1] tensor modifications.
"""

import torch
import torch.nn as nn
import sys
import traceback
from pathlib import Path
import functools

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework
from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.data.dataset import LazyMidiDataset

# Global tracker for tensor operations
TENSOR_TRACKER = {}
OPERATION_LOG = []

def track_tensor_version(tensor, name, operation=""):
    """Track tensor versions to detect in-place modifications."""
    if torch.is_tensor(tensor) and tensor.requires_grad:
        tensor_id = id(tensor)
        current_version = tensor._version
        
        if tensor_id in TENSOR_TRACKER:
            old_version = TENSOR_TRACKER[tensor_id]['version']
            if current_version != old_version:
                log_msg = f"🔍 TENSOR VERSION CHANGE: {name} {tensor.shape} v{old_version}→v{current_version} ({operation})"
                OPERATION_LOG.append(log_msg)
                print(log_msg)
                
                # If this is a [64, 1] or [batch_size, 1] tensor, flag it
                if len(tensor.shape) == 2 and tensor.shape[1] == 1:
                    critical_msg = f"🚨 CRITICAL: [batch_size, 1] tensor modified! {name} {tensor.shape}"
                    OPERATION_LOG.append(critical_msg)
                    print(critical_msg)
        
        TENSOR_TRACKER[tensor_id] = {
            'version': current_version,
            'name': name,
            'shape': tensor.shape,
            'operation': operation
        }

def monkey_patch_operations():
    """Monkey patch tensor operations to track modifications."""
    
    # Store original methods
    original_setitem = torch.Tensor.__setitem__
    original_iadd = torch.Tensor.__iadd__
    original_isub = torch.Tensor.__isub__
    original_imul = torch.Tensor.__imul__
    original_idiv = torch.Tensor.__itruediv__
    
    def tracked_setitem(self, key, value):
        track_tensor_version(self, f"tensor_setitem", f"[{key}] = {value}")
        return original_setitem(self, key, value)
    
    def tracked_iadd(self, other):
        track_tensor_version(self, f"tensor_iadd", f"+= {other}")
        result = original_iadd(self, other)
        track_tensor_version(result, f"tensor_iadd_result", f"after +=")
        return result
    
    def tracked_isub(self, other):
        track_tensor_version(self, f"tensor_isub", f"-= {other}")
        result = original_isub(self, other)
        track_tensor_version(result, f"tensor_isub_result", f"after -=")
        return result
    
    def tracked_imul(self, other):
        track_tensor_version(self, f"tensor_imul", f"*= {other}")
        result = original_imul(self, other)
        track_tensor_version(result, f"tensor_imul_result", f"after *=")
        return result
    
    def tracked_idiv(self, other):
        track_tensor_version(self, f"tensor_idiv", f"/= {other}")
        result = original_idiv(self, other)
        track_tensor_version(result, f"tensor_idiv_result", f"after /=")
        return result
    
    # Apply patches
    torch.Tensor.__setitem__ = tracked_setitem
    torch.Tensor.__iadd__ = tracked_iadd
    torch.Tensor.__isub__ = tracked_isub
    torch.Tensor.__imul__ = tracked_imul
    torch.Tensor.__itruediv__ = tracked_idiv
    
    return {
        'setitem': original_setitem,
        'iadd': original_iadd,
        'isub': original_isub,
        'imul': original_imul,
        'idiv': original_idiv
    }

def restore_operations(originals):
    """Restore original tensor operations."""
    torch.Tensor.__setitem__ = originals['setitem']
    torch.Tensor.__iadd__ = originals['iadd']
    torch.Tensor.__isub__ = originals['isub']
    torch.Tensor.__imul__ = originals['imul']
    torch.Tensor.__itruediv__ = originals['idiv']

def create_real_model_and_data():
    """Create the real model and data that's causing issues."""
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
    
    # Create dataset
    data_config = config["data"]
    dataset = LazyMidiDataset(
        data_dir=data_config.get("train_dir", "data/raw"),
        sequence_length=data_config.get("sequence_length", 512),
        overlap=data_config.get("overlap", 64),
        cache_dir=data_config.get("cache_dir", "data/cache"),
        num_workers=data_config.get("num_workers", 1),
        vocab_size=model_config.get("vocab_size", 774)
    )
    
    return model, dataset, config

def hunt_the_bug():
    """Systematically hunt down the in-place operation bug."""
    print("🎯 HUNTING THE IN-PLACE OPERATION BUG")
    print("=" * 60)
    
    # Clear tracking data
    global TENSOR_TRACKER, OPERATION_LOG
    TENSOR_TRACKER.clear()
    OPERATION_LOG.clear()
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Monkey patch operations
    originals = monkey_patch_operations()
    
    try:
        print("📋 Step 1: Creating real model and data...")
        model, dataset, config = create_real_model_and_data()
        model.train()
        
        print("📋 Step 2: Creating data loader...")
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=4,  # Smaller batch to see if issue persists
            shuffle=False,
            num_workers=0  # No multiprocessing for debugging
        )
        
        print("📋 Step 3: Getting first batch...")
        batch = next(iter(dataloader))
        tokens = batch['tokens']
        print(f"Batch shape: {tokens.shape}")
        
        # Track initial tensors
        track_tensor_version(tokens, "input_tokens", "initial")
        
        print("📋 Step 4: Testing forward pass...")
        with torch.autograd.detect_anomaly():
            # Forward pass
            output = model(tokens, return_latent=True)
            print(f"Forward pass successful! Output keys: {list(output.keys())}")
            
            # Track output tensors
            for key, value in output.items():
                if torch.is_tensor(value):
                    track_tensor_version(value, f"output_{key}", "forward_pass")
        
        print("📋 Step 5: Testing loss computation...")
        with torch.autograd.detect_anomaly():
            # Compute loss using model's built-in method
            losses = model.compute_loss(tokens)
            print(f"Loss computation successful! Loss keys: {list(losses.keys())}")
            
            total_loss = losses['total_loss']
            print(f"Total loss: {total_loss.item()}")
            
            # Track loss tensor
            track_tensor_version(total_loss, "total_loss", "loss_computation")
        
        print("📋 Step 6: Testing backward pass...")
        print("🔥 CRITICAL POINT: About to call loss.backward()...")
        print("Current tensor tracker state:")
        for tid, info in TENSOR_TRACKER.items():
            if len(info['shape']) == 2 and info['shape'][1] == 1:
                print(f"  🎯 TARGET: {info['name']} {info['shape']} v{info['version']}")
        
        # This is where the error occurs
        with torch.autograd.detect_anomaly():
            total_loss.backward()
        
        print("✅ Backward pass successful!")
        
    except Exception as e:
        print(f"\n❌ ERROR CAUGHT: {e}")
        print("\n📊 OPERATION LOG:")
        for i, log_entry in enumerate(OPERATION_LOG):
            print(f"{i+1:3d}. {log_entry}")
        
        print(f"\n🔍 TENSOR TRACKER STATE ({len(TENSOR_TRACKER)} tensors):")
        for tid, info in TENSOR_TRACKER.items():
            if len(info['shape']) == 2 and info['shape'][1] == 1:
                print(f"  🎯 CRITICAL: {info['name']} {info['shape']} v{info['version']} - {info['operation']}")
        
        print("\n🚨 FULL TRACEBACK:")
        traceback.print_exc()
        
    finally:
        # Restore original operations
        restore_operations(originals)
        torch.autograd.set_detect_anomaly(False)

def test_comprehensive_loss_framework():
    """Test the comprehensive loss framework specifically."""
    print("\n🧪 TESTING COMPREHENSIVE LOSS FRAMEWORK")
    print("=" * 50)
    
    try:
        # Load config
        config = load_config("configs/training_configs/quick_test.yaml")
        
        # Extract loss configuration
        loss_config = config.get('loss', {})
        
        # Create loss framework with safe parameters
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,  # Explicit vocab size
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
        
        print("✅ Loss framework created successfully")
        
        # Create model
        model, dataset, config = create_real_model_and_data()
        model.train()
        
        # Get test data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        tokens = batch['tokens']
        
        # Test with framework
        print("Testing loss framework computation...")
        
        # Monkey patch for tracking
        originals = monkey_patch_operations()
        
        try:
            with torch.autograd.detect_anomaly():
                # Forward pass
                output = model(tokens, return_latent=True)
                
                # Use loss framework
                framework_losses = loss_framework.compute_loss(output, tokens)
                
                total_loss = framework_losses['total_loss']
                print(f"Framework total loss: {total_loss.item()}")
                
                # Backward pass
                total_loss.backward()
                
            print("✅ Loss framework test passed!")
            
        finally:
            restore_operations(originals)
            
    except Exception as e:
        print(f"❌ Loss framework test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 SYSTEMATIC IN-PLACE OPERATION BUG HUNT")
    print("=" * 60)
    
    # Hunt the main bug
    hunt_the_bug()
    
    # Test loss framework separately
    test_comprehensive_loss_framework()
    
    print("\n🎯 Bug hunt completed!")