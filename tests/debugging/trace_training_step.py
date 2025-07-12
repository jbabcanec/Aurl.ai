#!/usr/bin/env python3
"""
Trace a single training step to find the exact inplace operation.
"""

import torch
import torch.nn as nn
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework
from src.data.dataset import LazyMidiDataset

# Monkey patch to track tensor modifications
class TensorVersionTracker:
    def __init__(self):
        self.tracked_tensors = {}
        self.modifications = []
    
    def track(self, tensor, name):
        if torch.is_tensor(tensor) and tensor.requires_grad:
            tensor_id = id(tensor)
            current_version = tensor._version
            
            if tensor_id in self.tracked_tensors:
                old_version = self.tracked_tensors[tensor_id]['version']
                if current_version != old_version:
                    mod_info = {
                        'name': name,
                        'shape': tensor.shape,
                        'old_version': old_version,
                        'new_version': current_version,
                        'stack': traceback.format_stack()[-3:-1]  # Get calling location
                    }
                    self.modifications.append(mod_info)
                    
                    # Flag critical [batch_size, 1] tensors
                    if len(tensor.shape) == 2 and tensor.shape[1] == 1:
                        print(f"🚨 CRITICAL MODIFICATION: {name} {tensor.shape} v{old_version}→v{current_version}")
                        print(f"   Called from: {mod_info['stack'][-1].strip()}")
            
            self.tracked_tensors[tensor_id] = {
                'name': name,
                'version': current_version,
                'shape': tensor.shape
            }
    
    def get_critical_modifications(self):
        """Get modifications to [batch_size, 1] tensors."""
        critical = []
        for mod in self.modifications:
            if len(mod['shape']) == 2 and mod['shape'][1] == 1:
                critical.append(mod)
        return critical

tracker = TensorVersionTracker()

# Monkey patch key tensor operations
original_setitem = torch.Tensor.__setitem__
original_iadd = torch.Tensor.__iadd__

def tracked_setitem(self, key, value):
    tracker.track(self, f"setitem_{self.shape}")
    result = original_setitem(self, key, value)
    tracker.track(self, f"setitem_{self.shape}_after")
    return result

def tracked_iadd(self, other):
    tracker.track(self, f"iadd_{self.shape}")
    result = original_iadd(self, other)
    tracker.track(self, f"iadd_{self.shape}_after")
    return result

torch.Tensor.__setitem__ = tracked_setitem
torch.Tensor.__iadd__ = tracked_iadd

def create_test_scenario():
    """Create the exact scenario that causes the bug."""
    # Load config
    config = load_config("configs/training_configs/quick_test.yaml")
    
    # Create model
    model_config = config["model"]
    model = MusicTransformerVAEGAN(
        vocab_size=model_config.get("vocab_size", 774),
        d_model=model_config.get("hidden_dim", 256),
        n_layers=model_config.get("num_layers", 4),
        n_heads=model_config.get("num_heads", 4),
        latent_dim=model_config.get("latent_dim", 66),
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
    
    # Create dataset
    dataset = LazyMidiDataset(
        data_dir="data/raw",
        sequence_length=512,
        overlap=64,
        cache_dir="data/cache",
        vocab_size=774
    )
    
    return model, dataset, config

def test_single_training_step():
    """Test a single training step with comprehensive tracking."""
    print("🔍 TRACING SINGLE TRAINING STEP")
    print("=" * 50)
    
    try:
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        print("📋 Creating model and data...")
        model, dataset, config = create_test_scenario()
        model.train()
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=4,  # Use exact batch size from config
            shuffle=False,
            num_workers=0
        )
        
        print("📋 Getting batch...")
        batch = next(iter(dataloader))
        tokens = batch['tokens']
        print(f"Batch shape: {tokens.shape}")
        
        # Track input
        tracker.track(tokens, "input_tokens")
        
        print("📋 Forward pass...")
        # Forward pass
        output = model(tokens, return_latent=True)
        
        # Track outputs
        for key, value in output.items():
            if torch.is_tensor(value):
                tracker.track(value, f"output_{key}")
        
        print("📋 Loss computation...")
        # Loss computation
        losses = model.compute_loss(tokens)
        total_loss = losses['total_loss']
        
        # Track loss
        tracker.track(total_loss, "total_loss")
        
        print(f"Total loss: {total_loss.item()}")
        
        print("📋 Backward pass (THIS IS WHERE THE ERROR OCCURS)...")
        # This should trigger the error
        total_loss.backward()
        
        print("✅ Training step completed successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        
        print("\n🚨 CRITICAL TENSOR MODIFICATIONS:")
        critical_mods = tracker.get_critical_modifications()
        for i, mod in enumerate(critical_mods):
            print(f"{i+1}. {mod['name']} {mod['shape']} v{mod['old_version']}→v{mod['new_version']}")
            print(f"   Stack: {mod['stack'][-1].strip()}")
        
        print(f"\n📊 ALL MODIFICATIONS ({len(tracker.modifications)} total):")
        for i, mod in enumerate(tracker.modifications[-10:]):  # Show last 10
            print(f"{i+1}. {mod['name']} {mod['shape']} v{mod['old_version']}→v{mod['new_version']}")
        
        print("\n🚨 FULL TRACEBACK:")
        traceback.print_exc()
        
        return False
    
    finally:
        # Restore original methods
        torch.Tensor.__setitem__ = original_setitem
        torch.Tensor.__iadd__ = original_iadd
        torch.autograd.set_detect_anomaly(False)
    
    return True

def test_loss_framework_only():
    """Test just the loss framework computation."""
    print("\n🧪 TESTING LOSS FRAMEWORK ONLY")
    print("=" * 40)
    
    try:
        # Create loss framework
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            musical_weighting=True,
            perceptual_emphasis=2.0
        )
        
        # Create dummy data that matches real training
        batch_size = 4
        seq_len = 64
        vocab_size = 774
        latent_dim = 66
        
        # Create mock model output
        output = {
            'logits': torch.randn(batch_size, seq_len, vocab_size, requires_grad=True),
            'mu': torch.randn(batch_size, latent_dim, requires_grad=True),
            'logvar': torch.randn(batch_size, latent_dim, requires_grad=True),
            'z': torch.randn(batch_size, latent_dim, requires_grad=True),
            'kl_loss': torch.randn(1, requires_grad=True)
        }
        
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Track all tensors
        for key, value in output.items():
            tracker.track(value, f"mock_{key}")
        tracker.track(tokens, "mock_tokens")
        
        print("Testing loss framework computation...")
        
        # Use loss framework
        losses = loss_framework.compute_loss(output, tokens)
        total_loss = losses['total_loss']
        
        tracker.track(total_loss, "framework_total_loss")
        
        print(f"Framework loss: {total_loss.item()}")
        
        # Backward pass
        total_loss.backward()
        
        print("✅ Loss framework test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Loss framework test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TRAINING STEP TRACER")
    print("=" * 30)
    
    # Test single training step
    success = test_single_training_step()
    
    if not success:
        print("\n🔄 Testing loss framework in isolation...")
        test_loss_framework_only()
    
    print("\n🎯 Tracing completed!")