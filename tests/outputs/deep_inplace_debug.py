#!/usr/bin/env python3
"""
Deep debugging script for inplace operation issues.
Based on latest research about AsStridedBackward0 errors.
"""

import torch
import torch.nn as nn
import sys
import traceback
from pathlib import Path
import numpy as np
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework

class TensorTracker:
    """Track tensor operations and modifications."""
    
    def __init__(self):
        self.tensor_versions = {}
        self.operations = []
        
    def track_tensor(self, tensor, name):
        """Track a tensor's version."""
        if torch.is_tensor(tensor):
            self.tensor_versions[name] = tensor._version
            self.operations.append(f"Tracking {name}: version {tensor._version}, shape {tensor.shape}")
    
    def check_version_changes(self, tensor, name):
        """Check if tensor version has changed."""
        if torch.is_tensor(tensor) and name in self.tensor_versions:
            old_version = self.tensor_versions[name]
            new_version = tensor._version
            if old_version != new_version:
                self.operations.append(f"VERSION CHANGE: {name} from {old_version} to {new_version}")
                return True
        return False
    
    def print_operations(self):
        """Print all tracked operations."""
        print("=== TENSOR OPERATIONS TRACE ===")
        for op in self.operations:
            print(op)
        print("=== END TRACE ===")

@contextmanager
def track_tensor_operations():
    """Context manager to track all tensor operations."""
    tracker = TensorTracker()
    
    # Monkey patch common operations
    original_setitem = torch.Tensor.__setitem__
    original_add = torch.Tensor.__add__
    original_iadd = torch.Tensor.__iadd__
    original_mul = torch.Tensor.__mul__
    original_imul = torch.Tensor.__imul__
    
    def tracked_setitem(self, key, value):
        tracker.operations.append(f"SETITEM: {self.shape}[{key}] = {value.shape if torch.is_tensor(value) else value}")
        return original_setitem(self, key, value)
    
    def tracked_add(self, other):
        result = original_add(self, other)
        tracker.operations.append(f"ADD: {self.shape} + {other.shape if torch.is_tensor(other) else other} -> {result.shape}")
        return result
    
    def tracked_iadd(self, other):
        tracker.operations.append(f"IADD: {self.shape} += {other.shape if torch.is_tensor(other) else other}")
        return original_iadd(self, other)
    
    def tracked_mul(self, other):
        result = original_mul(self, other)
        tracker.operations.append(f"MUL: {self.shape} * {other.shape if torch.is_tensor(other) else other} -> {result.shape}")
        return result
    
    def tracked_imul(self, other):
        tracker.operations.append(f"IMUL: {self.shape} *= {other.shape if torch.is_tensor(other) else other}")
        return original_imul(self, other)
    
    # Apply monkey patches
    torch.Tensor.__setitem__ = tracked_setitem
    torch.Tensor.__add__ = tracked_add
    torch.Tensor.__iadd__ = tracked_iadd
    torch.Tensor.__mul__ = tracked_mul
    torch.Tensor.__imul__ = tracked_imul
    
    try:
        yield tracker
    finally:
        # Restore original methods
        torch.Tensor.__setitem__ = original_setitem
        torch.Tensor.__add__ = original_add
        torch.Tensor.__iadd__ = original_iadd
        torch.Tensor.__mul__ = original_mul
        torch.Tensor.__imul__ = original_imul

def create_minimal_model():
    """Create a minimal model for testing."""
    model = MusicTransformerVAEGAN(
        vocab_size=100,  # Smaller vocab
        d_model=64,      # Smaller model
        n_layers=1,      # Single layer
        n_heads=2,       # Fewer heads
        latent_dim=12,   # Divisible by 3
        max_sequence_length=32,
        mode="vae_gan",
        discriminator_layers=1,
        beta=1.0,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.1,
        attention_type="hierarchical",
        local_window_size=16,
        global_window_size=8,
        sliding_window_size=24
    )
    return model

def test_specific_tensor_operations():
    """Test specific tensor operations that might cause issues."""
    print("Testing specific tensor operations...")
    
    # Test 1: Tensor slicing and modification
    print("\n=== Test 1: Tensor slicing ===")
    try:
        x = torch.randn(4, 10, requires_grad=True)
        print(f"Original tensor version: {x._version}")
        
        # This should NOT cause inplace error
        y = x[:, :5]  # Create a view
        print(f"View tensor version: {y._version}")
        
        # This WILL cause inplace error if y is used later in computation
        # y[0] = torch.zeros(5)  # Commenting this out
        
        # Instead, create a new tensor
        y_new = torch.cat([torch.zeros(1, 5), y[1:]], dim=0)
        print(f"New tensor version: {y_new._version}")
        
        loss = y_new.sum()
        loss.backward()
        print("✅ Test 1 passed")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Test tensor indexing with masks
    print("\n=== Test 2: Tensor indexing ===")
    try:
        x = torch.randn(4, 10, requires_grad=True)
        mask = torch.randint(0, 2, (4, 10)).bool()
        
        # This should be safe
        y = x[mask]
        loss = y.sum()
        loss.backward()
        print("✅ Test 2 passed")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Test tensor concatenation and reshaping
    print("\n=== Test 3: Tensor operations ===")
    try:
        x = torch.randn(4, 10, requires_grad=True)
        
        # Safe operations
        y = x.view(4, 10)  # View - potentially problematic
        z = y.contiguous()  # Make contiguous
        w = z.reshape(2, 20)  # Reshape
        
        loss = w.sum()
        loss.backward()
        print("✅ Test 3 passed")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

def test_model_components():
    """Test individual model components."""
    print("\n=== Testing Model Components ===")
    
    try:
        model = create_minimal_model()
        model.eval()
        
        # Test encoder
        print("Testing encoder...")
        tokens = torch.randint(0, 100, (2, 16))
        embeddings = model.embedding(tokens)
        
        with track_tensor_operations() as tracker:
            encoder_output = model.encoder(embeddings)
            tracker.print_operations()
        
        print(f"Encoder output keys: {list(encoder_output.keys())}")
        print("✅ Encoder test passed")
        
        # Test decoder
        print("Testing decoder...")
        z = encoder_output['z']
        
        with track_tensor_operations() as tracker:
            decoder_output = model.decoder(z, embeddings)
            tracker.print_operations()
        
        print(f"Decoder output shape: {decoder_output.shape}")
        print("✅ Decoder test passed")
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        traceback.print_exc()

def test_loss_computation():
    """Test loss computation specifically."""
    print("\n=== Testing Loss Computation ===")
    
    try:
        model = create_minimal_model()
        model.eval()
        
        tokens = torch.randint(0, 100, (2, 16))
        
        print("Testing forward pass...")
        with track_tensor_operations() as tracker:
            output = model(tokens, return_latent=True)
            tracker.print_operations()
        
        print(f"Forward pass output keys: {list(output.keys())}")
        
        print("Testing loss computation...")
        with track_tensor_operations() as tracker:
            losses = model.compute_loss(tokens)
            tracker.print_operations()
        
        print(f"Loss computation output keys: {list(losses.keys())}")
        
        print("Testing backward pass...")
        total_loss = losses['total_loss']
        print(f"Total loss: {total_loss.item()}")
        
        # Enable anomaly detection for backward pass
        with torch.autograd.detect_anomaly():
            total_loss.backward()
        
        print("✅ Loss computation test passed")
        
    except Exception as e:
        print(f"❌ Loss computation test failed: {e}")
        traceback.print_exc()

def test_full_training_step():
    """Test a full training step."""
    print("\n=== Testing Full Training Step ===")
    
    try:
        model = create_minimal_model()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        tokens = torch.randint(0, 100, (2, 16))
        
        print("Testing full training step...")
        with track_tensor_operations() as tracker:
            optimizer.zero_grad()
            
            # Forward pass
            losses = model.compute_loss(tokens)
            total_loss = losses['total_loss']
            
            print(f"Loss: {total_loss.item()}")
            
            # Backward pass
            with torch.autograd.detect_anomaly():
                total_loss.backward()
            
            # Check gradients
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
            
            print(f"Gradient norm: {grad_norm}")
            
            optimizer.step()
            
            tracker.print_operations()
        
        print("✅ Full training step test passed")
        
    except Exception as e:
        print(f"❌ Full training step test failed: {e}")
        traceback.print_exc()

def test_loss_framework():
    """Test the comprehensive loss framework."""
    print("\n=== Testing Loss Framework ===")
    
    try:
        # Load config
        config = load_config("configs/training_configs/quick_test.yaml")
        
        # Create model
        model = create_minimal_model()
        model.train()
        
        # Create loss framework with explicit parameters
        loss_framework = ComprehensiveLossFramework(
            vocab_size=100,  # Match our minimal model
            musical_weighting=True,
            perceptual_emphasis=2.0
        )
        
        tokens = torch.randint(0, 100, (2, 16))
        
        print("Testing loss framework...")
        with track_tensor_operations() as tracker:
            # Forward pass
            output = model(tokens, return_latent=True)
            
            # Compute losses using framework
            losses = loss_framework.compute_loss(output, tokens)
            
            print(f"Framework losses: {list(losses.keys())}")
            
            total_loss = losses['total_loss']
            print(f"Total loss: {total_loss.item()}")
            
            # Backward pass
            with torch.autograd.detect_anomaly():
                total_loss.backward()
            
            tracker.print_operations()
        
        print("✅ Loss framework test passed")
        
    except Exception as e:
        print(f"❌ Loss framework test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Deep Inplace Operation Debugging")
    print("=" * 50)
    
    # Test specific tensor operations
    test_specific_tensor_operations()
    
    # Test model components
    test_model_components()
    
    # Test loss computation
    test_loss_computation()
    
    # Test full training step
    test_full_training_step()
    
    # Test loss framework
    test_loss_framework()
    
    print("\n🎯 Deep debugging completed!")