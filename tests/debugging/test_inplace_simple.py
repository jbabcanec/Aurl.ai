#!/usr/bin/env python3
"""
Simple test to verify inplace operation fixes in loss components.
"""

import torch
import torch.nn as nn
from src.training.core.losses import (
    AdaptiveKLScheduler, 
    AdversarialStabilizer,
    MultiObjectiveLossBalancer,
    PerceptualReconstructionLoss
)

def test_kl_scheduler():
    """Test KL scheduler for inplace operations."""
    print("Testing AdaptiveKLScheduler...")
    scheduler = AdaptiveKLScheduler()
    
    # Test step operation
    for i in range(5):
        scheduler.step(kl_divergence=10.0 + i)
    
    # Test KL loss computation
    kl_tensor = torch.randn(64, 128, requires_grad=True)
    kl_loss = scheduler.get_kl_loss(kl_tensor)
    
    # Test backward
    kl_loss.backward()
    print("✓ AdaptiveKLScheduler passed")
    return True

def test_adversarial_stabilizer():
    """Test adversarial stabilizer for inplace operations."""
    print("Testing AdversarialStabilizer...")
    stabilizer = AdversarialStabilizer()
    
    # Test balance losses
    gen_loss = torch.tensor(1.0, requires_grad=True)
    disc_loss = torch.tensor(0.5, requires_grad=True)
    
    for i in range(5):
        balanced_gen, balanced_disc = stabilizer.balance_losses(gen_loss, disc_loss)
    
    # Test backward
    total = balanced_gen + balanced_disc
    total.backward()
    print("✓ AdversarialStabilizer passed")
    return True

def test_multi_objective_balancer():
    """Test multi-objective loss balancer for inplace operations."""
    print("Testing MultiObjectiveLossBalancer...")
    balancer = MultiObjectiveLossBalancer(num_objectives=3)
    
    # Test forward with multiple losses
    losses = {
        'reconstruction': torch.tensor(1.0, requires_grad=True),
        'kl_divergence': torch.tensor(0.5, requires_grad=True),
        'adversarial_gen': torch.tensor(0.3, requires_grad=True)
    }
    
    total_loss, weights = balancer(losses)
    
    # Test backward
    total_loss.backward()
    print("✓ MultiObjectiveLossBalancer passed")
    return True

def test_perceptual_reconstruction():
    """Test perceptual reconstruction loss for inplace operations."""
    print("Testing PerceptualReconstructionLoss...")
    loss_fn = PerceptualReconstructionLoss(vocab_size=774, musical_weighting=True)
    
    # Create test data
    batch_size, seq_len, vocab_size = 64, 512, 774
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    
    # Compute loss
    losses = loss_fn(logits, targets, mask)
    total_loss = losses['total']
    
    # Test backward
    total_loss.backward()
    print("✓ PerceptualReconstructionLoss passed")
    return True

def main():
    """Run all tests."""
    print("Testing inplace operation fixes...\n")
    
    tests = [
        test_kl_scheduler,
        test_adversarial_stabilizer,
        test_multi_objective_balancer,
        test_perceptual_reconstruction
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except RuntimeError as e:
            if "inplace operation" in str(e):
                print(f"✗ {test.__name__} failed with inplace error: {e}")
                all_passed = False
            else:
                raise e
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Inplace operations have been fixed.")
    else:
        print("\n✗ Some tests failed.")

if __name__ == "__main__":
    main()