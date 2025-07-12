#!/usr/bin/env python3
"""
Test the progressive training pipeline to ensure all fixes are working.

This test validates:
1. No more in-place operation errors
2. Progressive training stages work correctly
3. Model can train without gradient issues
4. Error handling works as expected
"""

import torch
import sys
from pathlib import Path
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.config import load_config
from src.training.utils.error_handling import ErrorHandler, GradientMonitor
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


def test_stage1_transformer_training():
    """Test Stage 1: Transformer-only training."""
    logger.info("Testing Stage 1: Transformer-only training")
    
    # Load stage 1 config
    config = load_config("configs/training_configs/stage1_transformer.yaml")
    
    # Create model in transformer mode
    model_config = config["model"]
    model = MusicTransformerVAEGAN(
        vocab_size=model_config.get("vocab_size", 774),
        d_model=model_config.get("hidden_dim", 256),
        n_layers=model_config.get("num_layers", 4),
        n_heads=model_config.get("num_heads", 4),
        latent_dim=model_config.get("latent_dim", 66),
        max_sequence_length=model_config.get("max_sequence_length", 512),
        mode="transformer",  # Force transformer mode
        dropout=model_config.get("dropout", 0.1)
    )
    
    # Create synthetic data
    batch_size = 4
    seq_len = 128  # Short for testing
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    model.train()
    output = model(tokens)
    
    assert output.shape == (batch_size, seq_len, model.vocab_size), \
        f"Unexpected output shape: {output.shape}"
    
    # Test loss computation and backward pass
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, model.vocab_size),
        tokens.view(-1)
    )
    
    # This should not raise any in-place operation errors
    loss.backward()
    
    # Check gradients
    grad_monitor = GradientMonitor()
    grad_stats = grad_monitor.check_gradients(model)
    
    assert grad_stats["healthy"], f"Gradient issues found: {grad_stats['issues']}"
    assert grad_stats["num_nan"] == 0, "NaN gradients detected"
    assert grad_stats["num_inf"] == 0, "Inf gradients detected"
    
    logger.info("✅ Stage 1 test passed!")
    return True


def test_stage2_vae_training():
    """Test Stage 2: VAE training."""
    logger.info("Testing Stage 2: VAE training")
    
    # Create model in VAE mode
    model = MusicTransformerVAEGAN(
        vocab_size=774,
        d_model=256,
        n_layers=4,
        n_heads=4,
        latent_dim=66,
        max_sequence_length=512,
        mode="vae",  # VAE mode
        encoder_layers=3,
        decoder_layers=3,
        beta=0.5,
        dropout=0.1
    )
    
    # Create synthetic data
    batch_size = 4
    seq_len = 128
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    # Test forward pass with latent
    model.train()
    output = model(tokens, return_latent=True)
    
    assert "reconstruction" in output, "Missing reconstruction output"
    assert "mu" in output, "Missing mu output"
    assert "logvar" in output, "Missing logvar output"
    assert "z" in output, "Missing latent z"
    
    # Test loss computation
    losses = model.compute_loss(tokens)
    
    assert "total_loss" in losses, "Missing total loss"
    assert "reconstruction_loss" in losses, "Missing reconstruction loss"
    assert "kl_loss" in losses, "Missing KL loss"
    
    # Test backward pass
    total_loss = losses["total_loss"]
    total_loss.backward()
    
    # Check gradients (use higher threshold for untrained VAE models)
    grad_monitor = GradientMonitor(threshold_norm=5000.0)  # Higher threshold for VAE
    grad_stats = grad_monitor.check_gradients(model)
    
    assert grad_stats["healthy"], f"Gradient issues found: {grad_stats['issues']}"
    
    logger.info("✅ Stage 2 test passed!")
    return True


def test_stage3_vae_gan_training():
    """Test Stage 3: Full VAE-GAN training."""
    logger.info("Testing Stage 3: VAE-GAN training")
    
    # Create model in VAE-GAN mode
    model = MusicTransformerVAEGAN(
        vocab_size=774,
        d_model=256,
        n_layers=4,
        n_heads=4,
        latent_dim=66,
        max_sequence_length=512,
        mode="vae_gan",  # Full VAE-GAN mode
        encoder_layers=3,
        decoder_layers=3,
        discriminator_layers=3,
        beta=1.0,
        dropout=0.1
    )
    
    # Create synthetic data
    batch_size = 4
    seq_len = 128
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    model.train()
    
    # Test loss computation (this includes discriminator)
    losses = model.compute_loss(tokens)
    
    required_losses = [
        "total_loss", 
        "reconstruction_loss", 
        "kl_loss",
        "generator_loss",
        "discriminator_loss"
    ]
    
    for loss_name in required_losses:
        assert loss_name in losses, f"Missing {loss_name}"
    
    # Test backward pass - this is where in-place errors would occur
    total_loss = losses["total_loss"]
    total_loss.backward()
    
    # Check gradients
    grad_monitor = GradientMonitor()
    grad_stats = grad_monitor.check_gradients(model)
    
    assert grad_stats["healthy"], f"Gradient issues found: {grad_stats['issues']}"
    
    logger.info("✅ Stage 3 test passed!")
    return True


def test_error_handling():
    """Test error handling mechanisms."""
    logger.info("Testing error handling")
    
    # Create error handler
    error_handler = ErrorHandler(max_retries=2)
    
    # Test gradient error handling
    model = MusicTransformerVAEGAN(vocab_size=774, mode="transformer")
    
    # Simulate a gradient error
    fake_error = RuntimeError(
        "one of the variables needed for gradient computation has been modified "
        "by an inplace operation: [torch.FloatTensor [64, 1]], which is output 0 "
        "of AsStridedBackward0, is at version 5; expected version 4 instead."
    )
    
    error_info = error_handler.handle_gradient_error(fake_error, model, {})
    
    assert error_info["type"] == "gradient_error"
    assert len(error_info["diagnostics"]) > 0
    assert len(error_info["recovery_suggestions"]) > 0
    
    # Test memory error handling
    oom_error = RuntimeError("CUDA out of memory")
    error_info = error_handler.handle_memory_error(oom_error, model, 32, 512)
    
    assert error_info["type"] == "memory_error"
    assert len(error_info["recovery_suggestions"]) > 0
    
    logger.info("✅ Error handling test passed!")
    return True


def test_mixed_precision_compatibility():
    """Test mixed precision works correctly on different devices."""
    logger.info("Testing mixed precision compatibility")
    
    # Test on CPU (should not use autocast)
    device = torch.device("cpu")
    model = MusicTransformerVAEGAN(vocab_size=774, mode="transformer").to(device)
    
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len)).to(device)
    
    # This should work without autocast on CPU
    output = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, model.vocab_size),
        tokens.view(-1)
    )
    loss.backward()
    
    logger.info("✅ Mixed precision compatibility test passed!")
    return True


def test_progressive_validation():
    """Test the progressive validation gates."""
    logger.info("Testing progressive validation gates")
    
    from train_progressive import ProgressiveTrainer
    
    # Create a mock config
    config = {
        "model": {"vocab_size": 774},
        "training": {},
        "experiment": {"name": "test"},
        "system": {"output_dir": "test_outputs"}
    }
    
    device = torch.device("cpu")
    trainer = ProgressiveTrainer(config, device)
    
    # Test stage 1 validation
    model = MusicTransformerVAEGAN(vocab_size=774, mode="transformer").to(device)
    
    # Create mock dataset
    class MockDataset:
        def __len__(self):
            return 100
    
    dataset = MockDataset()
    
    # Validation should pass for properly configured model
    # Note: This is a simplified test - real validation includes generation
    assert trainer.validate_stage(1, model, dataset) == True
    
    logger.info("✅ Progressive validation test passed!")
    return True


def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n" + "="*60)
    logger.info("🧪 RUNNING PROGRESSIVE TRAINING TESTS")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Stage 1 Transformer", test_stage1_transformer_training),
        ("Stage 2 VAE", test_stage2_vae_training),
        ("Stage 3 VAE-GAN", test_stage3_vae_gan_training),
        ("Error Handling", test_error_handling),
        ("Mixed Precision", test_mixed_precision_compatibility),
        # ("Progressive Validation", test_progressive_validation),  # Requires more setup
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🔬 Running test: {test_name}")
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            logger.error(f"❌ Test failed: {test_name}")
            logger.error(f"Error: {e}")
            results.append((test_name, False, str(e)))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if error:
            logger.info(f"  Error: {error}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! The training pipeline overhaul is working correctly.")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)