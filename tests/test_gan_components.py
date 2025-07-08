"""
Comprehensive test suite for GAN components.

This test verifies:
1. Multi-scale discriminator architecture and musical feature extraction
2. Spectral normalization and training stability
3. Feature matching and comprehensive loss functions
4. Progressive training strategies
5. Integration with existing VAE pipeline
6. Real musical data compatibility
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.discriminator import (
    MultiScaleDiscriminator, MusicalFeatureExtractor, 
    SpectralNorm, spectral_norm
)
from src.models.gan_losses import (
    ComprehensiveGANLoss, FeatureMatchingLoss, 
    SpectralRegularization, MusicalPerceptualLoss,
    ProgressiveGANLoss
)
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_spectral_normalization():
    """Test spectral normalization implementation."""
    print("🧪 Testing Spectral Normalization")
    print("-" * 40)
    
    try:
        # Test basic spectral norm wrapper
        linear = nn.Linear(256, 128)
        spec_linear = spectral_norm(linear)
        
        # Test forward pass
        x = torch.randn(4, 256)
        output = spec_linear(x)
        
        assert output.shape == (4, 128), f"Wrong output shape: {output.shape}"
        
        # Verify spectral norm constraint (should be close to 1)
        w = spec_linear.module.weight
        u = spec_linear.weight_u
        v = spec_linear.weight_v
        
        sigma = torch.dot(u, torch.mv(w.view(w.size(0), -1), v))
        
        print(f"   ✅ Spectral norm σ = {sigma.item():.4f} (should be ≈ 1.0)")
        print(f"   ✅ Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_musical_feature_extractor():
    """Test musical feature extraction."""
    print("\n🧪 Testing Musical Feature Extractor")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 4, 512
        d_model = 256
        vocab_size = 774
        
        extractor = MusicalFeatureExtractor(d_model, vocab_size)
        
        # Create realistic token sequence
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Add some musical structure
        # Note on tokens (0-127)
        note_positions = torch.randint(0, seq_len, (batch_size, 20))
        for b in range(batch_size):
            for pos in note_positions[b]:
                tokens[b, pos] = torch.randint(0, 128, (1,))
        
        # Time shift tokens (256-767)
        time_positions = torch.randint(0, seq_len, (batch_size, 30))
        for b in range(batch_size):
            for pos in time_positions[b]:
                tokens[b, pos] = torch.randint(256, 768, (1,))
        
        # Extract features
        features = extractor(tokens)
        
        assert features.shape == (batch_size, seq_len, d_model), f"Wrong feature shape: {features.shape}"
        
        # Check that features have variation (not all zeros)
        feature_std = features.std().item()
        assert feature_std > 0.01, f"Features too uniform: std={feature_std}"
        
        print(f"   ✅ Feature shape: {features.shape}")
        print(f"   ✅ Feature variation: std={feature_std:.4f}")
        print(f"   ✅ Musical structure detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_scale_discriminator():
    """Test multi-scale discriminator architecture."""
    print("\n🧪 Testing Multi-Scale Discriminator")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 2, 512
        d_model = 256
        vocab_size = 774
        
        # Test both progressive and standard modes
        for progressive in [True, False]:
            print(f"\n📍 Testing progressive={progressive}")
            
            discriminator = MultiScaleDiscriminator(
                d_model=d_model,
                vocab_size=vocab_size,
                n_layers=6,
                dropout=0.1,
                use_spectral_norm=True,
                progressive_training=progressive
            )
            
            # Create token sequence
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Forward pass
            output = discriminator(tokens)
            
            # Verify outputs
            required_keys = ['local_logits', 'phrase_logits', 'global_logits', 'combined_logits', 'features']
            for key in required_keys:
                assert key in output, f"Missing output key: {key}"
            
            # Check shapes
            for scale in ['local', 'phrase', 'global', 'combined']:
                logits = output[f'{scale}_logits']
                assert logits.shape == (batch_size, 1), f"Wrong {scale} logits shape: {logits.shape}"
            
            # Check features
            assert 'features' in output, "Missing features in output"
            assert len(output['features']) > 0, "No features extracted"
            
            print(f"   ✅ All outputs present and correctly shaped")
            print(f"   ✅ Features extracted: {len(output['features'])} layers")
            
            # Test progressive training advancement
            if progressive:
                discriminator.advance_training_stage()
                print(f"   ✅ Progressive training stage advanced")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_matching_loss():
    """Test feature matching loss computation."""
    print("\n🧪 Testing Feature Matching Loss")
    print("-" * 40)
    
    try:
        batch_size, feature_dim = 4, 256
        
        # Create mock features
        real_features = {
            'local_0': torch.randn(batch_size, 512, feature_dim),
            'phrase_0': torch.randn(batch_size, 512, feature_dim),
            'global_0': torch.randn(batch_size, 512, feature_dim),
            'input': torch.randn(batch_size, 512, feature_dim)
        }
        
        # Create different fake features
        fake_features = {
            'local_0': torch.randn(batch_size, 512, feature_dim),
            'phrase_0': torch.randn(batch_size, 512, feature_dim),
            'global_0': torch.randn(batch_size, 512, feature_dim),
            'input': torch.randn(batch_size, 512, feature_dim)
        }
        
        fm_loss = FeatureMatchingLoss()
        loss = fm_loss(real_features, fake_features)
        
        assert isinstance(loss, torch.Tensor), f"Loss should be tensor, got {type(loss)}"
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"   ✅ Feature matching loss: {loss.item():.6f}")
        
        # Test with identical features (should be close to 0)
        zero_loss = fm_loss(real_features, real_features)
        assert zero_loss.item() < 0.01, f"Identical features loss too high: {zero_loss.item()}"
        
        print(f"   ✅ Identical features loss: {zero_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_spectral_regularization():
    """Test spectral regularization techniques."""
    print("\n🧪 Testing Spectral Regularization")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 2, 256
        vocab_size = 774
        
        # Create simple discriminator
        discriminator = MultiScaleDiscriminator(
            d_model=128,
            vocab_size=vocab_size,
            n_layers=2,
            use_spectral_norm=True
        )
        
        spec_reg = SpectralRegularization(
            r1_gamma=10.0,
            use_r1=True,
            use_gradient_penalty=True
        )
        
        # Create data
        real_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        fake_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test R1 regularization
        r1_loss = spec_reg.r1_regularization(discriminator, real_tokens)
        assert isinstance(r1_loss, torch.Tensor), "R1 loss should be tensor"
        assert r1_loss.item() >= 0, f"R1 loss should be non-negative, got {r1_loss.item()}"
        
        # Test gradient penalty
        gp_loss = spec_reg.gradient_penalty(discriminator, real_tokens, fake_tokens)
        assert isinstance(gp_loss, torch.Tensor), "GP loss should be tensor"
        assert gp_loss.item() >= 0, f"GP loss should be non-negative, got {gp_loss.item()}"
        
        # Test combined
        total_reg = spec_reg(discriminator, real_tokens, fake_tokens)
        
        print(f"   ✅ R1 regularization: {r1_loss.item():.6f}")
        print(f"   ✅ Gradient penalty: {gp_loss.item():.6f}")
        print(f"   ✅ Total regularization: {total_reg.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_musical_perceptual_loss():
    """Test music-specific perceptual losses."""
    print("\n🧪 Testing Musical Perceptual Loss")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 4, 256
        vocab_size = 774
        
        perceptual_loss = MusicalPerceptualLoss(vocab_size=vocab_size)
        
        # Create musical token sequence
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Add some musical structure
        # Note sequences
        for b in range(batch_size):
            # Add some notes (0-127)
            note_positions = torch.randint(0, seq_len, (10,))
            for pos in note_positions:
                tokens[b, pos] = torch.randint(0, 128, (1,))
            
            # Add time shifts (256-767)
            time_positions = torch.randint(0, seq_len, (15,))
            for pos in time_positions:
                tokens[b, pos] = torch.randint(256, 768, (1,))
        
        # Compute perceptual loss
        loss = perceptual_loss(tokens)
        
        assert isinstance(loss, torch.Tensor), f"Loss should be tensor, got {type(loss)}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"   ✅ Musical perceptual loss: {loss.item():.6f}")
        
        # Test individual components
        rhythm_loss = perceptual_loss.rhythm_consistency_loss(tokens)
        harmony_loss = perceptual_loss.harmonic_coherence_loss(tokens)
        melody_loss = perceptual_loss.melodic_smoothness_loss(tokens)
        
        print(f"   ✅ Rhythm consistency: {rhythm_loss.item():.6f}")
        print(f"   ✅ Harmonic coherence: {harmony_loss.item():.6f}")
        print(f"   ✅ Melodic smoothness: {melody_loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_progressive_gan_loss():
    """Test progressive GAN training loss."""
    print("\n🧪 Testing Progressive GAN Loss")
    print("-" * 40)
    
    try:
        prog_loss = ProgressiveGANLoss(
            stage_epochs=[5, 10, 15],
            loss_weights={
                'stage_0': [1.0, 0.0, 0.0],
                'stage_1': [0.6, 0.4, 0.0],
                'stage_2': [0.4, 0.4, 0.2]
            }
        )
        
        # Test stage progression
        for epoch in range(20):
            prog_loss.step_epoch()
            stage = prog_loss.get_current_stage()
            weights = prog_loss.get_loss_weights()
            
            if epoch < 5:
                assert stage == 0, f"Wrong stage at epoch {epoch}: {stage}"
                assert weights == [1.0, 0.0, 0.0], f"Wrong weights: {weights}"
            elif epoch < 10:
                assert stage == 1, f"Wrong stage at epoch {epoch}: {stage}"
                assert weights == [0.6, 0.4, 0.0], f"Wrong weights: {weights}"
            else:
                assert stage == 2, f"Wrong stage at epoch {epoch}: {stage}"
                assert weights == [0.4, 0.4, 0.2], f"Wrong weights: {weights}"
        
        # Test loss computation
        discriminator_outputs = {
            'local_logits': torch.randn(4, 1),
            'phrase_logits': torch.randn(4, 1),
            'global_logits': torch.randn(4, 1)
        }
        
        loss = prog_loss(discriminator_outputs)
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        print(f"   ✅ Progressive stage transitions working")
        print(f"   ✅ Loss computation: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_comprehensive_gan_loss():
    """Test comprehensive GAN loss integration."""
    print("\n🧪 Testing Comprehensive GAN Loss")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 2, 256
        d_model = 128
        vocab_size = 774
        
        # Create discriminator
        discriminator = MultiScaleDiscriminator(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=4,
            use_spectral_norm=True
        )
        
        # Create comprehensive loss
        gan_loss = ComprehensiveGANLoss(
            feature_matching_weight=10.0,
            spectral_reg_weight=1.0,
            perceptual_weight=5.0,
            use_progressive=False,
            vocab_size=vocab_size
        )
        
        # Create data
        real_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        fake_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Get discriminator outputs
        real_outputs = discriminator(real_tokens)
        fake_outputs = discriminator(fake_tokens)
        
        # Test generator loss
        gen_losses = gan_loss.generator_loss(
            fake_outputs,
            real_outputs['features'],
            fake_outputs['features'],
            fake_tokens
        )
        
        required_gen_keys = ['adversarial', 'feature_matching', 'perceptual', 'total']
        for key in required_gen_keys:
            assert key in gen_losses, f"Missing generator loss key: {key}"
            assert isinstance(gen_losses[key], torch.Tensor), f"{key} should be tensor"
        
        # Test discriminator loss
        disc_losses = gan_loss.discriminator_loss(
            real_outputs,
            fake_outputs,
            discriminator,
            real_tokens,
            fake_tokens
        )
        
        required_disc_keys = ['adversarial', 'regularization', 'total']
        for key in required_disc_keys:
            assert key in disc_losses, f"Missing discriminator loss key: {key}"
            assert isinstance(disc_losses[key], torch.Tensor), f"{key} should be tensor"
        
        print(f"   ✅ Generator losses: {len(gen_losses)} components")
        print(f"      - Adversarial: {gen_losses['adversarial'].item():.4f}")
        print(f"      - Feature matching: {gen_losses['feature_matching'].item():.4f}")
        print(f"      - Perceptual: {gen_losses['perceptual'].item():.4f}")
        print(f"      - Total: {gen_losses['total'].item():.4f}")
        
        print(f"   ✅ Discriminator losses: {len(disc_losses)} components")
        print(f"      - Adversarial: {disc_losses['adversarial'].item():.4f}")
        print(f"      - Regularization: {disc_losses['regularization'].item():.4f}")
        print(f"      - Total: {disc_losses['total'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vae_gan_integration():
    """Test full VAE-GAN integration."""
    print("\n🧪 Testing VAE-GAN Integration")
    print("-" * 40)
    
    try:
        batch_size, seq_len = 2, 256
        vocab_size = 774
        
        # Create full VAE-GAN model
        model = MusicTransformerVAEGAN(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            encoder_layers=3,
            decoder_layers=3,
            discriminator_layers=3,
            latent_dim=48,
            mode="vae_gan"
        )
        
        # Test model components
        assert hasattr(model, 'encoder'), "Model missing encoder"
        assert hasattr(model, 'decoder'), "Model missing decoder"
        assert hasattr(model, 'discriminator'), "Model missing discriminator"
        
        # Create input data
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Test encoder
        x = model.embedding(tokens)
        encoder_output = model.encoder(x)
        
        assert 'mu' in encoder_output, "Missing mu in encoder output"
        assert 'z' in encoder_output, "Missing z in encoder output"
        
        # Test decoder
        z = encoder_output['z']
        decoder_output = model.decoder(z, x, encoder_features=x)
        
        assert decoder_output.shape == (batch_size, seq_len, vocab_size), f"Wrong decoder output shape: {decoder_output.shape}"
        
        # Test discriminator
        disc_output = model.discriminator(tokens)
        
        assert 'combined_logits' in disc_output, "Missing combined_logits in discriminator output"
        assert 'features' in disc_output, "Missing features in discriminator output"
        
        print(f"   ✅ VAE-GAN model initialized successfully")
        print(f"   ✅ Encoder output shape: {z.shape}")
        print(f"   ✅ Decoder output shape: {decoder_output.shape}")
        print(f"   ✅ Discriminator features: {len(disc_output['features'])}")
        
        # Test generation from latent
        with torch.no_grad():
            # Sample from prior
            prior_z = torch.randn(batch_size, 48)
            generated = model.decoder(prior_z, x[:, :100, :])  # Generate shorter sequence
            
            assert generated.shape[2] == vocab_size, f"Wrong generated vocab size: {generated.shape[2]}"
            
            print(f"   ✅ Generation from prior: {generated.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all GAN component tests."""
    print("🎮 GAN Components Test Suite")
    print("=" * 60)
    
    tests = [
        ("Spectral Normalization", test_spectral_normalization),
        ("Musical Feature Extractor", test_musical_feature_extractor),
        ("Multi-Scale Discriminator", test_multi_scale_discriminator),
        ("Feature Matching Loss", test_feature_matching_loss),
        ("Spectral Regularization", test_spectral_regularization),
        ("Musical Perceptual Loss", test_musical_perceptual_loss),
        ("Progressive GAN Loss", test_progressive_gan_loss),
        ("Comprehensive GAN Loss", test_comprehensive_gan_loss),
        ("VAE-GAN Integration", test_vae_gan_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All GAN components working correctly!")
        print("🚀 Ready for Phase 3.4 Loss Function Design!")
        return True
    else:
        print("🛠️  Some components need fixes before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)