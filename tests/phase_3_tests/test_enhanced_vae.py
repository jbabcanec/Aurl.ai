"""
Test suite for enhanced VAE components.

This test verifies:
1. Enhanced encoder with β-VAE and hierarchical latents
2. Enhanced decoder with skip connections
3. Musical priors and latent analysis tools
4. Integration with existing pipeline
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.encoder import EnhancedMusicEncoder, LatentRegularizer
from src.models.decoder import EnhancedMusicDecoder
from src.models.vae_components import MusicalPrior, LatentAnalyzer, AdaptiveBeta
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


def test_enhanced_encoder():
    """Test enhanced encoder functionality."""
    print("🧪 Testing Enhanced Encoder")
    print("-" * 40)
    
    batch_size, seq_len, d_model = 2, 512, 256
    latent_dim = 64
    
    # Test both hierarchical and standard modes
    for hierarchical in [True, False]:
        print(f"\n📍 Testing hierarchical={hierarchical}")
        
        try:
            encoder = EnhancedMusicEncoder(
                d_model=d_model,
                latent_dim=latent_dim if not hierarchical else 63,  # Divisible by 3
                n_layers=2,
                beta=1.5,
                hierarchical=hierarchical,
                free_bits=0.2
            )
            
            # Test forward pass
            x = torch.randn(batch_size, seq_len, d_model)
            output = encoder(x)
            
            # Verify outputs
            assert 'mu' in output, "Missing mu in encoder output"
            assert 'logvar' in output, "Missing logvar in encoder output"
            assert 'z' in output, "Missing z in encoder output"
            assert 'kl_loss' in output, "Missing kl_loss in encoder output"
            
            print(f"   ✅ Output shapes: mu={output['mu'].shape}, z={output['z'].shape}")
            print(f"   ✅ KL loss shape: {output['kl_loss'].shape}")
            print(f"   ✅ Active dims: {output['latent_info']['active_dims']:.2f}")
            
            if hierarchical:
                assert 'global_mu' in output['latent_info'], "Missing hierarchical info"
                print(f"   ✅ Hierarchical: global_kl={output['latent_info']['global_kl']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    return True


def test_enhanced_decoder():
    """Test enhanced decoder functionality."""
    print("\n🧪 Testing Enhanced Decoder")
    print("-" * 40)
    
    batch_size, seq_len, d_model = 2, 256, 256
    latent_dim = 63  # Divisible by 3
    vocab_size = 774
    
    # Test both modes
    for hierarchical in [True, False]:
        print(f"\n📍 Testing hierarchical={hierarchical}")
        
        try:
            decoder = EnhancedMusicDecoder(
                d_model=d_model,
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                n_layers=2,
                hierarchical=hierarchical,
                use_skip_connection=True
            )
            
            # Test inputs
            latent = torch.randn(batch_size, latent_dim)
            target_embeddings = torch.randn(batch_size, seq_len, d_model)
            encoder_features = torch.randn(batch_size, seq_len, d_model)
            
            # Test forward pass
            logits = decoder(latent, target_embeddings, encoder_features=encoder_features)
            
            # Verify output
            expected_shape = (batch_size, seq_len, vocab_size)
            assert logits.shape == expected_shape, f"Wrong output shape: {logits.shape}"
            
            print(f"   ✅ Output shape: {logits.shape}")
            print(f"   ✅ Output range: {logits.min().item():.2f} - {logits.max().item():.2f}")
            
            # Test with return_hidden
            output_dict = decoder(latent, target_embeddings, return_hidden=True)
            assert 'logits' in output_dict, "Missing logits in detailed output"
            assert 'hidden_states' in output_dict, "Missing hidden states"
            
            print(f"   ✅ Hidden states: {len(output_dict['hidden_states'])} layers")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    return True


def test_musical_prior():
    """Test musical prior implementations."""
    print("\n🧪 Testing Musical Prior")
    print("-" * 40)
    
    latent_dim = 32
    batch_size = 4
    
    # Test different prior types
    prior_types = ["standard", "mixture", "flow"]
    
    for prior_type in prior_types:
        print(f"\n📍 Testing {prior_type} prior")
        
        try:
            prior = MusicalPrior(
                latent_dim=latent_dim,
                prior_type=prior_type,
                num_modes=4,
                flow_layers=2
            )
            
            # Test sampling
            samples = prior.sample(batch_size, torch.device('cpu'))
            assert samples.shape == (batch_size, latent_dim), f"Wrong sample shape: {samples.shape}"
            
            # Test log probability
            log_probs = prior.log_prob(samples)
            assert log_probs.shape == (batch_size,), f"Wrong log prob shape: {log_probs.shape}"
            
            print(f"   ✅ Samples shape: {samples.shape}")
            print(f"   ✅ Log prob range: {log_probs.min().item():.2f} - {log_probs.max().item():.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    return True


def test_latent_analyzer():
    """Test latent analysis tools."""
    print("\n🧪 Testing Latent Analyzer")
    print("-" * 40)
    
    # Create mock model
    class MockVAE:
        def eval(self): pass
    
    try:
        analyzer = LatentAnalyzer(MockVAE())
        
        # Test dimension traversal
        base_latent = torch.randn(1, 32)
        values = torch.linspace(-2, 2, 5)
        
        def mock_decode(z):
            return torch.randn(1, 100, 774)  # Mock output
        
        traversals = analyzer.traverse_dimension(base_latent, dim=5, values=values, decode_fn=mock_decode)
        assert len(traversals) == 5, f"Wrong number of traversals: {len(traversals)}"
        
        print(f"   ✅ Dimension traversal: {len(traversals)} steps")
        
        # Test interpolation
        z1 = torch.randn(1, 32)
        z2 = torch.randn(1, 32)
        interpolations = analyzer.interpolate(z1, z2, steps=10, decode_fn=mock_decode)
        assert len(interpolations) == 10, f"Wrong interpolation steps: {len(interpolations)}"
        
        print(f"   ✅ Interpolation: {len(interpolations)} steps")
        
        # Test disentanglement metrics
        latents = torch.randn(100, 32)
        factors = torch.randn(100, 4)
        metrics = analyzer.compute_disentanglement_metrics(latents, factors)
        
        assert 'mig' in metrics, "Missing MIG metric"
        assert 'sap' in metrics, "Missing SAP metric"
        assert 'active_dims' in metrics, "Missing active dims metric"
        
        print(f"   ✅ Metrics: MIG={metrics['mig']:.3f}, SAP={metrics['sap']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    return True


def test_adaptive_beta():
    """Test adaptive β scheduling."""
    print("\n🧪 Testing Adaptive Beta")
    print("-" * 40)
    
    schedules = ["linear", "exponential", "cyclical"]
    
    for schedule in schedules:
        print(f"\n📍 Testing {schedule} schedule")
        
        try:
            beta_scheduler = AdaptiveBeta(
                beta_start=0.0,
                beta_end=2.0,
                warmup_epochs=10,
                schedule_type=schedule
            )
            
            # Test progression
            betas = []
            for epoch in range(15):
                beta = beta_scheduler.get_beta()
                betas.append(beta)
                beta_scheduler.step()
            
            assert betas[0] == 0.0, f"Beta should start at 0, got {betas[0]}"
            assert betas[-1] == 2.0, f"Beta should end at 2, got {betas[-1]}"
            
            print(f"   ✅ Beta progression: {betas[0]:.2f} → {betas[5]:.2f} → {betas[-1]:.2f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    return True


def test_latent_regularizer():
    """Test latent regularization."""
    print("\n🧪 Testing Latent Regularizer")
    print("-" * 40)
    
    try:
        regularizer = LatentRegularizer(
            latent_dim=32,
            mi_penalty=0.1,
            ortho_penalty=0.01,
            sparsity_penalty=0.01
        )
        
        # Test with latent codes (use multiple of 4 for orthogonality test)
        z = torch.randn(16, 32)
        losses = regularizer(z)
        
        assert 'mi_loss' in losses, "Missing MI loss"
        assert 'ortho_loss' in losses, "Missing orthogonality loss"
        assert 'sparsity_loss' in losses, "Missing sparsity loss"
        
        print(f"   ✅ MI loss: {losses['mi_loss'].item():.6f}")
        print(f"   ✅ Ortho loss: {losses['ortho_loss'].item():.6f}")
        print(f"   ✅ Sparsity loss: {losses['sparsity_loss'].item():.6f}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    return True


def test_integration():
    """Test integration of all enhanced components."""
    print("\n🧪 Testing Full Integration")
    print("-" * 40)
    
    try:
        # Setup
        batch_size, seq_len, d_model = 2, 256, 256
        latent_dim = 48  # Divisible by 3
        vocab_size = 774
        
        # Create components
        encoder = EnhancedMusicEncoder(
            d_model=d_model,
            latent_dim=latent_dim,
            n_layers=2,
            beta=1.0,
            hierarchical=True,
            free_bits=0.1
        )
        
        decoder = EnhancedMusicDecoder(
            d_model=d_model,
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            n_layers=2,
            hierarchical=True,
            use_skip_connection=True
        )
        
        prior = MusicalPrior(latent_dim, "mixture", num_modes=4)
        regularizer = LatentRegularizer(latent_dim)
        
        # Test full pipeline
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Encode
        enc_output = encoder(x)
        mu, logvar, z = enc_output['mu'], enc_output['logvar'], enc_output['z']
        
        # Decode
        logits = decoder(z, x, encoder_features=x)
        
        # Prior sampling
        z_prior = prior.sample(batch_size, x.device)
        
        # Regularization
        reg_losses = regularizer(z)
        
        print(f"   ✅ Encode: {x.shape} → {z.shape}")
        print(f"   ✅ Decode: {z.shape} → {logits.shape}")
        print(f"   ✅ Prior sample: {z_prior.shape}")
        print(f"   ✅ Regularization losses: {len(reg_losses)}")
        
        # Test loss computation
        recon_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            torch.randint(0, vocab_size, (batch_size * seq_len,))
        )
        
        kl_loss = enc_output['kl_loss'].mean()
        total_loss = recon_loss + kl_loss + sum(reg_losses.values())
        
        print(f"   ✅ Losses: recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
        print(f"   ✅ Total loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("🧬 Enhanced VAE Components Test Suite")
    print("=" * 60)
    
    tests = [
        ("Enhanced Encoder", test_enhanced_encoder),
        ("Enhanced Decoder", test_enhanced_decoder),
        ("Musical Prior", test_musical_prior),
        ("Latent Analyzer", test_latent_analyzer),
        ("Adaptive Beta", test_adaptive_beta),
        ("Latent Regularizer", test_latent_regularizer),
        ("Full Integration", test_integration)
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
        print("🎉 All enhanced VAE components working correctly!")
        return True
    else:
        print("🛠️  Some components need fixes before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)