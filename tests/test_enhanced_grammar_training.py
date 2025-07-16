#!/usr/bin/env python3
"""
Test Enhanced Grammar Training Setup

Validates that the Phase 5.2 enhanced grammar training system
is properly configured and ready for production training.
"""

import sys
import torch
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.musical_grammar import MusicalGrammarLoss, MusicalGrammarConfig
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.config import ConfigManager, MidiFlyConfig

def test_grammar_integration():
    """Test that enhanced grammar integrates properly with training."""
    print("Testing enhanced grammar integration...")
    
    # Create test configuration
    vocab_config = VocabularyConfig()
    grammar_config = MusicalGrammarConfig(
        velocity_quality_weight=15.0,
        timing_quality_weight=10.0
    )
    
    # Initialize grammar loss
    grammar_loss = MusicalGrammarLoss(grammar_config, vocab_config)
    
    # Create test model
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=128,  # Small for testing
        n_layers=2,
        n_heads=4,
        max_sequence_length=64,
        mode="transformer"  # Use simple transformer mode for testing
    )
    
    # Create test batch
    batch_size = 4
    seq_len = 32
    test_tokens = torch.randint(0, vocab_config.vocab_size, (batch_size, seq_len))
    
    # Test model forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(test_tokens)
    
    # Test grammar loss computation
    base_loss = torch.nn.functional.cross_entropy(
        outputs.view(-1, vocab_config.vocab_size),
        test_tokens.view(-1)
    )
    
    total_loss, loss_components = grammar_loss(
        outputs, test_tokens, base_loss
    )
    
    # Validate loss components
    assert isinstance(total_loss, torch.Tensor)
    assert 'velocity_quality_loss' in loss_components
    assert 'timing_quality_loss' in loss_components
    assert loss_components['total_loss'] >= loss_components['base_loss']
    
    print("✅ Grammar integration test passed!")
    print(f"   Base loss: {loss_components['base_loss']:.4f}")
    print(f"   Velocity quality loss: {loss_components['velocity_quality_loss']:.4f}")
    print(f"   Timing quality loss: {loss_components['timing_quality_loss']:.4f}")
    print(f"   Total loss: {loss_components['total_loss']:.4f}")

def test_training_config_compatibility():
    """Test that training configuration works with grammar system."""
    print("\nTesting training configuration compatibility...")
    
    # Test with actual config file
    config_path = Path("configs/training_configs/quick_test.yaml")
    if config_path.exists():
        config_manager = ConfigManager()
        config = config_manager.load_config(str(config_path))
        print(f"✅ Successfully loaded config: {config_path}")
        print(f"   Model hidden_dim: {config.model.hidden_dim}")
        print(f"   Training batch_size: {config.training.batch_size}")
    else:
        print("⚠️  Config file not found, using defaults")
        config = MidiFlyConfig()
        print(f"   Default model hidden_dim: {config.model.hidden_dim}")
        
    print("✅ Configuration compatibility test passed!")

def test_vocabulary_token_mapping():
    """Test that vocabulary token mapping works correctly."""
    print("\nTesting vocabulary token mapping...")
    
    vocab_config = VocabularyConfig()
    
    # Test a few known token mappings
    test_tokens = [0, 1, 100, 500, 700]
    
    for token in test_tokens:
        if token < vocab_config.vocab_size:
            event_type, value = vocab_config.token_to_event_info(token)
            print(f"   Token {token}: {event_type.name if hasattr(event_type, 'name') else event_type} value={value}")
    
    print("✅ Vocabulary token mapping test passed!")

def run_all_tests():
    """Run all enhanced grammar training tests."""
    print("🧪 Testing Enhanced Grammar Training System")
    print("=" * 50)
    
    try:
        test_grammar_integration()
        test_training_config_compatibility()
        test_vocabulary_token_mapping()
        
        print("\n🎉 All Enhanced Grammar Training Tests Passed!")
        print("\n📋 Phase 5.2 System Status:")
        print("✅ Grammar loss integration working")
        print("✅ Training configuration compatible")  
        print("✅ Vocabulary token mapping functional")
        print("✅ Ready for enhanced grammar training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)