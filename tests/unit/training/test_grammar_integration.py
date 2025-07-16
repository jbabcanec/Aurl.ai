"""
Unit tests for Grammar Integration components.

Tests the Section 5.3 Enhanced Training Pipeline implementation:
- GrammarEnhancedTraining functionality
- GrammarTrainingConfig validation
- Automatic rollback mechanism
- Integration with AdvancedTrainer

Author: Claude Code Assistant
Phase: 5.3 Enhanced Training Pipeline
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.training.utils.grammar_integration import (
    GrammarEnhancedTraining, 
    GrammarTrainingConfig, 
    GrammarTrainingState
)
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN


class TestGrammarTrainingConfig:
    """Test configuration class for grammar training."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GrammarTrainingConfig()
        
        assert config.grammar_loss_weight == 1.0
        assert config.grammar_validation_frequency == 50
        assert config.collapse_threshold == 0.5
        assert config.enable_rollback == True
        assert config.max_rollbacks == 3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GrammarTrainingConfig(
            grammar_loss_weight=2.0,
            collapse_threshold=0.7,
            enable_rollback=False
        )
        
        assert config.grammar_loss_weight == 2.0
        assert config.collapse_threshold == 0.7
        assert config.enable_rollback == False


class TestGrammarTrainingState:
    """Test grammar training state management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.state = GrammarTrainingState()
        self.config = GrammarTrainingConfig(
            collapse_threshold=0.5,
            collapse_patience=2,
            enable_rollback=True,
            max_rollbacks=3
        )
    
    def test_initial_state(self):
        """Test initial state values."""
        assert self.state.recent_grammar_scores == []
        assert self.state.consecutive_bad_scores == 0
        assert self.state.total_rollbacks == 0
        assert self.state.best_grammar_score == 0.0
        assert self.state.checkpoint_history == []
    
    def test_good_score_update(self):
        """Test updating with good grammar scores."""
        # Update with good score
        should_rollback = self.state.update_grammar_score(0.8, self.config)
        
        assert not should_rollback
        assert self.state.best_grammar_score == 0.8
        assert self.state.consecutive_bad_scores == 0
        assert len(self.state.recent_grammar_scores) == 1
    
    def test_bad_score_update(self):
        """Test updating with bad grammar scores."""
        # First bad score
        should_rollback = self.state.update_grammar_score(0.3, self.config)
        assert not should_rollback
        assert self.state.consecutive_bad_scores == 1
        
        # Second bad score - should trigger rollback
        self.state.add_checkpoint({'test': 'checkpoint'})  # Add checkpoint for rollback
        should_rollback = self.state.update_grammar_score(0.2, self.config)
        assert should_rollback
        assert self.state.consecutive_bad_scores == 2
    
    def test_emergency_stop(self):
        """Test emergency stop condition."""
        config = GrammarTrainingConfig(min_grammar_score=0.3)
        
        # Score below emergency threshold
        should_rollback = self.state.update_grammar_score(0.1, config)
        assert should_rollback
    
    def test_checkpoint_management(self):
        """Test checkpoint history management."""
        # Add multiple checkpoints
        for i in range(7):
            self.state.add_checkpoint({'checkpoint': i})
        
        # Should maintain max_history limit
        assert len(self.state.checkpoint_history) == 5  # max_history default
        assert self.state.checkpoint_history[0]['checkpoint'] == 2  # Oldest kept
        assert self.state.checkpoint_history[-1]['checkpoint'] == 6  # Newest
    
    def test_rollback_checkpoint_retrieval(self):
        """Test retrieving checkpoints for rollback."""
        # Add checkpoints
        for i in range(3):
            self.state.add_checkpoint({'step': i})
        
        # Get rollback checkpoint
        rollback_cp = self.state.get_rollback_checkpoint(steps=1)
        assert rollback_cp is not None
        assert rollback_cp['step'] == 1  # Second-to-last
        
        # Test insufficient history
        rollback_cp = self.state.get_rollback_checkpoint(steps=5)
        assert rollback_cp is None


class TestGrammarEnhancedTraining:
    """Test the main grammar enhancement class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.vocab_config = VocabularyConfig()
        
        # Create simple model for testing
        self.model = MusicTransformerVAEGAN(
            vocab_size=self.vocab_config.vocab_size,
            d_model=256,
            n_layers=2,
            n_heads=4,
            mode="transformer"
        )
        
        self.grammar_config = GrammarTrainingConfig(
            grammar_validation_frequency=5,  # Frequent for testing
            collapse_threshold=0.5
        )
        
        self.grammar_trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.grammar_trainer.model is self.model
        assert self.grammar_trainer.device == self.device
        assert self.grammar_trainer.vocab_config is self.vocab_config
        assert self.grammar_trainer.grammar_config is self.grammar_config
        assert self.grammar_trainer.batch_count == 0
        
        # Check components are initialized
        assert self.grammar_trainer.grammar_loss is not None
        assert self.grammar_trainer.sampler is not None
        assert self.grammar_trainer.grammar_state is not None
    
    def test_enhanced_loss_calculation(self):
        """Test enhanced loss calculation."""
        batch_size, seq_len, vocab_size = 2, 32, self.vocab_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        base_loss = torch.tensor(2.5)
        
        enhanced_loss, metrics = self.grammar_trainer.calculate_enhanced_loss(
            logits, targets, base_loss
        )
        
        assert isinstance(enhanced_loss, torch.Tensor)
        assert enhanced_loss.item() >= base_loss.item()  # Should be >= base loss
        
        # Check metrics
        assert 'base_loss' in metrics
        assert 'grammar_loss' in metrics
        assert 'total_loss' in metrics
        assert 'grammar_weight' in metrics
        
        assert metrics['base_loss'] == base_loss.item()
        assert metrics['total_loss'] == enhanced_loss.item()
    
    def test_validation_frequency(self):
        """Test grammar validation frequency."""
        # Initially should not validate
        assert not self.grammar_trainer.should_validate_grammar()
        
        # After reaching frequency, should validate
        self.grammar_trainer.batch_count = 5  # matches frequency
        assert self.grammar_trainer.should_validate_grammar()
        
        # Not on other batches
        self.grammar_trainer.batch_count = 6
        assert not self.grammar_trainer.should_validate_grammar()
    
    @patch('src.training.utils.grammar_integration.validate_generated_sequence')
    def test_generation_validation(self, mock_validate):
        """Test generation quality validation."""
        # Mock validation function
        mock_validate.return_value = {
            'grammar_score': 0.75,
            'note_pairing_score': 0.8,
            'timing_score': 0.7
        }
        
        # Mock generation to avoid actual model inference
        with patch.object(self.grammar_trainer.sampler, 'generate') as mock_generate:
            mock_generate.return_value = [torch.randint(0, 100, (32,))]
            
            validation_results = self.grammar_trainer.validate_generation_quality()
            
            assert validation_results['grammar_score'] == 0.75
            assert validation_results['num_samples'] == 3  # default validation_samples
            assert len(validation_results['individual_scores']) == 3
    
    def test_rollback_functionality(self):
        """Test automatic rollback mechanism."""
        # Create mock model and optimizer
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Save initial state
        initial_model_state = self.model.state_dict().copy()
        initial_optimizer_state = optimizer.state_dict().copy()
        
        # Save checkpoint for rollback
        self.grammar_trainer.save_checkpoint_for_rollback(
            model_state=initial_model_state,
            optimizer_state=initial_optimizer_state,
            epoch=0,
            batch_idx=0,
            grammar_score=0.8
        )
        
        # Modify model state
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Perform rollback
        success = self.grammar_trainer.perform_rollback(self.model, optimizer)
        
        assert success
        assert self.grammar_trainer.grammar_state.total_rollbacks == 1
        
        # Check state was restored (approximately, due to floating point precision)
        restored_state = self.model.state_dict()
        for key in initial_model_state:
            assert torch.allclose(initial_model_state[key], restored_state[key], atol=1e-6)
    
    def test_grammar_summary(self):
        """Test grammar summary generation."""
        # Update some state
        self.grammar_trainer.batch_count = 10
        self.grammar_trainer.grammar_state.best_grammar_score = 0.85
        
        summary = self.grammar_trainer.get_grammar_summary()
        
        assert summary['total_batches'] == 10
        assert summary['best_grammar_score'] == 0.85
        assert 'recent_grammar_scores' in summary
        assert 'consecutive_bad_scores' in summary
        assert 'checkpoint_history_length' in summary


# Integration test
class TestGrammarIntegrationIntegration:
    """Test integration with other components."""
    
    def test_import_all_components(self):
        """Test that all components can be imported."""
        from src.training.utils import (
            GrammarEnhancedTraining,
            GrammarTrainingConfig,
            GrammarTrainingState
        )
        
        # Should not raise ImportError
        assert GrammarEnhancedTraining is not None
        assert GrammarTrainingConfig is not None
        assert GrammarTrainingState is not None
    
    def test_config_file_compatibility(self):
        """Test that our config file can be loaded."""
        from src.utils.config import ConfigManager
        
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "training_configs" / "grammar_enhanced.yaml"
        
        if config_path.exists():
            config_manager = ConfigManager()
            config = config_manager.load_config(str(config_path))
            
            # Check that grammar-specific config exists
            assert hasattr(config, 'grammar')
            assert hasattr(config.grammar, 'grammar_loss_weight')
            assert hasattr(config.grammar, 'collapse_threshold')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])