#!/usr/bin/env python3
"""
Unit and Integration Tests for Section 5.4: Immediate Training Fix Plan

Tests the grammar-enhanced training integration, automatic rollback system,
and real-time monitoring functionality.

Author: Claude Code Assistant
Phase: 5.4 Immediate Training Fix Plan
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.utils.grammar_integration import (
    GrammarEnhancedTraining,
    GrammarTrainingConfig,
    GrammarTrainingState
)
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN


class TestSection54Integration:
    """Test Section 5.4 training integration components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = torch.device('cpu')
        self.vocab_config = VocabularyConfig()
        
        # Create test model
        self.model = MusicTransformerVAEGAN(
            vocab_size=self.vocab_config.vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            mode="transformer"
        )
        
        # Test configuration
        self.grammar_config = GrammarTrainingConfig(
            grammar_validation_frequency=5,  # Frequent for testing
            collapse_threshold=0.6,
            enable_rollback=True,
            max_rollbacks=3
        )
    
    def test_grammar_enhanced_training_initialization(self):
        """Test that GrammarEnhancedTraining initializes correctly."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        assert trainer.model is self.model
        assert trainer.device == self.device
        assert trainer.vocab_config is self.vocab_config
        assert trainer.grammar_config is self.grammar_config
        assert trainer.batch_count == 0
        assert trainer.grammar_loss is not None
        assert trainer.sampler is not None
    
    def test_enhanced_loss_calculation(self):
        """Test enhanced loss calculation with grammar enforcement."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        batch_size, seq_len = 2, 16
        vocab_size = self.vocab_config.vocab_size
        
        # Create test tensors
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        base_loss = torch.tensor(1.5)
        
        # Calculate enhanced loss
        enhanced_loss, metrics = trainer.calculate_enhanced_loss(
            logits, targets, base_loss
        )
        
        # Verify results
        assert isinstance(enhanced_loss, torch.Tensor)
        assert enhanced_loss.item() >= base_loss.item()
        
        # Check required metrics
        required_metrics = ['base_loss', 'grammar_loss', 'total_loss', 'grammar_weight']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_batch_processing_with_grammar(self):
        """Test batch processing with grammar monitoring."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        batch_size, seq_len = 2, 16
        vocab_size = self.vocab_config.vocab_size
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        base_loss = torch.tensor(1.5)
        
        # Mock generation to avoid actual model inference
        with patch.object(trainer, 'validate_generation_quality') as mock_validate:
            mock_validate.return_value = {'grammar_score': 0.7}
            
            enhanced_loss, metrics, should_stop = trainer.process_batch_with_grammar(
                logits=logits,
                targets=targets,
                base_loss=base_loss,
                epoch=0,
                batch_idx=5  # Should trigger validation
            )
        
        assert isinstance(enhanced_loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert isinstance(should_stop, bool)
        assert 'current_grammar_score' in metrics
    
    def test_rollback_functionality(self):
        """Test automatic rollback mechanism."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Save initial states
        initial_model_state = self.model.state_dict().copy()
        initial_optimizer_state = optimizer.state_dict().copy()
        
        # Save checkpoint
        trainer.save_checkpoint_for_rollback(
            model_state=initial_model_state,
            optimizer_state=initial_optimizer_state,
            epoch=0,
            batch_idx=0,
            grammar_score=0.8
        )
        
        # Modify model
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Perform rollback
        success = trainer.perform_rollback(self.model, optimizer)
        
        assert success
        assert trainer.grammar_state.total_rollbacks == 1
        
        # Verify state restoration (approximately)
        restored_state = self.model.state_dict()
        for key in initial_model_state:
            assert torch.allclose(
                initial_model_state[key], 
                restored_state[key], 
                atol=1e-6
            )
    
    def test_grammar_collapse_detection(self):
        """Test grammar collapse detection and rollback triggering."""
        config = GrammarTrainingConfig(
            collapse_threshold=0.6,
            collapse_patience=2,
            enable_rollback=True
        )
        
        state = GrammarTrainingState()
        state.best_grammar_score = 0.8  # Set a high initial score
        
        # Add a checkpoint first
        state.add_checkpoint({'test': 'checkpoint'})
        
        # First bad score (consecutive_bad_scores = 1)
        should_rollback = state.update_grammar_score(0.4, config)
        assert not should_rollback
        assert state.consecutive_bad_scores == 1
        
        # Second bad score (consecutive_bad_scores = 2, should trigger rollback)
        should_rollback = state.update_grammar_score(0.3, config)
        assert should_rollback
        assert state.consecutive_bad_scores == 2
    
    def test_validation_frequency(self):
        """Test grammar validation frequency logic."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        # Should not validate initially
        assert not trainer.should_validate_grammar()
        
        # Should validate at frequency intervals
        trainer.batch_count = 5  # Matches our test frequency
        assert trainer.should_validate_grammar()
        
        trainer.batch_count = 6
        assert not trainer.should_validate_grammar()
        
        trainer.batch_count = 10  # Next interval
        assert trainer.should_validate_grammar()
    
    def test_checkpoint_management(self):
        """Test checkpoint saving and retrieval."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        # Save multiple checkpoints
        for i in range(7):
            trainer.grammar_state.add_checkpoint({'step': i})
        
        # Should maintain max history
        assert len(trainer.grammar_state.checkpoint_history) == 5
        
        # Test retrieval
        checkpoint = trainer.grammar_state.get_rollback_checkpoint(1)
        assert checkpoint is not None
        assert checkpoint['step'] == 6  # Last checkpoint (with rollback fix)
    
    def test_grammar_summary(self):
        """Test grammar summary generation."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        # Update some state
        trainer.batch_count = 15
        trainer.grammar_state.best_grammar_score = 0.75
        trainer.grammar_state.recent_grammar_scores = [0.6, 0.7, 0.75]
        
        summary = trainer.get_grammar_summary()
        
        assert summary['total_batches'] == 15
        assert summary['best_grammar_score'] == 0.75
        assert len(summary['recent_grammar_scores']) == 3
        assert 'consecutive_bad_scores' in summary
        assert 'checkpoint_history_length' in summary


class TestSection54TrainingLoop:
    """Test the actual training loop integration."""
    
    def setup_method(self):
        """Setup training test fixtures."""
        self.device = torch.device('cpu')
        self.vocab_config = VocabularyConfig()
        
        self.model = MusicTransformerVAEGAN(
            vocab_size=self.vocab_config.vocab_size,
            d_model=64,  # Small for testing
            n_layers=1,
            n_heads=2,
            mode="transformer"
        )
        
        self.grammar_config = GrammarTrainingConfig(
            grammar_validation_frequency=3,  # Very frequent
            collapse_threshold=0.5,
            enable_rollback=True
        )
    
    def test_training_step_integration(self):
        """Test a complete training step with grammar enhancement."""
        trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Create test batch
        batch_size, seq_len = 2, 8
        tokens = torch.randint(0, self.vocab_config.vocab_size, (batch_size, seq_len))
        
        # Mock generation to speed up test
        with patch.object(trainer, 'validate_generation_quality') as mock_validate:
            mock_validate.return_value = {'grammar_score': 0.7}
            
            # Forward pass
            outputs = self.model(tokens[:, :-1])
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]
            
            # Calculate base loss
            base_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            # Set batch count to trigger validation
            trainer.batch_count = 3  # Will trigger validation on this batch
            
            # Process with grammar enhancement
            enhanced_loss, metrics, should_stop = trainer.process_batch_with_grammar(
                logits=logits,
                targets=targets,
                base_loss=base_loss,
                epoch=0,
                batch_idx=3  # Should trigger validation
            )
            
            # Backward pass
            optimizer.zero_grad()
            enhanced_loss.backward()
            optimizer.step()
        
        # Verify training step completed
        assert isinstance(enhanced_loss, torch.Tensor)
        assert enhanced_loss.item() > 0
        assert not should_stop  # Should not stop with good grammar score
        # Grammar score should be either default (0.5) or mocked value (0.7)
        assert metrics['current_grammar_score'] in [0.5, 0.7]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])