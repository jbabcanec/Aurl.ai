"""
Musical Grammar Integration for Advanced Training Pipeline

This module integrates musical grammar enforcement with the AdvancedTrainer
to address Section 5.3 requirements:
- Enhanced token sequence validation during training
- Automatic model rollback on collapse detection
- Integration with advanced training infrastructure

Author: Claude Code Assistant
Phase: 5.3 Enhanced Training Pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import copy

from ..musical_grammar import MusicalGrammarLoss, MusicalGrammarConfig, validate_generated_sequence
from ...generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from ...data.representation import VocabularyConfig
from ...utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class GrammarTrainingConfig:
    """Configuration for grammar-enforced training."""
    
    # Grammar enforcement
    grammar_loss_weight: float = 1.0
    grammar_validation_frequency: int = 50  # Validate every N batches
    
    # Collapse detection
    collapse_threshold: float = 0.5
    collapse_patience: int = 3  # Number of consecutive bad scores before rollback
    min_grammar_score: float = 0.3  # Absolute minimum before emergency stop
    
    # Rollback mechanism
    enable_rollback: bool = True
    rollback_steps: int = 1  # Number of checkpoints to roll back
    max_rollbacks: int = 3  # Maximum rollbacks per training session
    
    # Enhanced validation
    validation_sequence_length: int = 32
    validation_temperature: float = 1.0
    validation_samples: int = 3  # Number of samples to validate per check


class GrammarTrainingState:
    """Tracks grammar-related training state for rollback."""
    
    def __init__(self):
        self.recent_grammar_scores: List[float] = []
        self.consecutive_bad_scores: int = 0
        self.total_rollbacks: int = 0
        self.best_grammar_score: float = 0.0
        self.checkpoint_history: List[Dict] = []  # Store recent checkpoints for rollback
        
    def update_grammar_score(self, score: float, config: GrammarTrainingConfig) -> bool:
        """
        Update grammar score and check if rollback is needed.
        
        Returns:
            bool: True if rollback is recommended
        """
        self.recent_grammar_scores.append(score)
        
        # Keep only recent scores
        if len(self.recent_grammar_scores) > 20:
            self.recent_grammar_scores.pop(0)
            
        # Update best score
        if score > self.best_grammar_score:
            self.best_grammar_score = score
            self.consecutive_bad_scores = 0
        else:
            # Check if this is a bad score
            if score < config.collapse_threshold:
                self.consecutive_bad_scores += 1
            else:
                self.consecutive_bad_scores = 0
        
        # Check for rollback conditions
        should_rollback = (
            config.enable_rollback and
            self.consecutive_bad_scores >= config.collapse_patience and
            self.total_rollbacks < config.max_rollbacks and
            len(self.checkpoint_history) > 0
        )
        
        # Emergency stop condition
        if score < config.min_grammar_score:
            logger.error(f"EMERGENCY STOP: Grammar score {score:.3f} below minimum {config.min_grammar_score}")
            return True
            
        return should_rollback
    
    def add_checkpoint(self, checkpoint_data: Dict, max_history: int = 5):
        """Add checkpoint to history for potential rollback."""
        self.checkpoint_history.append(checkpoint_data)
        if len(self.checkpoint_history) > max_history:
            self.checkpoint_history.pop(0)
    
    def get_rollback_checkpoint(self, steps: int = 1):
        """Get checkpoint for rollback."""
        if len(self.checkpoint_history) >= steps:
            # Get the checkpoint 'steps' positions back from the end
            return self.checkpoint_history[-steps]
        return None


class GrammarEnhancedTraining:
    """
    Mixin class to add grammar enforcement to AdvancedTrainer.
    
    This class provides grammar-aware training capabilities that can be
    integrated into the existing AdvancedTrainer without major refactoring.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 vocab_config: VocabularyConfig,
                 grammar_config: Optional[GrammarTrainingConfig] = None):
        self.model = model
        self.device = device
        self.vocab_config = vocab_config
        self.grammar_config = grammar_config or GrammarTrainingConfig()
        
        # Initialize grammar components
        musical_config = MusicalGrammarConfig(
            note_pairing_weight=25.0,
            velocity_quality_weight=20.0,
            timing_quality_weight=15.0,
            repetition_penalty_weight=30.0
        )
        self.grammar_loss = MusicalGrammarLoss(musical_config, vocab_config)
        self.sampler = MusicSampler(model, device)
        
        # Training state
        self.grammar_state = GrammarTrainingState()
        self.batch_count = 0
        
        logger.info("Initialized GrammarEnhancedTraining")
        logger.info(f"Grammar validation frequency: {self.grammar_config.grammar_validation_frequency}")
        logger.info(f"Collapse threshold: {self.grammar_config.collapse_threshold}")
        logger.info(f"Rollback enabled: {self.grammar_config.enable_rollback}")
    
    def calculate_enhanced_loss(self, 
                               logits: torch.Tensor, 
                               targets: torch.Tensor,
                               base_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate loss with grammar enhancement.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]
            base_loss: Base loss from main loss framework
            
        Returns:
            enhanced_loss: Combined loss with grammar enforcement
            metrics: Dictionary of loss components and grammar metrics
        """
        # Apply musical grammar loss - pass logits directly
        # The MusicalGrammarLoss expects logits and will do argmax internally
        grammar_loss, grammar_metrics = self.grammar_loss(
            logits,  # Pass logits, not converted tokens
            targets,
            base_loss
        )
        
        # Combine losses
        total_loss = base_loss + self.grammar_config.grammar_loss_weight * grammar_loss
        
        # Enhanced metrics
        metrics = {
            'base_loss': base_loss.item(),
            'grammar_loss': grammar_loss.item(),
            'total_loss': total_loss.item(),
            'grammar_weight': self.grammar_config.grammar_loss_weight,
            **grammar_metrics
        }
        
        return total_loss, metrics
    
    def validate_generation_quality(self) -> Dict[str, float]:
        """
        Validate current model generation quality.
        
        Returns:
            Dictionary with validation metrics
        """
        try:
            self.model.eval()
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_length=self.grammar_config.validation_sequence_length,
                    strategy=SamplingStrategy.TEMPERATURE,
                    temperature=self.grammar_config.validation_temperature
                )
                
                scores = []
                for _ in range(self.grammar_config.validation_samples):
                    # Generate sample
                    generated_tokens = self.sampler.generate(config=generation_config)
                    if len(generated_tokens) > 0:
                        sample_tokens = generated_tokens[0].cpu().numpy()
                        
                        # Validate sequence
                        validation = validate_generated_sequence(sample_tokens, self.vocab_config)
                        scores.append(validation['grammar_score'])
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                return {
                    'grammar_score': avg_score,
                    'num_samples': len(scores),
                    'individual_scores': scores
                }
                
        except Exception as e:
            logger.warning(f"Generation validation failed: {e}")
            return {'grammar_score': 0.0, 'num_samples': 0, 'individual_scores': []}
        finally:
            self.model.train()
    
    def should_validate_grammar(self) -> bool:
        """Check if we should validate grammar this batch."""
        return self.batch_count > 0 and self.batch_count % self.grammar_config.grammar_validation_frequency == 0
    
    def process_batch_with_grammar(self,
                                  logits: torch.Tensor,
                                  targets: torch.Tensor,
                                  base_loss: torch.Tensor,
                                  epoch: int,
                                  batch_idx: int) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
        """
        Process batch with grammar enforcement.
        
        Args:
            logits: Model output logits
            targets: Target tokens
            base_loss: Base loss
            epoch: Current epoch
            batch_idx: Current batch index
            
        Returns:
            enhanced_loss: Loss with grammar enforcement
            metrics: Training metrics
            should_stop: Whether training should stop due to grammar collapse
        """
        self.batch_count += 1
        
        # Calculate enhanced loss
        enhanced_loss, metrics = self.calculate_enhanced_loss(logits, targets, base_loss)
        
        # Validate generation quality periodically
        grammar_score = 0.5  # Default
        should_stop = False
        
        if self.should_validate_grammar():
            validation_results = self.validate_generation_quality()
            grammar_score = validation_results['grammar_score']
            
            # Update grammar state and check for rollback
            should_rollback = self.grammar_state.update_grammar_score(
                grammar_score, 
                self.grammar_config
            )
            
            if should_rollback:
                logger.warning(f"Grammar collapse detected! Score: {grammar_score:.3f}")
                logger.warning(f"Consecutive bad scores: {self.grammar_state.consecutive_bad_scores}")
                should_stop = True  # Signal need for rollback
            
            # Log validation results
            logger.info(f"Grammar validation at batch {self.batch_count}: "
                       f"score={grammar_score:.3f}, "
                       f"best={self.grammar_state.best_grammar_score:.3f}")
        
        # Add grammar score to metrics
        metrics['current_grammar_score'] = grammar_score
        metrics['best_grammar_score'] = self.grammar_state.best_grammar_score
        metrics['consecutive_bad_scores'] = self.grammar_state.consecutive_bad_scores
        
        return enhanced_loss, metrics, should_stop
    
    def save_checkpoint_for_rollback(self, 
                                   model_state: Dict,
                                   optimizer_state: Dict,
                                   epoch: int,
                                   batch_idx: int,
                                   grammar_score: float):
        """Save checkpoint that can be used for rollback."""
        checkpoint_data = {
            'model_state_dict': copy.deepcopy(model_state),
            'optimizer_state_dict': copy.deepcopy(optimizer_state),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'grammar_score': grammar_score,
            'batch_count': self.batch_count
        }
        
        self.grammar_state.add_checkpoint(checkpoint_data)
        logger.debug(f"Saved checkpoint for potential rollback (epoch {epoch}, batch {batch_idx})")
    
    def perform_rollback(self, 
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer) -> bool:
        """
        Perform model rollback to previous good state.
        
        Returns:
            bool: True if rollback was successful
        """
        try:
            rollback_checkpoint = self.grammar_state.get_rollback_checkpoint(
                self.grammar_config.rollback_steps
            )
            
            if rollback_checkpoint is None:
                logger.error("No checkpoint available for rollback!")
                return False
            
            # Restore model and optimizer state
            model.load_state_dict(rollback_checkpoint['model_state_dict'])
            optimizer.load_state_dict(rollback_checkpoint['optimizer_state_dict'])
            
            # Update training state
            self.batch_count = rollback_checkpoint['batch_count']
            self.grammar_state.total_rollbacks += 1
            self.grammar_state.consecutive_bad_scores = 0
            
            logger.info(f"Successfully rolled back to epoch {rollback_checkpoint['epoch']}, "
                       f"batch {rollback_checkpoint['batch_idx']}")
            logger.info(f"Previous grammar score: {rollback_checkpoint['grammar_score']:.3f}")
            logger.info(f"Total rollbacks this session: {self.grammar_state.total_rollbacks}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_grammar_summary(self) -> Dict[str, Any]:
        """Get summary of grammar training statistics."""
        return {
            'total_batches': self.batch_count,
            'best_grammar_score': self.grammar_state.best_grammar_score,
            'total_rollbacks': self.grammar_state.total_rollbacks,
            'recent_grammar_scores': self.grammar_state.recent_grammar_scores[-10:],
            'consecutive_bad_scores': self.grammar_state.consecutive_bad_scores,
            'checkpoint_history_length': len(self.grammar_state.checkpoint_history)
        }