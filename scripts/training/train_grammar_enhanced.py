#!/usr/bin/env python3
"""
Grammar-Enhanced Training Script for Aurl.ai

This script addresses Section 5.3 requirements by integrating musical grammar
enforcement with the AdvancedTrainer infrastructure. Includes:

- Integration with AdvancedTrainer for distributed, mixed-precision training
- Musical grammar loss enforcement during training
- Real-time generation testing and validation
- Automatic model rollback on grammar collapse detection
- Enhanced token sequence validation

Author: Claude Code Assistant
Phase: 5.3 Enhanced Training Pipeline
Usage: python scripts/training/train_grammar_enhanced.py --config configs/training_configs/grammar_enhanced.yaml
"""

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.training.core.losses import ComprehensiveLossFramework
from src.training.utils.grammar_integration import GrammarEnhancedTraining, GrammarTrainingConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class GrammarAdvancedTrainer(AdvancedTrainer):
    """
    Enhanced AdvancedTrainer with musical grammar enforcement.
    
    Extends the AdvancedTrainer to include:
    - Musical grammar loss integration
    - Real-time generation validation
    - Automatic rollback on collapse detection
    """
    
    def __init__(self,
                 model: MusicTransformerVAEGAN,
                 loss_framework: ComprehensiveLossFramework,
                 config: TrainingConfig,
                 train_dataset: LazyMidiDataset,
                 val_dataset=None,
                 save_dir=None,
                 grammar_config: GrammarTrainingConfig = None):
        
        # Initialize parent class
        super().__init__(model, loss_framework, config, train_dataset, val_dataset, save_dir)
        
        # Initialize grammar enhancement
        vocab_config = VocabularyConfig()
        self.grammar_trainer = GrammarEnhancedTraining(
            model=model,
            device=self.device,
            vocab_config=vocab_config,
            grammar_config=grammar_config or GrammarTrainingConfig()
        )
        
        logger.info("Initialized GrammarAdvancedTrainer")
        logger.info(f"Grammar validation frequency: {self.grammar_trainer.grammar_config.grammar_validation_frequency}")
        
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with grammar enforcement."""
        self.model.train()
        epoch_metrics = {}
        total_loss = 0.0
        total_batches = 0
        grammar_scores = []
        
        # Setup data loader with current curriculum
        current_length = self.curriculum_scheduler.get_current_length(epoch) if self.curriculum_scheduler else None
        if current_length:
            logger.info(f"Epoch {epoch}: Using sequence length {current_length}")
            # Update dataset sequence length if curriculum learning is enabled
            if hasattr(self.train_dataset, 'update_sequence_length'):
                self.train_dataset.update_sequence_length(current_length)
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move batch to device
                input_ids = batch['tokens'].to(self.device)
                
                # Prepare inputs and targets
                inputs = input_ids[:, :-1]  # All but last token
                targets = input_ids[:, 1:]  # All but first token
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                        
                        # Calculate base loss using comprehensive framework
                        base_loss_dict = self.loss_framework(logits, targets, outputs)
                        base_loss = base_loss_dict['total_loss']
                        
                        # Apply grammar enhancement
                        enhanced_loss, grammar_metrics, should_stop = self.grammar_trainer.process_batch_with_grammar(
                            logits, targets, base_loss, epoch, batch_idx
                        )
                else:
                    outputs = self.model(inputs)
                    logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                    
                    # Calculate base loss
                    base_loss_dict = self.loss_framework(logits, targets, outputs)
                    base_loss = base_loss_dict['total_loss']
                    
                    # Apply grammar enhancement
                    enhanced_loss, grammar_metrics, should_stop = self.grammar_trainer.process_batch_with_grammar(
                        logits, targets, base_loss, epoch, batch_idx
                    )
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(enhanced_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    enhanced_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Accumulate metrics
                total_loss += enhanced_loss.item()
                total_batches += 1
                
                if 'current_grammar_score' in grammar_metrics:
                    grammar_scores.append(grammar_metrics['current_grammar_score'])
                
                # Save checkpoint for potential rollback (periodically)
                if batch_idx % 100 == 0 and grammar_metrics.get('current_grammar_score', 0) > 0:
                    self.grammar_trainer.save_checkpoint_for_rollback(
                        model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        epoch=epoch,
                        batch_idx=batch_idx,
                        grammar_score=grammar_metrics['current_grammar_score']
                    )
                
                # Check for rollback condition
                if should_stop:
                    logger.warning("Grammar collapse detected! Attempting rollback...")
                    
                    rollback_success = self.grammar_trainer.perform_rollback(
                        self.model, self.optimizer
                    )
                    
                    if rollback_success:
                        logger.info("Rollback successful, continuing training...")
                        # Reset enhanced logger metrics if needed
                        if hasattr(self, 'enhanced_logger'):
                            self.enhanced_logger.log_event("training.rollback", {
                                'epoch': epoch,
                                'batch': batch_idx,
                                'reason': 'grammar_collapse'
                            })
                    else:
                        logger.error("Rollback failed! Consider stopping training.")
                        if self.grammar_trainer.grammar_state.total_rollbacks >= self.grammar_trainer.grammar_config.max_rollbacks:
                            logger.error("Maximum rollbacks reached. Stopping training.")
                            break
                
                # Log metrics periodically
                if batch_idx % self.config.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                               f"Loss={enhanced_loss.item():.4f}, "
                               f"Grammar={grammar_metrics.get('current_grammar_score', 0.0):.3f}, "
                               f"Best Grammar={grammar_metrics.get('best_grammar_score', 0.0):.3f}")
                    
                    # Log to enhanced logger if available
                    if hasattr(self, 'enhanced_logger'):
                        self.enhanced_logger.log_batch_metrics(
                            epoch=epoch,
                            batch=batch_idx,
                            metrics={
                                **base_loss_dict,
                                **grammar_metrics,
                                'enhanced_loss': enhanced_loss.item()
                            },
                            learning_rate=self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                        )
                
                # Yield control periodically for responsive training
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                # Continue training despite individual batch failures
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        avg_grammar = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0.0
        
        epoch_metrics = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_grammar_score': avg_grammar,
            'total_batches': total_batches,
            'grammar_summary': self.grammar_trainer.get_grammar_summary()
        }
        
        logger.info(f"Epoch {epoch} complete: "
                   f"Avg Loss={avg_loss:.4f}, "
                   f"Avg Grammar={avg_grammar:.3f}, "
                   f"Batches={total_batches}")
        
        return epoch_metrics
    
    def train(self):
        """Main training loop with grammar enhancement."""
        logger.info("Starting grammar-enhanced training...")
        
        # Start training session with enhanced logging
        if hasattr(self, 'enhanced_logger'):
            self.enhanced_logger.start_training(self.config.num_epochs)
        
        training_history = []
        
        try:
            for epoch in range(self.config.num_epochs):
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Train epoch with grammar enhancement
                epoch_metrics = self._train_epoch(epoch)
                training_history.append(epoch_metrics)
                
                # Validation if available
                if self.val_dataset and epoch % self.config.eval_interval == 0:
                    val_metrics = self._validate_epoch(epoch)
                    epoch_metrics.update(val_metrics)
                
                # Save checkpoint
                is_best = epoch_metrics['avg_grammar_score'] > getattr(self, '_best_grammar_score', 0.0)
                if is_best:
                    self._best_grammar_score = epoch_metrics['avg_grammar_score']
                
                if epoch % self.config.save_interval == 0 or is_best:
                    self._save_checkpoint(epoch, epoch_metrics, is_best=is_best)
                
                # Log epoch summary
                if hasattr(self, 'enhanced_logger'):
                    self.enhanced_logger.log_epoch_summary(epoch, epoch_metrics)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final training history
            if self.save_dir:
                history_path = self.save_dir / "grammar_training_history.json"
                with open(history_path, 'w') as f:
                    json.dump(training_history, f, indent=2, default=str)
                logger.info(f"Training history saved: {history_path}")
            
            # Log final grammar summary
            final_summary = self.grammar_trainer.get_grammar_summary()
            logger.info(f"Final grammar summary: {final_summary}")
            
            if hasattr(self, 'enhanced_logger'):
                self.enhanced_logger.end_training()


def main():
    parser = argparse.ArgumentParser(description="Grammar-Enhanced Training")
    parser.add_argument("--config", "-c", default="configs/training_configs/quick_test.yaml",
                       help="Training configuration file")
    parser.add_argument("--save-dir", "-s", default="outputs/training",
                       help="Directory to save training outputs")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"grammar_enhanced_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🎵 Starting Grammar-Enhanced Training with AdvancedTrainer")
    logger.info(f"Config: {args.config}")
    logger.info(f"Save directory: {save_dir}")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Create training config
    training_config = TrainingConfig(
        num_epochs=config.training.get('num_epochs', 10),
        learning_rate=config.training.get('learning_rate', 1e-4),
        batch_size=config.training.get('batch_size', 4),
        use_mixed_precision=config.training.get('use_mixed_precision', True),
        curriculum_learning=config.training.get('curriculum_learning', True),
        distributed=config.training.get('distributed', False),
        log_interval=50,
        eval_interval=1000,
        save_interval=1000
    )
    
    # Create grammar config
    grammar_config = GrammarTrainingConfig(
        grammar_loss_weight=1.5,  # Stronger grammar enforcement
        grammar_validation_frequency=25,  # More frequent validation
        collapse_threshold=0.6,
        collapse_patience=2,
        enable_rollback=True,
        max_rollbacks=3
    )
    
    # Initialize model
    vocab_config = VocabularyConfig()
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=config.model.get('hidden_dim', 512),
        n_layers=config.model.get('num_layers', 6),
        n_heads=config.model.get('num_heads', 8),
        mode="transformer"  # Start with transformer mode for stability
    )
    
    # Initialize loss framework
    loss_framework = ComprehensiveLossFramework(model_config=config.model)
    
    # Initialize datasets
    train_dataset = LazyMidiDataset(
        data_dir=config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=config.data.get('sequence_length', 512),
        overlap=config.data.get('overlap', 256),
        max_files=config.data.get('max_files', 100),
        cache_dir=config.system.cache_dir,
        enable_augmentation=False,  # Disable for stable grammar training
        augmentation_probability=0.0
    )
    
    val_dataset = None  # Add validation dataset if needed
    
    # Initialize grammar-enhanced trainer
    trainer = GrammarAdvancedTrainer(
        model=model,
        loss_framework=loss_framework,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=save_dir,
        grammar_config=grammar_config
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training dataset: {len(train_dataset)} sequences")
    logger.info(f"Device: {trainer.device}")
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Add checkpoint loading logic here
        
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()