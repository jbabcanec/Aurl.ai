#!/usr/bin/env python3
"""
Aurl.ai Section 5.4: Immediate Training Fix Plan

Integrates Section 5.3 GrammarEnhancedTraining with AdvancedTrainer
to provide production-ready training with automatic rollback and
real-time grammar monitoring.

Section 5.4 Goals:
- Replace basic training loop with AdvancedTrainer integration
- Use GrammarEnhancedTraining for automatic model rollback
- Add real-time grammar monitoring every 10-50 batches
- Implement grammar-based early stopping
"""

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Section 5.3 Grammar Integration
from src.training.utils.grammar_integration import (
    GrammarEnhancedTraining, 
    GrammarTrainingConfig, 
    GrammarTrainingState
)

# Advanced Training Infrastructure
from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.training.core.losses import ComprehensiveLossFramework

# Core Components
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

class GrammarAdvancedTrainer(AdvancedTrainer):
    """
    Section 5.4: Integration of AdvancedTrainer with GrammarEnhancedTraining.
    
    Combines the production-ready AdvancedTrainer infrastructure with
    Section 5.3 grammar enforcement for automatic rollback and monitoring.
    """
    
    def __init__(self, 
                 model,
                 loss_framework,
                 config: TrainingConfig,
                 train_dataset,
                 val_dataset=None,
                 save_dir=None,
                 grammar_config: GrammarTrainingConfig = None):
        # Initialize AdvancedTrainer first
        super().__init__(model, loss_framework, config, train_dataset, val_dataset, save_dir)
        
        # Initialize grammar enhancement
        self.vocab_config = VocabularyConfig()
        self.grammar_config = grammar_config or GrammarTrainingConfig(
            grammar_validation_frequency=25,  # Check every 25 batches
            collapse_threshold=0.6,
            enable_rollback=True,
            max_rollbacks=3
        )
        
        # Initialize grammar-enhanced training
        self.grammar_trainer = GrammarEnhancedTraining(
            model=self.model,
            device=self.device,
            vocab_config=self.vocab_config,
            grammar_config=self.grammar_config
        )
        
        logger.info("Initialized GrammarAdvancedTrainer with automatic rollback")
        logger.info(f"Grammar validation frequency: {self.grammar_config.grammar_validation_frequency}")
        logger.info(f"Rollback enabled: {self.grammar_config.enable_rollback}")
    
    def compute_loss(self, batch):
        """
        Override AdvancedTrainer loss computation to include grammar enhancement.
        """
        tokens = batch['tokens'].to(self.device)
        
        # Forward pass
        outputs = self.model(tokens[:, :-1])  # Input: all but last token
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        targets = tokens[:, 1:]  # Target: all but first token
        
        # Calculate base loss using comprehensive framework
        base_loss = self.loss_framework(logits, targets, self.model)
        
        # Apply grammar enhancement
        enhanced_loss, metrics = self.grammar_trainer.calculate_enhanced_loss(
            logits, targets, base_loss
        )
        
        return enhanced_loss, metrics
    
    def training_step(self, batch, batch_idx):
        """
        Override training step to include grammar monitoring and rollback.
        """
        # Compute enhanced loss
        loss, metrics = self.compute_loss(batch)
        
        # Process batch with grammar monitoring
        enhanced_loss, enhanced_metrics, should_stop = self.grammar_trainer.process_batch_with_grammar(
            logits=None,  # Already calculated in compute_loss
            targets=None,  # Already calculated in compute_loss
            base_loss=loss,
            epoch=self.current_epoch,
            batch_idx=batch_idx
        )
        
        # Use enhanced loss for backward pass
        loss = enhanced_loss
        metrics.update(enhanced_metrics)
        
        # Check for grammar collapse and trigger rollback if needed
        if should_stop and self.grammar_config.enable_rollback:
            logger.warning("Grammar collapse detected - triggering rollback!")
            success = self.grammar_trainer.perform_rollback(self.model, self.optimizer)
            
            if success:
                logger.info("Model rollback successful - continuing training")
                # Reset metrics after rollback
                metrics['rollback_triggered'] = True
            else:
                logger.error("Model rollback failed - stopping training")
                self.should_stop = True
        
        # Save checkpoint periodically for potential rollback
        if batch_idx % 50 == 0:  # Every 50 batches
            current_grammar_score = metrics.get('current_grammar_score', 0.5)
            self.grammar_trainer.save_checkpoint_for_rollback(
                model_state=copy.deepcopy(self.model.state_dict()),
                optimizer_state=copy.deepcopy(self.optimizer.state_dict()),
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                grammar_score=current_grammar_score
            )
        
        return loss, metrics
    
    def on_epoch_end(self, epoch, metrics):
        """
        Override epoch end to include grammar summary.
        """
        super().on_epoch_end(epoch, metrics)
        
        # Add grammar summary to epoch metrics
        grammar_summary = self.grammar_trainer.get_grammar_summary()
        
        logger.info(f"Epoch {epoch} Grammar Summary:")
        logger.info(f"  Best grammar score: {grammar_summary['best_grammar_score']:.3f}")
        logger.info(f"  Total rollbacks: {grammar_summary['total_rollbacks']}")
        logger.info(f"  Recent scores: {grammar_summary['recent_grammar_scores'][-3:]}")
        
        # Add to metrics for tracking
        metrics.update({
            'grammar_best_score': grammar_summary['best_grammar_score'],
            'grammar_total_rollbacks': grammar_summary['total_rollbacks'],
            'grammar_recent_avg': sum(grammar_summary['recent_grammar_scores'][-5:]) / max(1, len(grammar_summary['recent_grammar_scores'][-5:]))
        })

def create_section_5_4_configs():
    """
    Create Section 5.4 configurations for grammar-enhanced training.
    
    Returns:
        Tuple of (TrainingConfig, GrammarTrainingConfig)
    """
    # Grammar configuration for Section 5.4
    grammar_config = GrammarTrainingConfig(
        grammar_loss_weight=1.5,           # Moderate grammar weight
        grammar_validation_frequency=25,    # Check every 25 batches (Section 5.4 req)
        collapse_threshold=0.6,             # Stricter than basic training
        collapse_patience=3,                # Allow 3 bad scores before rollback
        enable_rollback=True,               # Enable automatic rollback
        rollback_steps=1,                   # Roll back 1 checkpoint
        max_rollbacks=3,                    # Maximum 3 rollbacks per session
        validation_sequence_length=32,      # Generate 32 tokens for validation
        validation_temperature=1.0,         # Standard temperature
        validation_samples=3                # Test 3 samples per validation
    )
    
    # Training configuration using AdvancedTrainer
    training_config = TrainingConfig(
        num_epochs=10,
        batch_size=8,                       # Small batch for stable training
        learning_rate=1e-4,                 # Conservative learning rate
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,                  # Gradient clipping
        gradient_accumulation_steps=2,      # Effective batch size of 16
        use_mixed_precision=False,          # Disable for stability
        fp16=False,                         # Disable for stability
        distributed=False,                  # Single GPU training
        log_interval=10,
        eval_interval=50,
        save_interval=100
    )
    
    return training_config, grammar_config

def main():
    parser = argparse.ArgumentParser(description="Section 5.4: Immediate Training Fix Plan")
    parser.add_argument("--config", "-c", default="configs/training_configs/quick_test.yaml")
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--output", "-o", default="outputs/training")
    parser.add_argument("--max-files", "-f", type=int, default=20, help="Maximum files for training")
    parser.add_argument("--max-batches", "-b", type=int, default=50, help="Maximum batches per epoch")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"section_5_4_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🎵 Section 5.4: Immediate Training Fix Plan")
    logger.info("🔧 Integrating AdvancedTrainer with GrammarEnhancedTraining")
    logger.info(f"📁 Output: {output_dir}")
    
    # Load base config
    config_manager = ConfigManager()
    base_config = config_manager.load_config(args.config)
    
    # Create Section 5.4 configurations
    training_config, grammar_config = create_section_5_4_configs()
    
    # Override with CLI arguments
    training_config.num_epochs = args.epochs
    
    # Initialize components
    vocab_config = VocabularyConfig()
    
    # Model
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=base_config.model.hidden_dim,
        n_layers=base_config.model.num_layers,
        n_heads=base_config.model.num_heads,
        mode="transformer"
    )
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Dataset with custom collate function
    def collate_fn(batch):
        """Custom collate function for grammar-enhanced training."""
        return {
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'file_path': [item['file_path'] for item in batch],
            'sequence_idx': torch.tensor([item['sequence_idx'] for item in batch]),
            'augmented': torch.tensor([item['augmented'] for item in batch])
        }
    
    # Training dataset
    train_dataset = LazyMidiDataset(
        data_dir=base_config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=base_config.data.sequence_length,
        overlap=base_config.data.overlap,
        max_files=args.max_files,
        cache_dir=base_config.system.cache_dir,
        enable_augmentation=False,  # Disable for stable training
        augmentation_probability=0.0
    )
    
    # Note: AdvancedTrainer creates its own DataLoader internally
    
    logger.info(f"Dataset: {len(train_dataset)} sequences")
    
    # Initialize loss framework
    loss_framework = ComprehensiveLossFramework(vocab_config)
    
    # Initialize Section 5.4 trainer
    trainer = GrammarAdvancedTrainer(
        model=model,
        loss_framework=loss_framework,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=None,  # No validation for this test
        save_dir=output_dir,
        grammar_config=grammar_config
    )
    
    logger.info("✅ Section 5.4 trainer initialized with:")
    logger.info(f"   - Automatic rollback: {grammar_config.enable_rollback}")
    logger.info(f"   - Grammar validation frequency: {grammar_config.grammar_validation_frequency}")
    logger.info(f"   - Collapse threshold: {grammar_config.collapse_threshold}")
    logger.info(f"   - Maximum rollbacks: {grammar_config.max_rollbacks}")
    
    # Start training with limited batches for testing
    try:
        logger.info("🚀 Starting Section 5.4 training...")
        trainer.max_batches_per_epoch = args.max_batches  # Limit for testing
        
        final_metrics = trainer.train()
        
        logger.info("✅ Section 5.4 training completed successfully!")
        
        # Get final grammar summary
        grammar_summary = trainer.grammar_trainer.get_grammar_summary()
        
        logger.info("📊 Final Section 5.4 Results:")
        logger.info(f"   Best grammar score: {grammar_summary['best_grammar_score']:.3f}")
        logger.info(f"   Total rollbacks performed: {grammar_summary['total_rollbacks']}")
        logger.info(f"   Final batch count: {grammar_summary['total_batches']}")
        
        # Save final results
        results = {
            'section': '5.4',
            'description': 'Immediate Training Fix Plan',
            'completion_time': datetime.now().isoformat(),
            'grammar_summary': grammar_summary,
            'final_metrics': final_metrics,
            'config': {
                'training': training_config.__dict__,
                'grammar': grammar_config.__dict__
            }
        }
        
        import json
        results_path = output_dir / "section_5_4_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to: {results_path}")
        
        # Success criteria
        if grammar_summary['best_grammar_score'] > 0.6:
            logger.info("🎉 SUCCESS: Section 5.4 completed - grammar-enhanced training working!")
        else:
            logger.warning("⚠️ PARTIAL: Section 5.4 completed but grammar scores need improvement")
            
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"❌ Section 5.4 training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    output_dir = main()
    print(f"\n🎵 Section 5.4 Immediate Training Fix Plan completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review training logs and grammar scores")
    print(f"   2. Test generation quality with new model")
    print(f"   3. Proceed to Section 5.5: Training Pipeline Integration")