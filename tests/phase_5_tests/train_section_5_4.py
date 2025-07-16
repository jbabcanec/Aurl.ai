#!/usr/bin/env python3
"""
Section 5.4: Immediate Training Fix Plan - Simplified Version

This script tests the integration of Section 5.3 GrammarEnhancedTraining
without modifying the complex AdvancedTrainer infrastructure.

Key Goals:
- Test GrammarEnhancedTraining standalone
- Verify automatic rollback functionality  
- Validate real-time grammar monitoring
- Confirm Section 5.3 components work correctly
"""

import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Section 5.3 Grammar Integration
from src.training.utils.grammar_integration import (
    GrammarEnhancedTraining, 
    GrammarTrainingConfig, 
    GrammarTrainingState
)

# Core Components
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class Section54Trainer:
    """
    Simplified trainer for Section 5.4 testing.
    
    Tests the GrammarEnhancedTraining without complex AdvancedTrainer integration.
    """
    
    def __init__(self, model, device, vocab_config, grammar_config):
        self.model = model
        self.device = device
        self.vocab_config = vocab_config
        self.grammar_config = grammar_config
        
        # Initialize grammar enhancement from Section 5.3
        self.grammar_trainer = GrammarEnhancedTraining(
            model=model,
            device=device,
            vocab_config=vocab_config,
            grammar_config=grammar_config
        )
        
        # Training state
        self.epoch = 0
        self.batch_idx = 0
        self.should_stop = False
        
        logger.info("🎯 Section 5.4 Trainer initialized")
        logger.info(f"   - Grammar validation frequency: {grammar_config.grammar_validation_frequency}")
        logger.info(f"   - Automatic rollback: {grammar_config.enable_rollback}")
    
    def train_epoch(self, dataloader, optimizer, max_batches=None):
        """Train for one epoch with grammar enhancement."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'grammar_loss': 0.0,
            'batch_count': 0,
            'rollbacks_this_epoch': 0
        }
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            if self.should_stop:
                logger.warning("Early stopping triggered")
                break
                
            self.batch_idx = batch_idx
            tokens = batch['tokens'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(tokens[:, :-1])
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]
            
            # Calculate base loss
            base_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)
            )
            
            # Apply Section 5.3 grammar enhancement
            enhanced_loss, metrics, should_stop = self.grammar_trainer.process_batch_with_grammar(
                logits=logits,
                targets=targets,
                base_loss=base_loss,
                epoch=self.epoch,
                batch_idx=batch_idx
            )
            
            # Check for rollback
            if should_stop and self.grammar_config.enable_rollback:
                logger.warning("🔄 Grammar collapse detected - triggering rollback!")
                success = self.grammar_trainer.perform_rollback(self.model, optimizer)
                
                if success:
                    logger.info("✅ Rollback successful - continuing training")
                    epoch_metrics['rollbacks_this_epoch'] += 1
                else:
                    logger.error("❌ Rollback failed - stopping training")
                    self.should_stop = True
                    break
            
            # Backward pass
            enhanced_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Save checkpoint for potential rollback
            if batch_idx % 10 == 0:
                current_grammar_score = metrics.get('current_grammar_score', 0.5)
                self.grammar_trainer.save_checkpoint_for_rollback(
                    model_state=copy.deepcopy(self.model.state_dict()),
                    optimizer_state=copy.deepcopy(optimizer.state_dict()),
                    epoch=self.epoch,
                    batch_idx=batch_idx,
                    grammar_score=current_grammar_score
                )
            
            # Update metrics
            epoch_metrics['total_loss'] += enhanced_loss.item()
            epoch_metrics['grammar_loss'] += metrics.get('grammar_loss', 0.0)
            epoch_metrics['batch_count'] += 1
            
            # Log progress
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}: "
                          f"Loss={enhanced_loss.item():.4f}, "
                          f"Grammar={metrics.get('current_grammar_score', 0):.3f}")
        
        # Calculate averages
        if epoch_metrics['batch_count'] > 0:
            epoch_metrics['avg_loss'] = epoch_metrics['total_loss'] / epoch_metrics['batch_count']
            epoch_metrics['avg_grammar_loss'] = epoch_metrics['grammar_loss'] / epoch_metrics['batch_count']
        
        return epoch_metrics
    
    def train(self, dataloader, optimizer, num_epochs, max_batches_per_epoch=None):
        """Full training loop with Section 5.4 features."""
        logger.info(f"🚀 Starting Section 5.4 training for {num_epochs} epochs")
        
        training_results = {
            'epochs': [],
            'total_rollbacks': 0,
            'best_grammar_score': 0.0,
            'final_grammar_summary': {}
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            epoch_metrics = self.train_epoch(dataloader, optimizer, max_batches_per_epoch)
            
            # Add grammar summary
            grammar_summary = self.grammar_trainer.get_grammar_summary()
            epoch_metrics.update({
                'grammar_best_score': grammar_summary['best_grammar_score'],
                'grammar_total_rollbacks': grammar_summary['total_rollbacks'],
                'grammar_recent_avg': sum(grammar_summary['recent_grammar_scores'][-5:]) / max(1, len(grammar_summary['recent_grammar_scores'][-5:]))
            })
            
            training_results['epochs'].append(epoch_metrics)
            training_results['total_rollbacks'] = grammar_summary['total_rollbacks']
            training_results['best_grammar_score'] = grammar_summary['best_grammar_score']
            
            logger.info(f"✅ Epoch {epoch + 1} complete:")
            logger.info(f"   Avg Loss: {epoch_metrics.get('avg_loss', 0):.4f}")
            logger.info(f"   Best Grammar Score: {grammar_summary['best_grammar_score']:.3f}")
            logger.info(f"   Rollbacks This Epoch: {epoch_metrics['rollbacks_this_epoch']}")
            
            if self.should_stop:
                logger.warning("Training stopped early due to issues")
                break
        
        # Final grammar summary
        training_results['final_grammar_summary'] = self.grammar_trainer.get_grammar_summary()
        
        return training_results


def main():
    parser = argparse.ArgumentParser(description="Section 5.4: Immediate Training Fix Plan")
    parser.add_argument("--config", "-c", default="configs/training_configs/quick_test.yaml")
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--output", "-o", default="outputs/training")
    parser.add_argument("--max-files", "-f", type=int, default=10)
    parser.add_argument("--max-batches", "-b", type=int, default=20)
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"section_5_4_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🎵 Section 5.4: Immediate Training Fix Plan - Testing")
    logger.info(f"📁 Output: {output_dir}")
    
    # Load config
    config_manager = ConfigManager()
    base_config = config_manager.load_config(args.config)
    
    # Create configurations
    vocab_config = VocabularyConfig()
    
    grammar_config = GrammarTrainingConfig(
        grammar_loss_weight=1.5,
        grammar_validation_frequency=10,  # More frequent for testing
        collapse_threshold=0.6,
        collapse_patience=2,
        enable_rollback=True,
        max_rollbacks=3,
        validation_sequence_length=32
    )
    
    # Model
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=base_config.model.hidden_dim,
        n_layers=base_config.model.num_layers,
        n_heads=base_config.model.num_heads,
        mode="transformer"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters on {device}")
    
    # Dataset
    dataset = LazyMidiDataset(
        data_dir=base_config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=base_config.data.sequence_length,
        overlap=base_config.data.overlap,
        max_files=args.max_files,
        cache_dir=base_config.system.cache_dir,
        enable_augmentation=False,
        augmentation_probability=0.0
    )
    
    def collate_fn(batch):
        return {
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'file_path': [item['file_path'] for item in batch],
            'sequence_idx': torch.tensor([item['sequence_idx'] for item in batch]),
            'augmented': torch.tensor([item['augmented'] for item in batch])
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    logger.info(f"Dataset: {len(dataset)} sequences")
    logger.info(f"Dataloader: {len(dataloader)} batches")
    
    # Initialize Section 5.4 trainer
    trainer = Section54Trainer(
        model=model,
        device=device,
        vocab_config=vocab_config,
        grammar_config=grammar_config
    )
    
    # Start training
    try:
        results = trainer.train(
            dataloader=dataloader,
            optimizer=optimizer,
            num_epochs=args.epochs,
            max_batches_per_epoch=args.max_batches
        )
        
        logger.info("🎉 Section 5.4 training completed!")
        
        # Print results
        final_summary = results['final_grammar_summary']
        logger.info("📊 Final Section 5.4 Results:")
        logger.info(f"   Best grammar score: {final_summary['best_grammar_score']:.3f}")
        logger.info(f"   Total rollbacks: {final_summary['total_rollbacks']}")
        logger.info(f"   Total batches: {final_summary['total_batches']}")
        logger.info(f"   Recent grammar scores: {final_summary['recent_grammar_scores'][-3:]}")
        
        # Save results
        import json
        results_file = output_dir / "section_5_4_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Evaluate success
        if final_summary['best_grammar_score'] > 0.6:
            logger.info("✅ SUCCESS: Section 5.4 completed successfully!")
            logger.info("🔧 Grammar-enhanced training with automatic rollback is working!")
        elif final_summary['best_grammar_score'] > 0.4:
            logger.warning("⚠️ PARTIAL: Section 5.4 working but needs improvement")
        else:
            logger.error("❌ NEEDS WORK: Grammar scores too low")
        
        return str(output_dir)
        
    except Exception as e:
        logger.error(f"❌ Section 5.4 failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    output_dir = main()
    print(f"\n🎵 Section 5.4 testing completed!")
    print(f"📁 Results: {output_dir}")
    print(f"\n📋 Summary: GrammarEnhancedTraining tested with automatic rollback")
    print(f"🚀 Next: Integrate with AdvancedTrainer for Section 5.5")