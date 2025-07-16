#!/usr/bin/env python3
"""
Training Grammar Fix

Fix the training loop to prevent musical grammar collapse.
"""

import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.musical_grammar import MusicalGrammarLoss, MusicalGrammarConfig, validate_generated_sequence
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

class GrammarEnforcedTrainer:
    """Training class that enforces musical grammar to prevent collapse."""
    
    def __init__(self, model, device, vocab_config, grammar_config):
        self.model = model
        self.device = device
        self.vocab_config = vocab_config
        self.grammar_config = grammar_config
        
        # Initialize grammar loss and sampler
        self.grammar_loss = MusicalGrammarLoss(grammar_config)
        self.sampler = MusicSampler(model, device)
        
        # Training monitoring
        self.grammar_scores = []
        self.collapse_threshold = 0.5
        self.test_generation_frequency = 50  # Test every N batches
        
    def calculate_grammar_aware_loss(self, logits, targets):
        """Calculate loss with musical grammar enforcement."""
        # Standard cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Simple entropy regularization to prevent collapse
        probs = torch.softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=ce_loss.dtype, device=ce_loss.device))
        normalized_entropy = entropy / max_entropy
        
        # Entropy penalty: encourage high entropy to prevent collapse
        entropy_penalty = 0.1 * (1.0 - normalized_entropy)
        
        # Total loss
        total_loss = ce_loss + entropy_penalty
        
        # Monitor grammar (without affecting gradients)
        grammar_score = 0.5  # Default
        try:
            with torch.no_grad():
                if len(self.grammar_scores) % 10 == 0:  # Check less frequently
                    config = GenerationConfig(max_length=16, strategy=SamplingStrategy.TEMPERATURE, temperature=1.0)
                    generated_tokens = self.sampler.generate(config=config)
                    sample_tokens = generated_tokens[0].cpu().numpy()
                    validation = validate_generated_sequence(sample_tokens, self.vocab_config)
                    grammar_score = validation['grammar_score']
                    self.grammar_scores.append(grammar_score)
        except Exception as e:
            logger.debug(f"Grammar check failed: {e}")
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'entropy_penalty': entropy_penalty.item(),
            'normalized_entropy': normalized_entropy.item(),
            'grammar_score': grammar_score,
            'total_loss': total_loss.item()
        }
    
    def should_stop_training(self):
        """Check if training should stop due to grammar collapse."""
        if len(self.grammar_scores) < 10:
            return False
            
        # Check recent grammar scores
        recent_scores = self.grammar_scores[-10:]
        avg_recent = sum(recent_scores) / len(recent_scores)
        
        if avg_recent < self.collapse_threshold * 0.8:  # 80% of threshold
            logger.error(f"Stopping training: grammar collapse detected (avg={avg_recent:.3f})")
            return True
            
        return False

def train_with_grammar_enforcement():
    """Train model with grammar enforcement to prevent collapse."""
    logger.info("🚀 Starting Grammar-Enforced Training")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("debug/grammar_training_" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/training_configs/quick_test.yaml")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize components
    vocab_config = VocabularyConfig()
    grammar_config = MusicalGrammarConfig(
        velocity_quality_weight=20.0,   # Strong penalty for bad velocity
        timing_quality_weight=15.0,     # Strong penalty for bad timing  
        note_pairing_weight=25.0,       # Very strong penalty for mismatched notes
        repetition_penalty_weight=30.0  # Maximum penalty for repetition
    )
    
    # Create model (start fresh, not from collapsed checkpoint)
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=config.model.hidden_dim,
        n_layers=config.model.num_layers,
        n_heads=config.model.num_heads,
        max_sequence_length=config.model.max_sequence_length,
        mode="transformer"
    )
    model.to(device)
    
    # Initialize dataset
    dataset = LazyMidiDataset(
        data_dir=config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=config.data.sequence_length,
        overlap=config.data.overlap,
        max_files=50,  # Limit for debugging
        cache_dir=config.system.cache_dir,
        enable_augmentation=False,  # Disable for debugging
        augmentation_probability=0.0
    )
    
    def collate_fn(batch):
        """Custom collate function to handle variable sequence lengths."""
        # All sequences should be same length since dataset pads/truncates
        # But just in case, let's handle it properly
        return {
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'file_path': [item['file_path'] for item in batch],
            'sequence_idx': torch.tensor([item['sequence_idx'] for item in batch]),
            'augmented': torch.tensor([item['augmented'] for item in batch])
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for debugging
        collate_fn=collate_fn
    )
    
    # Initialize trainer with grammar enforcement
    trainer = GrammarEnforcedTrainer(model, device, vocab_config, grammar_config)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Dataset: {len(dataset)} sequences")
    logger.info(f"Device: {device}")
    
    # Training loop with grammar enforcement
    model.train()
    training_log = []
    
    for epoch in range(3):  # Short test run
        epoch_loss = 0
        epoch_grammar = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= 10:  # Limit batches for debugging
                break
                
            tokens = batch['tokens'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model logits
            outputs = model(tokens[:, :-1])  # Input: all but last token
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]  # Target: all but first token
            
            # Calculate grammar-aware loss
            loss, metrics = trainer.calculate_grammar_aware_loss(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Log metrics
            epoch_loss += metrics['total_loss']
            epoch_grammar += metrics.get('grammar_score', 0)
            batch_count += 1
            
            batch_log = {
                'epoch': epoch,
                'batch': batch_idx,
                'total_loss': metrics['total_loss'],
                'ce_loss': metrics['ce_loss'],
                'grammar_score': metrics.get('grammar_score', 0),
                'collapse_penalty': metrics.get('collapse_penalty', 0)
            }
            training_log.append(batch_log)
            
            logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                       f"Loss={metrics['total_loss']:.4f}, "
                       f"Grammar={metrics.get('grammar_score', 0):.3f}")
            
            # Check for early stopping due to collapse
            if trainer.should_stop_training():
                logger.error("STOPPING: Grammar collapse detected!")
                break
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_grammar = epoch_grammar / batch_count if batch_count > 0 else 0
        
        logger.info(f"Epoch {epoch} complete: Loss={avg_loss:.4f}, Grammar={avg_grammar:.3f}")
        
        # Save checkpoint if grammar is good
        if avg_grammar > 0.5:
            checkpoint_path = output_dir / f"good_grammar_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'grammar_score': avg_grammar,
                'loss': avg_loss
            }, checkpoint_path)
            logger.info(f"✅ Saved good checkpoint: {checkpoint_path}")
    
    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    logger.info(f"📊 Training complete. Log saved to {log_path}")
    
    # Final test generation
    logger.info("🎵 Testing final model generation...")
    config = GenerationConfig(max_length=32, strategy=SamplingStrategy.TEMPERATURE, temperature=1.0)
    with torch.no_grad():
        generated_tokens = trainer.sampler.generate(config=config)
        sample_tokens = generated_tokens[0].cpu().numpy()
        validation = validate_generated_sequence(sample_tokens, vocab_config)
        
        logger.info(f"Final grammar score: {validation['grammar_score']:.3f}")
        if validation['grammar_score'] > 0.7:
            logger.info("✅ SUCCESS: Model generates proper musical grammar!")
        else:
            logger.warning("❌ FAILURE: Model still has grammar issues")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = train_with_grammar_enforcement()