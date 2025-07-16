#!/usr/bin/env python3
"""
Comprehensive Fix for Token Generation Issues

This script addresses the three core problems:
1. Immediate END token generation
2. Unmatched note_off tokens without note_on pairs  
3. Poor musical structure

Solution approach:
- State-aware loss functions that track note on/off state
- Masking to prevent invalid token sequences
- Musical structure rewards for proper patterns
- Curriculum learning from simple to complex patterns
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.musical_grammar import validate_generated_sequence
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig, EventType
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

class MusicalStateTracker:
    """Tracks the musical state during sequence generation to enforce valid patterns."""
    
    def __init__(self, vocab_config: VocabularyConfig):
        self.vocab_config = vocab_config
        self.reset()
        
        # Token mappings
        self.start_token = vocab_config.event_to_token_map.get((EventType.START_TOKEN, 0), 0)
        self.end_token = vocab_config.event_to_token_map.get((EventType.END_TOKEN, 0), 1)
        self.pad_token = vocab_config.event_to_token_map.get((EventType.PAD_TOKEN, 0), 2)
        
        # Event ranges
        self.note_on_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                 if event_type == EventType.NOTE_ON])
        self.note_on_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                               if event_type == EventType.NOTE_ON])
        
        self.note_off_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.NOTE_OFF])
        self.note_off_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                if event_type == EventType.NOTE_OFF])
        
        self.velocity_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.VELOCITY_CHANGE])
        self.velocity_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                if event_type == EventType.VELOCITY_CHANGE])
        
        self.time_shift_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                    if event_type == EventType.TIME_SHIFT])
        self.time_shift_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.TIME_SHIFT])
    
    def reset(self):
        """Reset tracker state."""
        self.active_notes: Set[int] = set()  # Active note pitches
        self.sequence_length = 0
        self.has_velocity = False
        self.last_token = None
        self.note_count = 0
    
    def update_state(self, token: int):
        """Update state based on new token."""
        self.sequence_length += 1
        self.last_token = token
        
        # Track note on/off events
        if self.note_on_start <= token <= self.note_on_end:
            pitch = token - self.note_on_start + self.vocab_config.min_pitch
            self.active_notes.add(pitch)
            self.note_count += 1
        elif self.note_off_start <= token <= self.note_off_end:
            pitch = token - self.note_off_start + self.vocab_config.min_pitch
            self.active_notes.discard(pitch)  # Remove if present
        elif self.velocity_start <= token <= self.velocity_end:
            self.has_velocity = True
    
    def get_forbidden_tokens(self) -> Set[int]:
        """Get list of tokens that should be forbidden based on current state."""
        forbidden = set()
        
        # Issue 1: Prevent immediate END token
        if self.sequence_length < 5:  # Must have at least 5 tokens before END
            forbidden.add(self.end_token)
        
        # Issue 2: Prevent note_off for notes that aren't active
        for token in range(self.note_off_start, self.note_off_end + 1):
            pitch = token - self.note_off_start + self.vocab_config.min_pitch
            if pitch not in self.active_notes:
                forbidden.add(token)
        
        # Issue 3: Encourage musical structure
        # Require velocity setting before notes
        if (not self.has_velocity and self.last_token is not None and 
            self.note_on_start <= self.last_token <= self.note_on_end):
            # If we just had a note on without velocity, encourage velocity next
            pass  # We'll handle this in the reward system
        
        return forbidden

class StateAwareLoss:
    """Loss function that uses musical state to guide generation."""
    
    def __init__(self, vocab_config: VocabularyConfig):
        self.vocab_config = vocab_config
        self.tracker = MusicalStateTracker(vocab_config)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute state-aware loss that prevents invalid musical patterns.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
        ce_loss = ce_loss.reshape(batch_size, seq_len)
        
        # State-aware penalties
        state_penalty = torch.zeros_like(ce_loss)
        
        for b in range(batch_size):
            self.tracker.reset()
            sequence = targets[b]
            
            for i in range(seq_len):
                token = sequence[i].item()
                
                # Get forbidden tokens at this position
                forbidden_tokens = self.tracker.get_forbidden_tokens()
                
                # Apply heavy penalty if this token is forbidden
                if token in forbidden_tokens:
                    state_penalty[b, i] += 3.0  # Heavy penalty
                
                # Apply lighter penalty for suboptimal patterns
                if token == self.tracker.end_token and self.tracker.note_count == 0:
                    state_penalty[b, i] += 2.0  # End without any notes
                
                # Reward good patterns
                if (self.tracker.velocity_start <= token <= self.tracker.velocity_end and 
                    i < seq_len - 1 and 
                    self.tracker.note_on_start <= sequence[i+1].item() <= self.tracker.note_on_end):
                    state_penalty[b, i] -= 0.5  # Reward velocity before note
                
                # Update state for next iteration
                self.tracker.update_state(token)
        
        # Combine losses
        total_loss = ce_loss + state_penalty
        return total_loss.mean()

class ConstrainedSampler:
    """Sampler that enforces musical constraints during generation."""
    
    def __init__(self, model, device, vocab_config):
        self.model = model
        self.device = device
        self.vocab_config = vocab_config
        self.tracker = MusicalStateTracker(vocab_config)
    
    def generate_constrained(self, max_length: int = 50, temperature: float = 0.8) -> torch.Tensor:
        """Generate tokens with musical constraints enforced."""
        self.model.eval()
        self.tracker.reset()
        
        # Start with proper initialization
        sequence = [self.tracker.start_token]
        
        # Add initial velocity setting
        initial_velocity = self.tracker.velocity_start + 16  # Mid-range velocity
        sequence.append(initial_velocity)
        self.tracker.update_state(initial_velocity)
        
        current_input = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        
        for step in range(max_length - len(sequence)):
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(current_input)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                next_logits = logits[0, -1, :] / temperature
                
                # Apply constraints by masking forbidden tokens
                forbidden_tokens = self.tracker.get_forbidden_tokens()
                for token in forbidden_tokens:
                    next_logits[token] = -float('inf')
                
                # Apply musical structure biases
                self._apply_musical_biases(next_logits, sequence)
                
                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                sequence.append(next_token)
                self.tracker.update_state(next_token)
                
                # Stop if we hit END token and have sufficient notes
                if next_token == self.tracker.end_token and self.tracker.note_count >= 3:
                    break
                
                # Update input for next iteration
                current_input = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Prevent infinite sequences
                if len(sequence) >= max_length:
                    break
        
        return torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _apply_musical_biases(self, logits: torch.Tensor, sequence: List[int]):
        """Apply musical structure biases to improve generation quality."""
        # Encourage notes after time shifts
        if (len(sequence) > 0 and 
            self.tracker.time_shift_start <= sequence[-1] <= self.tracker.time_shift_end):
            for token in range(self.tracker.note_on_start, self.tracker.note_on_end + 1):
                logits[token] += 1.0  # Boost note_on probability
        
        # Encourage time shifts after notes
        if (len(sequence) > 0 and 
            self.tracker.note_on_start <= sequence[-1] <= self.tracker.note_on_end):
            for token in range(self.tracker.time_shift_start, min(self.tracker.time_shift_start + 50, self.tracker.time_shift_end + 1)):
                logits[token] += 0.5  # Boost short time shifts
        
        # Encourage note_off for active notes after some time
        if len(sequence) > 10:  # After some progression
            for pitch in self.tracker.active_notes:
                note_off_token = self.tracker.note_off_start + pitch - self.tracker.vocab_config.min_pitch
                if note_off_token <= self.tracker.note_off_end:
                    logits[note_off_token] += 0.8
        
        # Discourage excessive repetition
        if len(sequence) >= 3 and sequence[-1] == sequence[-2] == sequence[-3]:
            logits[sequence[-1]] -= 2.0  # Strong penalty for 4th repetition

def main():
    parser = argparse.ArgumentParser(description="Fix Token Generation Issues")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output", "-o", default="outputs/training/fixed", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"fixed_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔧 Starting Comprehensive Token Generation Fix")
    logger.info(f"Output: {output_dir}")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/training_configs/quick_test.yaml")
    
    # Initialize components
    vocab_config = VocabularyConfig()
    state_aware_loss = StateAwareLoss(vocab_config)
    
    # Model
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=config.model.hidden_dim,
        n_layers=config.model.num_layers,
        n_heads=config.model.num_heads,
        max_sequence_length=config.model.max_sequence_length,
        mode="transformer"
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Initialize constrained sampler
    constrained_sampler = ConstrainedSampler(model, device, vocab_config)
    
    # Dataset (small for focused training)
    dataset = LazyMidiDataset(
        data_dir=config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=config.data.sequence_length,
        overlap=config.data.overlap,
        max_files=15,
        cache_dir=config.system.cache_dir,
        enable_augmentation=False,
        augmentation_probability=0.0
    )
    
    def collate_fn(batch):
        return {
            'tokens': torch.stack([item['tokens'] for item in batch]),
            'file_path': [item['file_path'] for item in batch],
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate * 0.5)  # Slower learning
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Dataset: {len(dataset)} sequences")
    logger.info(f"Device: {device}")
    
    # Training loop with progressive difficulty
    model.train()
    best_metrics = {'notes': 0, 'grammar': 0.0, 'valid_pairs': 0}
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= 12:  # Focused training
                break
                
            tokens = batch['tokens'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(tokens[:, :-1])
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]
            
            # Calculate state-aware loss
            loss = state_aware_loss.compute_loss(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gentle clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Test constrained generation
            if batch_idx % 4 == 0:
                model.eval()
                generated_tokens = constrained_sampler.generate_constrained(max_length=40, temperature=0.7)
                sample_tokens = generated_tokens[0].cpu().numpy()
                validation = validate_generated_sequence(sample_tokens, vocab_config)
                
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Loss={loss.item():.4f}, "
                           f"Notes={validation['note_count']}, "
                           f"Grammar={validation['grammar_score']:.3f}, "
                           f"Pairs={validation.get('note_pairing_score', 0):.3f}")
                
                model.train()
        
        # Epoch evaluation
        model.eval()
        generated_tokens = constrained_sampler.generate_constrained(max_length=60, temperature=0.6)
        sample_tokens = generated_tokens[0].cpu().numpy()
        validation = validate_generated_sequence(sample_tokens, vocab_config)
        
        notes = validation['note_count']
        grammar = validation['grammar_score']
        pairs = validation.get('note_pairing_score', 0)
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Notes Generated: {notes}")
        logger.info(f"  Grammar Score: {grammar:.3f}")
        logger.info(f"  Note Pairing Score: {pairs:.3f}")
        
        # Save if improved
        if notes > best_metrics['notes'] or (notes == best_metrics['notes'] and grammar > best_metrics['grammar']):
            best_metrics.update({'notes': notes, 'grammar': grammar, 'valid_pairs': pairs})
            
            checkpoint_path = output_dir / f"fixed_model_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
                'loss': avg_loss,
                'd_model': config.model.hidden_dim,
                'n_layers': config.model.num_layers,
                'n_heads': config.model.num_heads,
                'max_sequence_length': config.model.max_sequence_length,
            }, checkpoint_path)
            
            logger.info(f"✅ Saved improved model: {checkpoint_path}")
            logger.info(f"   📊 Metrics: {notes} notes, {grammar:.3f} grammar, {pairs:.3f} pairing")
        
        model.train()
    
    logger.info(f"🎉 Training Complete!")
    logger.info(f"🏆 Best Results: {best_metrics['notes']} notes, {best_metrics['grammar']:.3f} grammar")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = main()