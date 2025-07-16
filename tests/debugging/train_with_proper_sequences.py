#!/usr/bin/env python3
"""
Enhanced Training Script with Proper Sequence Structure
"""

import sys
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.musical_grammar import MusicalGrammarLoss, MusicalGrammarConfig, validate_generated_sequence
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig, EventType
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from src.utils.config import ConfigManager
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

class StructuralLoss:
    """Enhanced loss that enforces proper musical sequence structure."""
    
    def __init__(self, vocab_config):
        self.vocab_config = vocab_config
        
        # Get token IDs for key events
        self.start_token = vocab_config.event_to_token_map.get((EventType.START_TOKEN, 0), 0)
        self.end_token = vocab_config.event_to_token_map.get((EventType.END_TOKEN, 0), 1)
        self.pad_token = vocab_config.event_to_token_map.get((EventType.PAD_TOKEN, 0), 2)
        
        # Get ranges for different event types
        self.note_on_range = self._get_event_range(EventType.NOTE_ON)
        self.note_off_range = self._get_event_range(EventType.NOTE_OFF)
        self.velocity_range = self._get_event_range(EventType.VELOCITY_CHANGE)
        self.time_shift_range = self._get_event_range(EventType.TIME_SHIFT)
        
    def _get_event_range(self, event_type):
        """Get the token range for a specific event type."""
        tokens = []
        for (e_type, value), token in self.vocab_config.event_to_token_map.items():
            if e_type == event_type:
                tokens.append(token)
        return (min(tokens), max(tokens)) if tokens else (0, 0)
    
    def calculate_structural_loss(self, logits, targets):
        """Calculate loss that encourages proper sequence structure."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='none')
        ce_loss = ce_loss.reshape(batch_size, seq_len)
        
        # Structural penalties
        structure_penalty = torch.zeros_like(ce_loss)
        
        for b in range(batch_size):
            sequence = targets[b]
            
            # Penalty 1: Discourage END token early in sequence
            for i in range(min(10, seq_len)):  # First 10 positions
                if sequence[i] == self.end_token:
                    structure_penalty[b, i] += 2.0  # Strong penalty for early END
            
            # Penalty 2: Encourage proper note pairing
            active_notes = set()
            for i in range(seq_len):
                token = sequence[i].item()
                
                # Track note on/off pairs
                if self.note_on_range[0] <= token <= self.note_on_range[1]:
                    # Note ON
                    pitch = token - self.note_on_range[0] + self.vocab_config.min_pitch
                    active_notes.add(pitch)
                elif self.note_off_range[0] <= token <= self.note_off_range[1]:
                    # Note OFF
                    pitch = token - self.note_off_range[0] + self.vocab_config.min_pitch
                    if pitch not in active_notes:
                        # Note OFF without matching Note ON
                        structure_penalty[b, i] += 1.5
                    else:
                        active_notes.remove(pitch)
            
            # Penalty 3: Discourage too many consecutive identical tokens
            for i in range(1, seq_len):
                if sequence[i] == sequence[i-1] and i < seq_len - 1:
                    if sequence[i+1] == sequence[i]:  # 3 in a row
                        structure_penalty[b, i] += 0.5
        
        # Combine losses
        total_loss = ce_loss + structure_penalty
        return total_loss.mean()
    
    def get_structured_prompt(self):
        """Create a properly structured prompt for generation."""
        prompt_tokens = [
            self.start_token,  # START
            self.velocity_range[0] + 16,  # Set moderate velocity
        ]
        return torch.tensor(prompt_tokens, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Structured Sequence Training")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output", "-o", default="outputs/training/structured", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"structured_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎵 Starting Structured Sequence Training")
    logger.info(f"Output: {output_dir}")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/training_configs/quick_test.yaml")
    
    # Initialize components
    vocab_config = VocabularyConfig()
    structural_loss = StructuralLoss(vocab_config)
    
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
    
    # Initialize sampler for testing
    sampler = MusicSampler(model, device)
    
    # Dataset
    dataset = LazyMidiDataset(
        data_dir=config.system.data_dir,
        vocab_config=vocab_config,
        sequence_length=config.data.sequence_length,
        overlap=config.data.overlap,
        max_files=20,  # Small dataset for focused training
        cache_dir=config.system.cache_dir,
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
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Dataset: {len(dataset)} sequences")
    logger.info(f"Device: {device}")
    
    # Training loop
    model.train()
    training_log = []
    best_valid_notes = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= 15:  # Limit batches for focused training
                break
                
            tokens = batch['tokens'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(tokens[:, :-1])  # Input: all but last token
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]  # Target: all but first token
            
            # Calculate structural loss
            loss = structural_loss.calculate_structural_loss(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Test generation periodically
            if batch_idx % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Use structured prompt for better generation
                    prompt = structural_loss.get_structured_prompt().unsqueeze(0).to(device)
                    config_gen = GenerationConfig(
                        max_length=30, 
                        strategy=SamplingStrategy.TEMPERATURE, 
                        temperature=0.8
                    )
                    generated_tokens = sampler.generate(prompt=prompt, config=config_gen)
                    sample_tokens = generated_tokens[0].cpu().numpy()
                    validation = validate_generated_sequence(sample_tokens, vocab_config)
                    
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                               f"Loss={loss.item():.4f}, "
                               f"Grammar={validation['grammar_score']:.3f}, "
                               f"Notes={validation['note_count']}")
                
                model.train()
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        # Final epoch test
        model.eval()
        with torch.no_grad():
            prompt = structural_loss.get_structured_prompt().unsqueeze(0).to(device)
            config_gen = GenerationConfig(
                max_length=50, 
                strategy=SamplingStrategy.TEMPERATURE, 
                temperature=0.8
            )
            generated_tokens = sampler.generate(prompt=prompt, config=config_gen)
            sample_tokens = generated_tokens[0].cpu().numpy()
            validation = validate_generated_sequence(sample_tokens, vocab_config)
            
            note_count = validation['note_count']
            grammar_score = validation['grammar_score']
            
            logger.info(f"Epoch {epoch} complete: Loss={avg_loss:.4f}, "
                       f"Grammar={grammar_score:.3f}, Notes={note_count}")
            
            # Save checkpoint if we get valid notes
            if note_count > best_valid_notes:
                best_valid_notes = note_count
                checkpoint_path = output_dir / f"structured_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'note_count': note_count,
                    'grammar_score': grammar_score,
                    'loss': avg_loss,
                    'd_model': config.model.hidden_dim,
                    'n_layers': config.model.num_layers,
                    'n_heads': config.model.num_heads,
                    'max_sequence_length': config.model.max_sequence_length,
                }, checkpoint_path)
                logger.info(f"✅ Saved checkpoint with {note_count} notes: {checkpoint_path}")
        
        model.train()
    
    logger.info(f"🎉 Training complete! Best note count: {best_valid_notes}")
    return str(output_dir)

if __name__ == "__main__":
    output_dir = main()