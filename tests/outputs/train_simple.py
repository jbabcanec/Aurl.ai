#!/usr/bin/env python3
"""
Simple, Fast Training Pipeline - No Fancy Monitoring
Just reliable training with basic progress tracking and clean logs.
"""

import argparse
import sys
import torch
import time
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.base_logger import setup_logger
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from torch.utils.data import DataLoader

logger = setup_logger(__name__)


def clear_training_outputs():
    """Clear all training outputs and logs."""
    dirs_to_clear = [
        "outputs/training",
        "outputs/checkpoints", 
        "outputs/logs",
        "logs",
        "training_state.json"
    ]
    
    print("🧹 Clearing training outputs...")
    
    for dir_path in dirs_to_clear:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"   🗑️  Removed directory: {path}")
            else:
                path.unlink()
                print(f"   🗑️  Removed file: {path}")
    
    print("✅ Training outputs cleared!")


class SimpleTrainer:
    """Simple, fast trainer with minimal overhead."""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        
    def train(self) -> bool:
        """Run simple training loop."""
        print("\n🎼 SIMPLE TRAINING PIPELINE")
        print("=" * 50)
        print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        
        try:
            # Create minimal model
            print("\n🔧 Creating model...")
            model = self._create_model()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   📊 Parameters: {param_count:,}")
            print(f"   🧠 Mode: {self.config['model']['mode'].upper()}")
            
            # Create dataset with size limit
            print("\n📁 Loading dataset...")
            dataset = self._create_dataset()
            
            # Limit dataset size for speed
            max_samples = self.config.get('max_samples', 100)
            if len(dataset) > max_samples:
                dataset.sequences = dataset.sequences[:max_samples]
                print(f"   🔍 Limited to {max_samples} samples for speed")
            
            print(f"   📄 Training samples: {len(dataset)}")
            
            # Create custom collate function to handle variable length sequences
            def collate_fn(batch):
                # Pad sequences to same length within batch
                max_len = max(item['tokens'].size(0) for item in batch)
                
                padded_tokens = []
                for item in batch:
                    tokens = item['tokens']
                    if tokens.size(0) < max_len:
                        # Pad with token ID 2 (PAD token)
                        padding = torch.full((max_len - tokens.size(0),), 2, dtype=tokens.dtype)
                        tokens = torch.cat([tokens, padding])
                    padded_tokens.append(tokens)
                
                return {
                    'tokens': torch.stack(padded_tokens),
                    'lengths': torch.tensor([item['tokens'].size(0) for item in batch])
                }
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=0,  # No multiprocessing for simplicity
                pin_memory=False,
                collate_fn=collate_fn
            )
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training'].get('weight_decay', 1e-5)
            )
            
            epochs = self.config['training']['num_epochs']
            total_batches = len(dataloader) * epochs
            batch_count = 0
            
            print(f"\n🚀 Training {epochs} epochs, {len(dataloader)} batches each")
            print(f"   🎯 Total batches: {total_batches}")
            print("=" * 50)
            
            start_time = time.time()
            best_loss = float('inf')
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                epoch_start = time.time()
                
                for batch_idx, batch in enumerate(dataloader):
                    batch_count += 1
                    
                    tokens = batch['tokens'].to(self.device)
                    
                    # Forward pass
                    if self.config['model']['mode'] == 'transformer':
                        logits = model(tokens)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, model.vocab_size),
                            tokens.view(-1),
                            ignore_index=2
                        )
                    else:
                        # VAE or VAE-GAN mode
                        losses = model.compute_loss(tokens)
                        loss = losses['total_loss']
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['training'].get('gradient_clip_norm', 1.0)
                    )
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Progress update every 10 batches
                    if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                        progress_pct = (batch_count / total_batches) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / progress_pct * 100) - elapsed if progress_pct > 0 else 0
                        
                        print(f"🎵 Epoch {epoch+1}/{epochs} | "
                              f"Batch {batch_idx+1}/{len(dataloader)} | "
                              f"Loss: {loss.item():.4f} | "
                              f"Progress: {progress_pct:.1f}% | "
                              f"ETA: {eta/60:.1f}m")
                
                # Epoch summary
                avg_loss = epoch_loss / len(dataloader)
                epoch_time = time.time() - epoch_start
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Save best checkpoint
                    checkpoint_dir = Path("outputs/checkpoints")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, checkpoint_dir / 'best_model.pt')
                
                print(f"✅ Epoch {epoch+1} complete: "
                      f"Avg Loss: {avg_loss:.4f} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Best: {best_loss:.4f}")
                print("-" * 50)
            
            total_time = time.time() - start_time
            print("\n🎉 Training completed!")
            print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
            print(f"   📈 Best loss: {best_loss:.4f}")
            print("   💾 Best model saved to: outputs/checkpoints/best_model.pt")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_model(self) -> MusicTransformerVAEGAN:
        """Create model with minimal configuration."""
        model_config = self.config['model']
        
        model = MusicTransformerVAEGAN(
            vocab_size=model_config.get('vocab_size', 774),
            d_model=model_config.get('hidden_dim', 128),
            n_layers=model_config.get('num_layers', 2),
            n_heads=model_config.get('num_heads', 2),
            latent_dim=model_config.get('latent_dim', 32),
            max_sequence_length=model_config.get('max_sequence_length', 256),
            mode=model_config.get('mode', 'transformer'),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        return model
    
    def _create_dataset(self) -> LazyMidiDataset:
        """Create dataset with minimal configuration."""
        data_config = self.config['data']
        system_config = self.config['system']
        
        dataset = LazyMidiDataset(
            data_dir=system_config.get('data_dir', 'data/raw'),
            cache_dir=system_config.get('cache_dir', 'data/cache'),
            sequence_length=data_config.get('sequence_length', 256),
            overlap=data_config.get('overlap', 64),
            enable_augmentation=False,  # Disabled for speed
            augmentation_probability=0.0
        )
        
        return dataset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Fast Training")
    parser.add_argument("--config", default="configs/training_configs/speed_test.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/mps/auto)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-samples", type=int, default=50, help="Max training samples for speed")
    parser.add_argument("--clear", action="store_true", help="Clear all training outputs first")
    
    args = parser.parse_args()
    
    # Clear outputs if requested
    if args.clear:
        clear_training_outputs()
        return
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.max_samples:
        config['max_samples'] = args.max_samples
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Run training
    trainer = SimpleTrainer(config, device)
    success = trainer.train()
    
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()