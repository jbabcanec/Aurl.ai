#!/usr/bin/env python3
"""
Aurl.ai Master Training Pipeline
=================================

Single comprehensive training script with staged progression:
1. Stage 1: Base Training (no augmentation, fast)
2. Stage 2: Augmented Training (5x data variety)  
3. Stage 3: Advanced Training (VAE/GAN modes)

Features:
- Automatic progression through stages
- State persistence (resume from interruptions)
- Config-driven training parameters
- Clear progress tracking and monitoring
"""

import argparse
import json
import sys
import torch
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.base_logger import setup_logger
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from torch.utils.data import DataLoader

logger = setup_logger(__name__)

@dataclass
class TrainingState:
    """Tracks training progression and state."""
    current_stage: int = 1
    stage_epoch: int = 0
    total_epochs: int = 0
    best_loss: float = float('inf')
    stage_1_complete: bool = False
    stage_2_complete: bool = False  
    stage_3_complete: bool = False
    stage_1_epochs: int = 0
    stage_2_epochs: int = 0
    stage_3_epochs: int = 0
    last_checkpoint: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None

    def save(self, path: str = "training_state.json"):
        """Save training state to file."""
        self.updated_at = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "training_state.json") -> 'TrainingState':
        """Load training state from file."""
        if not Path(path).exists():
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class MasterTrainer:
    """Master training pipeline with staged progression."""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.state = TrainingState.load()
        self.config = None
        
        # Stage definitions
        self.stages = {
            1: {
                "name": "Base Training (No Augmentation)",
                "description": "Fast training on clean data to establish baseline",
                "config": "configs/training_configs/stage1_base.yaml",
                "augmentation": False,
                "mode": "transformer",
                "target_epochs": 20
            },
            2: {
                "name": "Augmented Training", 
                "description": "Rich training with 5x data augmentation",
                "config": "configs/training_configs/stage2_augmented.yaml",
                "augmentation": True,
                "mode": "transformer", 
                "target_epochs": 30
            },
            3: {
                "name": "Advanced Training (VAE-GAN)",
                "description": "High-quality generation with VAE-GAN architecture",
                "config": "configs/training_configs/stage3_advanced.yaml", 
                "augmentation": True,
                "mode": "vae_gan",
                "target_epochs": 50
            }
        }
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
    def clear_outputs(self):
        """Clear all training outputs."""
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
        
        # Reset training state
        self.state = TrainingState()
        print("✅ Training outputs cleared!")
    
    def status(self):
        """Show current training status."""
        print(f"\n🎼 AURL.AI TRAINING STATUS")
        print(f"=" * 50)
        print(f"📊 Current Stage: {self.state.current_stage}/3")
        
        if self.state.current_stage <= 3:
            stage = self.stages[self.state.current_stage]
            print(f"🎯 Stage: {stage['name']}")
            print(f"📝 Description: {stage['description']}")
            print(f"📈 Progress: {self.state.stage_epoch}/{stage['target_epochs']} epochs")
        
        print(f"🏆 Best Loss: {self.state.best_loss:.4f}")
        print(f"⏱️  Total Epochs: {self.state.total_epochs}")
        
        if self.state.started_at:
            print(f"🕐 Started: {self.state.started_at}")
        if self.state.updated_at:
            print(f"🔄 Last Update: {self.state.updated_at}")
        
        print(f"\n📋 Stage Completion:")
        print(f"   Stage 1 (Base): {'✅' if self.state.stage_1_complete else '⏳'} ({self.state.stage_1_epochs} epochs)")
        print(f"   Stage 2 (Augmented): {'✅' if self.state.stage_2_complete else '⏳'} ({self.state.stage_2_epochs} epochs)")  
        print(f"   Stage 3 (Advanced): {'✅' if self.state.stage_3_complete else '⏳'} ({self.state.stage_3_epochs} epochs)")
        
        if self.state.last_checkpoint:
            print(f"💾 Last Checkpoint: {self.state.last_checkpoint}")
    
    def train_stage(self, stage_num: int) -> bool:
        """Train a specific stage."""
        if stage_num not in self.stages:
            print(f"❌ Invalid stage: {stage_num}")
            return False
        
        stage = self.stages[stage_num]
        print(f"\n🚀 Starting {stage['name']}")
        print(f"📝 {stage['description']}")
        print(f"⚙️  Config: {stage['config']}")
        print(f"🔄 Augmentation: {'Enabled' if stage['augmentation'] else 'Disabled'}")
        print(f"🧠 Mode: {stage['mode']}")
        print(f"🎯 Target: {stage['target_epochs']} epochs")
        print("=" * 50)
        
        # Load stage-specific config
        try:
            self.config = load_config(stage['config'])
        except FileNotFoundError:
            print(f"❌ Config file not found: {stage['config']}")
            print("📝 Creating default config...")
            self._create_stage_config(stage_num)
            self.config = load_config(stage['config'])
        
        # Update config with stage settings
        self.config['model']['mode'] = stage['mode']
        self.config['data']['augmentation'] = stage['augmentation']
        
        # Train this stage
        success = self._run_training(stage_num, stage['target_epochs'])
        
        if success:
            # Mark stage complete
            if stage_num == 1:
                self.state.stage_1_complete = True
                self.state.stage_1_epochs = stage['target_epochs']
            elif stage_num == 2:
                self.state.stage_2_complete = True  
                self.state.stage_2_epochs = stage['target_epochs']
            elif stage_num == 3:
                self.state.stage_3_complete = True
                self.state.stage_3_epochs = stage['target_epochs']
            
            # Beautiful stage completion display
            total_progress = self._calculate_total_progress(stage_num, stage['target_epochs'], stage['target_epochs'])
            total_bar = self._create_progress_bar(total_progress, width=50)
            
            print(f"\n🎉 {stage['name']} COMPLETED! 🎉")
            print(f"🌟 Total Progress: [{total_bar}] {total_progress*100:.1f}%")
            print(f"🏆 Best Loss: {self.state.best_loss:.4f}")
            print("=" * 60)
            
            # Advance to next stage
            if stage_num < 3:
                self.state.current_stage = stage_num + 1
                self.state.stage_epoch = 0
                next_stage = self.stages[stage_num + 1]
                print(f"\n🚀 Next: {next_stage['name']}")
                print(f"📝 {next_stage['description']}")
                print("-" * 60)
        
        self.state.save()
        return success
    
    def _run_training(self, stage_num: int, target_epochs: int) -> bool:
        """Run training for current stage."""
        try:
            # Create model
            print("🔧 Creating model...")
            model = MusicTransformerVAEGAN(**self.config['model'])
            model.to(self.device)
            
            # Create dataset with stage-appropriate augmentation
            print("📁 Loading dataset...")
            
            # Enable augmentation based on config
            enable_augmentation = self.config['data'].get('augmentation', False)
            augmentation_config = None
            if enable_augmentation and isinstance(enable_augmentation, dict):
                augmentation_config = enable_augmentation
            
            dataset = LazyMidiDataset(
                data_dir=self.config['system']['data_dir'],
                cache_dir=self.config['system']['cache_dir'],
                sequence_length=self.config['data']['sequence_length'],
                enable_augmentation=bool(enable_augmentation),
                augmentation_config=augmentation_config
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['system']['num_workers'],
                pin_memory=self.config['system']['pin_memory'],
                collate_fn=self._collate_fn
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            # Load checkpoint if resuming
            start_epoch = self.state.stage_epoch
            if start_epoch > 0 and self.state.last_checkpoint:
                self._load_checkpoint(model, optimizer)
                print(f"📦 Resumed from epoch {start_epoch}")
            
            # Training loop
            model.train()
            for epoch in range(start_epoch, target_epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    batch = batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass - use compute_loss for proper loss calculation
                    loss_dict = model.compute_loss(batch)
                    loss = loss_dict['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if 'gradient_clip_norm' in self.config['training']:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            self.config['training']['gradient_clip_norm']
                        )
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # Progress update with progress bars
                    if batch_idx % 10 == 0:
                        # Calculate progress
                        epoch_progress = (batch_idx + 1) / len(dataloader)
                        stage_progress = (epoch + epoch_progress) / target_epochs
                        total_progress = self._calculate_total_progress(stage_num, epoch + epoch_progress, target_epochs)
                        
                        # Create progress bars
                        epoch_bar = self._create_progress_bar(epoch_progress, width=20)
                        stage_bar = self._create_progress_bar(stage_progress, width=30)
                        total_bar = self._create_progress_bar(total_progress, width=40)
                        
                        # Calculate ETA
                        eta_minutes = ((time.time() - epoch_start) / (batch_idx + 1) * (len(dataloader) - batch_idx - 1)) / 60
                        
                        # Multi-line progress display
                        print(f"\n🎵 Stage {stage_num}/3 | Epoch {epoch+1}/{target_epochs} | Batch {batch_idx+1}/{len(dataloader)}")
                        print(f"📊 Epoch:  [{epoch_bar}] {epoch_progress*100:.1f}%")
                        print(f"🎯 Stage:  [{stage_bar}] {stage_progress*100:.1f}%")
                        print(f"🌟 Total:  [{total_bar}] {total_progress*100:.1f}%")
                        print(f"📉 Loss: {loss.item():.4f} | ⏱️  ETA: {eta_minutes:.1f}m")
                
                # End of epoch
                avg_loss = epoch_loss / num_batches
                epoch_time = time.time() - epoch_start
                
                # Update best loss and save checkpoint
                is_best = False
                if avg_loss < self.state.best_loss:
                    self.state.best_loss = avg_loss
                    is_best = True
                    self._save_checkpoint(model, optimizer, epoch)
                
                # Update state
                self.state.stage_epoch = epoch + 1
                self.state.total_epochs += 1
                self.state.save()
                
                # Final progress bars for completed epoch
                stage_progress = (epoch + 1) / target_epochs
                total_progress = self._calculate_total_progress(stage_num, epoch + 1, target_epochs)
                stage_bar = self._create_progress_bar(stage_progress, width=30)
                total_bar = self._create_progress_bar(total_progress, width=40)
                
                print(f"\n✅ Epoch {epoch+1}/{target_epochs} Complete!")
                print(f"🎯 Stage:  [{stage_bar}] {stage_progress*100:.1f}%")
                print(f"🌟 Total:  [{total_bar}] {total_progress*100:.1f}%")
                print(f"📉 Loss: {avg_loss:.4f} | ⏱️  Time: {epoch_time:.1f}s | 🏆 Best: {self.state.best_loss:.4f}")
                if is_best:
                    print("🎉 New best model saved!")
                print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _collate_fn(self, batch):
        """Custom collate function for variable length sequences."""
        # Dataset returns dictionaries with 'tokens' key
        if isinstance(batch[0], dict):
            # Extract tokens from dictionaries
            sequences = [item['tokens'] for item in batch]
        elif isinstance(batch[0], (tuple, list)):
            # Extract first element if tuple/list
            sequences = [item[0] for item in batch]
        else:
            # Direct tensors
            sequences = batch
        
        # Find max length in batch
        max_len = max(seq.size(0) for seq in sequences)
        
        # Pad all sequences to max length
        padded = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.zeros(max_len - seq.size(0), dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            padded.append(seq)
        
        return torch.stack(padded)
    
    def _save_checkpoint(self, model, optimizer, epoch):
        """Save model checkpoint."""
        checkpoint_dir = Path("outputs/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "best_model.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.state.best_loss,
            'stage': self.state.current_stage
        }, checkpoint_path)
        
        self.state.last_checkpoint = str(checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, model, optimizer):
        """Load model checkpoint."""
        if not self.state.last_checkpoint or not Path(self.state.last_checkpoint).exists():
            print("⚠️  No checkpoint found to resume from")
            return
        
        checkpoint = torch.load(self.state.last_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"📦 Loaded checkpoint from {self.state.last_checkpoint}")
    
    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a visual progress bar."""
        filled = int(progress * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar
    
    def _calculate_total_progress(self, current_stage: int, current_epoch: float, stage_epochs: int) -> float:
        """Calculate overall progress across all stages."""
        stage_weights = {1: 20, 2: 30, 3: 50}  # Total epochs per stage
        total_epochs = sum(stage_weights.values())  # 100 total epochs
        
        # Calculate completed epochs from previous stages
        completed_epochs = 0
        for stage in range(1, current_stage):
            completed_epochs += stage_weights[stage]
        
        # Add current stage progress
        current_stage_progress = (current_epoch / stage_epochs) * stage_weights[current_stage]
        
        return (completed_epochs + current_stage_progress) / total_epochs
    
    def _create_stage_config(self, stage_num: int):
        """Create default config for stage."""
        # This would create stage-specific configs
        # For now, we'll use existing configs
        print(f"📝 Using default config structure for stage {stage_num}")
    
    def run_full_training(self):
        """Run complete training pipeline."""
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()
            self.state.save()
        
        print(f"\n🎼 AURL.AI MASTER TRAINING PIPELINE")
        print(f"=" * 50)
        print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        
        # Show current status
        self.status()
        
        # Run stages in sequence
        for stage_num in range(self.state.current_stage, 4):
            if stage_num == 1 and self.state.stage_1_complete:
                continue
            if stage_num == 2 and self.state.stage_2_complete:
                continue
            if stage_num == 3 and self.state.stage_3_complete:
                continue
            
            success = self.train_stage(stage_num)
            if not success:
                print(f"❌ Stage {stage_num} failed. Training stopped.")
                return False
        
        # Final completion display with progress bar
        final_bar = self._create_progress_bar(1.0, width=60)
        
        print(f"\n🎊 COMPLETE TRAINING PIPELINE FINISHED! 🎊")
        print(f"🌟 Final Progress: [{final_bar}] 100.0%")
        print(f"🏆 Final Best Loss: {self.state.best_loss:.4f}")
        print(f"📊 Total Epochs: {self.state.total_epochs}")
        print(f"💾 Best Model: {self.state.last_checkpoint}")
        print(f"⏱️  Total Time: {(time.time() - time.mktime(datetime.fromisoformat(self.state.started_at).timetuple())) / 3600:.1f} hours")
        print("🎵" * 60)
        print("🎼 Ready for music generation! 🎼")
        print("🎵" * 60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Aurl.ai Master Training Pipeline")
    parser.add_argument('--clear', action='store_true', help='Clear all training outputs')
    parser.add_argument('--status', action='store_true', help='Show training status')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='Train specific stage')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    trainer = MasterTrainer(args)
    
    if args.clear:
        trainer.clear_outputs()
        return
    
    if args.status:
        trainer.status()
        return
    
    if args.stage:
        trainer.train_stage(args.stage)
        return
    
    # Default: run full training pipeline
    trainer.run_full_training()

if __name__ == "__main__":
    main()