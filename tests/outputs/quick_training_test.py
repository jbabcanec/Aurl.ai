#!/usr/bin/env python3
"""
Quick training test - just a few batches to verify everything works
"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import MasterTrainer

def quick_test():
    """Run a quick training test"""
    
    class MockArgs:
        def __init__(self):
            self.clear = False
            self.status = False
            self.stage = 1
            self.resume = False
    
    trainer = MasterTrainer(MockArgs())
    
    # Load config and create model
    config = trainer.config = trainer.stages[1]
    from src.utils.config import load_config
    trainer.config = load_config(config['config'])
    trainer.config['model']['mode'] = config['mode']
    trainer.config['data']['augmentation'] = config['augmentation']
    
    # Create model
    from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
    model = MusicTransformerVAEGAN(**trainer.config['model'])
    model.to(trainer.device)
    
    # Create dataset
    from src.data.dataset import LazyMidiDataset
    dataset = LazyMidiDataset(
        data_dir=trainer.config['system']['data_dir'],
        cache_dir=trainer.config['system']['cache_dir'],
        sequence_length=512,  # Small for testing
        enable_augmentation=False
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid pickle issues
        pin_memory=False,
        collate_fn=trainer._collate_fn
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    print("🚀 Quick Training Test")
    print("=" * 40)
    
    # Test just a few batches
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only 3 batches
            break
            
        print(f"\n📦 Batch {batch_idx + 1}/3")
        print(f"   Shape: {batch.shape}")
        
        batch = batch.to(trainer.device)
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model.compute_loss(batch)
        loss = loss_dict['total_loss']
        
        print(f"   Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Show progress bar
        progress = (batch_idx + 1) / 3
        bar = trainer._create_progress_bar(progress, width=20)
        print(f"   Progress: [{bar}] {progress*100:.1f}%")
    
    print("\n✅ Quick training test completed successfully!")
    print("🎉 All components working correctly!")

if __name__ == "__main__":
    quick_test()