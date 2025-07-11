#!/usr/bin/env python3
"""
Quick test script to debug training issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from torch.utils.data import DataLoader
import torch

def test_training_components():
    print("🧪 Testing training components...")
    
    # Test config loading
    config = load_config("configs/training_configs/stage1_base.yaml")
    print(f"✅ Config loaded: {config['model']['mode']}")
    
    # Test model creation
    model = MusicTransformerVAEGAN(**config['model'])
    print(f"✅ Model created: {model.mode}")
    
    # Test dataset creation
    dataset = LazyMidiDataset(
        data_dir=config['system']['data_dir'],
        cache_dir=config['system']['cache_dir'],
        sequence_length=512,  # Smaller for testing
        enable_augmentation=False
    )
    print(f"✅ Dataset created: {len(dataset)} sequences")
    
    # Test dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch for testing
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn
    )
    print(f"✅ DataLoader created")
    
    # Test getting one batch
    try:
        batch = next(iter(dataloader))
        print(f"✅ Got batch: {batch.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch)
            print(f"✅ Forward pass: {output.shape}")
            
            # Test loss computation
            loss_dict = model.compute_loss(batch)
            print(f"✅ Loss computation: {loss_dict['total_loss'].item():.4f}")
    
    except Exception as e:
        print(f"❌ Batch/forward error: {e}")
        import traceback
        traceback.print_exc()

def collate_fn(batch):
    """Simple collate function."""
    # Extract tokens from dictionaries
    if isinstance(batch[0], dict):
        sequences = [item['tokens'] for item in batch]
    else:
        sequences = batch
    
    # Find max length
    max_len = max(seq.size(0) for seq in sequences)
    
    # Pad sequences
    padded = []
    for seq in sequences:
        if seq.size(0) < max_len:
            padding = torch.zeros(max_len - seq.size(0), dtype=seq.dtype)
            seq = torch.cat([seq, padding])
        padded.append(seq)
    
    return torch.stack(padded)

if __name__ == "__main__":
    test_training_components()