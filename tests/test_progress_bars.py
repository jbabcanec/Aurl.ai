#!/usr/bin/env python3
"""
Test progress bars functionality
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train import MasterTrainer

def test_progress_bars():
    """Test the progress bar functionality"""
    
    class MockArgs:
        def __init__(self):
            self.clear = False
            self.status = False
            self.stage = None
            self.resume = False
    
    trainer = MasterTrainer(MockArgs())
    
    # Test progress bar creation
    print("🧪 Testing progress bar creation...")
    
    # Test different progress levels
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bar = trainer._create_progress_bar(progress, width=30)
        print(f"Progress {progress*100:5.1f}%: [{bar}]")
    
    # Test total progress calculation
    print("\n🧪 Testing total progress calculation...")
    
    # Test different stage scenarios
    scenarios = [
        (1, 5, 20),    # Stage 1, epoch 5/20
        (2, 15, 30),   # Stage 2, epoch 15/30
        (3, 25, 50),   # Stage 3, epoch 25/50
        (3, 50, 50),   # Stage 3, completed
    ]
    
    for stage, epoch, max_epochs in scenarios:
        total_progress = trainer._calculate_total_progress(stage, epoch, max_epochs)
        bar = trainer._create_progress_bar(total_progress, width=40)
        print(f"Stage {stage}, Epoch {epoch}/{max_epochs}: [{bar}] {total_progress*100:.1f}%")
    
    # Test simulated training progress display
    print("\n🧪 Testing simulated training progress display...")
    
    # Simulate epoch progress
    stage_num = 1
    epoch = 0
    target_epochs = 20
    
    for batch_idx in range(0, 100, 10):  # Simulate 10 batches
        total_batches = 100
        
        # Calculate progress
        epoch_progress = batch_idx / total_batches
        stage_progress = (epoch + epoch_progress) / target_epochs
        total_progress = trainer._calculate_total_progress(stage_num, epoch + epoch_progress, target_epochs)
        
        # Create progress bars
        epoch_bar = trainer._create_progress_bar(epoch_progress, width=20)
        stage_bar = trainer._create_progress_bar(stage_progress, width=30)
        total_bar = trainer._create_progress_bar(total_progress, width=40)
        
        # Display (like in actual training)
        print(f"\n🎵 Stage {stage_num}/3 | Epoch {epoch+1}/{target_epochs} | Batch {batch_idx+1}/{total_batches}")
        print(f"📊 Epoch:  [{epoch_bar}] {epoch_progress*100:.1f}%")
        print(f"🎯 Stage:  [{stage_bar}] {stage_progress*100:.1f}%")
        print(f"🌟 Total:  [{total_bar}] {total_progress*100:.1f}%")
        print(f"📉 Loss: 6.234 | ⏱️  ETA: 2.5m")
        
        time.sleep(0.1)  # Brief pause to show progression
    
    print("\n✅ Progress bars test completed successfully!")

if __name__ == "__main__":
    test_progress_bars()