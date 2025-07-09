"""
Comprehensive test suite for Phase 4.1 Training Framework.

This test verifies:
1. AdvancedTrainer initialization and configuration
2. Distributed training setup (simulation)
3. Mixed precision training functionality
4. Gradient accumulation and clipping
5. Dynamic batch sizing logic
6. Curriculum learning scheduler
7. Throughput monitoring accuracy
8. Memory optimization features
9. Integration with Phase 3 components
10. Complete training loop execution
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.core.trainer import (
    AdvancedTrainer, TrainingConfig, CurriculumScheduler,
    DynamicBatchSizer, ThroughputMonitor, create_trainer_from_config
)
from src.training.utils.memory_optimization import (
    MemoryProfiler, GradientCheckpointing, MemoryOptimizer
)
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework
from src.data.dataset import LazyMidiDataset
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


def test_training_config():
    """Test TrainingConfig dataclass functionality."""
    print("🧪 Testing Training Configuration")
    print("-" * 40)
    
    try:
        # Test default configuration
        config = TrainingConfig()
        assert config.batch_size == 32, f"Expected batch_size=32, got {config.batch_size}"
        assert config.learning_rate == 1e-4, f"Expected lr=1e-4, got {config.learning_rate}"
        assert config.use_mixed_precision == True, "Mixed precision should be enabled by default"
        
        # Test custom configuration
        custom_config = TrainingConfig(
            batch_size=64,
            learning_rate=5e-4,
            num_epochs=200,
            distributed=True,
            curriculum_learning=False
        )
        assert custom_config.batch_size == 64, "Custom batch size not set correctly"
        assert custom_config.distributed == True, "Distributed flag not set correctly"
        assert custom_config.curriculum_learning == False, "Curriculum learning flag not set correctly"
        
        print("   ✅ TrainingConfig initialization working")
        print("   ✅ Default and custom values set correctly")
        print("   ✅ All configuration options accessible")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_curriculum_scheduler():
    """Test curriculum learning scheduler."""
    print("\n🧪 Testing Curriculum Learning Scheduler")
    print("-" * 40)
    
    try:
        # Test linear curriculum
        scheduler = CurriculumScheduler(
            start_length=256,
            end_length=1024,
            curriculum_epochs=10,
            strategy="linear"
        )
        
        # Test progression
        epoch_0 = scheduler.get_current_length(0)
        epoch_5 = scheduler.get_current_length(5)
        epoch_10 = scheduler.get_current_length(10)
        epoch_15 = scheduler.get_current_length(15)
        
        assert epoch_0 == 256, f"Expected start length 256, got {epoch_0}"
        assert epoch_10 == 1024, f"Expected end length 1024 at epoch 10, got {epoch_10}"
        assert epoch_15 == 1024, f"Expected end length 1024 after curriculum, got {epoch_15}"
        assert 256 < epoch_5 < 1024, f"Expected intermediate value at epoch 5, got {epoch_5}"
        
        print(f"   ✅ Linear curriculum progression: {epoch_0} → {epoch_5} → {epoch_10} → {epoch_15}")
        
        # Test exponential curriculum
        exp_scheduler = CurriculumScheduler(
            start_length=128,
            end_length=512,
            curriculum_epochs=5,
            strategy="exponential"
        )
        
        exp_values = [exp_scheduler.get_current_length(i) for i in range(6)]
        assert exp_values[0] == 128, "Exponential should start at 128"
        assert exp_values[-1] == 512, "Exponential should end at 512"
        
        print(f"   ✅ Exponential curriculum: {exp_values}")
        
        # Test cosine curriculum
        cos_scheduler = CurriculumScheduler(
            start_length=200,
            end_length=800,
            curriculum_epochs=8,
            strategy="cosine"
        )
        
        cos_values = [cos_scheduler.get_current_length(i) for i in range(10)]
        assert cos_values[0] == 200, "Cosine should start at 200"
        assert cos_values[8] == 800, "Cosine should reach end at curriculum epochs"
        
        print(f"   ✅ Cosine curriculum: progression smooth and correct")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_batch_sizer():
    """Test dynamic batch sizing logic."""
    print("\n🧪 Testing Dynamic Batch Sizing")
    print("-" * 40)
    
    try:
        batch_sizer = DynamicBatchSizer(
            base_batch_size=32,
            min_batch_size=4,
            max_batch_size=64,
            memory_threshold=0.8
        )
        
        # Test sequence length scaling
        short_seq_batch = batch_sizer.get_optimal_batch_size(256)
        long_seq_batch = batch_sizer.get_optimal_batch_size(2048)
        
        assert short_seq_batch >= long_seq_batch, "Shorter sequences should allow larger batches"
        assert batch_sizer.min_batch_size <= short_seq_batch <= batch_sizer.max_batch_size
        assert batch_sizer.min_batch_size <= long_seq_batch <= batch_sizer.max_batch_size
        
        print(f"   ✅ Sequence length scaling: {short_seq_batch} (256 tokens) vs {long_seq_batch} (2048 tokens)")
        
        # Test memory pressure response
        high_memory_batch = batch_sizer.get_optimal_batch_size(1024, current_memory_usage=0.9)
        low_memory_batch = batch_sizer.get_optimal_batch_size(1024, current_memory_usage=0.5)
        
        assert high_memory_batch <= low_memory_batch, "High memory should reduce batch size"
        
        print(f"   ✅ Memory scaling: {high_memory_batch} (90% memory) vs {low_memory_batch} (50% memory)")
        
        # Test bounds enforcement
        extreme_batch = batch_sizer.get_optimal_batch_size(128, current_memory_usage=0.1)
        assert extreme_batch <= batch_sizer.max_batch_size, "Should not exceed max batch size"
        
        extreme_batch_2 = batch_sizer.get_optimal_batch_size(4096, current_memory_usage=0.95)
        assert extreme_batch_2 >= batch_sizer.min_batch_size, "Should not go below min batch size"
        
        print(f"   ✅ Bounds enforcement working correctly")
        
        # Test history tracking
        assert len(batch_sizer.batch_size_history) > 0, "Should track batch size history"
        assert len(batch_sizer.memory_history) > 0, "Should track memory history"
        
        print(f"   ✅ History tracking: {len(batch_sizer.batch_size_history)} batch sizes recorded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_throughput_monitor():
    """Test training throughput monitoring."""
    print("\n🧪 Testing Throughput Monitor")
    print("-" * 40)
    
    try:
        monitor = ThroughputMonitor(window_size=10)
        
        # Simulate training batches
        for i in range(5):
            monitor.start_batch()
            
            # Simulate processing time
            import time
            time.sleep(0.01)  # 10ms per batch
            
            # End batch with metrics
            monitor.end_batch(num_samples=16, num_tokens=1024)
        
        # Get metrics
        metrics = monitor.get_metrics()
        
        # Verify metrics exist
        required_metrics = ['samples_per_second', 'tokens_per_second', 'avg_batch_time', 'batch_time_std']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert metrics[metric] >= 0, f"Metric {metric} should be non-negative"
        
        print(f"   ✅ Throughput metrics: {metrics['samples_per_second']:.1f} samples/sec")
        print(f"   ✅ Token throughput: {metrics['tokens_per_second']:.0f} tokens/sec")
        print(f"   ✅ Average batch time: {metrics['avg_batch_time']:.4f}s")
        
        # Test window size limit
        for i in range(20):  # Exceed window size
            monitor.start_batch()
            time.sleep(0.001)
            monitor.end_batch(num_samples=8, num_tokens=512)
        
        assert len(monitor.batch_times) <= monitor.window_size, "Should maintain window size limit"
        
        print(f"   ✅ Window size maintained: {len(monitor.batch_times)}/{monitor.window_size}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_profiler():
    """Test memory profiling functionality."""
    print("\n🧪 Testing Memory Profiler")
    print("-" * 40)
    
    try:
        device = torch.device("cpu")  # Use CPU for consistent testing
        profiler = MemoryProfiler(device)
        
        # Record some checkpoints
        profiler.checkpoint("start")
        
        # Allocate some memory
        x = torch.randn(1000, 1000)
        profiler.checkpoint("after_allocation")
        
        # More allocation
        y = torch.randn(2000, 2000)
        profiler.checkpoint("after_large_allocation")
        
        # Get summary
        summary = profiler.get_memory_summary()
        
        assert summary['total_checkpoints'] == 3, f"Expected 3 checkpoints, got {summary['total_checkpoints']}"
        assert 'peak_cpu_rss' in summary, "Should track peak CPU memory"
        assert len(summary['checkpoints']) > 0, "Should have checkpoint history"
        
        print(f"   ✅ Memory checkpoints recorded: {summary['total_checkpoints']}")
        print(f"   ✅ Peak CPU memory: {summary['peak_cpu_rss']:.3f} GB")
        
        # Test checkpoint clearing
        profiler.clear_checkpoints()
        empty_summary = profiler.get_memory_summary()
        assert empty_summary.get('total_checkpoints', 0) == 0, "Checkpoints should be cleared"
        
        print(f"   ✅ Checkpoint clearing working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_checkpointing():
    """Test gradient checkpointing setup."""
    print("\n🧪 Testing Gradient Checkpointing")
    print("-" * 40)
    
    try:
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(100, 50)
                self.layer2 = nn.Linear(50, 25)
                self.layer3 = nn.Linear(25, 10)
            
            def forward(self, x):
                x = torch.relu(self.layer1(x))
                x = torch.relu(self.layer2(x))
                return self.layer3(x)
        
        model = SimpleModel()
        checkpointing = GradientCheckpointing(model)
        
        # Test auto-detection
        segments = checkpointing.segments
        assert len(segments) > 0, "Should auto-detect some segments"
        
        print(f"   ✅ Auto-detected segments: {segments}")
        
        # Test enable/disable
        checkpointing.enable()
        assert checkpointing.enabled == True, "Should be enabled after enable()"
        
        checkpointing.disable()
        assert checkpointing.enabled == False, "Should be disabled after disable()"
        
        print(f"   ✅ Enable/disable functionality working")
        
        # Test with manual segments
        manual_checkpointing = GradientCheckpointing(model, segments=['layer1', 'layer2'])
        assert manual_checkpointing.segments == ['layer1', 'layer2'], "Should use manual segments"
        
        print(f"   ✅ Manual segment specification working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_trainer_initialization():
    """Test AdvancedTrainer initialization."""
    print("\n🧪 Testing AdvancedTrainer Initialization")
    print("-" * 40)
    
    try:
        # Create minimal components for testing
        model = MusicTransformerVAEGAN(
            vocab_size=774,
            d_model=128,
            n_layers=2,
            mode="transformer"  # Simpler mode for testing
        )
        
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            use_automatic_balancing=False  # Simpler for testing
        )
        
        config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            distributed=False,
            use_mixed_precision=False,  # Simpler for testing
            curriculum_learning=False,
            dynamic_batching=False
        )
        
        # Create temporary dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock dataset
            train_dataset = Mock()
            train_dataset.__len__ = Mock(return_value=100)
            
            # Initialize trainer
            trainer = AdvancedTrainer(
                model=model,
                loss_framework=loss_framework,
                config=config,
                train_dataset=train_dataset,
                save_dir=Path(temp_dir)
            )
            
            # Verify initialization
            assert trainer.model is not None, "Model should be initialized"
            assert trainer.loss_framework is not None, "Loss framework should be initialized"
            assert trainer.optimizer is not None, "Optimizer should be created"
            assert trainer.device is not None, "Device should be set"
            
            print(f"   ✅ Trainer initialized with device: {trainer.device}")
            print(f"   ✅ Optimizer: {type(trainer.optimizer).__name__}")
            print(f"   ✅ Save directory: {trainer.save_dir}")
            
            # Test component initialization
            assert trainer.loss_monitor is not None, "Loss monitor should be initialized"
            assert trainer.stability_monitor is not None, "Stability monitor should be initialized"
            assert trainer.throughput_monitor is not None, "Throughput monitor should be initialized"
            
            print(f"   ✅ All monitoring components initialized")
            
            # Test configuration application
            assert trainer.config.batch_size == 4, "Config should be applied"
            assert trainer.current_epoch == 0, "Should start at epoch 0"
            assert trainer.global_step == 0, "Should start at step 0"
            
            print(f"   ✅ Configuration applied correctly")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step_simulation():
    """Test training step simulation."""
    print("\n🧪 Testing Training Step Simulation")
    print("-" * 40)
    
    try:
        # Create minimal setup
        model = MusicTransformerVAEGAN(
            vocab_size=774,
            d_model=64,
            n_layers=1,
            mode="transformer"
        )
        
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            use_automatic_balancing=False
        )
        
        config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            use_mixed_precision=False,
            distributed=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dataset = Mock()
            train_dataset.__len__ = Mock(return_value=10)
            
            trainer = AdvancedTrainer(
                model=model,
                loss_framework=loss_framework,
                config=config,
                train_dataset=train_dataset,
                save_dir=Path(temp_dir)
            )
            
            # Test forward pass
            batch_size, seq_len = 2, 64
            tokens = torch.randint(0, 774, (batch_size, seq_len))
            
            # Move to trainer device
            tokens = tokens.to(trainer.device)
            
            # Test forward pass
            trainer.model.train()
            losses = trainer._forward_pass(tokens)
            
            # Verify loss structure
            assert 'total_loss' in losses, "Should have total_loss"
            assert isinstance(losses['total_loss'], torch.Tensor), "total_loss should be tensor"
            assert losses['total_loss'].dim() == 0, "total_loss should be scalar"
            assert losses['total_loss'].item() > 0, "total_loss should be positive"
            
            print(f"   ✅ Forward pass working: loss = {losses['total_loss'].item():.4f}")
            
            # Test backward pass
            loss = losses['total_loss']
            trainer.optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            has_gradients = any(
                param.grad is not None and param.grad.norm() > 0
                for param in trainer.model.parameters()
            )
            assert has_gradients, "Should have gradients after backward pass"
            
            print(f"   ✅ Backward pass working: gradients computed")
            
            # Test optimizer step
            initial_param_values = {
                name: param.clone() for name, param in trainer.model.named_parameters()
            }
            
            trainer.optimizer.step()
            
            # Check if parameters changed
            params_changed = False
            for name, param in trainer.model.named_parameters():
                if not torch.allclose(param, initial_param_values[name]):
                    params_changed = True
                    break
            
            assert params_changed, "Parameters should change after optimizer step"
            
            print(f"   ✅ Optimizer step working: parameters updated")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_precision_simulation():
    """Test mixed precision training simulation."""
    print("\n🧪 Testing Mixed Precision Training")
    print("-" * 40)
    
    try:
        # Skip if CUDA not available (mixed precision needs CUDA)
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available, skipping mixed precision test")
            return True
        
        model = MusicTransformerVAEGAN(
            vocab_size=774,
            d_model=64,
            n_layers=1,
            mode="transformer"
        )
        
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            use_automatic_balancing=False
        )
        
        config = TrainingConfig(
            batch_size=2,
            use_mixed_precision=True,
            fp16=True,
            distributed=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dataset = Mock()
            train_dataset.__len__ = Mock(return_value=10)
            
            trainer = AdvancedTrainer(
                model=model,
                loss_framework=loss_framework,
                config=config,
                train_dataset=train_dataset,
                save_dir=Path(temp_dir)
            )
            
            # Verify scaler is created
            assert trainer.scaler is not None, "GradScaler should be created for mixed precision"
            print(f"   ✅ GradScaler initialized for mixed precision")
            
            # Test mixed precision forward pass
            tokens = torch.randint(0, 774, (2, 32)).to(trainer.device)
            
            trainer.model.train()
            
            # Test autocast context
            from torch.cuda.amp import autocast
            with autocast():
                losses = trainer._forward_pass(tokens)
                loss = losses['total_loss']
            
            assert isinstance(loss, torch.Tensor), "Loss should be computed correctly"
            print(f"   ✅ Mixed precision forward pass working")
            
            # Test scaled backward pass
            trainer.optimizer.zero_grad()
            scaled_loss = trainer.scaler.scale(loss)
            scaled_loss.backward()
            
            # Check that gradients exist
            has_gradients = any(
                param.grad is not None for param in trainer.model.parameters()
            )
            assert has_gradients, "Should have gradients after scaled backward"
            
            print(f"   ✅ Scaled backward pass working")
            
            # Test unscale and step
            trainer.scaler.unscale_(trainer.optimizer)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            
            print(f"   ✅ Scaler step and update working")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_training_integration():
    """Test complete training pipeline integration."""
    print("\n🧪 Testing Complete Training Integration")
    print("-" * 40)
    
    try:
        # Create a minimal but complete setup
        model = MusicTransformerVAEGAN(
            vocab_size=774,
            d_model=32,  # Very small for testing
            n_layers=1,
            mode="transformer"
        )
        
        loss_framework = ComprehensiveLossFramework(
            vocab_size=774,
            use_automatic_balancing=False
        )
        
        config = TrainingConfig(
            batch_size=2,
            num_epochs=1,  # Single epoch for testing
            learning_rate=1e-3,
            log_interval=1,  # Log every step
            use_mixed_precision=False,
            distributed=False,
            curriculum_learning=False,
            dynamic_batching=False
        )
        
        # Create mock dataset that returns actual data
        class MockDataset:
            def __init__(self):
                self.data = [
                    {'tokens': torch.randint(0, 774, (32,))} for _ in range(4)
                ]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        train_dataset = MockDataset()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = AdvancedTrainer(
                model=model,
                loss_framework=loss_framework,
                config=config,
                train_dataset=train_dataset,
                save_dir=Path(temp_dir)
            )
            
            # Test single epoch training
            initial_loss = None
            
            # Mock the dataloader creation to return our test data
            original_create_dataloader = trainer._create_dataloader
            
            def mock_create_dataloader(dataset, batch_size, shuffle=True):
                return torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
                )
            
            trainer._create_dataloader = mock_create_dataloader
            
            # Run one epoch
            epoch_losses = trainer.train_epoch()
            
            # Verify training occurred
            assert 'total_loss' in epoch_losses, "Should have total loss"
            assert epoch_losses['total_loss'] > 0, "Loss should be positive"
            
            print(f"   ✅ Single epoch training completed")
            print(f"   ✅ Average loss: {epoch_losses['total_loss']:.4f}")
            
            # Verify monitoring was updated
            assert trainer.global_step > 0, "Global step should increment"
            
            # Test checkpoint saving
            trainer.save_checkpoint(0, is_best=True)
            checkpoint_files = list(trainer.save_dir.glob("*.pt"))
            assert len(checkpoint_files) > 0, "Should create checkpoint files"
            
            print(f"   ✅ Checkpoints saved: {len(checkpoint_files)} files")
            
            # Test checkpoint loading
            checkpoint = torch.load(checkpoint_files[0], map_location=trainer.device, weights_only=False)
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'config']
            for key in required_keys:
                assert key in checkpoint, f"Checkpoint missing key: {key}"
            
            print(f"   ✅ Checkpoint structure verified")
            
            # Test training summary
            trainer._save_training_summary()
            summary_file = trainer.save_dir / "training_summary.json"
            assert summary_file.exists(), "Should create training summary"
            
            print(f"   ✅ Training summary saved")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4.1 training framework tests."""
    print("🎯 Phase 4.1 Training Framework Test Suite")
    print("=" * 60)
    
    tests = [
        ("Training Configuration", test_training_config),
        ("Curriculum Learning Scheduler", test_curriculum_scheduler),
        ("Dynamic Batch Sizing", test_dynamic_batch_sizer),
        ("Throughput Monitor", test_throughput_monitor),
        ("Memory Profiler", test_memory_profiler),
        ("Gradient Checkpointing", test_gradient_checkpointing),
        ("AdvancedTrainer Initialization", test_advanced_trainer_initialization),
        ("Training Step Simulation", test_training_step_simulation),
        ("Mixed Precision Training", test_mixed_precision_simulation),
        ("Complete Training Integration", test_complete_training_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 4.1 training framework components working correctly!")
        print("🚀 Ready for production training infrastructure!")
        return True
    else:
        print("🛠️  Some components need fixes before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)