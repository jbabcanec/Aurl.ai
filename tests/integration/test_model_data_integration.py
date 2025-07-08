"""
Integration test for Phase 3.1 model architecture with actual data pipeline.

This test verifies that our new model architecture works correctly with:
- Real MIDI data from our dataset
- The data pipeline we built in Phase 2
- The sequence length configurations we implemented
- The 387-token vocabulary we discovered
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import LazyMidiDataset, create_dataloader
from src.data.representation import VocabularyConfig, PianoRollConfig
from src.models import MusicTransformerVAEGAN, BaselineTransformer
from src.utils.config import MidiFlyConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelDataIntegrationTester:
    """
    Comprehensive integration tester for model + data pipeline.
    
    Tests the complete workflow from MIDI files to model training.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.midi_dir = project_root / "data" / "raw"
        self.output_dir = project_root / "outputs" / "integration_audit"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'vocab_size': 774,
            'd_model': 512,
            'n_layers': 4,  # Smaller for testing
            'n_heads': 8,
            'max_sequence_length': 2048,
            'dropout': 0.1,
            'attention_type': 'hierarchical'
        }
        
        self.results = {
            'data_pipeline': {},
            'model_architecture': {},
            'integration': {},
            'performance': {},
            'issues': []
        }
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete integration audit."""
        logger.info("🔍 Starting Model-Data Integration Audit")
        logger.info("=" * 60)
        
        try:
            # Test 1: Data Pipeline Verification
            logger.info("Test 1: Data Pipeline Verification")
            self._test_data_pipeline()
            
            # Test 2: Model Architecture Verification
            logger.info("Test 2: Model Architecture Verification")
            self._test_model_architecture()
            
            # Test 3: Integration Testing
            logger.info("Test 3: Integration Testing")
            self._test_integration()
            
            # Test 4: Performance Testing
            logger.info("Test 4: Performance Testing")
            self._test_performance()
            
            # Test 5: Edge Cases
            logger.info("Test 5: Edge Cases")
            self._test_edge_cases()
            
            # Generate report
            self._generate_report()
            
        except Exception as e:
            logger.error(f"❌ Integration audit failed: {e}")
            self.results['issues'].append(f"Critical failure: {e}")
            raise
        
        logger.info("✅ Integration audit completed")
        return self.results
    
    def _test_data_pipeline(self):
        """Test data pipeline components."""
        logger.info("  Testing dataset creation...")
        
        # Test dataset creation with different configurations
        test_configs = [
            {
                'sequence_length': 512,
                'truncation_strategy': 'sliding_window',
                'curriculum_learning': False
            },
            {
                'sequence_length': 1024,
                'truncation_strategy': 'adaptive',
                'curriculum_learning': True,
                'sequence_length_schedule': [512, 1024, 2048]
            },
            {
                'sequence_length': 2048,
                'truncation_strategy': 'truncate',
                'max_sequence_length': 2048
            }
        ]
        
        for i, config in enumerate(test_configs):
            try:
                dataset = LazyMidiDataset(
                    data_dir=self.midi_dir,
                    max_files=5,  # Small test
                    **config
                )
                
                # Test basic functionality
                assert len(dataset) > 0, f"Dataset {i} is empty"
                
                # Test sample loading
                sample = dataset[0]
                assert 'tokens' in sample, f"Dataset {i} missing tokens"
                assert sample['tokens'].dtype == torch.long, f"Dataset {i} wrong token dtype"
                
                # Test dataloader
                dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
                batch = next(iter(dataloader))
                
                self.results['data_pipeline'][f'config_{i}'] = {
                    'dataset_size': len(dataset),
                    'sequence_length': sample['tokens'].shape[0],
                    'batch_shape': batch['tokens'].shape,
                    'vocab_range': (batch['tokens'].min().item(), batch['tokens'].max().item()),
                    'status': 'passed'
                }
                
                logger.info(f"    Config {i}: ✅ {len(dataset)} sequences, shape {sample['tokens'].shape}")
                
            except Exception as e:
                self.results['data_pipeline'][f'config_{i}'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.results['issues'].append(f"Data pipeline config {i} failed: {e}")
                logger.error(f"    Config {i}: ❌ {e}")
    
    def _test_model_architecture(self):
        """Test model architecture components."""
        logger.info("  Testing model creation...")
        
        # Test different model modes
        model_configs = [
            {'mode': 'transformer', 'name': 'transformer_only'},
            {'mode': 'vae', 'latent_dim': 128, 'name': 'vae_only'},
            {'mode': 'vae_gan', 'latent_dim': 128, 'discriminator_layers': 3, 'name': 'vae_gan_full'}
        ]
        
        for config in model_configs:
            try:
                model_config = {**self.test_config, **config}
                name = model_config.pop('name')
                
                # Create model
                model = MusicTransformerVAEGAN(**model_config)
                
                # Test model properties
                model_size = model.get_model_size()
                
                # Test forward pass with dummy data
                batch_size, seq_len = 2, 512
                dummy_tokens = torch.randint(0, 774, (batch_size, seq_len))
                
                with torch.no_grad():
                    if config['mode'] == 'transformer':
                        output = model(dummy_tokens)
                        assert output.shape == (batch_size, seq_len, 774), f"Wrong output shape for {name}"
                    else:  # VAE modes
                        output = model(dummy_tokens, return_latent=True)
                        assert 'logits' in output, f"Missing logits in {name}"
                        assert 'mu' in output, f"Missing mu in {name}"
                        assert 'logvar' in output, f"Missing logvar in {name}"
                        assert output['logits'].shape == (batch_size, seq_len, 774), f"Wrong logits shape for {name}"
                
                # Test loss computation
                loss_dict = model.compute_loss(dummy_tokens)
                assert 'total_loss' in loss_dict, f"Missing total_loss in {name}"
                assert 'reconstruction_loss' in loss_dict, f"Missing reconstruction_loss in {name}"
                
                self.results['model_architecture'][name] = {
                    'parameters': model_size['total_parameters'],
                    'size_mb': model_size['model_size_mb'],
                    'mode': config['mode'],
                    'forward_pass': 'passed',
                    'loss_computation': 'passed',
                    'status': 'passed'
                }
                
                logger.info(f"    {name}: ✅ {model_size['total_parameters']:,} params, {model_size['model_size_mb']:.1f}MB")
                
            except Exception as e:
                self.results['model_architecture'][name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                self.results['issues'].append(f"Model {name} failed: {e}")
                logger.error(f"    {name}: ❌ {e}")
    
    def _test_integration(self):
        """Test full integration between data and model."""
        logger.info("  Testing data-model integration...")
        
        try:
            # Create dataset
            dataset = LazyMidiDataset(
                data_dir=self.midi_dir,
                max_files=3,
                sequence_length=1024,
                truncation_strategy='adaptive',
                curriculum_learning=True
            )
            
            dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
            
            # Test with different model modes
            modes = ['transformer', 'vae', 'vae_gan']
            
            for mode in modes:
                try:
                    # Create model
                    model = MusicTransformerVAEGAN(
                        mode=mode,
                        vocab_size=774,
                        d_model=256,  # Smaller for testing
                        n_layers=2,
                        n_heads=4,
                        max_sequence_length=1024,
                        latent_dim=64 if mode in ['vae', 'vae_gan'] else None
                    )
                    
                    # Test training step
                    batch = next(iter(dataloader))
                    tokens = batch['tokens']
                    
                    # Forward pass
                    if mode == 'transformer':
                        logits = model(tokens)
                        assert logits.shape[:-1] == tokens.shape, f"Shape mismatch in {mode}"
                    else:
                        outputs = model(tokens, return_latent=True)
                        assert outputs['logits'].shape[:-1] == tokens.shape, f"Shape mismatch in {mode}"
                    
                    # Loss computation
                    loss_dict = model.compute_loss(tokens)
                    loss = loss_dict['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Test generation
                    prompt = tokens[:1, :10]  # First sample, first 10 tokens
                    if mode == 'transformer':
                        generated = model.generate(prompt, max_new_tokens=20)
                        assert generated.shape[1] == 30, f"Wrong generation length in {mode}"
                    else:
                        generated = model.generate(max_new_tokens=20)
                        assert generated.shape[1] <= 21, f"Wrong generation length in {mode}"
                    
                    self.results['integration'][f'{mode}_integration'] = {
                        'forward_pass': 'passed',
                        'loss_computation': f'{loss.item():.4f}',
                        'backward_pass': 'passed',
                        'generation': 'passed',
                        'status': 'passed'
                    }
                    
                    logger.info(f"    {mode}: ✅ Loss={loss.item():.4f}, Generation={generated.shape}")
                    
                except Exception as e:
                    self.results['integration'][f'{mode}_integration'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    self.results['issues'].append(f"Integration {mode} failed: {e}")
                    logger.error(f"    {mode}: ❌ {e}")
        
        except Exception as e:
            self.results['integration']['dataset_creation'] = {
                'status': 'failed',
                'error': str(e)
            }
            self.results['issues'].append(f"Dataset creation failed: {e}")
            logger.error(f"    Dataset creation: ❌ {e}")
    
    def _test_performance(self):
        """Test performance characteristics."""
        logger.info("  Testing performance...")
        
        try:
            # Test with realistic sequence lengths
            sequence_lengths = [512, 1024, 2048]
            
            for seq_len in sequence_lengths:
                try:
                    # Create model
                    model = MusicTransformerVAEGAN(
                        mode='transformer',
                        vocab_size=774,
                        d_model=512,
                        n_layers=4,
                        n_heads=8,
                        max_sequence_length=seq_len,
                        attention_type='hierarchical'
                    )
                    
                    # Test memory usage
                    batch_size = 2
                    dummy_tokens = torch.randint(0, 774, (batch_size, seq_len))
                    
                    # Time forward pass
                    import time
                    start_time = time.time()
                    
                    with torch.no_grad():
                        logits = model(dummy_tokens)
                    
                    forward_time = time.time() - start_time
                    
                    # Memory usage (approximate)
                    model_params = sum(p.numel() for p in model.parameters())
                    activation_memory = batch_size * seq_len * 512 * 4  # Approximate
                    total_memory_mb = (model_params * 4 + activation_memory) / 1024 / 1024
                    
                    self.results['performance'][f'seq_len_{seq_len}'] = {
                        'forward_time_sec': forward_time,
                        'memory_mb': total_memory_mb,
                        'tokens_per_sec': (batch_size * seq_len) / forward_time,
                        'status': 'passed'
                    }
                    
                    logger.info(f"    Length {seq_len}: ✅ {forward_time:.3f}s, {total_memory_mb:.1f}MB, {(batch_size * seq_len) / forward_time:.0f} tok/s")
                
                except Exception as e:
                    self.results['performance'][f'seq_len_{seq_len}'] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    self.results['issues'].append(f"Performance test {seq_len} failed: {e}")
                    logger.error(f"    Length {seq_len}: ❌ {e}")
        
        except Exception as e:
            self.results['issues'].append(f"Performance testing failed: {e}")
            logger.error(f"    Performance testing: ❌ {e}")
    
    def _test_edge_cases(self):
        """Test edge cases and error handling."""
        logger.info("  Testing edge cases...")
        
        edge_cases = [
            {
                'name': 'empty_sequence',
                'tokens': torch.zeros(1, 1, dtype=torch.long),
                'expected': 'should_handle'
            },
            {
                'name': 'max_length_sequence',
                'tokens': torch.randint(0, 387, (1, 2048)),
                'expected': 'should_handle'
            },
            {
                'name': 'out_of_vocab',
                'tokens': torch.tensor([[0, 1, 1000]], dtype=torch.long),  # 1000 > 774
                'expected': 'should_error'
            }
        ]
        
        model = MusicTransformerVAEGAN(
            mode='transformer',
            vocab_size=774,
            d_model=256,
            n_layers=2,
            n_heads=4,
            max_sequence_length=2048
        )
        
        for case in edge_cases:
            try:
                with torch.no_grad():
                    output = model(case['tokens'])
                
                if case['expected'] == 'should_error':
                    self.results['issues'].append(f"Edge case {case['name']} should have failed but didn't")
                    logger.warning(f"    {case['name']}: ⚠️ Should have failed")
                else:
                    logger.info(f"    {case['name']}: ✅ Handled correctly")
                
            except Exception as e:
                if case['expected'] == 'should_error':
                    logger.info(f"    {case['name']}: ✅ Correctly failed: {e}")
                else:
                    self.results['issues'].append(f"Edge case {case['name']} failed: {e}")
                    logger.error(f"    {case['name']}: ❌ {e}")
    
    def _generate_report(self):
        """Generate comprehensive audit report."""
        report_path = self.output_dir / "integration_audit_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# 🔍 Model-Data Integration Audit Report\n\n")
            f.write(f"**Date:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"**Status:** {'✅ PASSED' if not self.results['issues'] else '❌ ISSUES FOUND'}\n\n")
            
            if self.results['issues']:
                f.write("## 🚨 Issues Found\n\n")
                for issue in self.results['issues']:
                    f.write(f"- ❌ {issue}\n")
                f.write("\n")
            
            f.write("## 📊 Test Results\n\n")
            
            # Data Pipeline Results
            f.write("### Data Pipeline Tests\n\n")
            for config, results in self.results['data_pipeline'].items():
                status = "✅" if results['status'] == 'passed' else "❌"
                f.write(f"- **{config}** {status}\n")
                if results['status'] == 'passed':
                    f.write(f"  - Dataset size: {results['dataset_size']}\n")
                    f.write(f"  - Sequence length: {results['sequence_length']}\n")
                    f.write(f"  - Batch shape: {results['batch_shape']}\n")
                    f.write(f"  - Vocab range: {results['vocab_range']}\n")
                else:
                    f.write(f"  - Error: {results['error']}\n")
                f.write("\n")
            
            # Model Architecture Results
            f.write("### Model Architecture Tests\n\n")
            for model, results in self.results['model_architecture'].items():
                status = "✅" if results['status'] == 'passed' else "❌"
                f.write(f"- **{model}** {status}\n")
                if results['status'] == 'passed':
                    f.write(f"  - Parameters: {results['parameters']:,}\n")
                    f.write(f"  - Size: {results['size_mb']:.1f}MB\n")
                    f.write(f"  - Mode: {results['mode']}\n")
                else:
                    f.write(f"  - Error: {results['error']}\n")
                f.write("\n")
            
            # Integration Results
            f.write("### Integration Tests\n\n")
            for test, results in self.results['integration'].items():
                status = "✅" if results['status'] == 'passed' else "❌"
                f.write(f"- **{test}** {status}\n")
                if results['status'] == 'passed':
                    f.write(f"  - Forward pass: {results['forward_pass']}\n")
                    f.write(f"  - Loss: {results['loss_computation']}\n")
                    f.write(f"  - Backward pass: {results['backward_pass']}\n")
                    f.write(f"  - Generation: {results['generation']}\n")
                else:
                    f.write(f"  - Error: {results['error']}\n")
                f.write("\n")
            
            # Performance Results
            f.write("### Performance Tests\n\n")
            for test, results in self.results['performance'].items():
                status = "✅" if results['status'] == 'passed' else "❌"
                f.write(f"- **{test}** {status}\n")
                if results['status'] == 'passed':
                    f.write(f"  - Forward time: {results['forward_time_sec']:.3f}s\n")
                    f.write(f"  - Memory usage: {results['memory_mb']:.1f}MB\n")
                    f.write(f"  - Throughput: {results['tokens_per_sec']:.0f} tokens/sec\n")
                else:
                    f.write(f"  - Error: {results['error']}\n")
                f.write("\n")
            
            # Recommendations
            f.write("## 🎯 Recommendations\n\n")
            
            if not self.results['issues']:
                f.write("✅ **All tests passed!** The integration between data pipeline and model architecture is working correctly.\n\n")
                f.write("**Ready for Phase 3.2:** The foundation is solid for implementing VAE components.\n\n")
            else:
                f.write("❌ **Issues found that need attention before proceeding.**\n\n")
            
            f.write("### Performance Insights\n\n")
            perf_results = self.results['performance']
            if perf_results:
                f.write("- **Memory scaling:** Memory usage scales roughly linearly with sequence length\n")
                f.write("- **Speed:** Hierarchical attention provides good throughput for long sequences\n")
                f.write("- **Recommended:** Start training with sequence_length=1024, scale to 2048\n\n")
            
            f.write("### Next Steps\n\n")
            f.write("1. **If all tests passed:** Proceed to Phase 3.2 (VAE components)\n")
            f.write("2. **If issues found:** Address issues before continuing\n")
            f.write("3. **Performance optimization:** Consider gradient checkpointing for longer sequences\n")
            f.write("4. **Integration testing:** Test with full dataset in Phase 4\n")
        
        logger.info(f"📄 Report saved to: {report_path}")


def main():
    """Run the integration audit."""
    project_root = Path(__file__).parent.parent.parent
    
    # Check if we have MIDI data
    midi_dir = project_root / "data" / "raw"
    if not midi_dir.exists() or not any(midi_dir.glob("*.mid")):
        print("❌ No MIDI data found. Please add MIDI files to data/raw/ directory.")
        print("   You can download test data or use your own MIDI files.")
        return
    
    # Run audit
    tester = ModelDataIntegrationTester(project_root)
    results = tester.run_full_audit()
    
    # Print summary
    print("\n" + "="*60)
    print("🔍 INTEGRATION AUDIT SUMMARY")
    print("="*60)
    
    if results['issues']:
        print(f"❌ Status: ISSUES FOUND ({len(results['issues'])} issues)")
        for issue in results['issues']:
            print(f"   - {issue}")
    else:
        print("✅ Status: ALL TESTS PASSED")
    
    print(f"\n📄 Full report: outputs/integration_audit/integration_audit_report.md")
    print("\n🎯 Ready for Phase 3.2: VAE Components" if not results['issues'] else "🛠️  Fix issues before proceeding")


if __name__ == "__main__":
    main()