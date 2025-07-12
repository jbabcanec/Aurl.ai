#!/usr/bin/env python3
"""
Progressive Training Pipeline for Aurl.ai Music Generation.

This implements a staged training approach:
1. Stage 1: Transformer-only (simple autoregressive modeling)
2. Stage 2: Transformer + VAE (add latent variable modeling)
3. Stage 3: Full VAE-GAN (add adversarial training)

Each stage validates before proceeding to the next.
"""

import argparse
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Fix matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.base_logger import setup_logger
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset, create_dataloader
from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.training.core.losses import ComprehensiveLossFramework

logger = setup_logger(__name__)


class ProgressiveTrainer:
    """Manages progressive training through stages."""
    
    def __init__(self, base_config: Dict[str, Any], device: torch.device):
        self.base_config = base_config
        self.device = device
        self.current_stage = 0
        self.stage_results = {}
        
    def create_stage_config(self, stage: int) -> Dict[str, Any]:
        """Create configuration for a specific training stage."""
        config = self.base_config.copy()
        
        if stage == 1:
            # Stage 1: Transformer-only
            config['model']['mode'] = 'transformer'
            config['training']['num_epochs'] = 20  # Shorter for initial stage
            config['training']['learning_rate'] = 5e-4
            config['experiment']['name'] = f"{config['experiment']['name']}_stage1_transformer"
            
        elif stage == 2:
            # Stage 2: Transformer + VAE
            config['model']['mode'] = 'vae'
            config['training']['num_epochs'] = 30
            config['training']['learning_rate'] = 3e-4
            config['training']['kl_weight'] = 0.1  # Start with low KL weight
            config['experiment']['name'] = f"{config['experiment']['name']}_stage2_vae"
            
        elif stage == 3:
            # Stage 3: Full VAE-GAN
            config['model']['mode'] = 'vae_gan'
            config['training']['num_epochs'] = 50
            config['training']['learning_rate'] = 1e-4
            config['training']['adversarial_weight'] = 0.05  # Start with low adversarial weight
            config['experiment']['name'] = f"{config['experiment']['name']}_stage3_vae_gan"
            
        return config
    
    def validate_stage(self, stage: int, model: MusicTransformerVAEGAN, 
                      dataset: LazyMidiDataset) -> bool:
        """Validate that a stage trained successfully."""
        logger.info(f"Validating Stage {stage}...")
        
        model.eval()
        validation_passed = True
        
        # Create a small validation batch
        val_loader = create_dataloader(dataset, batch_size=4, shuffle=False)
        val_batch = next(iter(val_loader))
        tokens = val_batch['tokens'].to(self.device)
        
        with torch.no_grad():
            try:
                # Test forward pass
                if stage == 1:
                    # Transformer-only validation
                    output = model(tokens)
                    if output.shape != (tokens.shape[0], tokens.shape[1], model.vocab_size):
                        logger.error(f"Invalid output shape: {output.shape}")
                        validation_passed = False
                        
                elif stage == 2:
                    # VAE validation
                    output = model(tokens, return_latent=True)
                    if 'reconstruction' not in output or 'mu' not in output:
                        logger.error("Missing VAE outputs")
                        validation_passed = False
                        
                elif stage == 3:
                    # VAE-GAN validation
                    losses = model.compute_loss(tokens)
                    required_losses = ['total_loss', 'reconstruction_loss', 'kl_loss', 
                                     'generator_loss', 'discriminator_loss']
                    for loss_name in required_losses:
                        if loss_name not in losses:
                            logger.error(f"Missing loss: {loss_name}")
                            validation_passed = False
                
                # Test generation
                generated = model.generate(
                    batch_size=1,
                    max_length=64,
                    temperature=1.0,
                    device=self.device
                )
                
                if generated.shape[1] != 64:
                    logger.error(f"Generation failed: {generated.shape}")
                    validation_passed = False
                    
            except Exception as e:
                logger.error(f"Validation failed with error: {e}")
                validation_passed = False
        
        if validation_passed:
            logger.info(f"✅ Stage {stage} validation PASSED")
        else:
            logger.error(f"❌ Stage {stage} validation FAILED")
            
        return validation_passed
    
    def train_stage(self, stage: int, dataset: LazyMidiDataset, 
                   checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Train a single stage."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting Stage {stage} Training")
        logger.info(f"{'='*60}\n")
        
        # Create stage-specific configuration
        stage_config = self.create_stage_config(stage)
        
        # Create model
        model_config = stage_config['model']
        model = MusicTransformerVAEGAN(
            vocab_size=model_config.get('vocab_size', 774),
            d_model=model_config.get('hidden_dim', 256),
            n_layers=model_config.get('num_layers', 4),
            n_heads=model_config.get('num_heads', 4),
            latent_dim=model_config.get('latent_dim', 66),
            max_sequence_length=model_config.get('max_sequence_length', 512),
            mode=model_config.get('mode', 'transformer'),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Load checkpoint if continuing from previous stage
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Load only compatible weights
            model_state = model.state_dict()
            pretrained_state = checkpoint['model_state_dict']
            
            # Filter out incompatible keys
            compatible_state = {}
            for k, v in pretrained_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible_state[k] = v
                    
            model_state.update(compatible_state)
            model.load_state_dict(model_state)
            logger.info(f"Loaded {len(compatible_state)}/{len(model_state)} compatible weights")
        
        # Create trainer
        training_config = stage_config['training']
        trainer_config = TrainingConfig(
            batch_size=training_config.get('batch_size', 4),
            learning_rate=training_config.get('learning_rate', 5e-4),
            num_epochs=training_config.get('num_epochs', 20),
            warmup_steps=training_config.get('warmup_steps', 100),
            weight_decay=training_config.get('weight_decay', 1e-5),
            use_mixed_precision=training_config.get('mixed_precision', True),
            max_grad_norm=training_config.get('gradient_clip_norm', 1.0)
        )
        
        # Create loss framework
        loss_framework = ComprehensiveLossFramework(
            vocab_size=model_config.get('vocab_size', 774),
            adversarial_weight=training_config.get('adversarial_weight', 0.05),
            warmup_epochs=training_config.get('warmup_epochs', 10) // 100  # Convert steps to epochs
        ).to(self.device)
        
        trainer = AdvancedTrainer(
            model=model,
            config=trainer_config,
            loss_framework=loss_framework,
            train_dataset=dataset
        )
        
        # Training loop with monitoring
        logger.info(f"Training Stage {stage} for {trainer_config.num_epochs} epochs...")
        start_time = time.time()
        
        try:
            results = trainer.train()
            
            # Validate stage
            if self.validate_stage(stage, model, dataset):
                # Save checkpoint
                checkpoint_dir = Path(stage_config['system']['output_dir']) / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'stage{stage}_final.pt'
                
                torch.save({
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'config': stage_config,
                    'results': results
                }, checkpoint_path)
                
                logger.info(f"✅ Stage {stage} completed successfully!")
                logger.info(f"Checkpoint saved to: {checkpoint_path}")
                
                results['checkpoint_path'] = str(checkpoint_path)
                results['validation_passed'] = True
                results['training_time'] = time.time() - start_time
                
            else:
                results['validation_passed'] = False
                logger.error(f"❌ Stage {stage} validation failed!")
                
        except Exception as e:
            logger.error(f"❌ Stage {stage} training failed: {e}")
            results = {
                'validation_passed': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
            
        return results
    
    def run_progressive_training(self, dataset: LazyMidiDataset):
        """Run full progressive training pipeline."""
        logger.info("\n" + "="*80)
        logger.info("🎼 PROGRESSIVE TRAINING PIPELINE")
        logger.info("="*80 + "\n")
        
        checkpoint_path = None
        
        for stage in [1, 2, 3]:
            # Train stage
            results = self.train_stage(stage, dataset, checkpoint_path)
            self.stage_results[f'stage_{stage}'] = results
            
            # Check if stage passed
            if not results.get('validation_passed', False):
                logger.error(f"\n❌ Training halted at Stage {stage}")
                logger.error("Please fix the issues and retry.")
                break
                
            # Use this stage's checkpoint for next stage
            checkpoint_path = results.get('checkpoint_path')
            
            # Brief pause between stages
            if stage < 3:
                logger.info(f"\n⏳ Pausing before Stage {stage + 1}...")
                time.sleep(5)
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print training summary."""
        logger.info("\n" + "="*80)
        logger.info("📊 PROGRESSIVE TRAINING SUMMARY")
        logger.info("="*80)
        
        total_time = 0
        for stage in [1, 2, 3]:
            stage_key = f'stage_{stage}'
            if stage_key in self.stage_results:
                results = self.stage_results[stage_key]
                status = "✅ PASSED" if results.get('validation_passed') else "❌ FAILED"
                time_taken = results.get('training_time', 0)
                total_time += time_taken
                
                logger.info(f"\nStage {stage}: {status}")
                logger.info(f"  Time: {time_taken/60:.1f} minutes")
                
                if results.get('validation_passed'):
                    logger.info(f"  Checkpoint: {results.get('checkpoint_path', 'N/A')}")
                else:
                    logger.info(f"  Error: {results.get('error', 'Validation failed')}")
        
        logger.info(f"\nTotal training time: {total_time/60:.1f} minutes")
        
        # Check if all stages passed
        all_passed = all(
            self.stage_results.get(f'stage_{i}', {}).get('validation_passed', False)
            for i in [1, 2, 3]
        )
        
        if all_passed:
            logger.info("\n🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
            logger.info("The model is ready for generation.")
        else:
            logger.info("\n⚠️  Progressive training incomplete.")


def main():
    """Main entry point for progressive training."""
    parser = argparse.ArgumentParser(
        description="Progressive Training Pipeline for Aurl.ai"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_configs/quick_test.yaml",
        help="Base configuration file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, mps, auto)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create dataset
    data_config = config["data"]
    system_config = config["system"]
    
    dataset = LazyMidiDataset(
        data_dir=system_config.get("data_dir", "data/raw"),
        cache_dir=system_config.get("cache_dir", "data/cache"),
        sequence_length=data_config.get("sequence_length", 512),
        overlap=data_config.get("overlap", 128),
        enable_augmentation=True,  # Enable augmentation
        augmentation_probability=0.3  # Start with moderate augmentation
    )
    
    logger.info(f"Dataset created with {len(dataset)} sequences")
    
    # Create progressive trainer
    trainer = ProgressiveTrainer(config, device)
    
    # Run progressive training
    trainer.run_progressive_training(dataset)


if __name__ == "__main__":
    main()