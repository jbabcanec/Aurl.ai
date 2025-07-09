#!/usr/bin/env python3
"""
Aurl.ai Music Generation - Main Training Pipeline

This is the main entry point for training music generation models with Aurl.ai.
Supports all Phase 4 advanced training techniques including:
- Distributed training
- Mixed precision 
- Comprehensive monitoring
- Advanced optimizers
- Curriculum learning
- Hyperparameter optimization

Usage:
    python train_pipeline.py                                    # Default training
    python train_pipeline.py --config configs/quick_test.yaml   # Quick test
    python train_pipeline.py --config configs/high_quality.yaml # High quality
    python train_pipeline.py --resume outputs/checkpoints/latest.pt  # Resume training
"""

import argparse
import sys
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import load_config
from src.utils.base_logger import setup_logger
from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.training.utils.reproducibility import setup_reproducible_training
from src.training.utils.checkpoint_manager import CheckpointManager
from src.training.utils.hyperparameter_optimization import (
    HyperparameterOptimizer, create_quick_search_config
)
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from src.training.core.losses import ComprehensiveLossFramework
from src.training.monitoring.realtime_dashboard import RealTimeDashboard

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aurl.ai Music Generation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_pipeline.py                                    # Default training
  python train_pipeline.py --config configs/quick_test.yaml   # Quick test (fast)
  python train_pipeline.py --config configs/high_quality.yaml # High quality (slow)
  python train_pipeline.py --resume outputs/checkpoints/latest.pt  # Resume
  python train_pipeline.py --optimize-hyperparameters         # Auto-optimize
  python train_pipeline.py --gpu 0,1 --batch-size 64        # Multi-GPU with larger batch
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging and checkpoints"
    )
    
    # Training control
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    # Hardware
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU IDs to use (e.g., '0', '0,1', 'auto', 'cpu')"
    )
    
    # Advanced options
    parser.add_argument(
        "--optimize-hyperparameters",
        action="store_true",
        help="Run hyperparameter optimization before training"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and data without training"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    # Monitoring
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable real-time dashboard"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    return parser.parse_args()


def setup_device(gpu_arg: str) -> torch.device:
    """Setup compute device based on arguments."""
    if gpu_arg.lower() == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU for training")
        return device
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    
    if gpu_arg.lower() == "auto":
        device = torch.device("cuda:0")
        logger.info(f"Auto-selected GPU: {device}")
    else:
        try:
            gpu_id = int(gpu_arg.split(",")[0])
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Using GPU: {device}")
        except (ValueError, IndexError):
            logger.warning(f"Invalid GPU specification: {gpu_arg}, falling back to cuda:0")
            device = torch.device("cuda:0")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> MusicTransformerVAEGAN:
    """Create and initialize the model."""
    logger.info("Creating MusicTransformerVAEGAN model...")
    
    model_config = config["model"]
    model = MusicTransformerVAEGAN(
        vocab_size=model_config.get("vocab_size", 774),  # Use actual vocab size
        d_model=model_config.get("hidden_dim", 512),
        n_layers=model_config.get("num_layers", 8),
        n_heads=model_config.get("num_heads", 8),
        latent_dim=model_config.get("latent_dim", 128),
        max_seq_length=model_config.get("max_sequence_length", 2048),
        mode=model_config.get("mode", "vae_gan"),
        dropout=model_config.get("dropout", 0.1)
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    logger.info(f"Model mode: {model_config.get('mode', 'vae_gan')}")
    
    return model


def create_dataset(config: Dict[str, Any]) -> LazyMidiDataset:
    """Create the training dataset."""
    logger.info("Creating training dataset...")
    
    data_config = config["data"]
    system_config = config["system"]
    
    dataset = LazyMidiDataset(
        data_dir=system_config.get("data_dir", "data/raw"),
        cache_dir=system_config.get("cache_dir", "data/cache"),
        sequence_length=data_config.get("sequence_length", 2048),
        overlap=data_config.get("overlap", 256),
        augmentation_config=data_config.get("augmentation", {}),
        cache_size_gb=data_config.get("cache_size_gb", 5),
        num_workers=data_config.get("num_workers", 4)
    )
    
    logger.info(f"Dataset created with {len(dataset)} sequences")
    return dataset


def run_hyperparameter_optimization(config: Dict[str, Any], model: MusicTransformerVAEGAN, 
                                   dataset: LazyMidiDataset, device: torch.device) -> Dict[str, Any]:
    """Run hyperparameter optimization."""
    logger.info("🔍 Starting hyperparameter optimization...")
    
    hyperopt_config = create_quick_search_config()
    hyperopt_config.max_trials = 20  # Reasonable for initial optimization
    
    optimizer = HyperparameterOptimizer(hyperopt_config)
    
    def objective_function(params: Dict[str, Any]) -> Dict[str, Any]:
        """Objective function for hyperparameter optimization."""
        # Create temporary config with suggested parameters
        temp_config = config.copy()
        temp_config["training"]["learning_rate"] = params.get("learning_rate", 1e-4)
        temp_config["training"]["batch_size"] = params.get("batch_size", 32)
        temp_config["model"]["latent_dim"] = params.get("latent_dim", 128)
        
        # Run short training session (5 epochs) to evaluate
        temp_config["training"]["num_epochs"] = 5
        temp_config["training"]["early_stopping"] = False
        
        try:
            # Create trainer with temporary config
            trainer_config = TrainingConfig(**temp_config["training"])
            trainer = AdvancedTrainer(model, trainer_config, device)
            
            # Quick training
            results = trainer.train(dataset, epochs=5)
            
            # Return metrics for optimization
            return {
                "validation_loss": results.get("final_loss", float("inf")),
                "musical_quality": results.get("musical_quality", 0.0),
                "training_efficiency": results.get("samples_per_second", 0.0) / 100.0
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter trial failed: {e}")
            return {
                "validation_loss": float("inf"),
                "musical_quality": 0.0,
                "training_efficiency": 0.0
            }
    
    # Run optimization
    results = optimizer.optimize(objective_function)
    
    best_params = results["best_trial"]["parameters"]
    logger.info(f"✅ Hyperparameter optimization complete!")
    logger.info(f"Best parameters: {best_params}")
    
    # Update config with best parameters
    config["training"]["learning_rate"] = best_params.get("learning_rate", config["training"]["learning_rate"])
    config["training"]["batch_size"] = best_params.get("batch_size", config["training"]["batch_size"])
    config["model"]["latent_dim"] = best_params.get("latent_dim", config["model"]["latent_dim"])
    
    return config


def run_training(config: Dict[str, Any], model: MusicTransformerVAEGAN, 
                dataset: LazyMidiDataset, device: torch.device, 
                args: argparse.Namespace) -> None:
    """Run the main training loop."""
    
    # Setup reproducibility if requested
    if config.get("system", {}).get("seed"):
        setup_reproducible_training(config["system"]["seed"])
        logger.info(f"Reproducible training enabled with seed: {config['system']['seed']}")
    
    # Create checkpoint manager
    checkpoint_dir = Path(config["system"]["output_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        max_checkpoints=config["training"].get("keep_best_n_checkpoints", 3),
        save_every_n_epochs=config["training"].get("save_every_n_epochs", 5)
    )
    
    # Create loss framework
    loss_framework = ComprehensiveLossFramework(
        vocab_size=config["model"].get("vocab_size", 774),
        reconstruction_weight=config["training"].get("reconstruction_weight", 1.0),
        kl_weight=config["training"].get("kl_weight", 1.0),
        adversarial_weight=config["training"].get("adversarial_weight", 0.1)
    ).to(device)
    
    # Create trainer
    trainer_config = TrainingConfig(**config["training"])
    trainer = AdvancedTrainer(
        model=model,
        config=trainer_config,
        device=device,
        loss_framework=loss_framework,
        checkpoint_manager=checkpoint_manager
    )
    
    # Setup monitoring
    dashboard = None
    if not args.no_dashboard:
        try:
            dashboard = RealTimeDashboard(
                experiment_name=config["experiment"]["name"],
                output_dir=config["system"]["output_dir"]
            )
            dashboard.start()
            logger.info("🖥️  Real-time dashboard started")
        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = checkpoint_manager.load_checkpoint(args.resume)
            trainer.load_state_dict(checkpoint)
            start_epoch = checkpoint.get("epoch", 0)
            logger.info(f"✅ Resumed from checkpoint at epoch {start_epoch}")
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            return
    
    # Start training
    logger.info("🚀 Starting training...")
    logger.info(f"📊 Configuration:")
    logger.info(f"  - Model: {config['model']['mode']} ({sum(p.numel() for p in model.parameters()):,} params)")
    logger.info(f"  - Dataset: {len(dataset)} sequences")
    logger.info(f"  - Batch size: {config['training']['batch_size']}")
    logger.info(f"  - Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  - Epochs: {config['training']['num_epochs']}")
    logger.info(f"  - Device: {device}")
    
    try:
        # Run training
        results = trainer.train(
            dataset=dataset,
            epochs=config["training"]["num_epochs"] - start_epoch,
            start_epoch=start_epoch
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📈 Final results:")
        logger.info(f"  - Best loss: {results.get('best_loss', 'N/A')}")
        logger.info(f"  - Total epochs: {results.get('total_epochs', 'N/A')}")
        logger.info(f"  - Training time: {results.get('training_time', 'N/A'):.1f}s")
        
        # Save final model
        final_checkpoint_path = checkpoint_dir / "final_model.pt"
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=results.get('total_epochs', 0),
            loss=results.get('best_loss', float('inf')),
            additional_data={'training_complete': True, 'results': results}
        )
        logger.info(f"💾 Final model saved to: {final_checkpoint_path}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        # Save emergency checkpoint
        emergency_path = checkpoint_dir / "emergency_checkpoint.pt"
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=trainer.current_epoch,
            loss=trainer.best_loss,
            additional_data={'interrupted': True}
        )
        logger.info(f"💾 Emergency checkpoint saved to: {emergency_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
        
    finally:
        # Cleanup
        if dashboard:
            try:
                dashboard.stop()
            except:
                pass


def main():
    """Main training pipeline entry point."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging level
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"📝 Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.no_tensorboard:
        config["experiment"]["use_tensorboard"] = False
    if args.wandb:
        config["experiment"]["use_wandb"] = True
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Create model
    model = create_model(config, device)
    
    # Create dataset
    dataset = create_dataset(config)
    
    # Dry run validation
    if args.dry_run:
        logger.info("✅ Dry run successful - configuration and data are valid")
        logger.info(f"📊 Ready to train with {len(dataset)} sequences on {device}")
        return
    
    # Hyperparameter optimization
    if args.optimize_hyperparameters:
        config = run_hyperparameter_optimization(config, model, dataset, device)
        # Recreate model with optimized parameters
        model = create_model(config, device)
    
    # Run training
    run_training(config, model, dataset, device, args)


if __name__ == "__main__":
    main()