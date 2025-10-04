#!/usr/bin/env python3
"""
Enhanced Training Script with Dissolvable Guidance

This script integrates all the advanced training features:
- Dissolvable guidance that fades as model learns
- Adaptive constraints that relax with performance
- Training monitor with auto-correction
- Advanced grammar enforcement with note pairing focus
"""

import sys
import torch
import argparse
from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Core training infrastructure
from src.training.core.trainer import AdvancedTrainer, TrainingConfig
from src.training.core.losses import ComprehensiveLossFramework
from src.training.utils.grammar_integration import GrammarEnhancedTraining, GrammarTrainingConfig

# New dissolvable components
from src.training.dissolvable_guidance import (
    DissolvableGuidanceModule,
    DissolvableGuidanceConfig,
    AdaptiveGuidanceScheduler
)
from src.training.adaptive_constraints import (
    HybridConstraintSystem,
    AdaptiveConstraintConfig
)
from src.training.training_monitor import TrainingMonitor, MonitoringConfig

# Model and data
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class EnhancedGrammarTrainer(AdvancedTrainer):
    """
    Enhanced trainer with dissolvable guidance and adaptive constraints.
    """

    def __init__(
        self,
        model,
        loss_framework,
        config: TrainingConfig,
        train_dataset,
        val_dataset=None,
        save_dir=None
    ):
        # Initialize base trainer
        super().__init__(model, loss_framework, config, train_dataset, val_dataset, save_dir)

        # Load enhanced configuration
        self.enhanced_config = self._load_enhanced_config()

        # Initialize dissolvable guidance
        self.guidance_module = DissolvableGuidanceModule(
            VocabularyConfig(),
            DissolvableGuidanceConfig(**self.enhanced_config['dissolvable_guidance'])
        ).to(self.device)

        # Initialize adaptive constraints
        self.constraint_system = HybridConstraintSystem(
            DissolvableGuidanceConfig(**self.enhanced_config['dissolvable_guidance']),
            AdaptiveConstraintConfig(
                initial_harmonic_strictness=1.0,
                initial_melodic_strictness=1.0,
                initial_rhythmic_strictness=1.0,
                initial_dynamic_strictness=1.0
            )
        ).to(self.device)

        # Initialize training monitor
        self.monitor = TrainingMonitor(
            MonitoringConfig(
                check_frequency=100,
                enable_auto_correction=True,
                save_monitor_logs=True
            ),
            self.model,
            self.optimizer
        )

        # Grammar integration
        self.grammar_config = GrammarTrainingConfig(
            grammar_validation_frequency=10,
            collapse_threshold=0.5,
            enable_rollback=True
        )
        self.grammar_trainer = GrammarEnhancedTraining(
            self.model,
            self.device,
            VocabularyConfig(),
            self.grammar_config
        )

        logger.info("Initialized Enhanced Grammar Trainer with dissolvable guidance")

    def _load_enhanced_config(self) -> Dict:
        """Load enhanced configuration from yaml."""
        config_path = Path("configs/training_configs/dissolvable_grammar.yaml")
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            return {
                'dissolvable_guidance': {},
                'musical_grammar': {},
                'adaptive_constraints': {}
            }

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Enhanced training step with guidance and monitoring."""
        # Get tokens from batch
        tokens = batch['tokens'].to(self.device)
        batch_size, seq_len = tokens.shape

        # Forward pass
        outputs = self.model(tokens[:, :-1])
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        targets = tokens[:, 1:]

        # Apply dissolvable guidance to logits
        current_epoch = self.current_epoch
        performance = self._get_current_performance()

        # Apply guidance and constraints
        guided_logits = self.constraint_system(
            logits.reshape(-1, logits.size(-1)),
            tokens,
            current_epoch,
            performance,
            training=True
        ).reshape(batch_size, seq_len - 1, -1)

        # Calculate loss with guided logits
        loss_dict = self.loss_framework(guided_logits, targets, self.model)

        # Apply grammar enhancement
        enhanced_loss = self.grammar_trainer.apply_grammar_loss(
            loss_dict.get('total_loss', loss_dict.get('loss')),
            guided_logits,
            targets
        )

        # Monitor training and apply corrections
        with torch.no_grad():
            # Sample some tokens for monitoring
            sampled_tokens = torch.argmax(guided_logits, dim=-1)
            monitor_results = self.monitor.check(
                sampled_tokens.flatten()[:100],  # Check first 100 tokens
                enhanced_loss.item(),
                batch_idx
            )

        # Check for grammar collapse
        if batch_idx % self.grammar_config.grammar_validation_frequency == 0:
            grammar_valid = self.grammar_trainer.validate_grammar_learning(guided_logits)
            if not grammar_valid and self.grammar_config.enable_rollback:
                logger.warning("Grammar collapse detected, initiating rollback")
                self.grammar_trainer.rollback_on_collapse()

        # Backward pass
        self.optimizer.zero_grad()
        enhanced_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

        # Log metrics
        metrics = {
            'loss': enhanced_loss.item(),
            'guidance_strength': self.constraint_system.dissolvable_guidance.state.current_strength,
            'note_pairing_score': monitor_results['checks'].get('note_pairing_score', 0.0)
        }

        # Update guidance based on performance
        if batch_idx % 500 == 0:
            self._update_guidance_systems(metrics)

        return metrics

    def _get_current_performance(self) -> Dict:
        """Get current training performance metrics."""
        # This would typically come from validation or recent training metrics
        return {
            'note_pairing_score': getattr(self, 'last_note_pairing_score', 0.5),
            'grammar_score': getattr(self, 'last_grammar_score', 0.5),
            'loss': getattr(self, 'last_loss', 1.0)
        }

    def _update_guidance_systems(self, metrics: Dict):
        """Update guidance and constraint systems based on performance."""
        # Store recent metrics
        self.last_note_pairing_score = metrics.get('note_pairing_score', 0.5)
        self.last_loss = metrics.get('loss', 1.0)

        # Log guidance status
        guidance_stats = self.constraint_system.get_stats()
        logger.info(
            f"Guidance Status - Strength: {guidance_stats['guidance']['current_strength']:.3f}, "
            f"Dissolved: {guidance_stats['guidance']['is_dissolved']}"
        )

    def validation_step(self, batch: Dict) -> Dict:
        """Validation step with grammar checking."""
        with torch.no_grad():
            tokens = batch['tokens'].to(self.device)

            # Forward pass
            outputs = self.model(tokens[:, :-1])
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            targets = tokens[:, 1:]

            # Calculate loss (no guidance during validation)
            loss_dict = self.loss_framework(logits, targets, self.model)

            # Check grammar quality
            grammar_results = self.grammar_trainer.compute_grammar_metrics(logits, targets)

            return {
                'val_loss': loss_dict.get('total_loss', loss_dict.get('loss')).item(),
                'val_grammar_score': grammar_results['overall_score'],
                'val_note_pairing': grammar_results['note_pairing_score']
            }


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Music Generation Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_configs/dissolvable_grammar.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/training/enhanced',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=150,
        help='Maximum number of training epochs'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model_config = config['model']
    model = MusicTransformerVAEGAN(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['hidden_dim'],
        n_layers=model_config['num_layers'],
        n_heads=model_config['num_heads'],
        max_sequence_length=model_config['max_sequence_length'],
        mode=model_config['mode'],
        latent_dim=model_config['latent_dim'],
        encoder_layers=model_config['encoder_layers'],
        decoder_layers=model_config['decoder_layers']
    )

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize loss framework
    loss_framework = ComprehensiveLossFramework()

    # Initialize datasets
    data_config = config['data']
    train_dataset = LazyMidiDataset(
        data_dir=data_config['train_data_path'],
        cache_dir=data_config['cache_dir'],
        sequence_length=data_config['sequence_length'],
        overlap=data_config['overlap']
    )

    val_dataset = None
    if config['validation']['enabled']:
        # In practice, you'd split the data properly
        val_dataset = train_dataset  # Placeholder

    # Initialize training config
    training_config = TrainingConfig(
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=args.max_epochs,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        use_mixed_precision=config['training']['mixed_precision']
    )

    # Initialize enhanced trainer
    trainer = EnhancedGrammarTrainer(
        model=model,
        loss_framework=loss_framework,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=output_dir
    )

    # Start training
    logger.info("="*50)
    logger.info("Starting Enhanced Training with Dissolvable Guidance")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info("="*50)

    # Train
    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()