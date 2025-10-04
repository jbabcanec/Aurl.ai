#!/usr/bin/env python3
"""
Cumulative Progressive Training Script

This script trains the model progressively on each MIDI file, maintaining
state across sessions and tracking which files have been processed.
"""

import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import yaml
import time
from datetime import datetime
from typing import Dict, Optional
import signal
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Cumulative training system
from src.training.cumulative_trainer import CumulativeTrainer

# Enhanced training components
from src.training.dissolvable_guidance import (
    DissolvableGuidanceModule,
    DissolvableGuidanceConfig
)
from src.training.adaptive_constraints import (
    HybridConstraintSystem,
    AdaptiveConstraintConfig
)
from src.training.training_monitor import TrainingMonitor, MonitoringConfig

# Core training
from src.training.core.trainer import TrainingConfig
from src.training.core.losses import ComprehensiveLossFramework
from src.training.musical_grammar import MusicalGrammarLoss

# Model and data
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.dataset import LazyMidiDataset
from src.data.representation import VocabularyConfig
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ProgressiveTrainer:
    """
    Trains model progressively on individual MIDI files with cumulative learning.
    """

    def __init__(self, config_path: str, checkpoint_path: Optional[str] = None):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize cumulative trainer
        self.cumulative = CumulativeTrainer(
            state_file="outputs/training/cumulative_state.json",
            checkpoint_dir="outputs/checkpoints/cumulative"
        )

        # Initialize model
        self.model = self._init_model(checkpoint_path)

        # Initialize training components
        self.loss_framework = ComprehensiveLossFramework()
        self.grammar_loss = MusicalGrammarLoss()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=0.01
        )

        # Initialize guidance systems
        self.guidance = self._init_guidance_system()
        self.monitor = TrainingMonitor(
            MonitoringConfig(
                check_frequency=50,
                enable_auto_correction=True
            ),
            self.model,
            self.optimizer
        )

        # Training state
        self.global_step = 0
        self.should_stop = False

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _init_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        """Initialize or load model."""
        model_config = self.config['model']
        model = MusicTransformerVAEGAN(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['hidden_dim'],
            n_layers=model_config['num_layers'],
            n_heads=model_config['num_heads'],
            max_sequence_length=model_config['max_sequence_length'],
            mode=model_config.get('mode', 'vae_gan')
        ).to(self.device)

        # Load checkpoint
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif self.cumulative.state.last_checkpoint_path:
            # Resume from cumulative state
            checkpoint_path = self.cumulative.state.last_checkpoint_path
            if Path(checkpoint_path).exists():
                logger.info(f"Resuming from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def _init_guidance_system(self) -> HybridConstraintSystem:
        """Initialize guidance and constraint systems."""
        dissolvable_config = DissolvableGuidanceConfig(
            initial_guidance_strength=1.0,
            dissolution_start_epoch=self.cumulative.state.total_epochs + 10,
            force_note_pairing=True
        )

        adaptive_config = AdaptiveConstraintConfig(
            initial_harmonic_strictness=0.8,
            initial_melodic_strictness=0.8,
            initial_rhythmic_strictness=0.7
        )

        return HybridConstraintSystem(
            dissolvable_config,
            adaptive_config
        ).to(self.device)

    def _signal_handler(self, sig, frame):
        """Handle graceful shutdown."""
        logger.info("\n⚠️ Received interrupt signal. Saving state and shutting down...")
        self.should_stop = True

    def train_on_file(self, file_path: str, epochs_per_file: int = 5) -> Dict:
        """
        Train on a single MIDI file.

        Args:
            file_path: Path to MIDI file
            epochs_per_file: Number of epochs to train on this file

        Returns:
            Training metrics
        """
        start_time = time.time()
        file_name = Path(file_path).name

        logger.info("="*60)
        logger.info(f"Training on: {file_name}")
        logger.info("="*60)

        # Mark file as training
        self.cumulative.start_file_training(file_path)

        try:
            # Create dataset for single file
            dataset = LazyMidiDataset(
                data_dir=str(Path(file_path).parent),
                cache_dir=self.config['data']['cache_dir'],
                sequence_length=self.config['data']['sequence_length'],
                specific_files=[file_path]
            )

            if len(dataset) == 0:
                raise ValueError(f"No valid sequences in {file_name}")

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=2
            )

            # Training metrics
            metrics = {
                'losses': [],
                'grammar_scores': [],
                'note_pairing_scores': []
            }

            # Train for specified epochs
            for epoch in range(epochs_per_file):
                if self.should_stop:
                    break

                epoch_loss = 0
                epoch_grammar = 0
                batch_count = 0

                self.model.train()

                for batch_idx, batch in enumerate(dataloader):
                    if self.should_stop:
                        break

                    # Training step
                    loss, batch_metrics = self._training_step(batch, batch_idx)

                    # Accumulate metrics
                    epoch_loss += loss
                    epoch_grammar += batch_metrics.get('grammar_score', 0)
                    batch_count += 1

                    # Log progress
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"[{file_name}] Epoch {epoch+1}/{epochs_per_file} | "
                            f"Batch {batch_idx}/{len(dataloader)} | "
                            f"Loss: {loss:.4f} | "
                            f"Grammar: {batch_metrics.get('grammar_score', 0):.3f}"
                        )

                # Epoch summary
                avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
                avg_grammar = epoch_grammar / batch_count if batch_count > 0 else 0

                metrics['losses'].append(avg_loss)
                metrics['grammar_scores'].append(avg_grammar)

                logger.info(
                    f"[{file_name}] Epoch {epoch+1} Complete | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg Grammar: {avg_grammar:.3f}"
                )

                # Update cumulative epochs
                self.cumulative.state.total_epochs += 1

            # Calculate final metrics
            final_metrics = {
                'final_loss': metrics['losses'][-1] if metrics['losses'] else float('inf'),
                'grammar_score': metrics['grammar_scores'][-1] if metrics['grammar_scores'] else 0,
                'avg_loss': sum(metrics['losses']) / len(metrics['losses'])
                           if metrics['losses'] else float('inf'),
                'avg_grammar': sum(metrics['grammar_scores']) / len(metrics['grammar_scores'])
                              if metrics['grammar_scores'] else 0
            }

            # Complete file training
            training_time = time.time() - start_time
            self.cumulative.complete_file_training(
                file_path,
                final_metrics['final_loss'],
                final_metrics,
                training_time
            )

            # Save checkpoint after each file
            self._save_checkpoint(file_name)

            return final_metrics

        except Exception as e:
            logger.error(f"Error training on {file_name}: {str(e)}")
            self.cumulative.fail_file_training(file_path, str(e))
            return {}

    def _training_step(self, batch: Dict, batch_idx: int) -> tuple:
        """Single training step."""
        tokens = batch['tokens'].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(tokens[:, :-1])
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        targets = tokens[:, 1:]

        # Apply guidance
        performance_metrics = {
            'note_pairing_score': 0.7,  # Would come from actual validation
            'grammar_score': 0.7
        }

        guided_logits = self.guidance(
            logits.reshape(-1, logits.size(-1)),
            tokens,
            self.cumulative.state.total_epochs,
            performance_metrics,
            training=True
        ).reshape(logits.shape)

        # Calculate loss
        loss_dict = self.loss_framework(guided_logits, targets, self.model)
        total_loss = loss_dict.get('total_loss', loss_dict.get('loss'))

        # Add grammar loss
        grammar_results = self.grammar_loss.compute_grammar_score(tokens)
        grammar_loss = self.grammar_loss(guided_logits, targets)
        total_loss = total_loss + grammar_loss * 2.0

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Monitor training
        with torch.no_grad():
            sampled = torch.argmax(guided_logits, dim=-1)
            monitor_results = self.monitor.check(
                sampled.flatten()[:100],
                total_loss.item(),
                self.global_step
            )

        self.global_step += 1
        self.cumulative.state.total_batches += 1

        metrics = {
            'grammar_score': grammar_results['overall_score'],
            'note_pairing_score': monitor_results['checks'].get('note_pairing_score', 0)
        }

        return total_loss.item(), metrics

    def _save_checkpoint(self, tag: str = ""):
        """Save model checkpoint."""
        checkpoint_path = self.cumulative.checkpoint_dir / f"checkpoint_epoch{self.cumulative.state.total_epochs}_{tag}.pt"

        checkpoint = {
            'epoch': self.cumulative.state.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cumulative_state': {
                'completed_files': self.cumulative.state.completed_files,
                'total_files': self.cumulative.state.total_files,
                'best_loss': self.cumulative.state.best_loss,
                'best_grammar': self.cumulative.state.best_grammar_score
            }
        }

        torch.save(checkpoint, checkpoint_path)
        self.cumulative.update_checkpoint(str(checkpoint_path), self.cumulative.state.total_epochs)

        logger.info(f"💾 Saved checkpoint: {checkpoint_path.name}")

    def run_progressive_training(
        self,
        data_dir: str,
        epochs_per_file: int = 5,
        max_files: Optional[int] = None
    ):
        """
        Run progressive training on all MIDI files.

        Args:
            data_dir: Directory containing MIDI files
            epochs_per_file: Epochs to train per file
            max_files: Maximum number of files to process
        """
        logger.info("\n" + "="*60)
        logger.info("🎼 STARTING CUMULATIVE PROGRESSIVE TRAINING")
        logger.info("="*60)

        # Scan for files
        self.cumulative.scan_data_directory(data_dir)

        # Get progress
        progress = self.cumulative.get_progress_summary()
        logger.info(f"\n📊 Training Progress:")
        logger.info(f"  Total Files: {progress['total_files']}")
        logger.info(f"  Completed: {progress['completed']} ({progress['progress_percent']:.1f}%)")
        logger.info(f"  Pending: {progress['pending']}")
        logger.info(f"  Failed: {progress['failed']}")
        logger.info(f"  Best Loss: {progress['best_loss']:.4f}")
        logger.info(f"  Best Grammar: {progress['best_grammar']:.3f}")

        # Process files
        files_processed = 0
        while True:
            if self.should_stop:
                break

            if max_files and files_processed >= max_files:
                logger.info(f"Reached max files limit ({max_files})")
                break

            # Get next file
            next_file = self.cumulative.get_next_file()
            if not next_file:
                logger.info("✅ All files processed!")
                break

            file_path, record = next_file

            # Train on file
            metrics = self.train_on_file(file_path, epochs_per_file)
            files_processed += 1

            # Show progress
            progress = self.cumulative.get_progress_summary()
            logger.info(f"\n📈 Progress Update: {progress['completed']}/{progress['total_files']} files "
                       f"({progress['progress_percent']:.1f}%)")

        # Final report
        logger.info("\n" + "="*60)
        logger.info("🏁 TRAINING SESSION COMPLETE")
        logger.info("="*60)

        final_progress = self.cumulative.get_progress_summary()
        logger.info(f"Files Completed: {final_progress['completed']}/{final_progress['total_files']}")
        logger.info(f"Total Training Time: {final_progress['total_time_hours']:.2f} hours")
        logger.info(f"Best Loss Achieved: {final_progress['best_loss']:.4f}")
        logger.info(f"Best Grammar Score: {final_progress['best_grammar']:.3f}")

        # Export report
        report_path = "outputs/training/training_report.json"
        self.cumulative.export_training_report(report_path)
        logger.info(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Main entry point for cumulative training."""
    parser = argparse.ArgumentParser(description="Cumulative Progressive Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_configs/dissolvable_grammar.yaml',
        help='Training configuration file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing MIDI files'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint to resume from'
    )
    parser.add_argument(
        '--epochs-per-file',
        type=int,
        default=5,
        help='Number of epochs to train on each file'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset cumulative state and start fresh'
    )

    args = parser.parse_args()

    # Reset state if requested
    if args.reset:
        state_file = Path("outputs/training/cumulative_state.json")
        if state_file.exists():
            state_file.unlink()
            logger.info("Reset cumulative training state")

    # Initialize trainer
    trainer = ProgressiveTrainer(args.config, args.checkpoint)

    # Run training
    trainer.run_progressive_training(
        args.data_dir,
        args.epochs_per_file,
        args.max_files
    )


if __name__ == "__main__":
    main()