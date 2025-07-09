"""
Enhanced Training Logger for Aurl.ai Music Generation.

This module integrates all logging systems to provide comprehensive training monitoring:
- Structured logging with detailed information
- TensorBoard integration
- Real-time dashboard
- Experiment tracking
- Data usage monitoring
- Anomaly detection

Provides the exact logging format specified in the gameplan with full transparency.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import asdict
import torch
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

from src.training.core.experiment_tracker import (
    ComprehensiveExperimentTracker, DataUsageInfo, EpochSummary
)
from src.training.monitoring.tensorboard_logger import TensorBoardLogger
from src.training.monitoring.realtime_dashboard import RealTimeDashboard, ConsoleLogger
from src.training.monitoring.wandb_integration import WandBIntegration
from src.training.monitoring.musical_quality_tracker import MusicalQualityTracker
from src.training.monitoring.anomaly_detector import EnhancedAnomalyDetector, AnomalySeverity
from src.training.utils.experiment_comparison import ExperimentComparator
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

# Import for convenience
import torch


class EnhancedTrainingLogger:
    """
    Comprehensive training logger that integrates all monitoring systems.
    
    Features:
    - Structured logging in gameplan format
    - TensorBoard integration
    - Real-time dashboard
    - Experiment tracking
    - Data usage monitoring
    - Automatic anomaly detection
    - Progress estimation
    """
    
    def __init__(self,
                 experiment_name: str,
                 save_dir: Path,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 data_config: Dict[str, Any],
                 enable_tensorboard: bool = True,
                 enable_dashboard: bool = True,
                 enable_wandb: bool = True,
                 wandb_project: str = "aurl-ai-music",
                 log_level: str = "INFO"):
        
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracker
        self.experiment_tracker = ComprehensiveExperimentTracker(
            experiment_name=experiment_name,
            save_dir=save_dir,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config
        )
        
        self.experiment_id = self.experiment_tracker.experiment_id
        
        # Initialize TensorBoard logger
        self.tensorboard_logger = None
        if enable_tensorboard:
            try:
                self.tensorboard_logger = TensorBoardLogger(
                    log_dir=save_dir,
                    experiment_id=self.experiment_id,
                    comment=experiment_name
                )
                logger.info("TensorBoard logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
        
        # Initialize real-time dashboard
        self.dashboard = None
        if enable_dashboard:
            try:
                self.dashboard = RealTimeDashboard(
                    save_dir=save_dir,
                    experiment_id=self.experiment_id,
                    update_interval=2.0,
                    enable_gui=True
                )
                logger.info("Real-time dashboard enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize dashboard: {e}")
        
        # Initialize console logger
        self.console_logger = ConsoleLogger(self.experiment_id)
        
        # Initialize Weights & Biases logger
        self.wandb_logger = None
        if enable_wandb:
            try:
                config_dict = {
                    "model": model_config,
                    "training": training_config,
                    "data": data_config
                }
                self.wandb_logger = WandBIntegration(
                    project_name=wandb_project,
                    experiment_name=experiment_name,
                    config=config_dict,
                    tags=["vae-gan", "music-generation", "aurl-ai"]
                )
                logger.info("Weights & Biases logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
        
        # Setup structured file logging
        self._setup_structured_logging(log_level)
        
        # Initialize musical quality tracker
        self.musical_quality_tracker = MusicalQualityTracker(
            vocab_size=model_config.get('vocab_size', 774),
            quality_threshold=0.5,
            alert_on_degradation=True
        )
        
        # Initialize enhanced anomaly detector
        self.anomaly_detector = EnhancedAnomalyDetector(
            window_size=100,
            z_score_threshold=3.0,
            enable_adaptive_thresholds=True,
            alert_callback=self._handle_anomaly_alert
        )
        
        # Training state
        self.training_start_time = datetime.now()
        self.current_epoch = 0
        self.total_epochs = training_config.get('num_epochs', 100)
        self.current_batch = 0
        self.total_batches = 0
        
        # Performance tracking
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Sample tracking
        self.samples_generated = []
        
        logger.info(f"Enhanced training logger initialized for experiment: {self.experiment_id}")
        logger.info(f"Logging directory: {self.save_dir}")
    
    def _setup_structured_logging(self, log_level: str):
        """Setup structured file logging in gameplan format."""
        
        # Create logs directory
        logs_dir = self.save_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup structured logger
        self.structured_logger = logging.getLogger(f"structured_{self.experiment_id}")
        self.structured_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.structured_logger.handlers[:]:
            self.structured_logger.removeHandler(handler)
        
        # Add rotating file handler
        log_file = logs_dir / f"training_{self.experiment_id}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=100*1024*1024, backupCount=5  # 100MB per file, 5 backups
        )
        
        # Custom formatter for gameplan format
        formatter = StructuredFormatter()
        file_handler.setFormatter(formatter)
        self.structured_logger.addHandler(file_handler)
        
        # Add console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.structured_logger.addHandler(console_handler)
        
        logger.info(f"Structured logging setup: {log_file}")
    
    def start_training(self, total_epochs: int):
        """Start training session."""
        self.total_epochs = total_epochs
        self.training_start_time = datetime.now()
        
        # Start dashboard
        if self.dashboard:
            self.dashboard.update_training_state(0, total_epochs)
            self.dashboard.start_dashboard()
        
        # Log training start
        self.structured_logger.info(
            f"[TRAINING_START] Starting training session\n"
            f"  - Experiment: {self.experiment_name}\n"
            f"  - ID: {self.experiment_id}\n"
            f"  - Total epochs: {total_epochs}\n"
            f"  - Start time: {self.training_start_time.isoformat()}"
        )
        
        logger.info(f"Started training session: {total_epochs} epochs")
    
    def start_epoch(self, epoch: int, total_batches: int):
        """Start epoch logging."""
        self.current_epoch = epoch
        self.total_batches = total_batches
        self.epoch_start_time = datetime.now()
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_training_state(epoch, self.total_epochs)
        
        # Log epoch start
        self.structured_logger.info(
            f"[EPOCH_START] Epoch {epoch}/{self.total_epochs}\n"
            f"  - Total batches: {total_batches}\n"
            f"  - Start time: {self.epoch_start_time.isoformat()}"
        )
        
        logger.info(f"Started epoch {epoch}/{self.total_epochs} with {total_batches} batches")
    
    def log_batch(self,
                  batch: int,
                  losses: Dict[str, float],
                  learning_rate: float,
                  gradient_norm: float,
                  throughput_metrics: Dict[str, float],
                  memory_metrics: Dict[str, float],
                  data_stats: Optional[Dict[str, Any]] = None):
        """
        Log batch-level metrics in structured format.
        
        This implements the exact logging format specified in the gameplan:
        [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [MODULE] Message
        - Epoch: X/Y, Batch: A/B
        - Files processed: N (M augmented)
        - Losses: {recon: X.XX, kl: X.XX, adv: X.XX}
        - Memory: GPU X.XGB/Y.YGB, RAM: X.XGB
        - Samples saved: path/to/sample.mid
        """
        
        self.current_batch = batch
        timestamp = datetime.now()
        
        # Format losses for display
        loss_display = {}
        for key, value in losses.items():
            if 'recon' in key.lower():
                loss_display['recon'] = value
            elif 'kl' in key.lower():
                loss_display['kl'] = value
            elif 'adv' in key.lower() or 'gan' in key.lower():
                loss_display['adv'] = value
            elif 'total' in key.lower():
                loss_display['total'] = value
        
        # Format memory information
        gpu_allocated = memory_metrics.get('gpu_allocated', 0)
        gpu_max = memory_metrics.get('gpu_max_allocated', 1)
        cpu_memory = memory_metrics.get('cpu_rss', 0)
        
        memory_str = f"GPU {gpu_allocated:.2f}GB/{gpu_max:.2f}GB, RAM: {cpu_memory:.2f}GB"
        
        # Format data statistics
        files_processed = data_stats.get('files_processed', 0) if data_stats else 0
        files_augmented = data_stats.get('files_augmented', 0) if data_stats else 0
        
        # Calculate ETA
        eta_str = self._calculate_eta()
        
        # Format losses string
        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_display.items()])
        
        # Structured log entry in exact gameplan format
        log_message = (
            f"Training progress\n"
            f"  - Epoch: {self.current_epoch}/{self.total_epochs}, Batch: {batch}/{self.total_batches}\n"
            f"  - Files processed: {files_processed} ({files_augmented} augmented)\n"
            f"  - Losses: {{recon: {loss_display.get('recon', 0.0):.2f}, kl: {loss_display.get('kl', 0.0):.2f}, adv: {loss_display.get('adv', 0.0):.2f}}}\n"
            f"  - Memory: {memory_str}\n"
            f"  - ETA: {eta_str}"
        )
        
        self.structured_logger.info(log_message)
        
        # Log additional metrics at different intervals
        if batch % 10 == 0:
            self.structured_logger.debug(
                f"Extended metrics\n"
                f"  - Throughput: {throughput_metrics.get('samples_per_second', 0):.1f} samples/sec, "
                f"{throughput_metrics.get('tokens_per_second', 0):.0f} tokens/sec\n"
                f"  - Learning Rate: {learning_rate:.2e}, Gradient Norm: {gradient_norm:.4f}"
            )
        
        # Update all monitoring systems
        combined_metrics = {
            **losses,
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            **throughput_metrics,
            **memory_metrics
        }
        
        # Update experiment tracker
        self.experiment_tracker.log_batch_metrics(
            epoch=self.current_epoch,
            batch=batch,
            total_batches=self.total_batches,
            losses=losses,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            throughput_metrics=throughput_metrics,
            memory_metrics=memory_metrics
        )
        
        # Update W&B
        if self.wandb_logger:
            global_step = self.current_epoch * self.total_batches + batch
            self.wandb_logger.log_batch_metrics(
                epoch=self.current_epoch,
                batch=batch,
                global_step=global_step,
                losses=losses,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                throughput_metrics=throughput_metrics,
                memory_metrics=memory_metrics
            )
        
        # Update TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_training_metrics(
                epoch=self.current_epoch,
                batch=batch,
                losses=losses,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                throughput_metrics=throughput_metrics,
                memory_metrics=memory_metrics
            )
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_metrics(combined_metrics)
        
        # Console progress logging
        eta_minutes = self._calculate_eta_minutes()
        self.console_logger.log_training_progress(
            epoch=self.current_epoch,
            batch=batch,
            total_batches=self.total_batches,
            losses=losses,
            throughput=throughput_metrics,
            eta_minutes=eta_minutes
        )
        
        # Check for anomalies
        all_metrics = {
            **losses,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            **throughput_metrics,
            **memory_metrics
        }
        
        anomalies = self.anomaly_detector.check_metrics(
            all_metrics,
            epoch=self.current_epoch,
            batch=batch
        )
        
        # Log critical anomalies
        for anomaly in anomalies:
            if anomaly.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.FATAL]:
                self.log_training_anomaly(
                    anomaly.anomaly_type.value,
                    anomaly.details
                )
    
    def log_data_usage(self, data_info: DataUsageInfo):
        """Log detailed data usage information."""
        
        # Log to experiment tracker
        self.experiment_tracker.log_data_usage(data_info)
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_data_usage(data_info)
        
        # Log structured entry
        aug_applied = [k for k, v in data_info.augmentation_applied.items() if v]
        aug_str = ", ".join(aug_applied) if aug_applied else "none"
        
        self.structured_logger.debug(
            f"[DATA_USAGE] Processed file: {data_info.file_name}\n"
            f"  - Original length: {data_info.original_length}, Processed: {data_info.processed_length}\n"
            f"  - Augmentation: {aug_str}\n"
            f"  - Transposition: {data_info.transposition} semitones\n"
            f"  - Time stretch: {data_info.time_stretch:.2f}x\n"
            f"  - Velocity scale: {data_info.velocity_scale:.2f}x\n"
            f"  - Processing time: {data_info.processing_time:.3f}s\n"
            f"  - Cache hit: {'yes' if data_info.cache_hit else 'no'}"
        )
    
    def end_epoch(self, 
                  train_losses: Dict[str, float],
                  val_losses: Dict[str, float],
                  data_stats: Dict[str, Any],
                  is_best_model: bool = False,
                  checkpoint_saved: bool = False):
        """End epoch and log summary."""
        
        if self.epoch_start_time is None:
            epoch_duration = 0
        else:
            epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        # Create epoch summary
        epoch_summary = EpochSummary(
            epoch=self.current_epoch,
            start_time=self.epoch_start_time,
            end_time=datetime.now(),
            duration=epoch_duration,
            losses=train_losses,
            validation_losses=val_losses,
            files_processed=data_stats.get('files_processed', 0),
            files_augmented=data_stats.get('files_augmented', 0),
            total_tokens=data_stats.get('total_tokens', 0),
            average_sequence_length=data_stats.get('average_sequence_length', 0),
            learning_rate=data_stats.get('learning_rate', 0),
            gradient_norm=data_stats.get('gradient_norm', 0),
            samples_per_second=data_stats.get('samples_per_second', 0),
            tokens_per_second=data_stats.get('tokens_per_second', 0),
            memory_usage=data_stats.get('memory_usage', {}),
            model_parameters=data_stats.get('model_parameters', 0),
            checkpoint_saved=checkpoint_saved,
            is_best_model=is_best_model
        )
        
        # Log to experiment tracker
        self.experiment_tracker.log_epoch_summary(epoch_summary)
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_epoch_summary(epoch_summary)
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_epoch_summary(
                epoch=self.current_epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                epoch_duration=epoch_duration,
                data_stats=data_stats
            )
        
        # Structured log entry
        train_loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        val_loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
        
        status_indicators = []
        if is_best_model:
            status_indicators.append("🏆 NEW BEST")
        if checkpoint_saved:
            status_indicators.append("💾 SAVED")
        
        status_str = " ".join(status_indicators) if status_indicators else ""
        
        log_message = (
            f"[EPOCH_COMPLETE] Epoch {self.current_epoch} completed {status_str}\n"
            f"  - Duration: {epoch_duration:.1f}s\n"
            f"  - Train losses: {{{train_loss_str}}}\n"
            f"  - Val losses: {{{val_loss_str}}}\n"
            f"  - Files processed: {data_stats.get('files_processed', 0)} "
            f"({data_stats.get('files_augmented', 0)} augmented)\n"
            f"  - Total tokens: {data_stats.get('total_tokens', 0):,}\n"
            f"  - Avg sequence length: {data_stats.get('average_sequence_length', 0):.1f}\n"
            f"  - Performance: {data_stats.get('samples_per_second', 0):.1f} samples/sec, "
            f"{data_stats.get('tokens_per_second', 0):.0f} tokens/sec"
        )
        
        self.structured_logger.info(log_message)
        
        # Console logging
        self.console_logger.log_epoch_complete(
            epoch=self.current_epoch,
            duration=epoch_duration,
            losses=train_losses
        )
        
        logger.info(f"Completed epoch {self.current_epoch} in {epoch_duration:.1f}s")
    
    def log_sample_generated(self, sample_path: Path, sample_info: Dict[str, Any]):
        """Log generated sample information."""
        
        # Add to samples list
        self.samples_generated.append({
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'path': str(sample_path),
            'info': sample_info,
            'timestamp': datetime.now().isoformat()
        })
        
        # Structured log in exact gameplan format
        self.structured_logger.info(
            f"Sample generation\n"
            f"  - Samples saved: {sample_path}"
        )
        
        # Log additional details separately
        self.structured_logger.debug(
            f"Sample details\n"
            f"  - Epoch: {self.current_epoch}, Batch: {self.current_batch}\n"
            f"  - Length: {sample_info.get('sequence_length', 'unknown')} tokens\n"
            f"  - Generation time: {sample_info.get('generation_time', 'unknown')}s\n"
            f"  - Unique tokens: {sample_info.get('unique_tokens', 'unknown')}\n"
            f"  - Repetition rate: {sample_info.get('repetition_rate', 'unknown'):.2%}"
        )
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            self.tensorboard_logger.log_generated_samples_info(sample_info, self.current_epoch)
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_generated_sample(
                sample_path=sample_path,
                sample_info=sample_info,
                epoch=self.current_epoch
            )
    
    def log_model_architecture(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """Log model architecture information."""
        
        if self.tensorboard_logger:
            self.tensorboard_logger.log_model_architecture(model, input_sample)
        
        # Log hyperparameters to W&B
        if self.wandb_logger:
            config = {
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            self.wandb_logger.log_hyperparameters(config)
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.structured_logger.info(
            f"[MODEL_ARCHITECTURE] Model architecture logged\n"
            f"  - Total parameters: {total_params:,}\n"
            f"  - Trainable parameters: {trainable_params:,}\n"
            f"  - Model size: {total_params * 4 / 1024**2:.1f} MB"
        )
    
    def log_training_anomaly(self, anomaly_type: str, details: str):
        """Log training anomaly detection."""
        
        self.structured_logger.warning(
            f"[ANOMALY_DETECTED] Training anomaly detected\n"
            f"  - Type: {anomaly_type}\n"
            f"  - Details: {details}\n"
            f"  - Epoch: {self.current_epoch}, Batch: {self.current_batch}\n"
            f"  - Timestamp: {datetime.now().isoformat()}"
        )
        
        logger.warning(f"Training anomaly detected: {anomaly_type} - {details}")
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_training_anomaly(anomaly_type, details)
    
    def _handle_anomaly_alert(self, anomaly):
        """Handle anomaly alerts from detector."""
        # Get recovery suggestions
        suggestions = anomaly.recovery_suggestions
        
        if suggestions:
            self.structured_logger.info(
                f"[RECOVERY_SUGGESTIONS] For {anomaly.anomaly_type.value}:\n" +
                "\n".join([f"  - {s}" for s in suggestions])
            )
    
    def end_training(self):
        """End training session and generate final reports."""
        
        training_duration = datetime.now() - self.training_start_time
        
        # Generate final experiment report
        final_report_path = self.experiment_tracker.generate_final_report()
        
        # Save musical quality report
        quality_report_path = self.save_dir / "experiments" / "musical_quality_report.json"
        self.musical_quality_tracker.save_quality_report(quality_report_path)
        
        # Save anomaly report
        anomaly_report_path = self.save_dir / "experiments" / "anomaly_report.json"
        self.anomaly_detector.save_anomaly_report(anomaly_report_path)
        
        # Save dashboard snapshot
        if self.dashboard:
            dashboard_snapshot = self.dashboard.save_dashboard_snapshot()
            self.dashboard.stop_dashboard()
        
        # Generate final TensorBoard visualizations
        if self.tensorboard_logger:
            self.tensorboard_logger.log_loss_landscape(self.current_epoch)
            
            # Log hyperparameters with final metrics
            hparams = self.experiment_tracker.config.to_dict()
            metrics = self.experiment_tracker._calculate_experiment_statistics()
            
            if 'performance_metrics' in metrics and 'final_losses' in metrics['performance_metrics']:
                final_metrics = metrics['performance_metrics']['final_losses']
                self.tensorboard_logger.log_hyperparameters(hparams, final_metrics)
        
        # Final structured log
        self.structured_logger.info(
            f"[TRAINING_COMPLETE] Training session completed\n"
            f"  - Total duration: {training_duration}\n"
            f"  - Epochs completed: {self.current_epoch}\n"
            f"  - Samples generated: {len(self.samples_generated)}\n"
            f"  - Final report: {final_report_path}\n"
            f"  - Experiment ID: {self.experiment_id}"
        )
        
        logger.info(f"Training completed - Duration: {training_duration}")
        logger.info(f"Final report saved: {final_report_path}")
        
        # Close all loggers
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
        
        # Finish W&B run
        if self.wandb_logger:
            self.wandb_logger.finish()
    
    def _calculate_eta(self) -> str:
        """Calculate estimated time to completion."""
        
        if self.current_epoch == 0:
            return "Calculating..."
        
        # Calculate based on epoch progress
        elapsed = datetime.now() - self.training_start_time
        epochs_completed = self.current_epoch
        
        if epochs_completed > 0:
            avg_time_per_epoch = elapsed.total_seconds() / epochs_completed
            remaining_epochs = self.total_epochs - self.current_epoch
            
            # Add current epoch progress
            if self.total_batches > 0:
                epoch_progress = self.current_batch / self.total_batches
                remaining_epoch_time = avg_time_per_epoch * (1 - epoch_progress)
            else:
                remaining_epoch_time = 0
            
            total_remaining = remaining_epoch_time + (remaining_epochs * avg_time_per_epoch)
            
            # Format as human readable
            if total_remaining < 3600:  # Less than 1 hour
                return f"{int(total_remaining // 60)}min {int(total_remaining % 60)}s"
            else:
                hours = int(total_remaining // 3600)
                minutes = int((total_remaining % 3600) // 60)
                return f"{hours}h {minutes}min"
        
        return "Calculating..."
    
    def _calculate_eta_minutes(self) -> Optional[int]:
        """Calculate ETA in minutes for console logger."""
        
        if self.current_epoch == 0:
            return None
        
        elapsed = datetime.now() - self.training_start_time
        epochs_completed = self.current_epoch
        
        if epochs_completed > 0:
            avg_time_per_epoch = elapsed.total_seconds() / epochs_completed
            remaining_epochs = self.total_epochs - self.current_epoch
            
            if self.total_batches > 0:
                epoch_progress = self.current_batch / self.total_batches
                remaining_epoch_time = avg_time_per_epoch * (1 - epoch_progress)
            else:
                remaining_epoch_time = 0
            
            total_remaining = remaining_epoch_time + (remaining_epochs * avg_time_per_epoch)
            return int(total_remaining // 60)
        
        return None
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get current experiment summary."""
        summary = self.experiment_tracker.get_experiment_summary()
        
        # Add musical quality summary
        quality_trend = self.musical_quality_tracker.get_quality_trend()
        summary["musical_quality"] = quality_trend
        
        # Add anomaly summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        summary["anomalies"] = anomaly_summary
        
        return summary
    
    def log_generated_sample_quality(self, sample_tokens: torch.Tensor, epoch: int):
        """Evaluate and log musical quality of generated sample."""
        
        # Evaluate musical quality
        quality_metrics = self.musical_quality_tracker.evaluate_sample(
            sample_tokens,
            epoch=epoch
        )
        
        # Log quality metrics
        self.structured_logger.info(
            f"[MUSICAL_QUALITY] Sample quality assessment\n"
            f"  - Overall quality: {quality_metrics.overall_quality:.3f}\n"
            f"  - Rhythm consistency: {quality_metrics.rhythm_consistency:.3f}\n"
            f"  - Harmonic coherence: {quality_metrics.harmonic_coherence:.3f}\n"
            f"  - Melodic contour: {quality_metrics.melodic_contour:.3f}\n"
            f"  - Repetition score: {quality_metrics.repetition_score:.3f}"
        )
        
        # Log to experiment tracker
        if hasattr(self.experiment_tracker, 'log_musical_quality'):
            self.experiment_tracker.log_musical_quality(quality_metrics.to_dict())
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_musical_quality_metrics(
                epoch=epoch,
                metrics=quality_metrics.to_dict()
            )
        
        # Log to TensorBoard
        if self.tensorboard_logger:
            for key, value in quality_metrics.to_dict().items():
                self.tensorboard_logger.writer.add_scalar(
                    f"musical_quality/{key}",
                    value,
                    epoch
                )
        
        return quality_metrics


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging in gameplan format."""
    
    def format(self, record):
        # Create timestamp with milliseconds
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format level and module
        level = record.levelname
        module = record.name.split('.')[-1] if '.' in record.name else record.name
        
        # Format message
        message = record.getMessage()
        
        # Return in gameplan format
        return f"[{timestamp}] [{level}] [{module}] {message}"


def compare_experiments(experiments_dir: Path, experiment_ids: Optional[List[str]] = None):
    """Compare multiple experiments and generate insights."""
    
    comparator = ExperimentComparator(experiments_dir)
    
    # Generate comparison report
    comparison_report = comparator.compare_experiments(experiment_ids)
    
    # Create visualizations
    viz_dir = experiments_dir / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_paths = comparator.visualize_comparison(comparison_report, viz_dir)
    
    # Save report
    report_path = viz_dir / "comparison_report.json"
    comparator.save_comparison_report(comparison_report, report_path)
    
    # Find best configuration
    best_config = comparator.find_best_configuration()
    
    logger.info(f"Experiment comparison complete - Report: {report_path}")
    logger.info(f"Generated {len(viz_paths)} visualizations")
    
    if best_config:
        logger.info(f"Best configuration: {best_config.experiment_id}")
    
    return comparison_report, viz_paths