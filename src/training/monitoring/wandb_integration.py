"""
Weights & Biases Integration for Aurl.ai Music Generation.

This module provides comprehensive experiment tracking with W&B including:
- Hyperparameter logging
- Metric tracking
- Model checkpointing
- Audio sample logging
- Experiment comparison
- Custom charts and visualizations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import torch
import numpy as np
from dataclasses import asdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.training.core.experiment_tracker import DataUsageInfo, EpochSummary
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class WandBIntegration:
    """
    Weights & Biases integration for comprehensive experiment tracking.
    
    Features:
    - Automatic hyperparameter logging
    - Real-time metric tracking
    - Model artifact versioning
    - Audio sample logging
    - Custom visualizations
    - Experiment comparison
    """
    
    def __init__(self,
                 project_name: str = "aurl-ai-music-generation",
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 mode: str = "online"):
        """
        Initialize W&B integration.
        
        Args:
            project_name: W&B project name
            experiment_name: Name for this run
            config: Configuration dictionary to log
            tags: List of tags for this run
            notes: Notes about this run
            mode: W&B mode (online, offline, disabled)
        """
        
        self.enabled = WANDB_AVAILABLE
        self.run = None
        
        if not self.enabled:
            logger.warning("wandb not installed. Weights & Biases logging disabled.")
            logger.info("Install with: pip install wandb")
            return
        
        try:
            # Initialize W&B run
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=config or {},
                tags=tags or [],
                notes=notes,
                mode=mode,
                reinit=True
            )
            
            # Define custom metrics
            self._define_custom_metrics()
            
            # Log system info
            self._log_system_info()
            
            logger.info(f"W&B initialized - Project: {project_name}, Run: {self.run.name}")
            logger.info(f"View run at: {self.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False
            self.run = None
    
    def _define_custom_metrics(self):
        """Define custom W&B metrics for better visualization."""
        if not self.enabled or not self.run:
            return
        
        # Define custom x-axis for different metrics
        wandb.define_metric("epoch")
        wandb.define_metric("batch")
        wandb.define_metric("global_step")
        
        # Training metrics
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="epoch")
        
        # Loss components
        wandb.define_metric("losses/*", step_metric="global_step")
        
        # Performance metrics
        wandb.define_metric("performance/*", step_metric="global_step")
        
        # Data usage metrics
        wandb.define_metric("data/*", step_metric="epoch")
        
        # Musical quality metrics
        wandb.define_metric("music/*", step_metric="epoch")
    
    def _log_system_info(self):
        """Log system information."""
        if not self.enabled or not self.run:
            return
        
        system_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            system_info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        wandb.config.update({"system": system_info})
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Flatten nested dictionaries
            flat_params = self._flatten_dict(hyperparams)
            wandb.config.update(flat_params)
            
            # Log as summary for experiment comparison
            for key, value in flat_params.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[f"hparam/{key}"] = value
                    
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters to W&B: {e}")
    
    def log_batch_metrics(self,
                         epoch: int,
                         batch: int,
                         global_step: int,
                         losses: Dict[str, float],
                         learning_rate: float,
                         gradient_norm: float,
                         throughput_metrics: Dict[str, float],
                         memory_metrics: Dict[str, float]):
        """Log batch-level metrics to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            metrics = {
                "epoch": epoch,
                "batch": batch,
                "global_step": global_step,
                
                # Losses
                **{f"losses/{k}": v for k, v in losses.items()},
                
                # Training metrics
                "train/learning_rate": learning_rate,
                "train/gradient_norm": gradient_norm,
                
                # Performance metrics
                **{f"performance/{k}": v for k, v in throughput_metrics.items()},
                
                # Memory metrics
                **{f"performance/memory_{k}": v for k, v in memory_metrics.items()},
            }
            
            wandb.log(metrics, step=global_step)
            
        except Exception as e:
            logger.warning(f"Failed to log batch metrics to W&B: {e}")
    
    def log_epoch_summary(self, summary: EpochSummary):
        """Log epoch summary to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Training losses
            train_metrics = {f"train/{k}": v for k, v in summary.losses.items()}
            
            # Validation losses
            val_metrics = {f"val/{k}": v for k, v in summary.validation_losses.items()}
            
            # Data usage statistics
            data_metrics = {
                "data/files_processed": summary.files_processed,
                "data/files_augmented": summary.files_augmented,
                "data/total_tokens": summary.total_tokens,
                "data/average_sequence_length": summary.average_sequence_length,
                "data/augmentation_rate": summary.files_augmented / max(summary.files_processed, 1)
            }
            
            # Performance metrics
            perf_metrics = {
                "performance/epoch_duration": summary.duration,
                "performance/samples_per_second": summary.samples_per_second,
                "performance/tokens_per_second": summary.tokens_per_second,
            }
            
            # Memory usage
            for key, value in summary.memory_usage.items():
                perf_metrics[f"performance/memory_{key}"] = value
            
            # Combined metrics
            all_metrics = {
                **train_metrics,
                **val_metrics,
                **data_metrics,
                **perf_metrics,
                "epoch": summary.epoch
            }
            
            wandb.log(all_metrics, step=summary.epoch)
            
            # Log best model status
            if summary.is_best_model:
                wandb.run.summary["best_epoch"] = summary.epoch
                for key, value in summary.losses.items():
                    wandb.run.summary[f"best_train_{key}"] = value
                for key, value in summary.validation_losses.items():
                    wandb.run.summary[f"best_val_{key}"] = value
            
        except Exception as e:
            logger.warning(f"Failed to log epoch summary to W&B: {e}")
    
    def log_data_usage(self, data_info: DataUsageInfo):
        """Log detailed data usage information."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Create data usage table if not exists
            if not hasattr(self, 'data_table'):
                self.data_table = wandb.Table(columns=[
                    "file_name", "original_length", "processed_length",
                    "transposition", "time_stretch", "velocity_scale",
                    "augmentations", "processing_time", "cache_hit"
                ])
            
            # Add row to table
            aug_str = ", ".join([k for k, v in data_info.augmentation_applied.items() if v])
            self.data_table.add_data(
                data_info.file_name,
                data_info.original_length,
                data_info.processed_length,
                data_info.transposition,
                data_info.time_stretch,
                data_info.velocity_scale,
                aug_str or "none",
                data_info.processing_time,
                data_info.cache_hit
            )
            
        except Exception as e:
            logger.debug(f"Failed to log data usage to W&B: {e}")
    
    def log_generated_sample(self,
                           sample_path: Path,
                           sample_info: Dict[str, Any],
                           epoch: int,
                           audio_data: Optional[np.ndarray] = None,
                           sample_rate: int = 22050):
        """Log generated music sample to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Log sample info
            sample_metrics = {
                f"music/sequence_length": sample_info.get('sequence_length', 0),
                f"music/unique_tokens": sample_info.get('unique_tokens', 0),
                f"music/repetition_rate": sample_info.get('repetition_rate', 0),
                f"music/generation_time": sample_info.get('generation_time', 0),
            }
            
            wandb.log(sample_metrics, step=epoch)
            
            # Log audio if provided
            if audio_data is not None:
                wandb.log({
                    "music/generated_sample": wandb.Audio(
                        audio_data,
                        sample_rate=sample_rate,
                        caption=f"Epoch {epoch} - Generated Sample"
                    )
                }, step=epoch)
            
            # Log sample file as artifact
            artifact = wandb.Artifact(
                name=f"generated_sample_epoch_{epoch}",
                type="music_sample",
                metadata=sample_info
            )
            artifact.add_file(str(sample_path))
            self.run.log_artifact(artifact)
            
        except Exception as e:
            logger.warning(f"Failed to log generated sample to W&B: {e}")
    
    def log_musical_quality_metrics(self,
                                  epoch: int,
                                  metrics: Dict[str, float]):
        """Log musical quality metrics."""
        if not self.enabled or not self.run:
            return
        
        try:
            quality_metrics = {f"music/quality_{k}": v for k, v in metrics.items()}
            quality_metrics["epoch"] = epoch
            
            wandb.log(quality_metrics, step=epoch)
            
            # Create custom chart for musical quality
            if not hasattr(self, 'quality_chart_logged'):
                wandb.log({
                    "charts/musical_quality": wandb.plot.line_series(
                        xs=[[epoch]],
                        ys=[[metrics.get('overall_quality', 0)]],
                        keys=["Overall Quality"],
                        title="Musical Quality Over Time",
                        xname="Epoch"
                    )
                })
                self.quality_chart_logged = True
                
        except Exception as e:
            logger.warning(f"Failed to log musical quality metrics to W&B: {e}")
    
    def log_training_anomaly(self, anomaly_type: str, details: str, severity: str = "warning"):
        """Log training anomaly to W&B."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Log as W&B alert
            wandb.alert(
                title=f"Training Anomaly: {anomaly_type}",
                text=details,
                level=severity.upper()
            )
            
            # Log to metrics
            if not hasattr(self, 'anomaly_count'):
                self.anomaly_count = {}
            
            self.anomaly_count[anomaly_type] = self.anomaly_count.get(anomaly_type, 0) + 1
            
            wandb.log({
                f"anomalies/{anomaly_type}_count": self.anomaly_count[anomaly_type],
                "anomalies/total_count": sum(self.anomaly_count.values())
            })
            
        except Exception as e:
            logger.warning(f"Failed to log anomaly to W&B: {e}")
    
    def log_model_checkpoint(self,
                           checkpoint_path: Path,
                           epoch: int,
                           metrics: Dict[str, float],
                           is_best: bool = False):
        """Log model checkpoint as W&B artifact."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Create artifact
            artifact_name = "model_best" if is_best else f"model_epoch_{epoch}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata={
                    "epoch": epoch,
                    "metrics": metrics,
                    "is_best": is_best,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add checkpoint file
            artifact.add_file(str(checkpoint_path))
            
            # Log artifact
            self.run.log_artifact(artifact, aliases=["latest", "best"] if is_best else ["latest"])
            
            logger.info(f"Logged checkpoint to W&B: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log checkpoint to W&B: {e}")
    
    def create_experiment_comparison(self):
        """Create experiment comparison visualizations."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Log final summary metrics for comparison
            final_summary = {
                "summary/final_train_loss": wandb.run.summary.get("train/total", 0),
                "summary/final_val_loss": wandb.run.summary.get("val/total", 0),
                "summary/best_train_loss": wandb.run.summary.get("best_train_total", 0),
                "summary/best_val_loss": wandb.run.summary.get("best_val_total", 0),
                "summary/total_epochs": wandb.run.summary.get("epoch", 0),
                "summary/total_samples_processed": wandb.run.summary.get("data/files_processed", 0),
            }
            
            wandb.run.summary.update(final_summary)
            
        except Exception as e:
            logger.warning(f"Failed to create experiment comparison: {e}")
    
    def finish(self, exit_code: int = 0):
        """Finish W&B run and upload final artifacts."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Log data usage table
            if hasattr(self, 'data_table'):
                wandb.log({"data_usage_details": self.data_table})
            
            # Create final comparison
            self.create_experiment_comparison()
            
            # Finish run
            wandb.finish(exit_code=exit_code)
            
            logger.info("W&B run finished successfully")
            
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dictionary for W&B config."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)