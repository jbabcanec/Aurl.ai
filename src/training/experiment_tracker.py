"""
Comprehensive Experiment Tracking System for Aurl.ai Music Generation.

This module provides detailed tracking of:
- Training progress and metrics
- Data usage and augmentation details
- Model configuration and architecture
- Real-time visualizations and dashboards
- Experiment comparison and analysis

Designed for production-grade music AI training with full transparency.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import torch
from collections import defaultdict, deque

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DataUsageInfo:
    """Track detailed data usage during training."""
    file_name: str
    original_length: int
    processed_length: int
    augmentation_applied: Dict[str, Any]
    transposition: int
    time_stretch: float
    velocity_scale: float
    instruments_used: List[str]
    processing_time: float
    cache_hit: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpochSummary:
    """Comprehensive epoch tracking."""
    epoch: int
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    
    # Loss tracking
    losses: Dict[str, float]
    validation_losses: Dict[str, float]
    
    # Data statistics
    files_processed: int
    files_augmented: int
    total_tokens: int
    average_sequence_length: float
    
    # Training metrics
    learning_rate: float
    gradient_norm: float
    samples_per_second: float
    tokens_per_second: float
    memory_usage: Dict[str, float]
    
    # Model state
    model_parameters: int
    checkpoint_saved: bool
    is_best_model: bool
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


@dataclass
class ExperimentConfig:
    """Complete experiment configuration tracking."""
    experiment_id: str
    experiment_name: str
    start_time: datetime
    
    # Model configuration
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    
    # Environment info
    device: str
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str]
    
    # Git info (if available)
    git_commit: Optional[str]
    git_branch: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        return result


class MetricsVisualizer:
    """Real-time metrics visualization dashboard."""
    
    def __init__(self, save_dir: Path, experiment_id: str):
        self.save_dir = save_dir
        self.experiment_id = experiment_id
        self.plots_dir = save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Initialize plot data storage
        self.metrics_history = defaultdict(list)
        self.timestamps = []
        
        logger.info(f"Initialized MetricsVisualizer for experiment {experiment_id}")
    
    def update_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Update metrics and refresh visualizations."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.timestamps.append(timestamp)
        
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Keep only last 1000 points for real-time plotting
        if len(self.timestamps) > 1000:
            self.timestamps = self.timestamps[-1000:]
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-1000:]
    
    def create_loss_dashboard(self) -> Path:
        """Create comprehensive loss tracking dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Metrics Dashboard - {self.experiment_id}', fontsize=16, fontweight='bold')
        
        # Training loss plot
        ax1 = axes[0, 0]
        if 'total_loss' in self.metrics_history:
            ax1.plot(self.timestamps, self.metrics_history['total_loss'], 
                    label='Total Loss', linewidth=2, color='red')
        if 'reconstruction_loss' in self.metrics_history:
            ax1.plot(self.timestamps, self.metrics_history['reconstruction_loss'], 
                    label='Reconstruction', linewidth=2, color='blue')
        if 'kl_loss' in self.metrics_history:
            ax1.plot(self.timestamps, self.metrics_history['kl_loss'], 
                    label='KL Divergence', linewidth=2, color='green')
        
        ax1.set_title('Training Losses', fontweight='bold')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        ax2 = axes[0, 1]
        if 'samples_per_second' in self.metrics_history:
            ax2.plot(self.timestamps, self.metrics_history['samples_per_second'], 
                    label='Samples/sec', linewidth=2, color='purple')
        if 'tokens_per_second' in self.metrics_history:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(self.timestamps, self.metrics_history['tokens_per_second'], 
                         label='Tokens/sec', linewidth=2, color='orange')
            ax2_twin.set_ylabel('Tokens/sec', color='orange')
        
        ax2.set_title('Training Performance', fontweight='bold')
        ax2.set_ylabel('Samples/sec', color='purple')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Memory usage
        ax3 = axes[1, 0]
        if 'gpu_memory_usage' in self.metrics_history:
            ax3.plot(self.timestamps, self.metrics_history['gpu_memory_usage'], 
                    label='GPU Memory', linewidth=2, color='red')
        if 'cpu_memory_usage' in self.metrics_history:
            ax3.plot(self.timestamps, self.metrics_history['cpu_memory_usage'], 
                    label='CPU Memory', linewidth=2, color='blue')
        
        ax3.set_title('Memory Usage (GB)', fontweight='bold')
        ax3.set_ylabel('Memory (GB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate and gradient norm
        ax4 = axes[1, 1]
        if 'learning_rate' in self.metrics_history:
            ax4.plot(self.timestamps, self.metrics_history['learning_rate'], 
                    label='Learning Rate', linewidth=2, color='green')
        if 'gradient_norm' in self.metrics_history:
            ax4_twin = ax4.twinx()
            ax4_twin.plot(self.timestamps, self.metrics_history['gradient_norm'], 
                         label='Gradient Norm', linewidth=2, color='red')
            ax4_twin.set_ylabel('Gradient Norm', color='red')
        
        ax4.set_title('Training Dynamics', fontweight='bold')
        ax4.set_ylabel('Learning Rate', color='green')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.plots_dir / f"training_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training dashboard: {dashboard_path}")
        return dashboard_path
    
    def create_data_usage_visualization(self, data_usage: List[DataUsageInfo]) -> Path:
        """Create detailed data usage visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Data Usage Analysis - {self.experiment_id}', fontsize=16, fontweight='bold')
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([info.to_dict() for info in data_usage])
        
        # Augmentation frequency
        ax1 = axes[0, 0]
        augmentation_counts = df['augmentation_applied'].apply(lambda x: len([k for k, v in x.items() if v])).value_counts()
        augmentation_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Augmentation Frequency', fontweight='bold')
        ax1.set_xlabel('Number of Augmentations Applied')
        ax1.set_ylabel('Number of Files')
        
        # Transposition distribution
        ax2 = axes[0, 1]
        df['transposition'].hist(bins=25, ax=ax2, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Transposition Distribution', fontweight='bold')
        ax2.set_xlabel('Semitones')
        ax2.set_ylabel('Frequency')
        ax2.axvline(0, color='red', linestyle='--', label='No Transposition')
        ax2.legend()
        
        # Time stretch distribution
        ax3 = axes[0, 2]
        df['time_stretch'].hist(bins=20, ax=ax3, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_title('Time Stretch Distribution', fontweight='bold')
        ax3.set_xlabel('Stretch Factor')
        ax3.set_ylabel('Frequency')
        ax3.axvline(1.0, color='red', linestyle='--', label='No Stretch')
        ax3.legend()
        
        # Processing time analysis
        ax4 = axes[1, 0]
        df['processing_time'].hist(bins=30, ax=ax4, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_title('Processing Time Distribution', fontweight='bold')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Frequency')
        
        # Cache hit rate
        ax5 = axes[1, 1]
        cache_stats = df['cache_hit'].value_counts()
        colors = ['lightcoral', 'lightblue']
        ax5.pie(cache_stats.values, labels=['Cache Miss', 'Cache Hit'], colors=colors, autopct='%1.1f%%')
        ax5.set_title('Cache Hit Rate', fontweight='bold')
        
        # Sequence length analysis
        ax6 = axes[1, 2]
        ax6.scatter(df['original_length'], df['processed_length'], alpha=0.6, c='red', label='Original vs Processed')
        ax6.plot([0, df['original_length'].max()], [0, df['original_length'].max()], 'k--', alpha=0.5, label='No Change')
        ax6.set_title('Sequence Length Changes', fontweight='bold')
        ax6.set_xlabel('Original Length')
        ax6.set_ylabel('Processed Length')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.plots_dir / f"data_usage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved data usage visualization: {viz_path}")
        return viz_path


class ComprehensiveExperimentTracker:
    """
    Main experiment tracking system for Aurl.ai training.
    
    Provides comprehensive tracking of:
    - Training metrics and progress
    - Data usage and augmentation details
    - Model architecture and configuration
    - Real-time visualizations
    - Experiment comparison
    """
    
    def __init__(self, 
                 experiment_name: str,
                 save_dir: Path,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 data_config: Dict[str, Any]):
        
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        self.save_dir = Path(save_dir)
        self.experiment_dir = self.save_dir / "experiments" / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.visualizer = MetricsVisualizer(self.experiment_dir, self.experiment_id)
        
        # Data tracking
        self.data_usage: List[DataUsageInfo] = []
        self.epoch_summaries: List[EpochSummary] = []
        self.metrics_buffer = deque(maxlen=1000)  # Real-time metrics buffer
        
        # Create experiment configuration
        self.config = ExperimentConfig(
            experiment_id=self.experiment_id,
            experiment_name=experiment_name,
            start_time=datetime.now(),
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            device=str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch()
        )
        
        # Save initial configuration
        self._save_config()
        
        # Initialize real-time logging
        self.last_dashboard_update = time.time()
        self.dashboard_update_interval = 30  # Update dashboard every 30 seconds
        
        logger.info(f"Initialized experiment tracker: {self.experiment_id}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.save_dir)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            import subprocess
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=self.save_dir)
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.experiment_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        
        logger.debug(f"Saved experiment configuration: {config_path}")
    
    def log_data_usage(self, data_info: DataUsageInfo):
        """Log detailed data usage information."""
        self.data_usage.append(data_info)
        
        # Save incremental data usage log
        data_log_path = self.experiment_dir / "data_usage.jsonl"
        with open(data_log_path, 'a') as f:
            f.write(json.dumps(data_info.to_dict()) + '\n')
    
    def log_batch_metrics(self, 
                         epoch: int, 
                         batch: int, 
                         total_batches: int,
                         losses: Dict[str, float],
                         learning_rate: float,
                         gradient_norm: float,
                         throughput_metrics: Dict[str, float],
                         memory_metrics: Dict[str, float]):
        """Log detailed batch-level metrics."""
        
        timestamp = datetime.now()
        
        batch_info = {
            'timestamp': timestamp.isoformat(),
            'experiment_id': self.experiment_id,
            'epoch': epoch,
            'batch': batch,
            'total_batches': total_batches,
            'progress': batch / total_batches,
            'losses': losses,
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            'throughput': throughput_metrics,
            'memory': memory_metrics
        }
        
        # Add to metrics buffer for real-time tracking
        self.metrics_buffer.append(batch_info)
        
        # Update visualizer
        combined_metrics = {
            **losses,
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            **{f'throughput_{k}': v for k, v in throughput_metrics.items()},
            **{f'memory_{k}': v for k, v in memory_metrics.items()}
        }
        self.visualizer.update_metrics(combined_metrics, timestamp)
        
        # Save incremental batch log
        batch_log_path = self.experiment_dir / "batch_metrics.jsonl"
        with open(batch_log_path, 'a') as f:
            f.write(json.dumps(batch_info, default=str) + '\n')
        
        # Update dashboard periodically
        if time.time() - self.last_dashboard_update > self.dashboard_update_interval:
            self._update_dashboard()
            self.last_dashboard_update = time.time()
        
        # Console logging with structured format
        self._log_batch_console(epoch, batch, total_batches, losses, 
                               learning_rate, throughput_metrics, memory_metrics)
    
    def _log_batch_console(self, epoch: int, batch: int, total_batches: int,
                          losses: Dict[str, float], learning_rate: float,
                          throughput: Dict[str, float], memory: Dict[str, float]):
        """Structured console logging."""
        
        # Progress calculation
        progress = batch / total_batches * 100
        eta = self._calculate_eta(epoch, batch, total_batches)
        
        # Format losses
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items() if "total" in k.lower() or "recon" in k.lower()])
        
        # Format memory
        gpu_mem = memory.get('gpu_allocated', 0)
        gpu_max = memory.get('gpu_max_allocated', 1)
        memory_str = f"GPU: {gpu_mem:.2f}GB/{gpu_max:.2f}GB"
        
        # Format throughput
        samples_sec = throughput.get('samples_per_second', 0)
        tokens_sec = throughput.get('tokens_per_second', 0)
        throughput_str = f"{samples_sec:.1f} samples/sec, {tokens_sec:.0f} tokens/sec"
        
        logger.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TRAINING] [BATCH]\n"
            f"  - Epoch: {epoch}, Batch: {batch}/{total_batches} ({progress:.1f}%)\n"
            f"  - Files processed: {len(self.data_usage)} ({sum(1 for d in self.data_usage if any(d.augmentation_applied.values()))} augmented)\n"
            f"  - Losses: {{{loss_str}}}\n"
            f"  - Memory: {memory_str}\n"
            f"  - Throughput: {throughput_str}\n"
            f"  - LR: {learning_rate:.2e}, ETA: {eta}"
        )
    
    def _calculate_eta(self, current_epoch: int, current_batch: int, total_batches: int) -> str:
        """Calculate estimated time to completion."""
        if not self.epoch_summaries:
            return "Calculating..."
        
        # Estimate based on recent epoch times
        recent_epochs = self.epoch_summaries[-3:]  # Last 3 epochs
        if not recent_epochs or not all(e.duration for e in recent_epochs):
            return "Calculating..."
        
        avg_epoch_time = np.mean([e.duration for e in recent_epochs if e.duration])
        
        # Current epoch progress
        epoch_progress = current_batch / total_batches
        remaining_epoch_time = avg_epoch_time * (1 - epoch_progress)
        
        # Remaining epochs (assuming we know total epochs from config)
        total_epochs = self.config.training_config.get('num_epochs', 100)
        remaining_epochs = total_epochs - current_epoch - 1
        
        total_remaining = remaining_epoch_time + (remaining_epochs * avg_epoch_time)
        
        # Format as hours:minutes
        hours = int(total_remaining // 3600)
        minutes = int((total_remaining % 3600) // 60)
        
        return f"{hours:02d}:{minutes:02d}"
    
    def log_epoch_summary(self, epoch_summary: EpochSummary):
        """Log comprehensive epoch summary."""
        self.epoch_summaries.append(epoch_summary)
        
        # Save epoch summary
        epoch_log_path = self.experiment_dir / "epoch_summaries.jsonl"
        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_summary.to_dict()) + '\n')
        
        # Console logging for epoch completion
        duration_str = f"{epoch_summary.duration:.1f}s" if epoch_summary.duration else "Unknown"
        
        logger.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [TRAINING] [EPOCH_COMPLETE]\n"
            f"  - Epoch: {epoch_summary.epoch} completed in {duration_str}\n"
            f"  - Files processed: {epoch_summary.files_processed} ({epoch_summary.files_augmented} augmented)\n"
            f"  - Total tokens: {epoch_summary.total_tokens:,}, Avg length: {epoch_summary.average_sequence_length:.1f}\n"
            f"  - Performance: {epoch_summary.samples_per_second:.1f} samples/sec, {epoch_summary.tokens_per_second:.0f} tokens/sec\n"
            f"  - Best model: {'YES' if epoch_summary.is_best_model else 'NO'}, Checkpoint saved: {'YES' if epoch_summary.checkpoint_saved else 'NO'}"
        )
    
    def _update_dashboard(self):
        """Update real-time dashboard."""
        try:
            dashboard_path = self.visualizer.create_loss_dashboard()
            
            # Create data usage visualization if we have enough data
            if len(self.data_usage) > 10:
                data_viz_path = self.visualizer.create_data_usage_visualization(self.data_usage)
        except Exception as e:
            logger.warning(f"Failed to update dashboard: {e}")
    
    def generate_final_report(self) -> Path:
        """Generate comprehensive final experiment report."""
        
        # Create data usage visualization
        if self.data_usage:
            data_viz_path = self.visualizer.create_data_usage_visualization(self.data_usage)
        
        # Create final dashboard
        final_dashboard = self.visualizer.create_loss_dashboard()
        
        # Generate comprehensive statistics
        stats = self._calculate_experiment_statistics()
        
        # Save comprehensive report
        report_path = self.experiment_dir / "final_experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Create markdown summary
        markdown_report = self._create_markdown_report(stats)
        markdown_path = self.experiment_dir / "EXPERIMENT_SUMMARY.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Generated final experiment report: {report_path}")
        logger.info(f"Generated markdown summary: {markdown_path}")
        
        return report_path
    
    def _calculate_experiment_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive experiment statistics."""
        
        # Basic experiment info
        stats = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.config.experiment_name,
            'total_duration': str(datetime.now() - self.config.start_time),
            'configuration': self.config.to_dict()
        }
        
        # Data usage statistics
        if self.data_usage:
            stats['data_statistics'] = {
                'total_files_processed': len(self.data_usage),
                'files_augmented': sum(1 for d in self.data_usage if any(d.augmentation_applied.values())),
                'augmentation_rate': sum(1 for d in self.data_usage if any(d.augmentation_applied.values())) / len(self.data_usage),
                'average_processing_time': np.mean([d.processing_time for d in self.data_usage]),
                'cache_hit_rate': np.mean([d.cache_hit for d in self.data_usage]),
                'transposition_range': {
                    'min': min(d.transposition for d in self.data_usage),
                    'max': max(d.transposition for d in self.data_usage),
                    'mean': np.mean([d.transposition for d in self.data_usage])
                },
                'time_stretch_range': {
                    'min': min(d.time_stretch for d in self.data_usage),
                    'max': max(d.time_stretch for d in self.data_usage),
                    'mean': np.mean([d.time_stretch for d in self.data_usage])
                }
            }
        
        # Training statistics
        if self.epoch_summaries:
            completed_epochs = [e for e in self.epoch_summaries if e.end_time]
            if completed_epochs:
                stats['training_statistics'] = {
                    'epochs_completed': len(completed_epochs),
                    'average_epoch_duration': np.mean([e.duration for e in completed_epochs if e.duration]),
                    'total_training_time': sum(e.duration for e in completed_epochs if e.duration),
                    'average_samples_per_second': np.mean([e.samples_per_second for e in completed_epochs]),
                    'average_tokens_per_second': np.mean([e.tokens_per_second for e in completed_epochs]),
                    'total_tokens_processed': sum(e.total_tokens for e in completed_epochs),
                    'best_epochs': [e.epoch for e in completed_epochs if e.is_best_model]
                }
        
        # Performance metrics
        if self.metrics_buffer:
            recent_metrics = list(self.metrics_buffer)[-100:]  # Last 100 batches
            if recent_metrics:
                stats['performance_metrics'] = {
                    'final_losses': recent_metrics[-1]['losses'],
                    'loss_trends': self._calculate_loss_trends(recent_metrics),
                    'throughput_stability': np.std([m['throughput'].get('samples_per_second', 0) for m in recent_metrics]),
                    'memory_efficiency': np.mean([m['memory'].get('gpu_allocated', 0) for m in recent_metrics])
                }
        
        return stats
    
    def _calculate_loss_trends(self, recent_metrics: List[Dict]) -> Dict[str, str]:
        """Calculate loss trends for the final period."""
        trends = {}
        
        for loss_name in ['total_loss', 'reconstruction_loss', 'kl_loss']:
            losses = [m['losses'].get(loss_name, 0) for m in recent_metrics if loss_name in m['losses']]
            if len(losses) >= 10:
                # Simple trend calculation
                early_avg = np.mean(losses[:len(losses)//2])
                late_avg = np.mean(losses[len(losses)//2:])
                
                if late_avg < early_avg * 0.95:
                    trends[loss_name] = "decreasing"
                elif late_avg > early_avg * 1.05:
                    trends[loss_name] = "increasing"
                else:
                    trends[loss_name] = "stable"
        
        return trends
    
    def _create_markdown_report(self, stats: Dict[str, Any]) -> str:
        """Create markdown experiment summary."""
        
        report = f"""# Experiment Report: {self.config.experiment_name}

**Experiment ID**: `{self.experiment_id}`  
**Date**: {self.config.start_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Duration**: {stats.get('total_duration', 'Unknown')}

## Configuration

### Model Configuration
```json
{json.dumps(self.config.model_config, indent=2)}
```

### Training Configuration  
```json
{json.dumps(self.config.training_config, indent=2)}
```

### Data Configuration
```json
{json.dumps(self.config.data_config, indent=2)}
```

## Data Usage Statistics
"""
        
        if 'data_statistics' in stats:
            data_stats = stats['data_statistics']
            report += f"""
- **Total Files Processed**: {data_stats['total_files_processed']:,}
- **Files Augmented**: {data_stats['files_augmented']:,} ({data_stats['augmentation_rate']:.1%})
- **Cache Hit Rate**: {data_stats['cache_hit_rate']:.1%}
- **Average Processing Time**: {data_stats['average_processing_time']:.3f}s
- **Transposition Range**: {data_stats['transposition_range']['min']} to {data_stats['transposition_range']['max']} semitones (avg: {data_stats['transposition_range']['mean']:.1f})
- **Time Stretch Range**: {data_stats['time_stretch_range']['min']:.2f} to {data_stats['time_stretch_range']['max']:.2f} (avg: {data_stats['time_stretch_range']['mean']:.2f})
"""
        
        if 'training_statistics' in stats:
            train_stats = stats['training_statistics']
            report += f"""
## Training Statistics

- **Epochs Completed**: {train_stats['epochs_completed']}
- **Average Epoch Duration**: {train_stats['average_epoch_duration']:.1f} seconds
- **Total Training Time**: {train_stats['total_training_time']:.1f} seconds
- **Average Performance**: {train_stats['average_samples_per_second']:.1f} samples/sec, {train_stats['average_tokens_per_second']:.0f} tokens/sec
- **Total Tokens Processed**: {train_stats['total_tokens_processed']:,}
- **Best Model Epochs**: {', '.join(map(str, train_stats['best_epochs']))}
"""
        
        if 'performance_metrics' in stats:
            perf_stats = stats['performance_metrics']
            report += f"""
## Final Performance Metrics

### Final Losses
```json
{json.dumps(perf_stats['final_losses'], indent=2)}
```

### Loss Trends
{chr(10).join([f"- **{loss}**: {trend}" for loss, trend in perf_stats['loss_trends'].items()])}

### System Performance
- **Throughput Stability**: {perf_stats['throughput_stability']:.2f} (std dev)
- **Average Memory Usage**: {perf_stats['memory_efficiency']:.2f} GB
"""
        
        report += f"""
## Environment Information

- **Device**: {self.config.device}
- **Python Version**: {self.config.python_version}  
- **PyTorch Version**: {self.config.pytorch_version}
- **CUDA Version**: {self.config.cuda_version or 'N/A'}
- **Git Commit**: {self.config.git_commit or 'N/A'}
- **Git Branch**: {self.config.git_branch or 'N/A'}

---
*Report generated automatically by Aurl.ai Experiment Tracker*
"""
        
        return report
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get current experiment summary."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.config.experiment_name,
            'start_time': self.config.start_time.isoformat(),
            'duration': str(datetime.now() - self.config.start_time),
            'data_files_processed': len(self.data_usage),
            'epochs_completed': len(self.epoch_summaries),
            'current_status': 'running',
            'experiment_directory': str(self.experiment_dir)
        }