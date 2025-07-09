"""
TensorBoard Integration for Aurl.ai Music Generation Training.

This module provides comprehensive TensorBoard logging with:
- Real-time training metrics visualization
- Loss landscape tracking
- Model architecture visualization
- Audio sample logging
- Data usage statistics
- Training progress monitoring

Designed for production-grade music AI training with detailed insights.
"""

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from io import BytesIO
import base64
import json

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TensorBoardLogger:
    """
    Comprehensive TensorBoard logging for music generation training.
    
    Features:
    - Real-time training metrics
    - Loss landscape visualization
    - Model architecture graphs
    - Audio sample tracking
    - Data usage analytics
    - Training progress monitoring
    """
    
    def __init__(self, log_dir: Path, experiment_id: str, comment: str = ""):
        self.log_dir = Path(log_dir) / "tensorboard" / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment
        )
        
        self.experiment_id = experiment_id
        self.step_counter = 0
        
        # Tracking for custom visualizations
        self.loss_history = {
            'train': {},
            'val': {}
        }
        
        logger.info(f"Initialized TensorBoard logger: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value to TensorBoard."""
        if step is None:
            step = self.step_counter
        
        self.writer.add_scalar(tag, value, step)
        
        # Store for custom visualizations
        if 'loss' in tag.lower():
            split = 'train' if 'train' in tag.lower() else 'val'
            if tag not in self.loss_history[split]:
                self.loss_history[split][tag] = []
            self.loss_history[split][tag].append((step, value))
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """Log multiple related scalars."""
        if step is None:
            step = self.step_counter
        
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Log individually for custom tracking
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(f"{main_tag}/{tag}", value, step)
    
    def log_training_metrics(self, 
                           epoch: int,
                           batch: int,
                           losses: Dict[str, float],
                           learning_rate: float,
                           gradient_norm: float,
                           throughput_metrics: Dict[str, float],
                           memory_metrics: Dict[str, float]):
        """Log comprehensive training metrics."""
        
        global_step = epoch * 1000 + batch  # Approximate global step
        
        # Loss metrics
        self.log_scalars("Training/Losses", losses, global_step)
        
        # Optimization metrics
        self.log_scalar("Training/LearningRate", learning_rate, global_step)
        self.log_scalar("Training/GradientNorm", gradient_norm, global_step)
        
        # Performance metrics
        self.log_scalars("Performance/Throughput", throughput_metrics, global_step)
        self.log_scalars("Performance/Memory", memory_metrics, global_step)
        
        # Training progress
        self.log_scalar("Progress/Epoch", epoch, global_step)
        self.log_scalar("Progress/Batch", batch, global_step)
    
    def log_epoch_summary(self,
                         epoch: int,
                         train_losses: Dict[str, float],
                         val_losses: Dict[str, float],
                         epoch_duration: float,
                         data_stats: Dict[str, Any]):
        """Log epoch-level summary metrics."""
        
        # Training losses
        for loss_name, value in train_losses.items():
            self.log_scalar(f"Epoch/Train_{loss_name}", value, epoch)
        
        # Validation losses
        for loss_name, value in val_losses.items():
            self.log_scalar(f"Epoch/Val_{loss_name}", value, epoch)
        
        # Performance metrics
        self.log_scalar("Epoch/Duration", epoch_duration, epoch)
        
        # Data statistics
        if 'files_processed' in data_stats:
            self.log_scalar("Data/FilesProcessed", data_stats['files_processed'], epoch)
        if 'files_augmented' in data_stats:
            self.log_scalar("Data/FilesAugmented", data_stats['files_augmented'], epoch)
            self.log_scalar("Data/AugmentationRate", 
                          data_stats['files_augmented'] / max(data_stats['files_processed'], 1), epoch)
        
        if 'average_sequence_length' in data_stats:
            self.log_scalar("Data/AvgSequenceLength", data_stats['average_sequence_length'], epoch)
        
        if 'total_tokens' in data_stats:
            self.log_scalar("Data/TotalTokens", data_stats['total_tokens'], epoch)
    
    def log_model_architecture(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """Log model architecture graph."""
        try:
            self.writer.add_graph(model, input_sample)
            logger.info("Logged model architecture to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")
    
    def log_model_parameters(self, model: torch.nn.Module, epoch: int):
        """Log model parameter statistics."""
        
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            # Parameter count
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
                
                # Parameter statistics
                self.log_scalar(f"Parameters/{name}/Mean", param.data.mean().item(), epoch)
                self.log_scalar(f"Parameters/{name}/Std", param.data.std().item(), epoch)
                self.log_scalar(f"Parameters/{name}/AbsMean", param.data.abs().mean().item(), epoch)
                
                # Gradient statistics
                if param.grad is not None:
                    self.log_scalar(f"Gradients/{name}/Mean", param.grad.mean().item(), epoch)
                    self.log_scalar(f"Gradients/{name}/Std", param.grad.std().item(), epoch)
                    self.log_scalar(f"Gradients/{name}/Norm", param.grad.norm().item(), epoch)
        
        # Overall model statistics
        self.log_scalar("Model/TotalParameters", total_params, epoch)
        self.log_scalar("Model/TrainableParameters", trainable_params, epoch)
        self.log_scalar("Model/NonTrainableParameters", total_params - trainable_params, epoch)
    
    def log_data_usage_visualization(self, data_usage_stats: Dict[str, Any], epoch: int):
        """Create and log data usage visualizations."""
        
        # Create data usage summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Data Usage Statistics - Epoch {epoch}', fontsize=14, fontweight='bold')
        
        # Augmentation frequency
        if 'augmentation_counts' in data_usage_stats:
            ax1 = axes[0, 0]
            aug_data = data_usage_stats['augmentation_counts']
            ax1.bar(range(len(aug_data)), aug_data.values(), color='skyblue')
            ax1.set_title('Augmentation Types Used')
            ax1.set_xlabel('Augmentation Type')
            ax1.set_ylabel('Frequency')
            ax1.set_xticks(range(len(aug_data)))
            ax1.set_xticklabels(aug_data.keys(), rotation=45)
        
        # Transposition distribution
        if 'transposition_distribution' in data_usage_stats:
            ax2 = axes[0, 1]
            trans_data = data_usage_stats['transposition_distribution']
            ax2.hist(trans_data, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            ax2.set_title('Transposition Distribution')
            ax2.set_xlabel('Semitones')
            ax2.set_ylabel('Frequency')
            ax2.axvline(0, color='red', linestyle='--', label='No Transposition')
            ax2.legend()
        
        # Processing time analysis
        if 'processing_times' in data_usage_stats:
            ax3 = axes[1, 0]
            proc_data = data_usage_stats['processing_times']
            ax3.hist(proc_data, bins=30, color='purple', alpha=0.7, edgecolor='black')
            ax3.set_title('Processing Time Distribution')
            ax3.set_xlabel('Processing Time (seconds)')
            ax3.set_ylabel('Frequency')
        
        # Cache hit rate pie chart
        if 'cache_stats' in data_usage_stats:
            ax4 = axes[1, 1]
            cache_data = data_usage_stats['cache_stats']
            colors = ['lightcoral', 'lightblue']
            ax4.pie(cache_data.values(), labels=cache_data.keys(), colors=colors, autopct='%1.1f%%')
            ax4.set_title('Cache Hit Rate')
        
        plt.tight_layout()
        
        # Convert to image and log
        self.log_matplotlib_figure("DataUsage/Statistics", fig, epoch)
        plt.close(fig)
    
    def log_loss_landscape(self, epoch: int):
        """Create and log loss landscape visualization."""
        
        if not any(self.loss_history['train'].values()):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Loss Landscape - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Training loss evolution
        ax1 = axes[0, 0]
        for loss_name, history in self.loss_history['train'].items():
            if history:
                steps, values = zip(*history)
                ax1.plot(steps, values, label=loss_name, linewidth=2)
        
        ax1.set_title('Training Loss Evolution')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss comparison
        ax2 = axes[0, 1]
        if self.loss_history['val']:
            for loss_name, history in self.loss_history['val'].items():
                if history:
                    steps, values = zip(*history)
                    ax2.plot(steps, values, label=f"Val {loss_name}", linewidth=2)
        
        ax2.set_title('Validation Loss Evolution')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Loss correlation heatmap
        ax3 = axes[1, 0]
        if len(self.loss_history['train']) > 1:
            # Create correlation matrix of recent losses
            recent_data = {}
            min_length = min(len(hist) for hist in self.loss_history['train'].values() if hist)
            if min_length > 10:
                for loss_name, history in self.loss_history['train'].items():
                    if history and len(history) >= min_length:
                        recent_data[loss_name] = [val for _, val in history[-min_length:]]
                
                if len(recent_data) > 1:
                    import pandas as pd
                    df = pd.DataFrame(recent_data)
                    corr_matrix = df.corr()
                    
                    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax3)
                    ax3.set_title('Loss Correlation Matrix')
        
        # Loss smoothness analysis
        ax4 = axes[1, 1]
        for loss_name, history in self.loss_history['train'].items():
            if history and len(history) > 10:
                values = [val for _, val in history]
                # Calculate rolling standard deviation as smoothness metric
                window_size = min(10, len(values) // 4)
                smoothness = []
                for i in range(window_size, len(values)):
                    window_std = np.std(values[i-window_size:i])
                    smoothness.append(window_std)
                
                if smoothness:
                    ax4.plot(range(window_size, len(values)), smoothness, 
                            label=f"{loss_name} smoothness", linewidth=2)
        
        ax4.set_title('Loss Smoothness (Rolling Std)')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Standard Deviation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.log_matplotlib_figure("Training/LossLandscape", fig, epoch)
        plt.close(fig)
    
    def log_matplotlib_figure(self, tag: str, figure: plt.Figure, step: int):
        """Convert matplotlib figure to image and log to TensorBoard."""
        
        # Convert figure to numpy array
        figure.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = figure.canvas.get_width_height()
        buf = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        
        # Roll the ALPHA channel to be last
        buf = np.roll(buf, 3, axis=2)
        
        # Convert to RGB (remove alpha channel)
        image = buf[:, :, :3]
        
        # TensorBoard expects CHW format
        image = np.transpose(image, (2, 0, 1))
        
        self.writer.add_image(tag, image, step)
    
    def log_attention_visualization(self, attention_weights: torch.Tensor, 
                                  layer_name: str, epoch: int):
        """Log attention weight visualizations."""
        
        if attention_weights.dim() != 4:  # [batch, heads, seq_len, seq_len]
            logger.warning(f"Unexpected attention weight dimensions: {attention_weights.shape}")
            return
        
        # Take first batch, average across heads
        attention = attention_weights[0].mean(dim=0).detach().cpu().numpy()
        
        # Create attention heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        ax.set_title(f'Attention Weights - {layer_name} (Epoch {epoch})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Log to TensorBoard
        self.log_matplotlib_figure(f"Attention/{layer_name}", fig, epoch)
        plt.close(fig)
    
    def log_generated_samples_info(self, sample_info: Dict[str, Any], epoch: int):
        """Log information about generated samples."""
        
        # Log generation statistics
        if 'generation_time' in sample_info:
            self.log_scalar("Generation/Time", sample_info['generation_time'], epoch)
        
        if 'sequence_length' in sample_info:
            self.log_scalar("Generation/SequenceLength", sample_info['sequence_length'], epoch)
        
        if 'unique_tokens' in sample_info:
            self.log_scalar("Generation/UniqueTokens", sample_info['unique_tokens'], epoch)
        
        if 'repetition_rate' in sample_info:
            self.log_scalar("Generation/RepetitionRate", sample_info['repetition_rate'], epoch)
        
        # Log musical characteristics
        if 'musical_stats' in sample_info:
            musical_stats = sample_info['musical_stats']
            for stat_name, value in musical_stats.items():
                self.log_scalar(f"Generation/Musical/{stat_name}", value, epoch)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics."""
        
        # Convert complex values to strings for TensorBoard
        hparams_clean = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_clean[key] = value
            else:
                hparams_clean[key] = str(value)
        
        self.writer.add_hparams(hparams_clean, metrics)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text, step)
    
    def log_training_progress_summary(self, 
                                    current_epoch: int,
                                    total_epochs: int,
                                    current_loss: float,
                                    best_loss: float,
                                    eta_minutes: int):
        """Log training progress summary text."""
        
        progress_text = f"""
## Training Progress Summary

**Epoch**: {current_epoch}/{total_epochs} ({current_epoch/total_epochs*100:.1f}%)  
**Current Loss**: {current_loss:.4f}  
**Best Loss**: {best_loss:.4f}  
**ETA**: {eta_minutes} minutes  

**Status**: {"🟢 On Track" if current_loss <= best_loss * 1.1 else "⚠️ Monitoring"}
        """
        
        self.log_text("Training/ProgressSummary", progress_text, current_epoch)
    
    def increment_step(self):
        """Increment internal step counter."""
        self.step_counter += 1
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            logger.info("Closed TensorBoard writer")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()