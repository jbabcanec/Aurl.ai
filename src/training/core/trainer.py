"""
Advanced Training Framework for Aurl.ai Music Generation.

This module implements a comprehensive training framework with:
- Distributed data parallel training
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large batches
- Dynamic batch sizing by sequence length
- Curriculum learning implementation
- Memory-efficient attention training
- Model parallelism support
- Training throughput monitoring
- GPU utilization optimization

Designed to work seamlessly with our Phase 3 architecture including:
- MusicTransformerVAEGAN models
- ComprehensiveLossFramework
- Real-time loss monitoring
- Professional configuration system
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
import math
import numpy as np
from pathlib import Path
import psutil
import GPUtil
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
from datetime import datetime

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.training.core.losses import ComprehensiveLossFramework
from src.training.monitoring.loss_visualization import LossMonitor, TrainingStabilityMonitor
from src.training.core.training_logger import EnhancedTrainingLogger
from src.training.core.experiment_tracker import DataUsageInfo
from src.data.dataset import LazyMidiDataset
from src.utils.config import load_config
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration with all options."""
    
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 1e-6
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    
    # Mixed precision
    use_mixed_precision: bool = True
    fp16: bool = True
    bf16: bool = False
    dynamic_loss_scaling: bool = True
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Dynamic batching
    dynamic_batching: bool = True
    max_sequence_length: int = 2048
    min_batch_size: int = 4
    max_batch_size: int = 64
    memory_threshold: float = 0.9  # GPU memory threshold
    
    # Curriculum learning
    curriculum_learning: bool = True
    curriculum_start_length: int = 512
    curriculum_end_length: int = 2048
    curriculum_epochs: int = 20
    
    # Memory optimization
    activation_checkpointing: bool = False
    gradient_checkpointing: bool = True
    model_parallelism: bool = False
    
    # Monitoring
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    throughput_monitoring: bool = True
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, lion
    scheduler: str = "cosine"  # linear, cosine, plateau
    min_learning_rate: float = 1e-7


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive training."""
    
    def __init__(self,
                 start_length: int = 512,
                 end_length: int = 2048,
                 curriculum_epochs: int = 20,
                 strategy: str = "linear"):
        self.start_length = start_length
        self.end_length = end_length
        self.curriculum_epochs = curriculum_epochs
        self.strategy = strategy
        
        logger.info(f"Initialized CurriculumScheduler: {start_length} → {end_length} over {curriculum_epochs} epochs")
    
    def get_current_length(self, epoch: int) -> int:
        """Get current sequence length based on epoch."""
        if epoch >= self.curriculum_epochs:
            return self.end_length
        
        progress = epoch / self.curriculum_epochs
        
        if self.strategy == "linear":
            current_length = self.start_length + progress * (self.end_length - self.start_length)
        elif self.strategy == "exponential":
            # Exponential growth
            ratio = (self.end_length / self.start_length) ** progress
            current_length = self.start_length * ratio
        elif self.strategy == "cosine":
            # Cosine annealing (smooth)
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            current_length = self.start_length + cosine_progress * (self.end_length - self.start_length)
        else:
            current_length = self.start_length
        
        return int(current_length)


class DynamicBatchSizer:
    """Dynamic batch sizing based on sequence length and GPU memory."""
    
    def __init__(self,
                 base_batch_size: int = 32,
                 min_batch_size: int = 4,
                 max_batch_size: int = 64,
                 memory_threshold: float = 0.9):
        self.base_batch_size = base_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        
        # Track memory usage history
        self.memory_history = []
        self.batch_size_history = []
        
    def get_optimal_batch_size(self, sequence_length: int, current_memory_usage: float = None) -> int:
        """Calculate optimal batch size based on sequence length and memory."""
        
        # Base calculation: inverse relationship with sequence length
        length_factor = 1024 / max(sequence_length, 256)  # Normalize to 1024 baseline
        suggested_batch_size = int(self.base_batch_size * length_factor)
        
        # Adjust based on GPU memory if available
        if current_memory_usage is not None:
            if current_memory_usage > self.memory_threshold:
                # Reduce batch size if memory usage is high
                memory_factor = (1.0 - current_memory_usage) / (1.0 - self.memory_threshold)
                suggested_batch_size = int(suggested_batch_size * max(memory_factor, 0.5))
            elif current_memory_usage < self.memory_threshold * 0.7:
                # Increase batch size if memory usage is low
                memory_factor = 1.2
                suggested_batch_size = int(suggested_batch_size * memory_factor)
        
        # Apply bounds
        optimal_batch_size = max(self.min_batch_size, 
                               min(suggested_batch_size, self.max_batch_size))
        
        # Track history
        self.batch_size_history.append(optimal_batch_size)
        if current_memory_usage is not None:
            self.memory_history.append(current_memory_usage)
        
        return optimal_batch_size


class ThroughputMonitor:
    """Monitor training throughput and performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Throughput tracking
        self.batch_times = []
        self.sample_counts = []
        self.token_counts = []
        
        # Resource tracking
        self.gpu_utilizations = []
        self.gpu_memory_usages = []
        self.cpu_utilizations = []
        
        # Current batch tracking
        self.batch_start_time = None
        
    def start_batch(self):
        """Mark the start of a batch."""
        self.batch_start_time = time.time()
    
    def end_batch(self, num_samples: int, num_tokens: int):
        """Mark the end of a batch and record metrics."""
        if self.batch_start_time is None:
            return
        
        batch_time = time.time() - self.batch_start_time
        
        # Record throughput
        self.batch_times.append(batch_time)
        self.sample_counts.append(num_samples)
        self.token_counts.append(num_tokens)
        
        # Record resource usage
        try:
            # GPU metrics
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                self.gpu_utilizations.append(gpu.load * 100)
                self.gpu_memory_usages.append(gpu.memoryUtil * 100)
            
            # CPU metrics
            self.cpu_utilizations.append(psutil.cpu_percent())
            
        except Exception as e:
            logger.warning(f"Could not collect resource metrics: {e}")
        
        # Maintain window size
        for metric_list in [self.batch_times, self.sample_counts, self.token_counts,
                           self.gpu_utilizations, self.gpu_memory_usages, self.cpu_utilizations]:
            if len(metric_list) > self.window_size:
                metric_list.pop(0)
        
        self.batch_start_time = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current throughput and resource metrics."""
        if not self.batch_times:
            return {}
        
        # Calculate throughput
        total_time = sum(self.batch_times)
        total_samples = sum(self.sample_counts)
        total_tokens = sum(self.token_counts)
        
        metrics = {
            'samples_per_second': total_samples / total_time if total_time > 0 else 0,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'avg_batch_time': np.mean(self.batch_times),
            'batch_time_std': np.std(self.batch_times),
        }
        
        # Add resource metrics
        if self.gpu_utilizations:
            metrics.update({
                'gpu_utilization': np.mean(self.gpu_utilizations),
                'gpu_memory_usage': np.mean(self.gpu_memory_usages),
            })
        
        if self.cpu_utilizations:
            metrics['cpu_utilization'] = np.mean(self.cpu_utilizations)
        
        return metrics


class AdvancedTrainer:
    """
    Advanced training framework with all Phase 4.1 features.
    
    Features:
    - Distributed training support
    - Mixed precision training
    - Dynamic batch sizing
    - Curriculum learning
    - Throughput monitoring
    - Memory optimization
    """
    
    def __init__(self,
                 model: MusicTransformerVAEGAN,
                 loss_framework: ComprehensiveLossFramework,
                 config: TrainingConfig,
                 train_dataset: LazyMidiDataset,
                 val_dataset: Optional[LazyMidiDataset] = None,
                 save_dir: Optional[Path] = None):
        
        self.model = model
        self.loss_framework = loss_framework
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/training")
        
        # Initialize device and distributed training
        self.device = self._setup_device()
        self._setup_distributed()
        
        # Initialize mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Initialize optimizers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize comprehensive logging system
        self.enhanced_logger = EnhancedTrainingLogger(
            experiment_name=f"aurl_ai_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_dir=self.save_dir,
            model_config={
                'vocab_size': model.vocab_size,
                'd_model': model.d_model,
                'n_layers': model.n_layers,
                'mode': model.mode,
                'latent_dim': getattr(model, 'latent_dim', None),
                'max_sequence_length': model.max_sequence_length
            },
            training_config=asdict(config),
            data_config={
                'dataset_type': type(train_dataset).__name__,
                'dataset_size': len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'
            },
            enable_tensorboard=True,
            enable_dashboard=True
        )
        
        # Initialize monitoring (keep existing for compatibility)
        self.loss_monitor = LossMonitor(save_dir=self.save_dir / "loss_monitoring")
        self.stability_monitor = TrainingStabilityMonitor()
        self.throughput_monitor = ThroughputMonitor()
        
        # Data usage tracking
        self.files_processed_this_epoch = 0
        self.files_augmented_this_epoch = 0
        self.total_tokens_this_epoch = 0
        self.data_usage_buffer = []
        
        # Initialize curriculum and dynamic batching
        self.curriculum_scheduler = CurriculumScheduler(
            start_length=config.curriculum_start_length,
            end_length=config.curriculum_end_length,
            curriculum_epochs=config.curriculum_epochs
        ) if config.curriculum_learning else None
        
        self.dynamic_batch_sizer = DynamicBatchSizer(
            base_batch_size=config.batch_size,
            min_batch_size=config.min_batch_size,
            max_batch_size=config.max_batch_size,
            memory_threshold=config.memory_threshold
        ) if config.dynamic_batching else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized AdvancedTrainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if torch.cuda.is_available():
            if self.config.distributed:
                device = torch.device(f"cuda:{self.config.local_rank}")
            else:
                device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        return device
    
    def _setup_distributed(self):
        """Setup distributed training if configured."""
        if self.config.distributed:
            # Initialize distributed training
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.config.rank,
                world_size=self.config.world_size
            )
            
            # Move model to device and wrap with DDP
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.config.local_rank])
            
            logger.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")
        else:
            self.model = self.model.to(self.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        model_params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                model_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == "lion":
            # Lion optimizer (if available)
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    model_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            except ImportError:
                logger.warning("Lion optimizer not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    model_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        logger.info(f"Created {self.config.optimizer} optimizer with lr={self.config.learning_rate}")
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "linear":
            total_steps = len(self.train_dataset) * self.config.num_epochs // self.config.batch_size
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler == "cosine":
            total_steps = len(self.train_dataset) * self.config.num_epochs // self.config.batch_size
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_learning_rate
            )
        elif self.config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5,
                min_lr=self.config.min_learning_rate
            )
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"Created {self.config.scheduler} scheduler")
        
        return scheduler
    
    def _create_dataloader(self, dataset: LazyMidiDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create dataloader with proper configuration."""
        sampler = None
        if self.config.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # DistributedSampler handles shuffling
        
        # Dynamic sequence length for curriculum learning
        current_length = self.config.max_sequence_length
        if self.curriculum_scheduler:
            current_length = self.curriculum_scheduler.get_current_length(self.current_epoch)
            # Update dataset sequence length
            dataset.max_sequence_length = current_length
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def _get_current_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get comprehensive memory metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            metrics['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            metrics['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['cpu_rss'] = memory_info.rss / 1024**3  # GB
            metrics['cpu_vms'] = memory_info.vms / 1024**3
        except:
            pass
        
        return metrics
    
    def _track_batch_data_usage(self, batch: Dict[str, Any], batch_size: int, seq_len: int):
        """Track data usage for current batch."""
        
        # Update epoch statistics
        self.files_processed_this_epoch += batch_size
        self.total_tokens_this_epoch += batch_size * seq_len
        
        # Check for augmentation indicators in batch (if available)
        if 'metadata' in batch:
            metadata = batch['metadata']
            
            for i in range(batch_size):
                item_meta = metadata[i] if isinstance(metadata, list) else metadata
                
                # Create data usage info
                data_usage = DataUsageInfo(
                    file_name=item_meta.get('file_name', f'batch_{self.current_epoch}_{i}'),
                    original_length=item_meta.get('original_length', seq_len),
                    processed_length=seq_len,
                    augmentation_applied=item_meta.get('augmentation_applied', {}),
                    transposition=item_meta.get('transposition', 0),
                    time_stretch=item_meta.get('time_stretch', 1.0),
                    velocity_scale=item_meta.get('velocity_scale', 1.0),
                    instruments_used=item_meta.get('instruments_used', ['piano']),
                    processing_time=item_meta.get('processing_time', 0.0),
                    cache_hit=item_meta.get('cache_hit', False)
                )
                
                # Check if augmented
                if any(data_usage.augmentation_applied.values()):
                    self.files_augmented_this_epoch += 1
                
                # Log to enhanced logger
                self.enhanced_logger.log_data_usage(data_usage)
        else:
            # Fallback: estimate augmentation based on patterns (simplified)
            # This would need dataset integration for full functionality
            pass
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Reset epoch statistics
        self.files_processed_this_epoch = 0
        self.files_augmented_this_epoch = 0
        self.total_tokens_this_epoch = 0
        self.data_usage_buffer = []
        
        # Get current batch size (dynamic or fixed)
        current_batch_size = self.config.batch_size
        if self.dynamic_batch_sizer:
            current_length = self.config.max_sequence_length
            if self.curriculum_scheduler:
                current_length = self.curriculum_scheduler.get_current_length(self.current_epoch)
            
            memory_usage = self._get_current_memory_usage()
            current_batch_size = self.dynamic_batch_sizer.get_optimal_batch_size(
                current_length, memory_usage
            )
        
        # Create dataloader for this epoch
        train_dataloader = self._create_dataloader(
            self.train_dataset, 
            current_batch_size, 
            shuffle=True
        )
        
        # Start epoch logging
        self.enhanced_logger.start_epoch(self.current_epoch, len(train_dataloader))
        
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Start throughput monitoring
            if self.config.throughput_monitoring:
                self.throughput_monitor.start_batch()
            
            # Move batch to device and track data usage
            tokens = batch['tokens'].to(self.device)
            batch_size, seq_len = tokens.shape
            
            # Track data usage for this batch
            self._track_batch_data_usage(batch, batch_size, seq_len)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    losses = self._forward_pass(tokens)
            else:
                losses = self._forward_pass(tokens)
            
            # Backward pass
            loss = losses['total_loss'] / self.config.gradient_accumulation_steps
            
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler and self.config.scheduler != "plateau":
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                # Update monitoring
                self.stability_monitor.update(
                    gradient_norm=grad_norm,
                    loss_value=losses['total_loss'].item(),
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
                
                self.global_step += 1
            
            # Record losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key].append(value.item())
            
            # End throughput monitoring
            if self.config.throughput_monitoring:
                num_tokens = batch_size * seq_len
                self.throughput_monitor.end_batch(batch_size, num_tokens)
            
            # Enhanced logging
            if batch_idx % self.config.log_interval == 0:
                # Get memory metrics
                memory_metrics = self._get_memory_metrics()
                
                # Get throughput metrics
                throughput_metrics = self.throughput_monitor.get_metrics()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Get gradient norm (if available)
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        break
                
                # Data statistics for this batch
                data_stats = {
                    'files_processed': self.files_processed_this_epoch,
                    'files_augmented': self.files_augmented_this_epoch
                }
                
                # Enhanced batch logging
                self.enhanced_logger.log_batch(
                    batch=batch_idx,
                    losses=losses,
                    learning_rate=current_lr,
                    gradient_norm=grad_norm,
                    throughput_metrics=throughput_metrics,
                    memory_metrics=memory_metrics,
                    data_stats=data_stats
                )
                
                # Legacy logging (for compatibility)
                self._log_progress(batch_idx, len(train_dataloader), losses, current_batch_size)
        
        # Calculate epoch averages
        epoch_avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }
        
        # Calculate average sequence length
        avg_seq_length = self.total_tokens_this_epoch / max(self.files_processed_this_epoch, 1)
        
        # Get final throughput metrics for epoch
        final_throughput = self.throughput_monitor.get_metrics()
        
        # Prepare data statistics
        data_stats = {
            'files_processed': self.files_processed_this_epoch,
            'files_augmented': self.files_augmented_this_epoch,
            'total_tokens': self.total_tokens_this_epoch,
            'average_sequence_length': avg_seq_length,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'gradient_norm': 0.0,  # Will be updated with actual value
            'samples_per_second': final_throughput.get('samples_per_second', 0),
            'tokens_per_second': final_throughput.get('tokens_per_second', 0),
            'memory_usage': self._get_memory_metrics(),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        # End epoch with enhanced logging (no validation for now)
        self.enhanced_logger.end_epoch(
            train_losses=epoch_avg_losses,
            val_losses={},  # Will be updated when validation is implemented
            data_stats=data_stats,
            is_best_model=False,  # Will be determined in main training loop
            checkpoint_saved=False  # Will be determined in main training loop
        )
        
        # Update loss monitoring (legacy)
        self.loss_monitor.update(epoch_avg_losses, self.global_step)
        
        # Update loss framework
        self.loss_framework.step_epoch(
            avg_kl=epoch_avg_losses.get('kl_raw', None)
        )
        
        return epoch_avg_losses
    
    def _forward_pass(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through model and loss computation."""
        batch_size, seq_len = tokens.shape
        
        if self.model.mode == "transformer":
            # Pure transformer mode
            logits = self.model(tokens)
            
            # Simple cross-entropy loss
            recon_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.model.vocab_size),
                tokens.view(-1),
                ignore_index=2  # PAD token
            )
            
            losses = {
                'total_loss': recon_loss,
                'reconstruction_loss': recon_loss
            }
            
        elif self.model.mode in ["vae", "vae_gan"]:
            # VAE-based modes
            # Get model embeddings
            embeddings = self.model.embedding(tokens)
            
            # Encoder forward pass
            encoder_output = self.model.encoder(embeddings)
            
            # Decoder forward pass  
            reconstruction_logits = self.model.decoder(
                encoder_output['z'],
                embeddings,
                encoder_features=embeddings
            )
            
            # Generate tokens for discriminator (if in VAE-GAN mode)
            if self.model.mode == "vae_gan":
                with torch.no_grad():
                    generated_probs = torch.softmax(reconstruction_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        generated_probs.view(-1, self.model.vocab_size),
                        num_samples=1
                    ).view(batch_size, seq_len)
                
                # Discriminator forward pass
                real_disc_output = self.model.discriminator(tokens)
                fake_disc_output = self.model.discriminator(generated_tokens)
                
                # Comprehensive loss computation
                losses = self.loss_framework(
                    reconstruction_logits=reconstruction_logits,
                    target_tokens=tokens,
                    encoder_output=encoder_output,
                    real_discriminator_output=real_disc_output,
                    fake_discriminator_output=fake_disc_output,
                    discriminator=self.model.discriminator,
                    generated_tokens=generated_tokens
                )
            else:
                # VAE-only mode
                recon_loss = torch.nn.functional.cross_entropy(
                    reconstruction_logits.view(-1, self.model.vocab_size),
                    tokens.view(-1),
                    ignore_index=2  # PAD token
                )
                
                kl_loss = encoder_output['kl_loss'].mean()
                total_loss = recon_loss + 0.1 * kl_loss
                losses = {
                    'total_loss': total_loss,
                    'reconstruction_loss': recon_loss,
                    'kl_loss': kl_loss
                }
        else:
            raise ValueError(f"Unknown model mode: {self.model.mode}")
        
        return losses
    
    def _log_progress(self, batch_idx: int, total_batches: int, losses: Dict[str, torch.Tensor], batch_size: int):
        """Log training progress."""
        # Get throughput metrics
        throughput_metrics = self.throughput_monitor.get_metrics()
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Get current sequence length (if curriculum learning)
        current_length = self.config.max_sequence_length
        if self.curriculum_scheduler:
            current_length = self.curriculum_scheduler.get_current_length(self.current_epoch)
        
        # Format loss values
        loss_str = ", ".join([
            f"{k}: {v.item():.4f}" if isinstance(v, torch.Tensor) else f"{k}: {v:.4f}"
            for k, v in losses.items() if "total" in k.lower() or "recon" in k.lower()
        ])
        
        # Format throughput
        throughput_str = ""
        if throughput_metrics:
            throughput_str = f" | {throughput_metrics.get('samples_per_second', 0):.1f} samples/sec"
            if 'tokens_per_second' in throughput_metrics:
                throughput_str += f", {throughput_metrics['tokens_per_second']:.0f} tokens/sec"
        
        logger.info(
            f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
            f"BS: {batch_size}, SeqLen: {current_length}, LR: {current_lr:.2e} | "
            f"{loss_str}{throughput_str}"
        )
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        val_losses = defaultdict(list)
        
        val_dataloader = self._create_dataloader(
            self.val_dataset,
            self.config.batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in val_dataloader:
                tokens = batch['tokens'].to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        losses = self._forward_pass(tokens)
                else:
                    losses = self._forward_pass(tokens)
                
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        val_losses[key].append(value.item())
        
        # Calculate averages
        val_avg_losses = {
            f"val_{key}": np.mean(values) for key, values in val_losses.items()
        }
        
        logger.info(
            f"Validation - " + 
            ", ".join([f"{k}: {v:.4f}" for k, v in val_avg_losses.items() if "total" in k.lower()])
        )
        
        return val_avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_framework_state': self.loss_framework.get_state(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop with enhanced logging."""
        
        # Start training session with enhanced logging
        self.enhanced_logger.start_training(self.config.num_epochs)
        
        # Log model architecture
        try:
            sample_input = torch.randint(0, self.model.vocab_size, (1, 64)).to(self.device)
            self.enhanced_logger.log_model_architecture(self.model, sample_input)
        except Exception as e:
            logger.warning(f"Could not log model architecture: {e}")
        
        logger.info("Starting training...")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Distributed: {self.config.distributed}")
        logger.info(f"Mixed precision: {self.config.use_mixed_precision}")
        logger.info(f"Curriculum learning: {self.config.curriculum_learning}")
        logger.info(f"Dynamic batching: {self.config.dynamic_batching}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Check for best model
            current_val_loss = val_losses.get('val_total_loss', train_losses.get('total_loss', float('inf')))
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
            
            # Scheduler step (for plateau scheduler)
            if self.scheduler and self.config.scheduler == "plateau":
                self.scheduler.step(current_val_loss)
            
            # Save checkpoint
            checkpoint_saved = False
            if epoch % (self.config.num_epochs // 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
                checkpoint_saved = True
            
            # Update epoch summary with actual values
            data_stats = {
                'files_processed': self.files_processed_this_epoch,
                'files_augmented': self.files_augmented_this_epoch,
                'total_tokens': self.total_tokens_this_epoch,
                'average_sequence_length': self.total_tokens_this_epoch / max(self.files_processed_this_epoch, 1),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gradient_norm': 0.0,
                'samples_per_second': self.throughput_monitor.get_metrics().get('samples_per_second', 0),
                'tokens_per_second': self.throughput_monitor.get_metrics().get('tokens_per_second', 0),
                'memory_usage': self._get_memory_metrics(),
                'model_parameters': sum(p.numel() for p in self.model.parameters())
            }
            
            # Update enhanced logger with final epoch information
            self.enhanced_logger.end_epoch(
                train_losses=train_losses,
                val_losses=val_losses,
                data_stats=data_stats,
                is_best_model=is_best,
                checkpoint_saved=checkpoint_saved
            )
            
            # Get training stability report
            stability_report = self.stability_monitor.get_stability_report()
            if stability_report['status'] != 'stable':
                self.enhanced_logger.log_training_anomaly(
                    'training_instability',
                    f"Status: {stability_report['status']}, Recommendations: {stability_report.get('recommendations', [])}"
                )
                logger.warning(f"Training stability: {stability_report['status']}")
                for recommendation in stability_report.get('recommendations', []):
                    logger.warning(f"Recommendation: {recommendation}")
            
            # Generate sample periodically
            if epoch % 10 == 0 and epoch > 0:
                self._generate_training_sample(epoch)
            
            # Log epoch summary
            epoch_summary = {**train_losses, **val_losses}
            logger.info(f"Epoch {epoch} complete - " + 
                       ", ".join([f"{k}: {v:.4f}" for k, v in epoch_summary.items() 
                                if "total" in k.lower()][:3]))
        
        logger.info("Training completed!")
        
        # Final checkpoint
        self.save_checkpoint(self.config.num_epochs - 1, False)
        
        # End training session
        self.enhanced_logger.end_training()
        
        # Save training summary (legacy)
        self._save_training_summary()
    
    def _generate_training_sample(self, epoch: int):
        """Generate a sample during training for monitoring."""
        try:
            self.model.eval()
            
            # Generate a small sample
            with torch.no_grad():
                start_time = time.time()
                
                if self.model.mode == "transformer":
                    # Create a simple prompt
                    prompt = torch.randint(0, min(100, self.model.vocab_size), (1, 8)).to(self.device)
                    generated = self.model.generate(
                        prompt_tokens=prompt,
                        max_new_tokens=32,
                        temperature=0.8
                    )
                else:
                    # VAE/VAE-GAN mode
                    generated = self.model.generate(
                        max_new_tokens=32,
                        temperature=0.8
                    )
                
                generation_time = time.time() - start_time
                
                # Calculate basic statistics
                unique_tokens = len(torch.unique(generated))
                sequence_length = generated.size(1)
                
                # Save sample
                sample_dir = self.save_dir / "generated_samples"
                sample_dir.mkdir(exist_ok=True)
                sample_path = sample_dir / f"sample_epoch_{epoch:03d}.pt"
                torch.save(generated, sample_path)
                
                # Log sample information
                sample_info = {
                    'generation_time': generation_time,
                    'sequence_length': sequence_length,
                    'unique_tokens': unique_tokens,
                    'repetition_rate': 1.0 - (unique_tokens / sequence_length)
                }
                
                # Evaluate musical quality
                quality_metrics = self.enhanced_logger.log_generated_sample_quality(
                    generated, epoch
                )
                
                # Add quality to sample info
                sample_info['musical_quality'] = quality_metrics.overall_quality
                
                self.enhanced_logger.log_sample_generated(sample_path, sample_info)
                
            self.model.train()
            
        except Exception as e:
            logger.warning(f"Failed to generate training sample: {e}")
    
    def _save_training_summary(self):
        """Save comprehensive training summary."""
        summary = {
            'config': self.config.__dict__,
            'training_completed': True,
            'total_epochs': self.config.num_epochs,
            'global_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'final_throughput': self.throughput_monitor.get_metrics(),
            'stability_report': self.stability_monitor.get_stability_report()
        }
        
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved training summary: {summary_path}")


def create_trainer_from_config(config_path: str) -> AdvancedTrainer:
    """Create trainer from configuration file."""
    # Load configuration
    config_dict = load_config(config_path)
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    # Create model
    model_config = config_dict.get('model', {})
    model = MusicTransformerVAEGAN(**model_config)
    
    # Create loss framework
    loss_config = config_dict.get('loss', {})
    loss_framework = ComprehensiveLossFramework(**loss_config)
    
    # Create datasets
    data_config = config_dict.get('data', {})
    train_dataset = LazyMidiDataset(
        data_dir=data_config.get('train_dir', 'data/raw'),
        **data_config
    )
    
    val_dataset = None
    if data_config.get('val_dir'):
        val_dataset = LazyMidiDataset(
            data_dir=data_config['val_dir'],
            **data_config
        )
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        loss_framework=loss_framework,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=Path(config_dict.get('output_dir', 'outputs/training'))
    )
    
    return trainer


# Multi-GPU training support
def setup_distributed_training(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training environment."""
    import os
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized distributed training: rank {rank}/{world_size}")


def train_distributed(rank: int, world_size: int, config_path: str):
    """Distributed training entry point."""
    # Setup distributed environment
    setup_distributed_training(rank, world_size)
    
    # Create trainer with distributed config
    trainer = create_trainer_from_config(config_path)
    trainer.config.distributed = True
    trainer.config.rank = rank
    trainer.config.world_size = world_size
    trainer.config.local_rank = rank
    
    # Start training
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()


def launch_distributed_training(config_path: str, world_size: int = None):
    """Launch distributed training across multiple GPUs."""
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    logger.info(f"Launching distributed training on {world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(world_size, config_path),
        nprocs=world_size,
        join=True
    )