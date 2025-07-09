"""
Phase 4.3: Advanced Checkpointing System

This module provides comprehensive checkpoint management for training:
- Full training state preservation and restoration
- Checkpoint averaging for ensemble benefits
- Automatic cleanup and versioning
- Musical quality-based checkpoint selection
- Distributed checkpoint coordination
- Compression and integrity validation

Key Features:
- Save/restore complete training state (model, optimizer, scheduler, etc.)
- Best model selection based on multiple criteria
- Checkpoint averaging for improved generalization
- Automatic cleanup policies with configurable retention
- Musical quality assessment for checkpoint ranking
- Distributed checkpoint coordination for multi-GPU training
- Compression and integrity checks for reliable storage
"""

import os
import json
import time
import shutil
import hashlib
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from packaging import version

# Import pipeline state manager for complete state recovery
try:
    from .pipeline_state_manager import PipelineStateManager, PipelineState
except ImportError:
    PipelineStateManager = None
    PipelineState = None


@dataclass
class CheckpointMetadata:
    """Comprehensive checkpoint metadata."""
    
    # Basic information
    checkpoint_id: str
    timestamp: str
    epoch: int
    batch: int
    step: int
    
    # Training state
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    
    # Model information
    model_parameters: int
    model_size_mb: float
    
    # Musical quality metrics
    musical_quality: Optional[float] = None
    rhythm_consistency: Optional[float] = None
    harmonic_coherence: Optional[float] = None
    melodic_contour: Optional[float] = None
    
    # Training configuration
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    
    # File information
    file_path: str = ""
    file_size_mb: float = 0.0
    checksum: str = ""
    compressed: bool = False
    
    # Selection criteria scores
    selection_score: float = 0.0
    is_best: bool = False
    
    # Additional metrics
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass 
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    # Core settings
    save_dir: str = "outputs/checkpoints"
    save_frequency: int = 1  # Save every N epochs
    max_checkpoints: int = 10  # Maximum number to keep
    
    # Selection criteria
    selection_metric: str = "val_loss"  # Primary metric for best model
    selection_mode: str = "min"  # "min" or "max"
    musical_quality_weight: float = 0.3  # Weight for musical quality in selection
    
    # Compression and validation
    compress_checkpoints: bool = True
    validate_integrity: bool = True
    
    # Cleanup policies
    cleanup_older_than_days: int = 30
    keep_best_n: int = 3  # Always keep N best checkpoints
    
    # Advanced features
    enable_averaging: bool = True
    averaging_window: int = 5  # Number of checkpoints to average
    distributed_saving: bool = False
    async_saving: bool = False
    
    # Versioning
    enable_versioning: bool = True
    max_versions_per_checkpoint: int = 3


class CheckpointManager:
    """
    Advanced checkpoint management system for music generation training.
    
    Features:
    - Complete training state preservation
    - Intelligent checkpoint selection and cleanup
    - Checkpoint averaging for ensemble benefits
    - Musical quality-based ranking
    - Distributed checkpoint coordination
    - Compression and integrity validation
    """
    
    def __init__(self, config: CheckpointConfig, enable_pipeline_state: bool = True):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_history: List[CheckpointMetadata] = []
        
        # Load existing checkpoints
        self._load_checkpoint_registry()
        
        # Distributed state
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # Averaging state
        self.averaged_models: Dict[str, torch.nn.Module] = {}
        
        # Pipeline state manager for complete resume functionality
        self.pipeline_state_manager = None
        if enable_pipeline_state and PipelineStateManager is not None:
            self.pipeline_state_manager = PipelineStateManager(str(self.save_dir))
        
        print(f"CheckpointManager initialized:")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Existing checkpoints: {len(self.checkpoints)}")
        print(f"  Distributed: {self.is_distributed} (rank {self.rank}/{self.world_size})")
        print(f"  Pipeline state manager: {'enabled' if self.pipeline_state_manager else 'disabled'}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        batch: int,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        musical_quality_metrics: Optional[Dict[str, float]] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        force_save: bool = False,
        # Pipeline state parameters for complete resume
        dataset: Optional[Any] = None,
        dataloader: Optional[Any] = None,
        augmenter: Optional[Any] = None,
        curriculum_scheduler: Optional[Any] = None
    ) -> CheckpointMetadata:
        """
        Save a comprehensive checkpoint with full training state.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            batch: Current batch
            step: Current step
            train_loss: Training loss
            val_loss: Validation loss
            musical_quality_metrics: Musical quality assessment
            additional_state: Additional state to save
            force_save: Force save even if frequency check fails
            dataset: Dataset for pipeline state capture
            dataloader: DataLoader for pipeline state capture
            augmenter: Augmentation manager for pipeline state capture
            curriculum_scheduler: Curriculum scheduler for pipeline state capture
            
        Returns:
            CheckpointMetadata object for the saved checkpoint
        """
        
        # Check if we should save (frequency check)
        if not force_save and epoch % self.config.save_frequency != 0:
            return None
        
        # Only save on main process in distributed setting
        if self.is_distributed and self.rank != 0:
            return None
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(epoch, step)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'batch': batch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'python_version': f"{version.parse('3.8')}",  # Simplified
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['learning_rate'] = scheduler.get_last_lr()[0]
        else:
            checkpoint_data['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Add additional state
        if additional_state:
            checkpoint_data['additional_state'] = additional_state
        
        # Add musical quality metrics
        if musical_quality_metrics:
            checkpoint_data['musical_quality_metrics'] = musical_quality_metrics
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=checkpoint_data['timestamp'],
            epoch=epoch,
            batch=batch,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=checkpoint_data['learning_rate'],
            model_parameters=sum(p.numel() for p in model.parameters()),
            model_size_mb=sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            musical_quality=musical_quality_metrics.get('overall_quality') if musical_quality_metrics else None,
            rhythm_consistency=musical_quality_metrics.get('rhythm_consistency') if musical_quality_metrics else None,
            harmonic_coherence=musical_quality_metrics.get('harmonic_coherence') if musical_quality_metrics else None,
            melodic_contour=musical_quality_metrics.get('melodic_contour') if musical_quality_metrics else None,
            additional_metrics=musical_quality_metrics or {}
        )
        
        # Save checkpoint file
        checkpoint_path = self._save_checkpoint_file(checkpoint_data, metadata)
        
        # Update metadata with file information
        metadata.file_path = str(checkpoint_path)
        metadata.file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Calculate checksum if validation enabled
        if self.config.validate_integrity:
            metadata.checksum = self._calculate_checksum(checkpoint_path)
        
        # Calculate selection score
        metadata.selection_score = self._calculate_selection_score(metadata)
        
        # Add to registry
        self.checkpoints.append(metadata)
        self.checkpoint_history.append(metadata)
        
        # Update best checkpoints
        self._update_best_checkpoints(metadata)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save registry
        self._save_checkpoint_registry()
        
        # Create averaged model if enabled
        if self.config.enable_averaging:
            self._update_averaged_model(model, metadata)
        
        # Save pipeline state for complete resume functionality
        if self.pipeline_state_manager is not None:
            try:
                pipeline_state = self.pipeline_state_manager.capture_state(
                    epoch=epoch,
                    batch=batch,
                    step=step,
                    dataset=dataset,
                    dataloader=dataloader,
                    augmenter=augmenter,
                    curriculum_scheduler=curriculum_scheduler
                )
                pipeline_path = self.pipeline_state_manager.save_pipeline_state(
                    checkpoint_id, pipeline_state
                )
                print(f"  Pipeline state saved: {pipeline_path.name}")
            except Exception as e:
                warnings.warn(f"Failed to save pipeline state: {e}")
        
        print(f"Checkpoint saved: {checkpoint_id}")
        print(f"  Path: {checkpoint_path}")
        print(f"  Size: {metadata.file_size_mb:.2f} MB")
        print(f"  Selection score: {metadata.selection_score:.4f}")
        
        return metadata
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        load_best: bool = False,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        strict: bool = True,
        # Pipeline state parameters for complete resume
        restore_pipeline_state: bool = False,
        dataset: Optional[Any] = None,
        dataloader: Optional[Any] = None,
        augmenter: Optional[Any] = None,
        curriculum_scheduler: Optional[Any] = None
    ) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """
        Load a checkpoint and restore training state.
        
        Args:
            checkpoint_path: Explicit path to checkpoint file
            checkpoint_id: ID of checkpoint to load
            load_best: Load the best checkpoint according to selection criteria
            model: Model to load state into
            optimizer: Optimizer to load state into  
            scheduler: Scheduler to load state into
            strict: Strict loading for model state dict
            restore_pipeline_state: Whether to restore complete pipeline state
            dataset: Dataset for pipeline state restoration
            dataloader: DataLoader for pipeline state restoration
            augmenter: Augmentation manager for pipeline state restoration
            curriculum_scheduler: Curriculum scheduler for pipeline state restoration
            
        Returns:
            Tuple of (checkpoint_data, metadata)
        """
        
        # Determine which checkpoint to load
        if load_best:
            metadata = self.get_best_checkpoint()
            if metadata is None:
                raise ValueError("No best checkpoint available")
            checkpoint_path = metadata.file_path
        elif checkpoint_id:
            metadata = self._find_checkpoint_by_id(checkpoint_id)
            if metadata is None:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            checkpoint_path = metadata.file_path
        elif checkpoint_path is None:
            # Load latest checkpoint
            metadata = self.get_latest_checkpoint()
            if metadata is None:
                raise ValueError("No checkpoints available")
            checkpoint_path = metadata.file_path
        else:
            # Find metadata for given path
            metadata = self._find_checkpoint_by_path(checkpoint_path)
        
        # Validate checkpoint integrity
        if self.config.validate_integrity and metadata and metadata.checksum:
            if not self._validate_checkpoint(checkpoint_path, metadata.checksum):
                raise ValueError(f"Checkpoint integrity check failed: {checkpoint_path}")
        
        # Load checkpoint data
        try:
            if checkpoint_path.endswith('.gz'):
                import gzip
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location='cpu', weights_only=False)
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")
        
        # Load model state
        if model is not None and 'model_state_dict' in checkpoint_data:
            try:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
                print(f"Model state loaded from {checkpoint_path}")
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to load model state: {e}")
                else:
                    warnings.warn(f"Partial model state loading: {e}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print(f"Optimizer state loaded from {checkpoint_path}")
            except Exception as e:
                warnings.warn(f"Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            try:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                print(f"Scheduler state loaded from {checkpoint_path}")
            except Exception as e:
                warnings.warn(f"Failed to load scheduler state: {e}")
        
        # Restore pipeline state for complete resume if requested
        if restore_pipeline_state and self.pipeline_state_manager is not None:
            if metadata:
                checkpoint_id = metadata.checkpoint_id
            else:
                # Extract checkpoint ID from path
                checkpoint_id = Path(checkpoint_path).stem.replace('_pipeline_state', '')
            
            try:
                pipeline_state = self.pipeline_state_manager.full_pipeline_restore(
                    checkpoint_id=checkpoint_id,
                    dataset=dataset,
                    dataloader=dataloader,
                    augmenter=augmenter,
                    curriculum_scheduler=curriculum_scheduler
                )
                print(f"  Pipeline state restored from epoch {pipeline_state.current_epoch}")
            except Exception as e:
                warnings.warn(f"Failed to restore pipeline state: {e}")
        
        print(f"Checkpoint loaded successfully:")
        print(f"  Path: {checkpoint_path}")
        print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
        print(f"  Step: {checkpoint_data.get('step', 'unknown')}")
        print(f"  Train Loss: {checkpoint_data.get('train_loss', 'unknown')}")
        
        return checkpoint_data, metadata
    
    def auto_resume(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        # Pipeline state parameters for complete resume
        restore_pipeline_state: bool = False,
        dataset: Optional[Any] = None,
        dataloader: Optional[Any] = None,
        augmenter: Optional[Any] = None,
        curriculum_scheduler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically resume from the latest checkpoint if available.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            restore_pipeline_state: Whether to restore complete pipeline state
            dataset: Dataset for pipeline state restoration
            dataloader: DataLoader for pipeline state restoration
            augmenter: Augmentation manager for pipeline state restoration
            curriculum_scheduler: Curriculum scheduler for pipeline state restoration
            
        Returns:
            Checkpoint data if resumed, None if no checkpoint found
        """
        
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint is None:
            print("No checkpoint found for auto-resume")
            return None
        
        print(f"Auto-resuming from checkpoint: {latest_checkpoint.checkpoint_id}")
        
        checkpoint_data, _ = self.load_checkpoint(
            checkpoint_path=latest_checkpoint.file_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=False,  # More forgiving for auto-resume
            restore_pipeline_state=restore_pipeline_state,
            dataset=dataset,
            dataloader=dataloader,
            augmenter=augmenter,
            curriculum_scheduler=curriculum_scheduler
        )
        
        return checkpoint_data
    
    def create_averaged_model(
        self,
        model: nn.Module,
        checkpoint_ids: Optional[List[str]] = None,
        window_size: Optional[int] = None
    ) -> nn.Module:
        """
        Create an averaged model from multiple checkpoints.
        
        Args:
            model: Base model architecture
            checkpoint_ids: Specific checkpoints to average (if None, uses recent best)
            window_size: Number of recent checkpoints to average
            
        Returns:
            Averaged model
        """
        
        if checkpoint_ids is None:
            # Use recent best checkpoints
            window_size = window_size or self.config.averaging_window
            best_checkpoints = self.get_best_checkpoints(limit=window_size)
            checkpoint_ids = [cp.checkpoint_id for cp in best_checkpoints]
        
        if len(checkpoint_ids) == 0:
            raise ValueError("No checkpoints available for averaging")
        
        print(f"Creating averaged model from {len(checkpoint_ids)} checkpoints")
        
        # Load all checkpoint states
        states = []
        for checkpoint_id in checkpoint_ids:
            metadata = self._find_checkpoint_by_id(checkpoint_id)
            if metadata is None:
                warnings.warn(f"Checkpoint {checkpoint_id} not found, skipping")
                continue
            
            if metadata.file_path.endswith('.gz'):
                import gzip
                with gzip.open(metadata.file_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location='cpu', weights_only=False)
            else:
                checkpoint_data = torch.load(metadata.file_path, map_location='cpu', weights_only=False)
            states.append(checkpoint_data['model_state_dict'])
        
        if len(states) == 0:
            raise ValueError("No valid checkpoints found for averaging")
        
        # Average the states
        averaged_state = {}
        for key in states[0].keys():
            # Check if all states have this key
            if all(key in state for state in states):
                tensors = [state[key] for state in states]
                
                # Average tensors
                if len(tensors) > 0 and isinstance(tensors[0], torch.Tensor):
                    averaged_state[key] = torch.stack(tensors).mean(dim=0)
                else:
                    # For non-tensor values, take the first one
                    averaged_state[key] = tensors[0]
        
        # Load averaged state into model
        model.load_state_dict(averaged_state)
        
        print(f"Model averaged from {len(states)} checkpoints")
        return model
    
    def get_best_checkpoint(self, metric: Optional[str] = None) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint according to selection criteria."""
        
        metric = metric or self.config.selection_metric
        
        if metric in self.best_checkpoints:
            return self.best_checkpoints[metric]
        
        return None
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the most recent checkpoint."""
        
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda x: x.step)
    
    def get_best_checkpoints(self, limit: int = 5) -> List[CheckpointMetadata]:
        """Get the top N checkpoints by selection score."""
        
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x.selection_score,
            reverse=True
        )
        
        return sorted_checkpoints[:limit]
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about checkpoints."""
        
        if not self.checkpoints:
            return {"total_checkpoints": 0}
        
        stats = {
            "total_checkpoints": len(self.checkpoints),
            "total_size_mb": sum(cp.file_size_mb for cp in self.checkpoints),
            "latest_checkpoint": self.get_latest_checkpoint().checkpoint_id,
            "best_checkpoint": self.get_best_checkpoint().checkpoint_id if self.get_best_checkpoint() else None,
            "average_file_size_mb": np.mean([cp.file_size_mb for cp in self.checkpoints]),
            "checkpoint_frequency": self.config.save_frequency,
            "selection_metric": self.config.selection_metric,
        }
        
        # Musical quality stats
        quality_scores = [cp.musical_quality for cp in self.checkpoints if cp.musical_quality is not None]
        if quality_scores:
            stats["musical_quality"] = {
                "mean": np.mean(quality_scores),
                "std": np.std(quality_scores),
                "min": np.min(quality_scores),
                "max": np.max(quality_scores)
            }
        
        return stats
    
    def get_pipeline_resume_info(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about pipeline state for resume.
        
        Args:
            checkpoint_id: Specific checkpoint to check (uses latest if None)
            
        Returns:
            Dictionary with pipeline resume information
        """
        
        if self.pipeline_state_manager is None:
            return {"status": "pipeline_state_disabled"}
        
        if checkpoint_id is None:
            latest = self.get_latest_checkpoint()
            if latest is None:
                return {"status": "no_checkpoints"}
            checkpoint_id = latest.checkpoint_id
        
        try:
            # Try to load pipeline state
            pipeline_state = self.pipeline_state_manager.load_pipeline_state(checkpoint_id)
            return self.pipeline_state_manager.get_resume_info(pipeline_state)
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cleanup_checkpoints(self, force: bool = False) -> Dict[str, int]:
        """
        Clean up old checkpoints according to policies.
        
        Args:
            force: Force cleanup even if not needed
            
        Returns:
            Statistics about cleanup operation
        """
        
        if not force and len(self.checkpoints) <= self.config.max_checkpoints:
            return {"removed": 0, "kept": len(self.checkpoints)}
        
        print("Starting checkpoint cleanup...")
        
        # Sort checkpoints by selection score (best first)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x.selection_score,
            reverse=True
        )
        
        # Always keep best N checkpoints
        keep_best = sorted_checkpoints[:self.config.keep_best_n]
        keep_best_ids = {cp.checkpoint_id for cp in keep_best}
        
        # Remove old checkpoints based on age and count
        cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_older_than_days)
        
        removed_count = 0
        kept_count = 0
        
        # Keep track of which ones to remove
        to_remove = []
        to_keep = []
        
        for checkpoint in sorted_checkpoints:  # Process best-to-worst order
            should_keep = False
            
            # Always keep the best N
            if checkpoint.checkpoint_id in keep_best_ids:
                should_keep = True
            # Keep if under limit and not too old
            elif kept_count < self.config.max_checkpoints:
                checkpoint_date = datetime.fromisoformat(checkpoint.timestamp)
                if checkpoint_date >= cutoff_date:
                    should_keep = True
            
            if should_keep:
                to_keep.append(checkpoint)
                kept_count += 1
            else:
                to_remove.append(checkpoint)
        
        # Remove checkpoints
        for checkpoint in to_remove:
            self._remove_checkpoint(checkpoint)
            removed_count += 1
        
        # Update checkpoint list
        remaining_checkpoints = to_keep
        
        self.checkpoints = remaining_checkpoints
        self._save_checkpoint_registry()
        
        print(f"Checkpoint cleanup completed: removed {removed_count}, kept {len(remaining_checkpoints)}")
        
        return {"removed": removed_count, "kept": len(remaining_checkpoints)}
    
    def _generate_checkpoint_id(self, epoch: int, step: int) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_epoch_{epoch:04d}_step_{step:08d}_{timestamp}"
    
    def _save_checkpoint_file(
        self,
        checkpoint_data: Dict[str, Any],
        metadata: CheckpointMetadata
    ) -> Path:
        """Save checkpoint data to file."""
        
        checkpoint_path = self.save_dir / f"{metadata.checkpoint_id}.pt"
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Compress if enabled
        if self.config.compress_checkpoints:
            compressed_path = self._compress_checkpoint(checkpoint_path)
            if compressed_path != checkpoint_path:
                checkpoint_path.unlink()  # Remove uncompressed version
                checkpoint_path = compressed_path
                metadata.compressed = True
        
        return checkpoint_path
    
    def _compress_checkpoint(self, checkpoint_path: Path) -> Path:
        """Compress checkpoint file."""
        import gzip
        
        compressed_path = checkpoint_path.with_suffix('.pt.gz')
        
        with open(checkpoint_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Check compression ratio
        original_size = checkpoint_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        ratio = compressed_size / original_size
        
        print(f"Compressed checkpoint: {ratio:.2%} of original size")
        
        return compressed_path
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        
        hash_sha256 = hashlib.sha256()
        
        # Handle compressed files
        if file_path.suffix == '.gz':
            import gzip
            with gzip.open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        else:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _validate_checkpoint(self, file_path: str, expected_checksum: str) -> bool:
        """Validate checkpoint integrity."""
        
        try:
            actual_checksum = self._calculate_checksum(Path(file_path))
            return actual_checksum == expected_checksum
        except Exception as e:
            warnings.warn(f"Checksum validation failed: {e}")
            return False
    
    def _calculate_selection_score(self, metadata: CheckpointMetadata) -> float:
        """Calculate selection score for checkpoint ranking."""
        
        score = 0.0
        
        # Primary metric (val_loss or train_loss)
        if metadata.val_loss is not None:
            primary_score = 1.0 / (1.0 + metadata.val_loss)  # Lower loss = higher score
        else:
            primary_score = 1.0 / (1.0 + metadata.train_loss)
        
        score += primary_score * (1.0 - self.config.musical_quality_weight)
        
        # Musical quality component
        if metadata.musical_quality is not None:
            quality_score = metadata.musical_quality  # Assuming 0-1 scale
            score += quality_score * self.config.musical_quality_weight
        
        return score
    
    def _update_best_checkpoints(self, metadata: CheckpointMetadata):
        """Update best checkpoint tracking."""
        
        # Check if this is the best for any metric
        metrics_to_track = ['val_loss', 'train_loss', 'musical_quality', 'selection_score']
        
        for metric in metrics_to_track:
            value = getattr(metadata, metric, None)
            if value is None:
                continue
            
            current_best = self.best_checkpoints.get(metric)
            
            is_better = False
            if current_best is None:
                is_better = True
            elif metric in ['val_loss', 'train_loss']:
                # Lower is better for losses
                is_better = value < getattr(current_best, metric)
            else:
                # Higher is better for quality metrics
                is_better = value > getattr(current_best, metric)
            
            if is_better:
                self.best_checkpoints[metric] = metadata
                if metric == self.config.selection_metric:
                    metadata.is_best = True
    
    def _update_averaged_model(self, model: nn.Module, metadata: CheckpointMetadata):
        """Update averaged model with new checkpoint."""
        
        # This is a simplified implementation
        # In practice, you might want more sophisticated averaging strategies
        
        model_id = f"averaged_{self.config.averaging_window}"
        
        if model_id not in self.averaged_models:
            self.averaged_models[model_id] = type(model)(model.config) if hasattr(model, 'config') else None
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints automatically."""
        
        if len(self.checkpoints) > self.config.max_checkpoints:
            self.cleanup_checkpoints(force=True)
    
    def _remove_checkpoint(self, metadata: CheckpointMetadata):
        """Remove checkpoint file and metadata."""
        
        try:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                file_path.unlink()
                print(f"Removed checkpoint file: {file_path}")
        except Exception as e:
            warnings.warn(f"Failed to remove checkpoint file {metadata.file_path}: {e}")
    
    def _find_checkpoint_by_id(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Find checkpoint metadata by ID."""
        
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        
        return None
    
    def _find_checkpoint_by_path(self, file_path: str) -> Optional[CheckpointMetadata]:
        """Find checkpoint metadata by file path."""
        
        for checkpoint in self.checkpoints:
            if checkpoint.file_path == file_path:
                return checkpoint
        
        return None
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk."""
        
        registry_path = self.save_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load checkpoints
            self.checkpoints = [
                CheckpointMetadata(**cp_data) 
                for cp_data in registry_data.get('checkpoints', [])
            ]
            
            # Load best checkpoints
            best_data = registry_data.get('best_checkpoints', {})
            self.best_checkpoints = {
                metric: CheckpointMetadata(**cp_data)
                for metric, cp_data in best_data.items()
            }
            
            print(f"Loaded checkpoint registry: {len(self.checkpoints)} checkpoints")
            
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint registry: {e}")
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk."""
        
        registry_path = self.save_dir / "checkpoint_registry.json"
        
        registry_data = {
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
            'best_checkpoints': {
                metric: asdict(cp) for metric, cp in self.best_checkpoints.items()
            },
            'config': asdict(self.config),
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint registry: {e}")


def create_checkpoint_manager(
    save_dir: str = "outputs/checkpoints",
    max_checkpoints: int = 10,
    save_frequency: int = 1,
    enable_compression: bool = True,
    enable_averaging: bool = True,
    musical_quality_weight: float = 0.3,
    enable_pipeline_state: bool = True
) -> CheckpointManager:
    """
    Create a checkpoint manager with common settings.
    
    Args:
        save_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        save_frequency: Save every N epochs
        enable_compression: Enable checkpoint compression
        enable_averaging: Enable checkpoint averaging
        musical_quality_weight: Weight for musical quality in selection
        enable_pipeline_state: Enable complete pipeline state management
        
    Returns:
        Configured CheckpointManager
    """
    
    config = CheckpointConfig(
        save_dir=save_dir,
        max_checkpoints=max_checkpoints,
        save_frequency=save_frequency,
        compress_checkpoints=enable_compression,
        enable_averaging=enable_averaging,
        musical_quality_weight=musical_quality_weight
    )
    
    return CheckpointManager(config, enable_pipeline_state=enable_pipeline_state)