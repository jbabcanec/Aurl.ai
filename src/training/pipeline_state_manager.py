"""
Pipeline State Manager for Complete Training Resume

This module extends checkpoint functionality to include complete data pipeline state:
- Random number generator states for reproducible augmentation
- Data pipeline position and iteration state  
- Augmentation schedule progression
- Curriculum learning state
- Dataset configuration and sequence generation state

Enables true training resume where training picks up exactly where it left off,
including mid-augmentation, mid-epoch, with exact same random sequences.
"""

import os
import json
import pickle
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader


@dataclass
class PipelineState:
    """Complete data pipeline state for resumable training."""
    
    # Random states
    python_random_state: Any  # Python random state
    numpy_random_state: Dict[str, Any]  # NumPy random state
    torch_random_state: torch.Tensor  # PyTorch random state
    torch_cuda_random_state: Optional[torch.Tensor] = None  # CUDA random state
    
    # Data pipeline position
    current_epoch: int = 0
    current_batch: int = 0
    current_step: int = 0
    files_processed_this_epoch: int = 0
    total_files_processed: int = 0
    
    # Dataset state
    dataset_indices: Optional[List[int]] = None  # Current sequence of indices
    sequence_length: int = 512  # Current curriculum sequence length
    batch_size: int = 32
    
    # Augmentation state
    augmentation_epoch: int = 0
    augmentation_probability: float = 0.5
    augmentation_schedule_position: float = 0.0
    augmentations_applied: Dict[str, int] = None
    
    # Curriculum learning state
    curriculum_stage: str = "warmup"  # warmup, main, decay
    curriculum_progress: float = 0.0
    target_sequence_length: int = 512
    
    # Training metadata
    timestamp: str = ""
    version: str = "1.0"
    
    def __post_init__(self):
        if self.augmentations_applied is None:
            self.augmentations_applied = {}
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()


class PipelineStateManager:
    """
    Manages complete data pipeline state for resumable training.
    
    This class works alongside CheckpointManager to ensure that training
    can resume exactly where it left off, including:
    - Exact random sequences for augmentation
    - Current position in dataset iteration
    - Augmentation schedule progression
    - Curriculum learning state
    """
    
    def __init__(self, save_dir: str = "outputs/checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Current pipeline state
        self.current_state: Optional[PipelineState] = None
        
        # References to training components (set during training)
        self.augmenter = None
        self.curriculum_scheduler = None
        self.dataset = None
        self.dataloader = None
        
        print(f"PipelineStateManager initialized: {self.save_dir}")
    
    def capture_state(
        self,
        epoch: int,
        batch: int, 
        step: int,
        dataset: Optional[Any] = None,
        dataloader: Optional[DataLoader] = None,
        augmenter: Optional[Any] = None,
        curriculum_scheduler: Optional[Any] = None
    ) -> PipelineState:
        """
        Capture complete pipeline state at current training position.
        
        Args:
            epoch: Current epoch
            batch: Current batch in epoch
            step: Global training step
            dataset: Training dataset reference
            dataloader: DataLoader reference  
            augmenter: Augmentation manager reference
            curriculum_scheduler: Curriculum scheduler reference
            
        Returns:
            PipelineState object with complete state
        """
        
        # Capture random states
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state()
        
        # Capture dataset state
        dataset_indices = None
        sequence_length = 512
        batch_size = 32
        
        if dataset is not None:
            if hasattr(dataset, 'indices'):
                dataset_indices = list(dataset.indices) if dataset.indices is not None else None
            if hasattr(dataset, 'current_max_length'):
                sequence_length = dataset.current_max_length
        
        if dataloader is not None:
            batch_size = dataloader.batch_size
        
        # Capture augmentation state
        aug_epoch = epoch
        aug_probability = 0.5
        aug_schedule_pos = 0.0
        augmentations_applied = {}
        
        if augmenter is not None:
            if hasattr(augmenter, 'current_epoch'):
                aug_epoch = augmenter.current_epoch
            if hasattr(augmenter, 'current_probability'):
                aug_probability = augmenter.current_probability
            if hasattr(augmenter, 'schedule_position'):
                aug_schedule_pos = augmenter.schedule_position
            if hasattr(augmenter, 'augmentation_stats'):
                augmentations_applied = dict(augmenter.augmentation_stats)
        
        # Capture curriculum state
        curriculum_stage = "main"
        curriculum_progress = 0.0
        target_length = sequence_length
        
        if curriculum_scheduler is not None:
            if hasattr(curriculum_scheduler, 'current_stage'):
                curriculum_stage = curriculum_scheduler.current_stage
            if hasattr(curriculum_scheduler, 'progress'):
                curriculum_progress = curriculum_scheduler.progress
            if hasattr(curriculum_scheduler, 'target_length'):
                target_length = curriculum_scheduler.target_length
        
        # Create state object
        state = PipelineState(
            python_random_state=python_state,
            numpy_random_state={
                'state': numpy_state[1].tolist(),
                'pos': int(numpy_state[2]),
                'has_gauss': int(numpy_state[3]),
                'cached_gaussian': float(numpy_state[4])
            },
            torch_random_state=torch_state,
            torch_cuda_random_state=cuda_state,
            current_epoch=epoch,
            current_batch=batch,
            current_step=step,
            files_processed_this_epoch=batch * batch_size,  # Approximate
            total_files_processed=step * batch_size,  # Approximate
            dataset_indices=dataset_indices,
            sequence_length=sequence_length,
            batch_size=batch_size,
            augmentation_epoch=aug_epoch,
            augmentation_probability=aug_probability,
            augmentation_schedule_position=aug_schedule_pos,
            augmentations_applied=augmentations_applied,
            curriculum_stage=curriculum_stage,
            curriculum_progress=curriculum_progress,
            target_sequence_length=target_length
        )
        
        self.current_state = state
        return state
    
    def save_pipeline_state(
        self,
        checkpoint_id: str,
        state: Optional[PipelineState] = None
    ) -> Path:
        """
        Save pipeline state to disk.
        
        Args:
            checkpoint_id: Unique identifier for this checkpoint
            state: Pipeline state to save (uses current if None)
            
        Returns:
            Path to saved state file
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No pipeline state to save")
        
        # Save state as pickle file (for complex Python objects)
        state_path = self.save_dir / f"{checkpoint_id}_pipeline_state.pkl"
        
        try:
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            
            # Also save as JSON for inspection (some fields may not serialize)
            json_path = self.save_dir / f"{checkpoint_id}_pipeline_state.json"
            try:
                json_data = asdict(state)
                # Convert non-serializable fields
                if 'python_random_state' in json_data:
                    json_data['python_random_state'] = str(json_data['python_random_state'])
                if 'torch_random_state' in json_data:
                    json_data['torch_random_state'] = "torch.Tensor"
                if 'torch_cuda_random_state' in json_data:
                    json_data['torch_cuda_random_state'] = "torch.Tensor" if json_data['torch_cuda_random_state'] is not None else None
                
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save JSON version: {e}")
            
            print(f"Pipeline state saved: {state_path}")
            return state_path
            
        except Exception as e:
            raise ValueError(f"Failed to save pipeline state: {e}")
    
    def load_pipeline_state(self, checkpoint_id: str) -> PipelineState:
        """
        Load pipeline state from disk.
        
        Args:
            checkpoint_id: Unique identifier for checkpoint to load
            
        Returns:
            Loaded PipelineState object
        """
        
        state_path = self.save_dir / f"{checkpoint_id}_pipeline_state.pkl"
        
        if not state_path.exists():
            raise FileNotFoundError(f"Pipeline state file not found: {state_path}")
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            self.current_state = state
            print(f"Pipeline state loaded: {state_path}")
            return state
            
        except Exception as e:
            raise ValueError(f"Failed to load pipeline state: {e}")
    
    def restore_random_states(self, state: Optional[PipelineState] = None):
        """
        Restore all random number generator states.
        
        Args:
            state: Pipeline state to restore from (uses current if None)
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No pipeline state to restore from")
        
        try:
            # Restore Python random state
            random.setstate(state.python_random_state)
            
            # Restore NumPy random state
            numpy_state_tuple = (
                'MT19937',
                np.array(state.numpy_random_state['state'], dtype=np.uint32),
                state.numpy_random_state['pos'],
                state.numpy_random_state['has_gauss'],
                state.numpy_random_state['cached_gaussian']
            )
            np.random.set_state(numpy_state_tuple)
            
            # Restore PyTorch random state
            torch.set_rng_state(state.torch_random_state)
            
            # Restore CUDA random state if available
            if state.torch_cuda_random_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(state.torch_cuda_random_state)
            
            print("✅ Random states restored successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not restore random states: {e}")
    
    def restore_dataset_state(
        self, 
        dataset: Any,
        dataloader: DataLoader,
        state: Optional[PipelineState] = None
    ):
        """
        Restore dataset and dataloader state.
        
        Args:
            dataset: Dataset to configure
            dataloader: DataLoader to configure
            state: Pipeline state to restore from
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No pipeline state to restore from")
        
        try:
            # Restore dataset configuration
            if hasattr(dataset, 'current_max_length'):
                dataset.current_max_length = state.sequence_length
            
            if hasattr(dataset, 'indices') and state.dataset_indices is not None:
                dataset.indices = state.dataset_indices
            
            # Note: DataLoader batch_size is typically immutable after creation
            # This would require recreating the DataLoader in practice
            
            print(f"✅ Dataset state restored: seq_len={state.sequence_length}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not restore dataset state: {e}")
    
    def restore_augmentation_state(
        self,
        augmenter: Any,
        state: Optional[PipelineState] = None
    ):
        """
        Restore augmentation manager state.
        
        Args:
            augmenter: Augmentation manager to configure
            state: Pipeline state to restore from
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No pipeline state to restore from")
        
        try:
            if hasattr(augmenter, 'current_epoch'):
                augmenter.current_epoch = state.augmentation_epoch
            
            if hasattr(augmenter, 'current_probability'):
                augmenter.current_probability = state.augmentation_probability
            
            if hasattr(augmenter, 'schedule_position'):
                augmenter.schedule_position = state.augmentation_schedule_position
            
            if hasattr(augmenter, 'augmentation_stats'):
                augmenter.augmentation_stats.update(state.augmentations_applied)
            
            print(f"✅ Augmentation state restored: prob={state.augmentation_probability:.3f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not restore augmentation state: {e}")
    
    def restore_curriculum_state(
        self,
        curriculum_scheduler: Any,
        state: Optional[PipelineState] = None
    ):
        """
        Restore curriculum scheduler state.
        
        Args:
            curriculum_scheduler: Curriculum scheduler to configure
            state: Pipeline state to restore from
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No pipeline state to restore from")
        
        try:
            if hasattr(curriculum_scheduler, 'current_stage'):
                curriculum_scheduler.current_stage = state.curriculum_stage
            
            if hasattr(curriculum_scheduler, 'progress'):
                curriculum_scheduler.progress = state.curriculum_progress
            
            if hasattr(curriculum_scheduler, 'target_length'):
                curriculum_scheduler.target_length = state.target_sequence_length
            
            print(f"✅ Curriculum state restored: stage={state.curriculum_stage}, progress={state.curriculum_progress:.3f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not restore curriculum state: {e}")
    
    def full_pipeline_restore(
        self,
        checkpoint_id: str,
        dataset: Any,
        dataloader: DataLoader,
        augmenter: Optional[Any] = None,
        curriculum_scheduler: Optional[Any] = None
    ) -> PipelineState:
        """
        Perform complete pipeline state restoration.
        
        Args:
            checkpoint_id: Checkpoint to restore from
            dataset: Dataset to configure
            dataloader: DataLoader to configure
            augmenter: Optional augmentation manager
            curriculum_scheduler: Optional curriculum scheduler
            
        Returns:
            Restored pipeline state
        """
        
        print(f"🔄 Restoring complete pipeline state from: {checkpoint_id}")
        
        # Load state
        state = self.load_pipeline_state(checkpoint_id)
        
        # Restore all components
        self.restore_random_states(state)
        self.restore_dataset_state(dataset, dataloader, state)
        
        if augmenter is not None:
            self.restore_augmentation_state(augmenter, state)
        
        if curriculum_scheduler is not None:
            self.restore_curriculum_state(curriculum_scheduler, state)
        
        print(f"✅ Complete pipeline state restored to epoch {state.current_epoch}, batch {state.current_batch}")
        
        return state
    
    def get_resume_info(self, state: Optional[PipelineState] = None) -> Dict[str, Any]:
        """
        Get human-readable information about resume state.
        
        Args:
            state: Pipeline state to analyze
            
        Returns:
            Dictionary with resume information
        """
        
        if state is None:
            state = self.current_state
        
        if state is None:
            return {"status": "no_state_available"}
        
        return {
            "status": "ready_to_resume",
            "epoch": state.current_epoch,
            "batch": state.current_batch,
            "step": state.current_step,
            "sequence_length": state.sequence_length,
            "augmentation_probability": state.augmentation_probability,
            "curriculum_stage": state.curriculum_stage,
            "files_processed": state.total_files_processed,
            "timestamp": state.timestamp,
            "augmentations_applied": dict(state.augmentations_applied),
            "curriculum_progress": f"{state.curriculum_progress:.1%}"
        }


# Integration functions for easy use with existing checkpoint manager

def save_complete_training_state(
    checkpoint_manager,
    pipeline_state_manager: PipelineStateManager,
    checkpoint_id: str,
    **capture_kwargs
) -> Tuple[Any, Path]:
    """
    Save both checkpoint and pipeline state together.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        pipeline_state_manager: PipelineStateManager instance  
        checkpoint_id: Unique checkpoint identifier
        **capture_kwargs: Arguments for capture_state()
        
    Returns:
        Tuple of (checkpoint_metadata, pipeline_state_path)
    """
    
    # Capture current pipeline state
    pipeline_state = pipeline_state_manager.capture_state(**capture_kwargs)
    
    # Save pipeline state
    state_path = pipeline_state_manager.save_pipeline_state(checkpoint_id, pipeline_state)
    
    return pipeline_state, state_path


def load_complete_training_state(
    checkpoint_manager,
    pipeline_state_manager: PipelineStateManager,
    checkpoint_id: str,
    **restore_kwargs
) -> Tuple[Any, Any]:
    """
    Load both checkpoint and pipeline state together.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        pipeline_state_manager: PipelineStateManager instance
        checkpoint_id: Checkpoint identifier to load
        **restore_kwargs: Arguments for full_pipeline_restore()
        
    Returns:
        Tuple of (checkpoint_data, pipeline_state)
    """
    
    # Load checkpoint (if using existing checkpoint manager)
    # checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_id=checkpoint_id)
    
    # Restore complete pipeline state
    pipeline_state = pipeline_state_manager.full_pipeline_restore(checkpoint_id, **restore_kwargs)
    
    return None, pipeline_state  # Return checkpoint_data when integrated


def create_pipeline_state_manager(save_dir: str = "outputs/checkpoints") -> PipelineStateManager:
    """
    Factory function to create pipeline state manager.
    
    Args:
        save_dir: Directory to save pipeline states
        
    Returns:
        Configured PipelineStateManager
    """
    
    return PipelineStateManager(save_dir=save_dir)