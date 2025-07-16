"""
Dataset classes for Aurl.ai with lazy loading and efficient memory management.

This module provides PyTorch Dataset implementations that handle large musical
datasets without loading everything into memory.
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from src.data.midi_parser import load_midi_file, MidiData, StreamingMidiParser
from src.data.representation import (
    MusicRepresentationConverter, MusicalRepresentation, 
    VocabularyConfig, PianoRollConfig, MusicalMetadata
)
from src.data.augmentation import MusicAugmenter, AugmentationConfig
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class LazyMidiDataset(Dataset):
    """
    Lazy-loading MIDI dataset that processes files on-demand.
    
    Designed for scalability with large datasets. Only loads and processes
    MIDI files when they are actually needed for training.
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 cache_dir: Union[str, Path] = None,
                 vocab_config: VocabularyConfig = None,
                 piano_roll_config: PianoRollConfig = None,
                 sequence_length: int = 2048,
                 max_sequence_length: Optional[int] = None,
                 overlap: int = 256,
                 file_extensions: List[str] = None,
                 max_files: Optional[int] = None,
                 enable_caching: bool = True,
                 truncation_strategy: str = "sliding_window",
                 curriculum_learning: bool = False,
                 sequence_length_schedule: Optional[List[int]] = None,
                 current_epoch: int = 0,
                 enable_augmentation: bool = True,
                 augmentation_config: Optional[AugmentationConfig] = None,
                 augmentation_probability: float = 0.5):
        """
        Initialize lazy MIDI dataset.
        
        Args:
            data_dir: Directory containing MIDI files
            cache_dir: Directory for caching processed data
            vocab_config: Vocabulary configuration for tokenization
            piano_roll_config: Piano roll configuration
            sequence_length: Base length of sequences for training
            max_sequence_length: Maximum allowed sequence length (None = no limit)
            overlap: Overlap between sequences
            file_extensions: Allowed file extensions
            max_files: Maximum number of files to include (for testing)
            enable_caching: Whether to cache processed representations
            truncation_strategy: How to handle long sequences ("sliding_window", "truncate", "adaptive")
            curriculum_learning: Enable progressive sequence length increase
            sequence_length_schedule: List of sequence lengths for curriculum learning
            current_epoch: Current training epoch (for curriculum learning)
            enable_augmentation: Whether to enable on-the-fly data augmentation
            augmentation_config: Configuration for data augmentation
            augmentation_probability: Base probability of applying augmentation
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir or (self.data_dir.parent / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.vocab_config = vocab_config or VocabularyConfig()
        self.piano_roll_config = piano_roll_config or PianoRollConfig()
        self.converter = MusicRepresentationConverter(
            self.vocab_config, self.piano_roll_config
        )
        
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.overlap = overlap
        self.enable_caching = enable_caching
        self.truncation_strategy = truncation_strategy
        self.curriculum_learning = curriculum_learning
        self.sequence_length_schedule = sequence_length_schedule or [512, 1024, 2048]
        self.current_epoch = current_epoch
        
        # Augmentation setup
        self.enable_augmentation = enable_augmentation
        self.augmentation_probability = augmentation_probability
        self.augmentation_config = augmentation_config or AugmentationConfig()
        
        # Initialize augmenter if enabled
        if self.enable_augmentation:
            self.augmenter = MusicAugmenter(self.augmentation_config)
            logger.info(f"Augmentation enabled with probability {augmentation_probability}")
        else:
            self.augmenter = None
            logger.info("Augmentation disabled")
        
        # Determine effective sequence length
        self.effective_sequence_length = self._get_effective_sequence_length()
        
        # File discovery
        self.file_extensions = file_extensions or ['.mid', '.midi']
        self.midi_files = self._discover_midi_files(max_files)
        
        # Validate configuration to prevent division by zero
        self._validate_configuration()
        
        # Sequence mapping (file_idx, start_idx, end_idx)
        self.sequences = []
        self._build_sequence_index()
        
        logger.info(f"LazyMidiDataset initialized with {len(self.midi_files)} files, "
                   f"{len(self.sequences)} sequences, effective_length={self.effective_sequence_length}")
    
    def _discover_midi_files(self, max_files: Optional[int] = None) -> List[Path]:
        """Discover MIDI files in the data directory."""
        midi_files = []
        
        for ext in self.file_extensions:
            pattern = f"**/*{ext}"
            files = list(self.data_dir.glob(pattern))
            midi_files.extend(files)
        
        # Sort for consistent ordering
        midi_files.sort()
        
        if max_files:
            midi_files = midi_files[:max_files]
        
        logger.info(f"Discovered {len(midi_files)} MIDI files")
        return midi_files
    
    def _validate_configuration(self):
        """Validate dataset configuration to prevent common errors."""
        # Check for potential division by zero in step_size calculation
        if self.effective_sequence_length <= self.overlap:
            logger.warning(f"Potential configuration issue detected:")
            logger.warning(f"  effective_sequence_length ({self.effective_sequence_length}) <= overlap ({self.overlap})")
            logger.warning(f"  This will cause step_size <= 0 and trigger fallback behavior")
            logger.warning(f"  Consider reducing overlap or increasing sequence_length/max_sequence_length")
        
        # Additional validations
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        
        if self.effective_sequence_length <= 0:
            raise ValueError(f"effective_sequence_length must be positive, got {self.effective_sequence_length}")
        
        logger.debug(f"Configuration validated: effective_length={self.effective_sequence_length}, overlap={self.overlap}")
    
    def _get_effective_sequence_length(self) -> int:
        """Get the effective sequence length considering curriculum learning."""
        if not self.curriculum_learning or not self.sequence_length_schedule:
            base_length = self.sequence_length
        else:
            # Curriculum learning: gradually increase sequence length
            schedule_idx = min(self.current_epoch // 10, len(self.sequence_length_schedule) - 1)
            base_length = self.sequence_length_schedule[schedule_idx]
            logger.debug(f"Curriculum learning: epoch {self.current_epoch}, using length {base_length}")
        
        # Apply maximum sequence length constraint
        if self.max_sequence_length is not None:
            return min(base_length, self.max_sequence_length)
        return base_length
    
    def update_epoch(self, epoch: int):
        """Update current epoch for curriculum learning and augmentation scheduling."""
        if epoch != self.current_epoch:
            old_epoch = self.current_epoch
            self.current_epoch = epoch
            
            # Update curriculum learning if enabled
            if self.curriculum_learning:
                old_length = self.effective_sequence_length
                self.effective_sequence_length = self._get_effective_sequence_length()
                
                if old_length != self.effective_sequence_length:
                    logger.info(f"Curriculum learning: Updated sequence length from {old_length} to {self.effective_sequence_length}")
                    # Rebuild sequence index with new length
                    self.sequences = []
                    self._build_sequence_index()
            
            # Update augmentation probability based on epoch (progressive augmentation)
            if self.enable_augmentation:
                self._update_augmentation_schedule()
                
            logger.debug(f"Dataset epoch updated: {old_epoch} -> {epoch}")
    
    def _update_augmentation_schedule(self):
        """Update augmentation probability based on training progress."""
        # Progressive augmentation: start with lower probability, increase over time
        # This allows the model to learn basic patterns before adding complexity
        base_prob = self.augmentation_probability
        
        # Increase augmentation probability over the first 20% of training
        if self.current_epoch < 20:
            # Start at 25% of base probability, linearly increase
            schedule_factor = 0.25 + (0.75 * (self.current_epoch / 20))
            current_prob = base_prob * schedule_factor
        else:
            # After epoch 20, use full augmentation probability
            current_prob = base_prob
        
        # Update augmenter probability
        if hasattr(self.augmenter, 'set_global_probability'):
            self.augmenter.set_global_probability(current_prob)
        
        logger.debug(f"Augmentation probability updated to {current_prob:.3f} for epoch {self.current_epoch}")
    
    def get_augmentation_state(self) -> Dict[str, Any]:
        """Get current augmentation state for checkpointing."""
        if not self.enable_augmentation or not self.augmenter:
            return {}
        
        return {
            'enabled': self.enable_augmentation,
            'probability': self.augmentation_probability,
            'current_epoch': self.current_epoch,
            'config': self.augmentation_config.__dict__ if self.augmentation_config else {},
            'augmenter_state': getattr(self.augmenter, 'get_state', lambda: {})()
        }
    
    def set_augmentation_state(self, state: Dict[str, Any]):
        """Restore augmentation state from checkpoint."""
        if not state or not self.enable_augmentation:
            return
        
        self.augmentation_probability = state.get('probability', 0.5)
        self.current_epoch = state.get('current_epoch', 0)
        
        if self.augmenter and 'augmenter_state' in state:
            if hasattr(self.augmenter, 'set_state'):
                self.augmenter.set_state(state['augmenter_state'])
        
        # Update augmentation schedule based on restored epoch
        self._update_augmentation_schedule()
        
        logger.info(f"Restored augmentation state for epoch {self.current_epoch}")
    
    def _build_sequence_index(self):
        """Build an index of all sequences across all files."""
        logger.info("Building sequence index...")
        
        for file_idx, midi_file in enumerate(self.midi_files):
            try:
                # Get file info without full parsing
                file_info = self._get_file_info(midi_file)
                estimated_tokens = file_info.get('estimated_tokens', 1000)
                
                # Calculate number of sequences for this file with length limiting
                effective_length = self.effective_sequence_length
                
                if estimated_tokens <= effective_length:
                    # Small file - one sequence
                    self.sequences.append((file_idx, 0, estimated_tokens))
                else:
                    # Large file - apply truncation strategy
                    if self.truncation_strategy == "sliding_window":
                        # Multiple overlapping sequences
                        step_size = effective_length - self.overlap
                        
                        # Prevent division by zero: ensure step_size > 0
                        if step_size <= 0:
                            logger.warning(f"Invalid step_size={step_size} for {midi_file}. "
                                         f"effective_length={effective_length}, overlap={self.overlap}. "
                                         f"Using single sequence fallback.")
                            # Fallback: create single sequence without overlap
                            self.sequences.append((file_idx, 0, min(effective_length, estimated_tokens)))
                            continue
                        
                        max_sequences = (estimated_tokens - self.overlap) // step_size
                        
                        # Limit number of sequences to prevent memory explosion
                        max_sequences_per_file = min(max_sequences, 20)  # Cap at 20 sequences per file
                        
                        for seq_idx in range(max_sequences_per_file):
                            start_idx = seq_idx * step_size
                            end_idx = min(start_idx + effective_length, estimated_tokens)
                            self.sequences.append((file_idx, start_idx, end_idx))
                    
                    elif self.truncation_strategy == "truncate":
                        # Single sequence, truncated to max length
                        self.sequences.append((file_idx, 0, effective_length))
                    
                    elif self.truncation_strategy == "adaptive":
                        # Adaptive strategy: use sliding window for moderately long pieces,
                        # truncate for extremely long pieces
                        if estimated_tokens <= effective_length * 3:
                            # Moderately long: sliding window
                            step_size = effective_length - self.overlap
                            
                            # Prevent division by zero: ensure step_size > 0
                            if step_size <= 0:
                                logger.warning(f"Invalid step_size={step_size} for {midi_file} in adaptive mode. "
                                             f"effective_length={effective_length}, overlap={self.overlap}. "
                                             f"Using truncate fallback.")
                                # Fallback: single sequence, truncated
                                self.sequences.append((file_idx, 0, min(effective_length, estimated_tokens)))
                                continue
                            
                            num_sequences = min(3, (estimated_tokens - self.overlap) // step_size)
                            
                            for seq_idx in range(num_sequences):
                                start_idx = seq_idx * step_size
                                end_idx = min(start_idx + effective_length, estimated_tokens)
                                self.sequences.append((file_idx, start_idx, end_idx))
                        else:
                            # Very long: truncate to avoid memory issues
                            self.sequences.append((file_idx, 0, effective_length))
            
            except Exception as e:
                logger.warning(f"Failed to index {midi_file}: {e}")
                continue
        
        logger.info(f"Built sequence index: {len(self.sequences)} sequences")
    
    def _get_file_info(self, midi_file: Path) -> Dict[str, Any]:
        """Get basic file information for indexing."""
        cache_key = self._get_cache_key(midi_file, "info")
        cache_file = self.cache_dir / f"{cache_key}.info"
        
        if self.enable_caching and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # Fall through to recalculate
        
        # Calculate file info
        try:
            streaming_parser = StreamingMidiParser()
            file_info = streaming_parser.get_file_info(midi_file)
            
            # Estimate number of tokens (rough approximation)
            estimated_notes = file_info.get('num_messages', 1000) // 4  # Rough estimate
            estimated_tokens = estimated_notes * 3  # Note on + note off + time shift
            file_info['estimated_tokens'] = estimated_tokens
            
            # Cache the info
            if self.enable_caching:
                with open(cache_file, 'wb') as f:
                    pickle.dump(file_info, f)
            
            return file_info
            
        except Exception as e:
            logger.warning(f"Failed to get info for {midi_file}: {e}")
            return {'estimated_tokens': 1000}  # Default fallback
    
    def _get_cache_key(self, midi_file: Path, suffix: str = "") -> str:
        """Generate cache key for a file."""
        # Include file path, modification time, and config in hash
        file_stat = midi_file.stat()
        content = f"{midi_file}_{file_stat.st_mtime}_{file_stat.st_size}"
        content += f"_{self.vocab_config.vocab_size}_{self.sequence_length}"
        if suffix:
            content += f"_{suffix}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_representation(self, midi_file: Path) -> MusicalRepresentation:
        """Load and convert MIDI file to representation with caching."""
        cache_key = self._get_cache_key(midi_file, "repr")
        cache_file = self.cache_dir / f"{cache_key}.repr"
        
        # Try to load from cache
        if self.enable_caching and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache miss for {midi_file}: {e}")
        
        # Load and process the file
        try:
            midi_data = load_midi_file(midi_file)
            representation = self.converter.midi_to_representation(midi_data)
            
            # Add metadata
            representation.metadata = MusicalMetadata(
                source_file=str(midi_file),
                title=midi_file.stem
            )
            
            # Cache the representation
            if self.enable_caching:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(representation, f)
                except Exception as e:
                    logger.warning(f"Failed to cache {midi_file}: {e}")
            
            return representation
            
        except Exception as e:
            logger.error(f"Failed to load {midi_file}: {e}")
            # Return empty representation as fallback
            return MusicalRepresentation()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with optional augmentation."""
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
        
        file_idx, start_idx, end_idx = self.sequences[idx]
        midi_file = self.midi_files[file_idx]
        
        # Initialize augmentation tracking
        augmentation_applied = {}
        original_midi_data = None
        
        try:
            # Load representation
            representation = self._load_representation(midi_file)
            
            # Apply augmentation if enabled and probability check passes
            should_augment = (self.enable_augmentation and 
                            self.augmenter is not None and 
                            np.random.random() < self.augmentation_probability)
            
            if should_augment:
                try:
                    # Load original MIDI data for augmentation
                    original_midi_data = load_midi_file(midi_file)
                    
                    # Apply augmentation
                    augmented_midi, augmentation_applied = self.augmenter.augment(original_midi_data, self.current_epoch)
                    
                    # Convert augmented MIDI to representation
                    representation = self.converter.midi_to_representation(augmented_midi)
                    representation.metadata = MusicalMetadata(
                        source_file=str(midi_file),
                        title=midi_file.stem,
                        augmented=True,
                        augmentation_info=augmentation_applied
                    )
                    
                except Exception as e:
                    logger.warning(f"Augmentation failed for {midi_file}: {e}, using original")
                    augmentation_applied = {}
            
            # Extract sequence
            if representation.tokens is not None and len(representation.tokens) > 0:
                tokens = representation.tokens
                
                # Handle sequence extraction
                actual_start = min(start_idx, len(tokens) - 1)
                actual_end = min(end_idx, len(tokens))
                
                if actual_start >= actual_end:
                    # Edge case: create minimal sequence
                    sequence = np.array([0, 1], dtype=np.int32)  # START + END tokens
                else:
                    sequence = tokens[actual_start:actual_end]
                
                # Pad or truncate to target length
                target_length = self.effective_sequence_length
                if len(sequence) < target_length:
                    # Pad with PAD tokens (token 2)
                    pad_length = target_length - len(sequence)
                    sequence = np.concatenate([
                        sequence, 
                        np.full(pad_length, 2, dtype=np.int32)  # PAD token
                    ])
                elif len(sequence) > target_length:
                    sequence = sequence[:target_length]
                
                # Get corresponding piano roll segment
                piano_roll_segment = None
                if representation.piano_roll is not None:
                    # Calculate time-based segment for piano roll
                    time_per_token = representation.duration / len(representation.tokens)
                    start_time = actual_start * time_per_token
                    end_time = actual_end * time_per_token
                    
                    start_step = self.piano_roll_config.time_to_step(start_time)
                    end_step = self.piano_roll_config.time_to_step(end_time)
                    
                    if start_step < representation.piano_roll.shape[0]:
                        end_step = min(end_step, representation.piano_roll.shape[0])
                        piano_roll_segment = representation.piano_roll[start_step:end_step]
                
            else:
                # Fallback for empty representation
                target_length = self.effective_sequence_length
                sequence = np.array([0, 1] + [2] * (target_length - 2), dtype=np.int32)
                piano_roll_segment = None
            
            # Prepare return data with augmentation information
            result = {
                'tokens': torch.from_numpy(sequence).long(),
                'file_path': str(midi_file),
                'sequence_idx': idx,
                'augmented': should_augment,
                'augmentation_info': augmentation_applied
            }
            
            if piano_roll_segment is not None:
                result['piano_roll'] = torch.from_numpy(piano_roll_segment).float()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get item {idx} from {midi_file}: {e}")
            
            # Return minimal fallback
            target_length = self.effective_sequence_length
            fallback_tokens = np.array([0, 1] + [2] * (target_length - 2), dtype=np.int32)
            return {
                'tokens': torch.from_numpy(fallback_tokens).long(),
                'file_path': str(midi_file),
                'sequence_idx': idx,
                'augmented': False,
                'augmentation_info': {}
            }
    
    def get_file_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        stats = {
            'num_files': len(self.midi_files),
            'num_sequences': len(self.sequences),
            'total_size_mb': 0,
            'file_sizes': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        for midi_file in self.midi_files:
            try:
                size_mb = midi_file.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                stats['file_sizes'].append(size_mb)
            except Exception:
                continue
        
        stats['avg_file_size_mb'] = stats['total_size_mb'] / max(1, len(stats['file_sizes']))
        stats['sequence_length'] = self.sequence_length
        stats['effective_sequence_length'] = self.effective_sequence_length
        stats['max_sequence_length'] = self.max_sequence_length
        stats['truncation_strategy'] = self.truncation_strategy
        stats['curriculum_learning'] = self.curriculum_learning
        stats['current_epoch'] = self.current_epoch
        stats['vocab_size'] = self.vocab_config.vocab_size
        
        # Augmentation statistics
        stats['augmentation_enabled'] = self.enable_augmentation
        stats['augmentation_probability'] = self.augmentation_probability
        if self.enable_augmentation and self.augmentation_config:
            stats['augmentation_config'] = {
                'transpose_range': self.augmentation_config.transpose_range,
                'time_stretch_range': self.augmentation_config.time_stretch_range,
                'velocity_scale_range': self.augmentation_config.velocity_scale_range,
                'transpose_prob': self.augmentation_config.transpose_probability,
                'time_stretch_prob': self.augmentation_config.time_stretch_probability,
                'velocity_scale_prob': self.augmentation_config.velocity_scale_probability
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the cache directory."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            logger.info("Cache cleared")


def midi_collate_fn(batch):
    """Custom collate function to handle variable-length piano rolls."""
    tokens = torch.stack([item['tokens'] for item in batch])
    file_paths = [item['file_path'] for item in batch]
    sequence_indices = torch.tensor([item['sequence_idx'] for item in batch])
    
    # Handle augmentation metadata
    augmented = torch.tensor([item.get('augmented', False) for item in batch], dtype=torch.bool)
    augmentation_info = [item.get('augmentation_info', {}) for item in batch]
    
    result = {
        'tokens': tokens,
        'file_paths': file_paths,
        'sequence_indices': sequence_indices,
        'augmented': augmented,
        'augmentation_info': augmentation_info
    }
    
    # Handle piano roll if present
    if 'piano_roll' in batch[0] and batch[0]['piano_roll'] is not None:
        # Find max time dimension and verify pitch dimensions match
        max_time = max(item['piano_roll'].shape[0] for item in batch 
                      if 'piano_roll' in item and item['piano_roll'] is not None)
        
        # Get pitch dimension from first non-None piano roll
        pitch_dim = next(item['piano_roll'].shape[1] for item in batch 
                        if 'piano_roll' in item and item['piano_roll'] is not None)
        
        # Pad piano rolls to same time dimension
        piano_rolls = []
        for item in batch:
            if 'piano_roll' in item and item['piano_roll'] is not None:
                pr = item['piano_roll']
                # Verify pitch dimensions match
                if pr.shape[1] != pitch_dim:
                    raise ValueError(f"Piano roll pitch dimension mismatch: {pr.shape[1]} vs {pitch_dim}")
                
                if pr.shape[0] < max_time:
                    # Pad with zeros
                    pad_shape = (max_time - pr.shape[0], pr.shape[1])
                    padding = torch.zeros(pad_shape, dtype=pr.dtype)
                    pr = torch.cat([pr, padding], dim=0)
                piano_rolls.append(pr)
            else:
                # Create empty piano roll with correct pitch dimension
                piano_rolls.append(torch.zeros((max_time, pitch_dim), dtype=torch.float32))
        
        result['piano_roll'] = torch.stack(piano_rolls)
    
    return result


def create_dataloader(dataset: LazyMidiDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for the MIDI dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=midi_collate_fn
    )