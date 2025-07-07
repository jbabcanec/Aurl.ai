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
from src.utils.logger import setup_logger

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
                 overlap: int = 256,
                 file_extensions: List[str] = None,
                 max_files: Optional[int] = None,
                 enable_caching: bool = True):
        """
        Initialize lazy MIDI dataset.
        
        Args:
            data_dir: Directory containing MIDI files
            cache_dir: Directory for caching processed data
            vocab_config: Vocabulary configuration for tokenization
            piano_roll_config: Piano roll configuration
            sequence_length: Length of sequences for training
            overlap: Overlap between sequences
            file_extensions: Allowed file extensions
            max_files: Maximum number of files to include (for testing)
            enable_caching: Whether to cache processed representations
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
        self.overlap = overlap
        self.enable_caching = enable_caching
        
        # File discovery
        self.file_extensions = file_extensions or ['.mid', '.midi']
        self.midi_files = self._discover_midi_files(max_files)
        
        # Sequence mapping (file_idx, start_idx, end_idx)
        self.sequences = []
        self._build_sequence_index()
        
        logger.info(f"LazyMidiDataset initialized with {len(self.midi_files)} files, "
                   f"{len(self.sequences)} sequences")
    
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
    
    def _build_sequence_index(self):
        """Build an index of all sequences across all files."""
        logger.info("Building sequence index...")
        
        for file_idx, midi_file in enumerate(self.midi_files):
            try:
                # Get file info without full parsing
                file_info = self._get_file_info(midi_file)
                estimated_tokens = file_info.get('estimated_tokens', 1000)
                
                # Calculate number of sequences for this file
                if estimated_tokens <= self.sequence_length:
                    # Small file - one sequence
                    self.sequences.append((file_idx, 0, estimated_tokens))
                else:
                    # Large file - multiple overlapping sequences
                    step_size = self.sequence_length - self.overlap
                    num_sequences = (estimated_tokens - self.overlap) // step_size
                    
                    for seq_idx in range(num_sequences):
                        start_idx = seq_idx * step_size
                        end_idx = min(start_idx + self.sequence_length, estimated_tokens)
                        self.sequences.append((file_idx, start_idx, end_idx))
            
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
        """Get a training sample."""
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.sequences)}")
        
        file_idx, start_idx, end_idx = self.sequences[idx]
        midi_file = self.midi_files[file_idx]
        
        try:
            # Load representation
            representation = self._load_representation(midi_file)
            
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
                if len(sequence) < self.sequence_length:
                    # Pad with PAD tokens (token 2)
                    pad_length = self.sequence_length - len(sequence)
                    sequence = np.concatenate([
                        sequence, 
                        np.full(pad_length, 2, dtype=np.int32)  # PAD token
                    ])
                elif len(sequence) > self.sequence_length:
                    sequence = sequence[:self.sequence_length]
                
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
                sequence = np.array([0, 1] + [2] * (self.sequence_length - 2), dtype=np.int32)
                piano_roll_segment = None
            
            # Prepare return data
            result = {
                'tokens': torch.from_numpy(sequence).long(),
                'file_path': str(midi_file),
                'sequence_idx': idx
            }
            
            if piano_roll_segment is not None:
                result['piano_roll'] = torch.from_numpy(piano_roll_segment).float()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get item {idx} from {midi_file}: {e}")
            
            # Return minimal fallback
            fallback_tokens = np.array([0, 1] + [2] * (self.sequence_length - 2), dtype=np.int32)
            return {
                'tokens': torch.from_numpy(fallback_tokens).long(),
                'file_path': str(midi_file),
                'sequence_idx': idx
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
        stats['vocab_size'] = self.vocab_config.vocab_size
        
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


def create_dataloader(dataset: LazyMidiDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for the MIDI dataset."""
    
    def collate_fn(batch):
        """Custom collate function to handle variable-length piano rolls."""
        tokens = torch.stack([item['tokens'] for item in batch])
        file_paths = [item['file_path'] for item in batch]
        sequence_indices = torch.tensor([item['sequence_idx'] for item in batch])
        
        result = {
            'tokens': tokens,
            'file_paths': file_paths,
            'sequence_indices': sequence_indices
        }
        
        # Handle piano roll if present
        if 'piano_roll' in batch[0] and batch[0]['piano_roll'] is not None:
            # Find max time dimension
            max_time = max(item['piano_roll'].shape[0] for item in batch 
                          if 'piano_roll' in item and item['piano_roll'] is not None)
            
            # Pad piano rolls to same time dimension
            piano_rolls = []
            for item in batch:
                if 'piano_roll' in item and item['piano_roll'] is not None:
                    pr = item['piano_roll']
                    if pr.shape[0] < max_time:
                        # Pad with zeros
                        pad_shape = (max_time - pr.shape[0], pr.shape[1])
                        padding = torch.zeros(pad_shape, dtype=pr.dtype)
                        pr = torch.cat([pr, padding], dim=0)
                    piano_rolls.append(pr)
                else:
                    # Create empty piano roll
                    piano_rolls.append(torch.zeros((max_time, 88), dtype=torch.float32))
            
            result['piano_roll'] = torch.stack(piano_rolls)
        
        return result
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )