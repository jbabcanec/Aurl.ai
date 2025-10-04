"""
Cumulative Training System with File Tracking

This module manages progressive training where each MIDI file is processed
once and the model continuously learns, maintaining a record of what's been
trained and what remains.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import shutil
from collections import OrderedDict

from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class FileTrainingRecord:
    """Record of a single file's training status."""
    file_path: str
    file_hash: str
    status: str  # 'pending', 'training', 'completed', 'error'
    epochs_trained: int = 0
    last_loss: float = float('inf')
    training_time: float = 0.0
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class CumulativeTrainingState:
    """Complete state of cumulative training."""
    # Training progress
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    current_file: Optional[str] = None

    # Model state
    total_epochs: int = 0
    total_batches: int = 0
    best_loss: float = float('inf')
    best_grammar_score: float = 0.0

    # Timing
    start_time: str = ""
    last_update: str = ""
    total_training_time: float = 0.0

    # File records
    file_records: Dict[str, FileTrainingRecord] = field(default_factory=dict)

    # Training queue
    pending_files: List[str] = field(default_factory=list)
    completed_files_list: List[str] = field(default_factory=list)

    # Checkpoint info
    last_checkpoint_path: Optional[str] = None
    checkpoint_epoch: int = 0


class CumulativeTrainer:
    """
    Manages cumulative training across multiple MIDI files with state persistence.
    """

    def __init__(
        self,
        state_file: str = "outputs/training/cumulative_state.json",
        checkpoint_dir: str = "outputs/checkpoints/cumulative"
    ):
        self.state_file = Path(state_file)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize state
        self.state = self._load_state()

        # File tracking
        self.processed_hashes = set()
        self._update_processed_hashes()

    def _load_state(self) -> CumulativeTrainingState:
        """Load training state from disk or create new."""
        if self.state_file.exists():
            logger.info(f"Loading cumulative state from {self.state_file}")
            with open(self.state_file) as f:
                data = json.load(f)

                # Convert file records
                file_records = {}
                for path, record_data in data.get('file_records', {}).items():
                    file_records[path] = FileTrainingRecord(**record_data)

                # Create state object
                state = CumulativeTrainingState(
                    total_files=data.get('total_files', 0),
                    completed_files=data.get('completed_files', 0),
                    failed_files=data.get('failed_files', 0),
                    current_file=data.get('current_file'),
                    total_epochs=data.get('total_epochs', 0),
                    total_batches=data.get('total_batches', 0),
                    best_loss=data.get('best_loss', float('inf')),
                    best_grammar_score=data.get('best_grammar_score', 0.0),
                    start_time=data.get('start_time', ''),
                    last_update=data.get('last_update', ''),
                    total_training_time=data.get('total_training_time', 0.0),
                    file_records=file_records,
                    pending_files=data.get('pending_files', []),
                    completed_files_list=data.get('completed_files_list', []),
                    last_checkpoint_path=data.get('last_checkpoint_path'),
                    checkpoint_epoch=data.get('checkpoint_epoch', 0)
                )

                logger.info(f"Resumed: {state.completed_files}/{state.total_files} files completed")
                return state
        else:
            logger.info("Starting new cumulative training session")
            return CumulativeTrainingState(start_time=datetime.now().isoformat())

    def _save_state(self):
        """Save current training state to disk."""
        # Convert state to dictionary
        state_dict = {
            'total_files': self.state.total_files,
            'completed_files': self.state.completed_files,
            'failed_files': self.state.failed_files,
            'current_file': self.state.current_file,
            'total_epochs': self.state.total_epochs,
            'total_batches': self.state.total_batches,
            'best_loss': self.state.best_loss,
            'best_grammar_score': self.state.best_grammar_score,
            'start_time': self.state.start_time,
            'last_update': datetime.now().isoformat(),
            'total_training_time': self.state.total_training_time,
            'pending_files': self.state.pending_files,
            'completed_files_list': self.state.completed_files_list,
            'last_checkpoint_path': self.state.last_checkpoint_path,
            'checkpoint_epoch': self.state.checkpoint_epoch,
            'file_records': {
                path: asdict(record)
                for path, record in self.state.file_records.items()
            }
        }

        # Write to temp file then move (atomic operation)
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state_dict, f, indent=2)

        # Move temp to actual
        shutil.move(str(temp_file), str(self.state_file))

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents for change detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _update_processed_hashes(self):
        """Update set of processed file hashes."""
        self.processed_hashes.clear()
        for record in self.state.file_records.values():
            if record.status == 'completed':
                self.processed_hashes.add(record.file_hash)

    def scan_data_directory(self, data_dir: str, pattern: str = "*.mid") -> List[Path]:
        """
        Scan directory for MIDI files and update training queue.

        Args:
            data_dir: Directory containing MIDI files
            pattern: File pattern to match

        Returns:
            List of new files to process
        """
        data_path = Path(data_dir)
        all_files = sorted(data_path.glob(f"**/{pattern}"))

        new_files = []
        updated_files = []

        for file_path in all_files:
            file_str = str(file_path)
            file_hash = self._get_file_hash(file_path)

            # Check if file is new or changed
            if file_str not in self.state.file_records:
                # New file
                record = FileTrainingRecord(
                    file_path=file_str,
                    file_hash=file_hash,
                    status='pending'
                )
                self.state.file_records[file_str] = record
                self.state.pending_files.append(file_str)
                new_files.append(file_path)

            elif self.state.file_records[file_str].file_hash != file_hash:
                # File changed - reset its training
                record = self.state.file_records[file_str]
                if record.status == 'completed':
                    self.state.completed_files -= 1
                    self.state.completed_files_list.remove(file_str)

                record.file_hash = file_hash
                record.status = 'pending'
                record.epochs_trained = 0
                record.completed_at = None

                if file_str not in self.state.pending_files:
                    self.state.pending_files.append(file_str)

                updated_files.append(file_path)

        # Update total files count
        self.state.total_files = len(self.state.file_records)

        # Save state
        self._save_state()

        if new_files:
            logger.info(f"Found {len(new_files)} new files to train")
        if updated_files:
            logger.info(f"Found {len(updated_files)} updated files to retrain")

        return new_files + updated_files

    def get_next_file(self) -> Optional[Tuple[str, FileTrainingRecord]]:
        """
        Get the next file to train.

        Returns:
            Tuple of (file_path, record) or None if no files pending
        """
        while self.state.pending_files:
            file_path = self.state.pending_files[0]
            record = self.state.file_records.get(file_path)

            if record and record.status == 'pending':
                return file_path, record
            else:
                # Remove from pending if already processed
                self.state.pending_files.pop(0)

        return None

    def start_file_training(self, file_path: str):
        """Mark a file as currently being trained."""
        if file_path in self.state.file_records:
            self.state.file_records[file_path].status = 'training'
            self.state.current_file = file_path
            self._save_state()
            logger.info(f"Started training on: {Path(file_path).name}")

    def complete_file_training(
        self,
        file_path: str,
        final_loss: float,
        metrics: Dict,
        training_time: float
    ):
        """Mark a file as completed with metrics."""
        if file_path in self.state.file_records:
            record = self.state.file_records[file_path]
            record.status = 'completed'
            record.last_loss = final_loss
            record.metrics = metrics
            record.training_time = training_time
            record.completed_at = datetime.now().isoformat()

            # Update state
            self.state.completed_files += 1
            self.state.completed_files_list.append(file_path)

            if file_path in self.state.pending_files:
                self.state.pending_files.remove(file_path)

            if self.state.current_file == file_path:
                self.state.current_file = None

            # Update best metrics
            if final_loss < self.state.best_loss:
                self.state.best_loss = final_loss

            grammar_score = metrics.get('grammar_score', 0)
            if grammar_score > self.state.best_grammar_score:
                self.state.best_grammar_score = grammar_score

            self.state.total_training_time += training_time

            self._save_state()

            logger.info(
                f"✅ Completed: {Path(file_path).name} | "
                f"Loss: {final_loss:.4f} | "
                f"Grammar: {grammar_score:.3f} | "
                f"Progress: {self.state.completed_files}/{self.state.total_files}"
            )

    def fail_file_training(self, file_path: str, error_message: str):
        """Mark a file as failed with error."""
        if file_path in self.state.file_records:
            record = self.state.file_records[file_path]
            record.status = 'error'
            record.error_message = error_message

            self.state.failed_files += 1

            if file_path in self.state.pending_files:
                self.state.pending_files.remove(file_path)

            if self.state.current_file == file_path:
                self.state.current_file = None

            self._save_state()

            logger.error(f"❌ Failed: {Path(file_path).name} - {error_message}")

    def update_checkpoint(self, checkpoint_path: str, epoch: int):
        """Update checkpoint information."""
        self.state.last_checkpoint_path = checkpoint_path
        self.state.checkpoint_epoch = epoch
        self._save_state()

    def get_progress_summary(self) -> Dict:
        """Get training progress summary."""
        return {
            'total_files': self.state.total_files,
            'completed': self.state.completed_files,
            'failed': self.state.failed_files,
            'pending': len(self.state.pending_files),
            'progress_percent': (
                self.state.completed_files / self.state.total_files * 100
                if self.state.total_files > 0 else 0
            ),
            'current_file': Path(self.state.current_file).name
                          if self.state.current_file else None,
            'best_loss': self.state.best_loss,
            'best_grammar': self.state.best_grammar_score,
            'total_time_hours': self.state.total_training_time / 3600
        }

    def get_completed_files(self) -> List[str]:
        """Get list of completed file paths."""
        return self.state.completed_files_list.copy()

    def get_pending_files(self) -> List[str]:
        """Get list of pending file paths."""
        return self.state.pending_files.copy()

    def reset_file(self, file_path: str):
        """Reset a file to pending status for retraining."""
        if file_path in self.state.file_records:
            record = self.state.file_records[file_path]

            if record.status == 'completed':
                self.state.completed_files -= 1
                self.state.completed_files_list.remove(file_path)
            elif record.status == 'error':
                self.state.failed_files -= 1

            record.status = 'pending'
            record.epochs_trained = 0
            record.completed_at = None
            record.error_message = None

            if file_path not in self.state.pending_files:
                self.state.pending_files.append(file_path)

            self._save_state()
            logger.info(f"Reset {Path(file_path).name} to pending")

    def export_training_report(self, output_file: str):
        """Export detailed training report."""
        report = {
            'summary': self.get_progress_summary(),
            'completed_files': [
                {
                    'file': Path(f).name,
                    'loss': self.state.file_records[f].last_loss,
                    'metrics': self.state.file_records[f].metrics,
                    'time': self.state.file_records[f].training_time,
                    'completed': self.state.file_records[f].completed_at
                }
                for f in self.state.completed_files_list
            ],
            'failed_files': [
                {
                    'file': Path(f).name,
                    'error': self.state.file_records[f].error_message
                }
                for f, r in self.state.file_records.items()
                if r.status == 'error'
            ],
            'state': asdict(self.state)
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported training report to {output_file}")