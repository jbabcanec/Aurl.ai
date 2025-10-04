"""
Progress Visualization for Cumulative Training

This module provides real-time visualization of training progress,
including which files have been processed and current metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ProgressVisualizer:
    """
    Visualizes training progress with ASCII art and statistics.
    """

    def __init__(self, state_file: str = "outputs/training/cumulative_state.json"):
        self.state_file = Path(state_file)

    def display_progress_bar(
        self,
        completed: int,
        total: int,
        width: int = 50,
        title: str = "Progress"
    ) -> str:
        """Create ASCII progress bar."""
        if total == 0:
            return f"{title}: No files to process"

        percent = completed / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)

        return f"{title}: [{bar}] {completed}/{total} ({percent*100:.1f}%)"

    def display_file_grid(self, file_records: Dict, max_files: int = 100) -> List[str]:
        """
        Create visual grid showing file status.
        ✅ = completed, 🔄 = training, ⏳ = pending, ❌ = error
        """
        lines = []
        lines.append("\n📁 File Status Grid:")
        lines.append("="*60)

        # Group files by status
        completed = []
        training = []
        pending = []
        error = []

        for path, record in list(file_records.items())[:max_files]:
            name = Path(path).stem[:20]  # Truncate long names
            if record['status'] == 'completed':
                completed.append(f"✅ {name}")
            elif record['status'] == 'training':
                training.append(f"🔄 {name}")
            elif record['status'] == 'pending':
                pending.append(f"⏳ {name}")
            elif record['status'] == 'error':
                error.append(f"❌ {name}")

        # Display groups
        if training:
            lines.append("Currently Training:")
            for item in training:
                lines.append(f"  {item}")

        if completed:
            lines.append(f"\nCompleted ({len(completed)}):")
            # Show recent completions
            for item in completed[-5:]:
                lines.append(f"  {item}")
            if len(completed) > 5:
                lines.append(f"  ... and {len(completed)-5} more")

        if pending:
            lines.append(f"\nPending ({len(pending)}):")
            # Show next few
            for item in pending[:3]:
                lines.append(f"  {item}")
            if len(pending) > 3:
                lines.append(f"  ... and {len(pending)-3} more")

        if error:
            lines.append(f"\nErrors ({len(error)}):")
            for item in error:
                lines.append(f"  {item}")

        return lines

    def display_metrics_dashboard(self, state: Dict) -> List[str]:
        """Create metrics dashboard."""
        lines = []
        lines.append("\n📊 Training Metrics Dashboard")
        lines.append("="*60)

        # Training stats
        lines.append("\n🎯 Performance:")
        lines.append(f"  Best Loss:         {state.get('best_loss', float('inf')):.4f}")
        lines.append(f"  Best Grammar:      {state.get('best_grammar_score', 0):.3f}")
        lines.append(f"  Total Epochs:      {state.get('total_epochs', 0)}")
        lines.append(f"  Total Batches:     {state.get('total_batches', 0)}")

        # Time stats
        total_time = state.get('total_training_time', 0)
        hours = total_time / 3600
        lines.append("\n⏱️ Time:")
        lines.append(f"  Training Time:     {hours:.2f} hours")
        lines.append(f"  Started:           {state.get('start_time', 'N/A')}")
        lines.append(f"  Last Update:       {state.get('last_update', 'N/A')}")

        # Current file details
        if state.get('current_file'):
            lines.append(f"\n🎵 Current File:     {Path(state['current_file']).name}")

        # Checkpoint info
        if state.get('last_checkpoint_path'):
            lines.append(f"\n💾 Last Checkpoint:  {Path(state['last_checkpoint_path']).name}")
            lines.append(f"  Checkpoint Epoch:  {state.get('checkpoint_epoch', 0)}")

        return lines

    def display_recent_completions(self, file_records: Dict, limit: int = 5) -> List[str]:
        """Show recently completed files with metrics."""
        lines = []
        lines.append("\n🏆 Recent Completions:")
        lines.append("-"*60)

        # Get completed files sorted by completion time
        completed = [
            (path, record) for path, record in file_records.items()
            if record.get('status') == 'completed' and record.get('completed_at')
        ]
        completed.sort(key=lambda x: x[1].get('completed_at', ''), reverse=True)

        for path, record in completed[:limit]:
            name = Path(path).name
            loss = record.get('last_loss', float('inf'))
            metrics = record.get('metrics', {})
            grammar = metrics.get('grammar_score', 0)
            time_taken = record.get('training_time', 0) / 60  # Convert to minutes

            lines.append(f"  {name[:30]:<30} Loss: {loss:.4f}  Grammar: {grammar:.3f}  Time: {time_taken:.1f}m")

        return lines

    def create_full_display(self) -> str:
        """Create full progress display."""
        if not self.state_file.exists():
            return "No training state file found. Start training to see progress."

        # Load state
        with open(self.state_file) as f:
            state = json.load(f)

        lines = []

        # Header
        lines.append("\n" + "="*60)
        lines.append("🎼 AURL CUMULATIVE TRAINING PROGRESS")
        lines.append("="*60)

        # Progress bar
        completed = state.get('completed_files', 0)
        total = state.get('total_files', 0)
        lines.append("")
        lines.append(self.display_progress_bar(completed, total))

        # File grid
        if 'file_records' in state:
            lines.extend(self.display_file_grid(state['file_records']))

        # Metrics dashboard
        lines.extend(self.display_metrics_dashboard(state))

        # Recent completions
        if 'file_records' in state:
            lines.extend(self.display_recent_completions(state['file_records']))

        # Footer
        lines.append("\n" + "="*60)
        lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(lines)

    def watch_progress(self, refresh_interval: int = 5):
        """
        Watch progress with auto-refresh.

        Args:
            refresh_interval: Seconds between refreshes
        """
        import os

        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')

                # Display progress
                print(self.create_full_display())

                # Wait
                print(f"\nRefreshing in {refresh_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nStopped watching progress.")


def visualize_training_progress():
    """Standalone function to visualize current progress."""
    visualizer = ProgressVisualizer()
    print(visualizer.create_full_display())


def watch_training_progress(refresh_interval: int = 5):
    """Standalone function to watch progress with auto-refresh."""
    visualizer = ProgressVisualizer()
    visualizer.watch_progress(refresh_interval)