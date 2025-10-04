#!/usr/bin/env python3
"""
Watch training progress with auto-refresh.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.progress_visualizer import watch_training_progress

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch training progress")
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )
    args = parser.parse_args()

    watch_training_progress(args.refresh)