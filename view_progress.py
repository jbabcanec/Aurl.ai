#!/usr/bin/env python3
"""
View current training progress.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.progress_visualizer import visualize_training_progress

if __name__ == "__main__":
    visualize_training_progress()