#!/usr/bin/env python3
"""
View current training progress.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.progress_visualizer import visualize_training_progress

if __name__ == "__main__":
    visualize_training_progress()