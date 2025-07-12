# Scripts Directory

This directory contains utility scripts for the Aurl.ai project.

## Structure

```
scripts/
├── training/           # Training pipeline scripts
│   ├── train_pipeline.py    # Original full training pipeline (deprecated)
│   ├── train_progressive.py # Progressive training (transformer→VAE→GAN)
│   └── train_enhanced.py    # Enhanced training with monitoring
├── data/              # Data processing scripts
│   └── preprocess_data.py   # Data preprocessing utilities
├── analyze_sequence_lengths.py  # Analyze MIDI sequence lengths
├── calculate_vocab_size.py      # Calculate vocabulary size
└── create_combined_dashboard.py # Create training dashboard
```

## Training Scripts

- **train_simple.py** (in root): Recommended simple, fast training script
- **train_progressive.py**: Multi-stage training (transformer→VAE→VAE-GAN)
- **train_enhanced.py**: Full-featured training with comprehensive monitoring
- **train_pipeline.py**: Original training pipeline (legacy)

## Usage

Most scripts can be run from the project root:

```bash
# Simple training (recommended)
python train_simple.py --epochs 5 --max-samples 100

# Progressive training
python scripts/training/train_progressive.py

# Enhanced training with monitoring
python scripts/training/train_enhanced.py
```