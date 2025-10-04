# Project Structure

## Root Directory Organization

```
Aurl/
├── src/                 # Core source code
│   ├── models/          # Model architectures (Transformer, VAE, GAN, etc.)
│   ├── training/        # Training modules and utilities
│   ├── data/            # Data loading and processing
│   └── utils/           # Utility functions
│
├── studies/             # Research and analysis
│   └── chord_analysis/  # Harmonic progression analysis from MIDI
│
├── scripts/             # Utility scripts
│   ├── analysis/        # Data analysis tools
│   ├── monitoring/      # Training progress monitoring
│   ├── training/        # Training pipeline scripts
│   └── data_processing/ # Data preprocessing
│
├── data/                # Data directory
│   ├── raw/             # Raw MIDI files (194 classical pieces)
│   ├── processed/       # Processed tensors
│   └── cache/           # Cached data
│
├── configs/             # Configuration files
│   ├── model/           # Model configurations
│   └── training/        # Training configurations
│
├── tests/               # Unit and integration tests
│
├── docs/                # Documentation
│
├── archive/             # Archived/old files
│
└── train.py             # Main training script
```

## Key Directories

### `/src/` - Core Implementation
- **models/**: Neural network architectures
  - `transformer.py`: Music transformer
  - `vae.py`: Variational autoencoder
  - `kv_cache.py`: KV-cache for fast generation
- **training/**: Training infrastructure
  - `dissolvable_guidance.py`: Guidance that fades over time
  - `cumulative_trainer.py`: Track training across files
  - `progress_visualizer.py`: Training progress tracking

### `/studies/` - Research & Analysis
- **chord_analysis/**: Complete harmonic analysis
  - Analyzed 194 MIDI files
  - 59,657 chord events
  - Progression probability diagrams

### `/scripts/` - Tools & Utilities
- **analysis/**: MIDI and model output analysis
- **monitoring/**: Real-time training monitoring
- **training/**: Various training approaches

### `/data/raw/` - Dataset
- 194 classical MIDI files
- Composers: Beethoven, Mozart, Chopin, Bach, Debussy, Schubert, etc.
- ~60,000 chord events total

## Important Files

- `train.py` - Main training entry point
- `generate.py` - Music generation script
- `GAMEPLAN.md` - Project roadmap
- `README.md` - Project overview

## Recent Additions

1. **Chord Analysis Study** (`/studies/chord_analysis/`)
   - Complete analysis of all MIDI files
   - Progression probability visualizations
   - Composer-specific harmonic signatures

2. **Training Enhancements** (`/src/training/`)
   - Dissolvable guidance system
   - Cumulative training tracker
   - KV-cache for 10-50x faster generation

3. **Monitoring Tools** (`/scripts/monitoring/`)
   - Real-time progress viewer
   - Auto-refresh training monitor