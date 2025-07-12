#!/usr/bin/env python3
"""
Preprocess all MIDI data for faster training.

This script preprocesses all MIDI files in the data/raw directory
and caches the results to eliminate the slow lazy loading during training.
"""

import argparse
import time
from pathlib import Path
import logging

from src.utils.config import load_config
from src.data.preprocessor import StreamingPreprocessor, PreprocessingOptions, QuantizationMode, VelocityNormalizationMode
from src.utils.base_logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Preprocess MIDI data for training")
    parser.add_argument("--config", type=str, default="configs/training_configs/quick_test.yaml",
                       help="Config file to use")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing even if cache exists")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(__name__)
    logger.info("Starting MIDI data preprocessing...")
    
    # Load configuration
    config = load_config(args.config)
    data_dir = Path(config.system.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Create preprocessing options optimized for training
    options = PreprocessingOptions(
        quantization_mode=QuantizationMode.GROOVE_PRESERVING,
        quantization_resolution=125,  # milliseconds
        quantization_strength=0.8,
        velocity_normalization=VelocityNormalizationMode.STYLE_PRESERVING,
        preserve_rubato=True,
        enable_chord_detection=False,  # Keep disabled for now
        max_polyphony=8,
        reduce_polyphony=True,
        max_sequence_length=config.data.sequence_length,
        overlap_ratio=0.1,
        cache_processed=True
    )
    
    # Create preprocessor
    preprocessor = StreamingPreprocessor(config, options)
    
    if args.force:
        logger.info("Clearing existing cache...")
        preprocessor.clear_cache()
    
    # Count MIDI files
    midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files to preprocess")
    
    if not midi_files:
        logger.warning("No MIDI files found in data directory")
        return 0
    
    # Process all files
    start_time = time.time()
    processed_count = 0
    cache_hits = 0
    errors = 0
    
    logger.info(f"Processing with {args.max_workers} workers...")
    
    for result in preprocessor.stream_process_directory(data_dir, max_workers=args.max_workers):
        try:
            processed_count += 1
            if result.cache_hit:
                cache_hits += 1
            
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                logger.info(f"Processed {processed_count}/{len(midi_files)} files "
                           f"({rate:.1f} files/sec, {cache_hits} cache hits)")
                
        except Exception as e:
            errors += 1
            logger.error(f"Error processing file: {e}")
    
    # Final statistics
    total_time = time.time() - start_time
    stats = preprocessor.get_statistics()
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {processed_count}")
    logger.info(f"Cache hits: {cache_hits}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per file: {total_time/max(1, processed_count):.3f} seconds")
    logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    logger.info("=" * 60)
    logger.info("Training data is now ready for fast loading!")
    
    return 0

if __name__ == "__main__":
    exit(main())