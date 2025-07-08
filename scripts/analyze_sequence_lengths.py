"""
Phase 3.1 Sequence Length Analysis - Critical Risk Assessment

Analyzes real MIDI sequence length distributions to inform model architecture decisions.
This addresses the identified risk of sequence length explosion with 32nd note precision.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.midi_parser import load_midi_file, MidiData
from src.data.representation import MusicRepresentationConverter, VocabularyConfig
from src.data.preprocessor import StreamingPreprocessor, PreprocessingOptions
from src.utils.config import MidiFlyConfig


@dataclass
class SequenceLengthAnalysis:
    """Results of sequence length analysis."""
    file_count: int
    total_duration: float
    avg_duration: float
    
    # Token sequence lengths
    token_lengths: List[int]
    avg_token_length: float
    max_token_length: int
    min_token_length: int
    
    # Timing resolution analysis
    simple_pieces: int
    complex_pieces: int
    
    # Memory projections
    memory_per_sequence_mb: float
    memory_per_batch_mb: float
    projected_gpu_memory_gb: float
    
    # Recommendations
    recommended_max_length: int
    recommended_batch_size: int
    truncation_loss_percent: float


def analyze_midi_sequence_lengths(midi_dir: Path, max_files: int = 50) -> SequenceLengthAnalysis:
    """
    Analyze sequence lengths from real MIDI files to inform model architecture.
    
    Args:
        midi_dir: Directory containing MIDI files
        max_files: Maximum number of files to analyze
        
    Returns:
        SequenceLengthAnalysis with detailed metrics
    """
    print(f"🔍 Analyzing sequence lengths from {midi_dir}")
    print(f"📁 Processing up to {max_files} MIDI files...")
    
    # Initialize components
    config = MidiFlyConfig()
    preprocessing_options = PreprocessingOptions()
    preprocessor = StreamingPreprocessor(config, preprocessing_options)
    vocab_config = VocabularyConfig()
    
    # Collect data
    sequence_data = []
    durations = []
    simple_count = 0
    complex_count = 0
    
    midi_files = list(midi_dir.glob("*.mid"))[:max_files]
    
    for i, midi_file in enumerate(midi_files):
        print(f"  Processing {i+1}/{len(midi_files)}: {midi_file.name}")
        
        try:
            # Load and preprocess MIDI
            midi_data = load_midi_file(midi_file)
            if not midi_data or not midi_data.instruments:
                continue
            
            # Check complexity
            complexity = vocab_config.detect_piece_complexity(midi_data)
            if complexity == "simple":
                simple_count += 1
            else:
                complex_count += 1
            
            # Convert to representation
            representation = preprocessor.converter.midi_to_representation(midi_data)
            
            # Collect metrics
            token_count = len(representation.tokens) if representation.tokens is not None else 0
            duration = midi_data.end_time
            
            sequence_data.append({
                'file': midi_file.name,
                'duration': duration,
                'token_count': token_count,
                'complexity': complexity,
                'note_count': sum(len(inst.notes) for inst in midi_data.instruments),
                'instrument_count': len(midi_data.instruments)
            })
            
            durations.append(duration)
            
        except Exception as e:
            print(f"    ⚠️  Error processing {midi_file.name}: {e}")
            continue
    
    if not sequence_data:
        raise ValueError("No valid MIDI files found for analysis")
    
    # Calculate statistics
    token_lengths = [d['token_count'] for d in sequence_data]
    avg_duration = np.mean(durations)
    avg_token_length = np.mean(token_lengths)
    max_token_length = max(token_lengths)
    min_token_length = min(token_lengths)
    
    # Memory calculations (assuming float32 embeddings)
    embedding_dim = 512  # Typical transformer embedding size
    memory_per_token_bytes = 4 * embedding_dim  # 4 bytes per float32 * embedding_dim
    memory_per_sequence_mb = (avg_token_length * memory_per_token_bytes) / 1024 / 1024
    
    # Batch size calculations
    target_gpu_memory_gb = 16  # Target <16GB per GPU
    available_memory_mb = target_gpu_memory_gb * 1024 * 0.8  # 80% utilization
    max_batch_size = int(available_memory_mb / memory_per_sequence_mb)
    
    # Recommendations
    # Target sequence length that keeps 95% of sequences
    sorted_lengths = sorted(token_lengths)
    percentile_95 = sorted_lengths[int(len(sorted_lengths) * 0.95)]
    recommended_max_length = min(percentile_95, 2048)  # Cap at 2048 for memory
    
    # Calculate truncation loss
    truncated_count = sum(1 for length in token_lengths if length > recommended_max_length)
    truncation_loss_percent = (truncated_count / len(token_lengths)) * 100
    
    # Final batch size with recommended length
    recommended_memory_mb = (recommended_max_length * memory_per_token_bytes) / 1024 / 1024
    recommended_batch_size = max(1, int(available_memory_mb / recommended_memory_mb))
    
    return SequenceLengthAnalysis(
        file_count=len(sequence_data),
        total_duration=sum(durations),
        avg_duration=avg_duration,
        token_lengths=token_lengths,
        avg_token_length=avg_token_length,
        max_token_length=max_token_length,
        min_token_length=min_token_length,
        simple_pieces=simple_count,
        complex_pieces=complex_count,
        memory_per_sequence_mb=memory_per_sequence_mb,
        memory_per_batch_mb=memory_per_sequence_mb * max_batch_size,
        projected_gpu_memory_gb=memory_per_sequence_mb * max_batch_size / 1024,
        recommended_max_length=recommended_max_length,
        recommended_batch_size=recommended_batch_size,
        truncation_loss_percent=truncation_loss_percent
    )


def create_sequence_length_visualization(analysis: SequenceLengthAnalysis, output_dir: Path):
    """Create visualizations of sequence length analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create histogram of sequence lengths
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(analysis.token_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(analysis.avg_token_length, color='red', linestyle='--', 
                label=f'Average: {analysis.avg_token_length:.0f}')
    plt.axvline(analysis.recommended_max_length, color='green', linestyle='--', 
                label=f'Recommended Max: {analysis.recommended_max_length}')
    plt.xlabel('Token Count')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Sequence Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 2)
    sorted_lengths = sorted(analysis.token_lengths)
    percentiles = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    plt.plot(sorted_lengths, percentiles, linewidth=2)
    plt.axvline(analysis.recommended_max_length, color='green', linestyle='--', 
                label=f'95th Percentile: {analysis.recommended_max_length}')
    plt.xlabel('Token Count')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Distribution of Sequence Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Memory usage projection
    plt.subplot(2, 2, 3)
    sequence_lengths = range(512, 4096, 256)
    memory_usage = [(length * 4 * 512) / 1024 / 1024 for length in sequence_lengths]  # MB per sequence
    batch_sizes = [min(32, int(12800 / mem)) for mem in memory_usage]  # Batch size that fits in ~12.8GB
    
    plt.plot(sequence_lengths, memory_usage, 'b-', linewidth=2, label='Memory per Sequence')
    plt.axhline(analysis.memory_per_sequence_mb, color='red', linestyle='--', 
                label=f'Current Avg: {analysis.memory_per_sequence_mb:.1f}MB')
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Batch size implications
    plt.subplot(2, 2, 4)
    plt.plot(sequence_lengths, batch_sizes, 'g-', linewidth=2, label='Max Batch Size')
    plt.axhline(analysis.recommended_batch_size, color='red', linestyle='--', 
                label=f'Recommended: {analysis.recommended_batch_size}')
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Batch Size')
    plt.title('Batch Size vs Sequence Length (16GB GPU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to {output_dir / 'sequence_length_analysis.png'}")


def generate_architecture_recommendations(analysis: SequenceLengthAnalysis) -> Dict:
    """Generate specific architecture recommendations based on analysis."""
    
    recommendations = {
        "sequence_configuration": {
            "max_sequence_length": analysis.recommended_max_length,
            "truncation_strategy": "sliding_window",
            "padding_strategy": "right_pad",
            "truncation_loss_percent": analysis.truncation_loss_percent
        },
        
        "memory_optimization": {
            "recommended_batch_size": analysis.recommended_batch_size,
            "gradient_accumulation_steps": max(1, 32 // analysis.recommended_batch_size),
            "gradient_checkpointing": True,
            "mixed_precision": "fp16"
        },
        
        "attention_architecture": {
            "use_hierarchical_attention": analysis.avg_token_length > 1024,
            "local_attention_window": 256,
            "global_attention_stride": 64,
            "sparse_attention_pattern": "sliding_window"
        },
        
        "model_scaling": {
            "start_with_small_model": True,
            "recommended_layers": 6 if analysis.avg_token_length < 1024 else 8,
            "recommended_heads": 8,
            "recommended_dim": 512,
            "feed_forward_expansion": 4
        },
        
        "training_strategy": {
            "curriculum_learning": True,
            "start_with_short_sequences": True,
            "sequence_length_schedule": [512, 1024, analysis.recommended_max_length],
            "adaptive_batch_sizing": True
        }
    }
    
    return recommendations


def main():
    """Main analysis function."""
    print("🎼 Phase 3.1 - Sequence Length Analysis for Model Architecture")
    print("=" * 70)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    midi_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs" / "sequence_analysis"
    
    # Check if MIDI directory exists
    if not midi_dir.exists():
        print(f"❌ MIDI directory not found: {midi_dir}")
        print("Please ensure you have MIDI files in data/raw/ directory")
        return
    
    try:
        # Run analysis
        analysis = analyze_midi_sequence_lengths(midi_dir, max_files=50)
        
        # Create visualization
        create_sequence_length_visualization(analysis, output_dir)
        
        # Generate recommendations
        recommendations = generate_architecture_recommendations(analysis)
        
        # Save results
        results = {
            "analysis": {
                "file_count": analysis.file_count,
                "avg_duration_seconds": analysis.avg_duration,
                "avg_token_length": analysis.avg_token_length,
                "max_token_length": analysis.max_token_length,
                "min_token_length": analysis.min_token_length,
                "simple_pieces": analysis.simple_pieces,
                "complex_pieces": analysis.complex_pieces,
                "complexity_ratio": analysis.complex_pieces / analysis.file_count,
                "memory_per_sequence_mb": analysis.memory_per_sequence_mb,
                "recommended_max_length": analysis.recommended_max_length,
                "recommended_batch_size": analysis.recommended_batch_size,
                "truncation_loss_percent": analysis.truncation_loss_percent
            },
            "recommendations": recommendations
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types in nested structures
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        results_serializable = recursive_convert(results)
        
        with open(output_dir / "sequence_analysis_results.json", 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Print summary
        print(f"\n📊 Analysis Results:")
        print(f"  Files analyzed: {analysis.file_count}")
        print(f"  Average duration: {analysis.avg_duration:.1f}s")
        print(f"  Average token length: {analysis.avg_token_length:.0f}")
        print(f"  Max token length: {analysis.max_token_length}")
        print(f"  Complex pieces: {analysis.complex_pieces}/{analysis.file_count} ({analysis.complex_pieces/analysis.file_count:.1%})")
        
        print(f"\n🧠 Memory Analysis:")
        print(f"  Memory per sequence: {analysis.memory_per_sequence_mb:.1f}MB")
        print(f"  Recommended max length: {analysis.recommended_max_length}")
        print(f"  Recommended batch size: {analysis.recommended_batch_size}")
        print(f"  Truncation loss: {analysis.truncation_loss_percent:.1f}%")
        
        print(f"\n🏗️  Architecture Recommendations:")
        print(f"  Use hierarchical attention: {recommendations['attention_architecture']['use_hierarchical_attention']}")
        print(f"  Start with {recommendations['model_scaling']['recommended_layers']} layers")
        print(f"  Gradient accumulation: {recommendations['memory_optimization']['gradient_accumulation_steps']} steps")
        print(f"  Curriculum learning: {recommendations['training_strategy']['curriculum_learning']}")
        
        print(f"\n✅ Analysis complete! Results saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()