"""
MIDI Data Audit Tool for Aurl.ai

This tool analyzes actual MIDI files and shows how they transform through our system,
providing insights into what the neural network will see during training.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pretty_midi
from collections import Counter


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.midi_parser import load_midi_file, MidiParser
from src.data.representation import (
    MusicRepresentationConverter, VocabularyConfig, PianoRollConfig
)
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MidiDataAuditor:
    """Audits actual MIDI data to understand training data characteristics."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/raw")
        self.vocab_config = VocabularyConfig()
        self.piano_roll_config = PianoRollConfig()
        self.converter = MusicRepresentationConverter(
            self.vocab_config, self.piano_roll_config
        )
        self.parser = MidiParser()
        
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "total_notes": 0,
            "musical_statistics": {},
            "transformation_examples": [],
            "vocabulary_usage": {},
            "data_quality_insights": {}
        }
    
    def create_sample_midi_files(self):
        """Create diverse sample MIDI files for testing."""
        sample_dir = self.data_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Simple melody
        pm = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0, name="Piano")
        
        # C major scale up and down
        scale = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
        for i, pitch in enumerate(scale):
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=i * 0.5,
                end=(i + 1) * 0.5 - 0.05
            )
            piano.notes.append(note)
        pm.instruments.append(piano)
        pm.write(str(sample_dir / "simple_melody.mid"))
        
        # 2. Piano piece with chords
        pm2 = pretty_midi.PrettyMIDI()
        piano2 = pretty_midi.Instrument(program=0, name="Piano")
        
        # Right hand melody
        melody = [72, 74, 76, 77, 79, 77, 76, 74]
        for i, pitch in enumerate(melody):
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=i * 0.5,
                end=(i + 1) * 0.5 - 0.05
            )
            piano2.notes.append(note)
        
        # Left hand chords (C, F, G, C progression)
        chords = [
            [48, 52, 55],  # C major
            [53, 57, 60],  # F major
            [55, 59, 62],  # G major
            [48, 52, 55]   # C major
        ]
        
        for i, chord in enumerate(chords):
            start_time = i * 2.0
            for pitch in chord:
                note = pretty_midi.Note(
                    velocity=70,
                    pitch=pitch,
                    start=start_time,
                    end=start_time + 1.8
                )
                piano2.notes.append(note)
        
        pm2.instruments.append(piano2)
        pm2.write(str(sample_dir / "piano_with_chords.mid"))
        
        # 3. Multi-instrument piece
        pm3 = pretty_midi.PrettyMIDI()
        
        # Piano
        piano3 = pretty_midi.Instrument(program=0, name="Piano")
        for i in range(8):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60 + (i % 8),
                start=i * 0.5,
                end=(i + 1) * 0.5 - 0.05
            )
            piano3.notes.append(note)
        
        # Bass
        bass = pretty_midi.Instrument(program=32, name="Bass")
        bass_notes = [36, 41, 43, 36]  # C, F, G, C
        for i, pitch in enumerate(bass_notes):
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=i * 2.0,
                end=(i + 1) * 2.0 - 0.1
            )
            bass.notes.append(note)
        
        # Drums
        drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        for i in range(16):
            # Kick on beats
            if i % 4 == 0:
                kick = pretty_midi.Note(
                    velocity=120,
                    pitch=36,
                    start=i * 0.25,
                    end=i * 0.25 + 0.1
                )
                drums.notes.append(kick)
            
            # Hi-hat on eighth notes
            hihat = pretty_midi.Note(
                velocity=60,
                pitch=42,
                start=i * 0.25,
                end=i * 0.25 + 0.05
            )
            drums.notes.append(hihat)
        
        pm3.instruments.append(piano3)
        pm3.instruments.append(bass)
        pm3.instruments.append(drums)
        pm3.write(str(sample_dir / "multi_instrument.mid"))
        
        return sample_dir
    
    def audit_midi_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit a single MIDI file."""
        try:
            # Parse MIDI
            midi_data = load_midi_file(file_path)
            
            # Transform to representation
            representation = self.converter.midi_to_representation(midi_data)
            
            # Collect statistics
            stats = {
                "filename": file_path.name,
                "duration_seconds": midi_data.end_time,
                "num_instruments": len(midi_data.instruments),
                "total_notes": sum(len(inst.notes) for inst in midi_data.instruments),
                "instruments": []
            }
            
            # Analyze each instrument
            for inst in midi_data.instruments:
                inst_stats = {
                    "name": inst.name,
                    "program": inst.program,
                    "is_drum": inst.is_drum,
                    "num_notes": len(inst.notes),
                    "pitch_range": [127, 0],
                    "velocity_range": [127, 0],
                    "avg_note_duration": 0,
                    "polyphony_max": 0
                }
                
                if inst.notes:
                    pitches = [n.pitch for n in inst.notes]
                    velocities = [n.velocity for n in inst.notes]
                    durations = [n.end - n.start for n in inst.notes]
                    
                    inst_stats["pitch_range"] = [min(pitches), max(pitches)]
                    inst_stats["velocity_range"] = [min(velocities), max(velocities)]
                    inst_stats["avg_note_duration"] = np.mean(durations)
                    
                    # Calculate max polyphony
                    note_times = [(n.start, 1) for n in inst.notes] + [(n.end, -1) for n in inst.notes]
                    note_times.sort()
                    current_notes = 0
                    max_notes = 0
                    for _, delta in note_times:
                        current_notes += delta
                        max_notes = max(max_notes, current_notes)
                    inst_stats["polyphony_max"] = max_notes
                
                stats["instruments"].append(inst_stats)
            
            # Analyze transformation
            transform_stats = {
                "num_events": len(representation.events),
                "num_tokens": len(representation.tokens) if representation.tokens is not None else 0,
                "unique_tokens": len(np.unique(representation.tokens)) if representation.tokens is not None else 0,
                "piano_roll_shape": representation.piano_roll.shape if representation.piano_roll is not None else None,
                "event_type_distribution": {},
                "token_examples": []
            }
            
            # Count event types
            event_counter = Counter([e.event_type.name for e in representation.events])
            transform_stats["event_type_distribution"] = dict(event_counter)
            
            # Sample tokens with meanings
            if representation.tokens is not None and len(representation.tokens) > 0:
                sample_indices = np.linspace(0, len(representation.tokens)-1, min(10, len(representation.tokens)), dtype=int)
                for idx in sample_indices:
                    token = int(representation.tokens[idx])
                    event_type, value = self.vocab_config.token_to_event_info(token)
                    transform_stats["token_examples"].append({
                        "position": int(idx),
                        "token": int(token),
                        "type": event_type.name,
                        "value": int(value)
                    })
            
            return {
                "basic_stats": stats,
                "transformation": transform_stats,
                "representation": representation
            }
            
        except Exception as e:
            logger.error(f"Failed to audit {file_path}: {e}")
            return None
    
    def generate_audit_report(self, output_dir: Path = None):
        """Generate comprehensive audit report for MIDI data."""
        if output_dir is None:
            output_dir = Path("data/audits")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample files if needed
        sample_dir = self.create_sample_midi_files()
        
        # Find all MIDI files
        midi_files = list(self.data_dir.glob("**/*.mid")) + list(self.data_dir.glob("**/*.midi"))
        
        if not midi_files:
            logger.warning("No MIDI files found, using only samples")
            midi_files = list(sample_dir.glob("*.mid"))
        
        print(f"\n🎵 Auditing {len(midi_files)} MIDI files...")
        
        # Audit each file
        all_pitches = []
        all_velocities = []
        all_durations = []
        token_usage = Counter()
        
        for midi_file in midi_files[:10]:  # Limit to 10 files for now
            print(f"  Analyzing: {midi_file.name}")
            
            result = self.audit_midi_file(midi_file)
            if result:
                self.audit_results["files_analyzed"] += 1
                
                # Collect global statistics
                for inst in result["basic_stats"]["instruments"]:
                    if inst["num_notes"] > 0:
                        self.audit_results["total_notes"] += inst["num_notes"]
                        
                        # Extract note data
                        midi_data = load_midi_file(midi_file)
                        for midi_inst in midi_data.instruments:
                            if midi_inst.name == inst["name"]:
                                for note in midi_inst.notes:
                                    all_pitches.append(note.pitch)
                                    all_velocities.append(note.velocity)
                                    all_durations.append(note.end - note.start)
                
                # Track token usage
                if result["transformation"]["num_tokens"] > 0:
                    representation = result["representation"]
                    for token in representation.tokens:
                        token_usage[int(token)] += 1
                
                # Store example transformations
                if len(self.audit_results["transformation_examples"]) < 3:
                    self.audit_results["transformation_examples"].append({
                        "file": midi_file.name,
                        "basic_stats": result["basic_stats"],
                        "transformation": result["transformation"]
                    })
        
        # Calculate aggregate statistics
        if all_pitches:
            self.audit_results["musical_statistics"] = {
                "pitch_analysis": {
                    "range": [min(all_pitches), max(all_pitches)],
                    "mean": float(np.mean(all_pitches)),
                    "std": float(np.std(all_pitches)),
                    "most_common": Counter(all_pitches).most_common(10)
                },
                "velocity_analysis": {
                    "range": [min(all_velocities), max(all_velocities)],
                    "mean": float(np.mean(all_velocities)),
                    "std": float(np.std(all_velocities)),
                    "distribution": np.histogram(all_velocities, bins=8)[0].tolist()
                },
                "duration_analysis": {
                    "range": [float(min(all_durations)), float(max(all_durations))],
                    "mean": float(np.mean(all_durations)),
                    "std": float(np.std(all_durations)),
                    "short_notes": sum(1 for d in all_durations if d < 0.25),
                    "medium_notes": sum(1 for d in all_durations if 0.25 <= d < 1.0),
                    "long_notes": sum(1 for d in all_durations if d >= 1.0)
                }
            }
        
        # Analyze vocabulary usage
        self.audit_results["vocabulary_usage"] = {
            "total_vocab_size": self.vocab_config.vocab_size,
            "unique_tokens_used": len(token_usage),
            "coverage_percentage": len(token_usage) / self.vocab_config.vocab_size * 100,
            "most_common_tokens": token_usage.most_common(20),
            "token_distribution": {
                "special_tokens": sum(1 for t in token_usage if t < 4),
                "note_events": sum(1 for t in token_usage if 4 <= t < 180),
                "time_events": sum(1 for t in token_usage if 180 <= t < 305),
                "other_events": sum(1 for t in token_usage if t >= 305)
            }
        }
        
        # Data quality insights
        avg_polyphony = []
        for example in self.audit_results["transformation_examples"]:
            for inst in example["basic_stats"]["instruments"]:
                if inst["polyphony_max"] > 0:
                    avg_polyphony.append(inst["polyphony_max"])
        
        self.audit_results["data_quality_insights"] = {
            "average_file_duration": float(np.mean([ex["basic_stats"]["duration_seconds"] 
                                                   for ex in self.audit_results["transformation_examples"]])),
            "average_instruments_per_file": float(np.mean([ex["basic_stats"]["num_instruments"] 
                                                          for ex in self.audit_results["transformation_examples"]])),
            "max_polyphony_observed": max(avg_polyphony) if avg_polyphony else 0,
            "average_max_polyphony": float(np.mean(avg_polyphony)) if avg_polyphony else 0,
            "recommendations": []
        }
        
        # Generate recommendations
        if self.audit_results["data_quality_insights"]["average_max_polyphony"] < 3:
            self.audit_results["data_quality_insights"]["recommendations"].append(
                "Low polyphony detected. Consider adding more complex piano pieces."
            )
        
        if self.audit_results["vocabulary_usage"]["coverage_percentage"] < 30:
            self.audit_results["data_quality_insights"]["recommendations"].append(
                f"Only {self.audit_results['vocabulary_usage']['coverage_percentage']:.1f}% of vocabulary used. "
                "Dataset may benefit from more diverse musical styles."
            )
        
        # Save report
        report_file = output_dir / f"midi_data_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2, cls=NumpyEncoder)
        
        # Generate visualizations
        self.create_visualizations(output_dir)
        
        print(f"\n📊 Audit complete! Report saved to: {report_file}")
        return self.audit_results
    
    def create_visualizations(self, output_dir: Path):
        """Create visual representations of the audit results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MIDI Data Analysis - Aurl.ai', fontsize=16)
        
        # 1. Pitch distribution
        ax1 = axes[0, 0]
        if "pitch_analysis" in self.audit_results["musical_statistics"]:
            pitches, counts = zip(*self.audit_results["musical_statistics"]["pitch_analysis"]["most_common"])
            ax1.bar(range(len(pitches)), counts)
            ax1.set_xticks(range(len(pitches)))
            ax1.set_xticklabels([self._pitch_to_note(p) for p in pitches], rotation=45)
            ax1.set_title('Most Common Pitches')
            ax1.set_ylabel('Frequency')
        
        # 2. Velocity distribution
        ax2 = axes[0, 1]
        if "velocity_analysis" in self.audit_results["musical_statistics"]:
            vel_dist = self.audit_results["musical_statistics"]["velocity_analysis"]["distribution"]
            ax2.bar(range(len(vel_dist)), vel_dist)
            ax2.set_title('Velocity Distribution')
            ax2.set_xlabel('Velocity Bins')
            ax2.set_ylabel('Count')
        
        # 3. Note duration distribution
        ax3 = axes[1, 0]
        if "duration_analysis" in self.audit_results["musical_statistics"]:
            dur_stats = self.audit_results["musical_statistics"]["duration_analysis"]
            categories = ['Short\n(<0.25s)', 'Medium\n(0.25-1s)', 'Long\n(>1s)']
            values = [dur_stats["short_notes"], dur_stats["medium_notes"], dur_stats["long_notes"]]
            ax3.bar(categories, values)
            ax3.set_title('Note Duration Categories')
            ax3.set_ylabel('Count')
        
        # 4. Token usage
        ax4 = axes[1, 1]
        token_dist = self.audit_results["vocabulary_usage"]["token_distribution"]
        labels = list(token_dist.keys())
        values = list(token_dist.values())
        ax4.pie(values, labels=labels, autopct='%1.1f%%')
        ax4.set_title('Token Type Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'midi_data_analysis.png', dpi=150)
        plt.close()
        
        print(f"  📈 Visualizations saved to: {output_dir}/midi_data_analysis.png")
    
    def _pitch_to_note(self, pitch: int) -> str:
        """Convert MIDI pitch to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note = note_names[pitch % 12]
        return f"{note}{octave}"
    
    def print_summary(self):
        """Print a human-readable summary of the audit."""
        print("\n" + "="*60)
        print("🎼 MIDI DATA AUDIT SUMMARY - Aurl.ai")
        print("="*60)
        
        print(f"\n📊 Dataset Overview:")
        print(f"  Files analyzed: {self.audit_results['files_analyzed']}")
        print(f"  Total notes: {self.audit_results['total_notes']:,}")
        
        if self.audit_results["musical_statistics"]:
            stats = self.audit_results["musical_statistics"]
            print(f"\n🎵 Musical Characteristics:")
            print(f"  Pitch range: {stats['pitch_analysis']['range'][0]} to {stats['pitch_analysis']['range'][1]}")
            print(f"  Average pitch: {stats['pitch_analysis']['mean']:.1f}")
            print(f"  Velocity range: {stats['velocity_analysis']['range'][0]} to {stats['velocity_analysis']['range'][1]}")
            print(f"  Average velocity: {stats['velocity_analysis']['mean']:.1f}")
            print(f"  Note durations: {stats['duration_analysis']['mean']:.3f}s average")
        
        print(f"\n🔤 Vocabulary Usage:")
        print(f"  Total vocabulary size: {self.audit_results['vocabulary_usage']['total_vocab_size']}")
        print(f"  Unique tokens used: {self.audit_results['vocabulary_usage']['unique_tokens_used']}")
        print(f"  Coverage: {self.audit_results['vocabulary_usage']['coverage_percentage']:.1f}%")
        
        print(f"\n📈 Data Quality:")
        insights = self.audit_results["data_quality_insights"]
        print(f"  Average file duration: {insights['average_file_duration']:.1f}s")
        print(f"  Average instruments: {insights['average_instruments_per_file']:.1f}")
        print(f"  Max polyphony: {insights['max_polyphony_observed']}")
        
        if insights["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in insights["recommendations"]:
                print(f"  • {rec}")
        
        print("\n" + "="*60)


def main():
    """Run the MIDI data audit."""
    print("🔍 Aurl.ai MIDI Data Audit Tool")
    print("Analyzing how MIDI files transform through our system...")
    
    auditor = MidiDataAuditor()
    auditor.generate_audit_report()
    auditor.print_summary()


if __name__ == "__main__":
    main()