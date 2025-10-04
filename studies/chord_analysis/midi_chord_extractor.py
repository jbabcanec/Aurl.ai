#!/usr/bin/env python3
"""
MIDI Chord Extractor - Analyzes actual MIDI files to extract chord progressions.

This tool reads raw MIDI files and extracts the harmonic content by:
1. Identifying simultaneous notes to form chords
2. Tracking chord progressions over time
3. Analyzing patterns across different composers
"""

import mido
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ChordEvent:
    """Represents a chord at a specific time."""
    time: float  # in beats
    pitches: Set[int]  # MIDI note numbers
    chord_name: str
    root: Optional[int] = None
    quality: Optional[str] = None
    inversion: int = 0


class MIDIChordAnalyzer:
    """Analyzes MIDI files to extract chord progressions."""

    def __init__(self):
        # Chord templates for recognition (intervals from root)
        self.chord_templates = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            '7': [0, 4, 7, 10],
            'dim7': [0, 3, 6, 9],
            'half-dim7': [0, 3, 6, 10],
            'maj6': [0, 4, 7, 9],
            'min6': [0, 3, 7, 9],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            'add9': [0, 4, 7, 14],
            'min9': [0, 3, 7, 10, 14],
        }

        # Note names for output
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Statistics collectors
        self.all_progressions = []
        self.composer_stats = defaultdict(lambda: {
            'chords': Counter(),
            'transitions': defaultdict(Counter),
            'chord_durations': [],
            'key_signatures': Counter()
        })

    def extract_chords_from_midi(self, midi_path: Path) -> List[ChordEvent]:
        """Extract chord sequence from a MIDI file."""
        try:
            mid = mido.MidiFile(str(midi_path))
        except:
            print(f"Could not read {midi_path}")
            return []

        # Merge all tracks
        events = []
        current_time = 0
        active_notes = {}  # pitch -> velocity

        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = msg.velocity
                    events.append((track_time, 'note_on', msg.note))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        del active_notes[msg.note]
                    events.append((track_time, 'note_off', msg.note))

        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Extract chords at regular intervals
        chords = []
        if not events:
            return chords

        # Sample every 480 ticks (typically a quarter note)
        sample_interval = 480
        max_time = events[-1][0]

        for sample_time in range(0, int(max_time), sample_interval):
            # Get all active notes at this time
            active_at_time = set()
            for time, event_type, pitch in events:
                if time > sample_time:
                    break
                if event_type == 'note_on':
                    active_at_time.add(pitch)
                elif event_type == 'note_off' and pitch in active_at_time:
                    active_at_time.discard(pitch)

            # Only analyze if we have 2+ notes
            if len(active_at_time) >= 2:
                chord = self.identify_chord(active_at_time)
                if chord:
                    chords.append(chord)

        return chords

    def identify_chord(self, pitches: Set[int]) -> Optional[ChordEvent]:
        """Identify a chord from a set of MIDI pitches."""
        if len(pitches) < 2:
            return None

        pitch_list = sorted(list(pitches))

        # Try to find the best matching chord
        best_match = None
        best_score = 0

        # Try each possible root note
        for root_idx, root in enumerate(pitch_list):
            # Get intervals from this root
            intervals = [(p - root) % 12 for p in pitch_list]
            intervals = sorted(set(intervals))

            # Check against templates
            for chord_type, template in self.chord_templates.items():
                # Calculate match score
                score = self._calculate_chord_match_score(intervals, template)

                if score > best_score:
                    best_score = score
                    root_name = self.note_names[root % 12]
                    best_match = ChordEvent(
                        time=0,  # Will be set by caller
                        pitches=pitches,
                        chord_name=f"{root_name}{chord_type}",
                        root=root,
                        quality=chord_type,
                        inversion=root_idx
                    )

        return best_match if best_score > 0.5 else None

    def _calculate_chord_match_score(self, intervals: List[int], template: List[int]) -> float:
        """Calculate how well a set of intervals matches a chord template."""
        # Simple scoring: count matching intervals
        matches = sum(1 for t in template if t in intervals)
        extra = sum(1 for i in intervals if i not in template)

        if len(template) == 0:
            return 0

        score = matches / len(template) - (extra * 0.1)
        return max(0, min(1, score))

    def analyze_progression_patterns(self, chords: List[ChordEvent]) -> Dict:
        """Analyze patterns in a chord progression."""
        if len(chords) < 2:
            return {}

        # Extract chord roots and qualities
        progression = []
        for chord in chords:
            if chord.root is not None:
                # Simplify to root and quality
                root_class = chord.root % 12
                progression.append((root_class, chord.quality))

        # Count transitions
        transitions = defaultdict(Counter)
        for i in range(len(progression) - 1):
            current = progression[i]
            next_chord = progression[i + 1]
            transitions[str(current)][str(next_chord)] += 1

        # Find common patterns (2-4 chord sequences)
        patterns = Counter()
        for length in [2, 3, 4]:
            for i in range(len(progression) - length + 1):
                pattern = tuple(progression[i:i + length])
                patterns[str(pattern)] += 1

        # Analyze chord frequencies
        chord_freq = Counter([str(c) for c in progression])

        return {
            'chord_frequencies': dict(chord_freq.most_common(20)),
            'transitions': {k: dict(v) for k, v in transitions.items()},
            'common_patterns': dict(patterns.most_common(20)),
            'total_chords': len(progression),
            'unique_chords': len(set(progression))
        }

    def analyze_all_midi_files(self, data_dir: Path) -> Dict:
        """Analyze all MIDI files in the data directory."""
        midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*.MID"))

        print(f"Found {len(midi_files)} MIDI files to analyze")

        all_results = {
            'files_analyzed': 0,
            'total_chords': 0,
            'composer_analysis': {},
            'global_patterns': {
                'all_chords': Counter(),
                'all_transitions': defaultdict(Counter),
                'pattern_lengths': Counter()
            }
        }

        for i, midi_file in enumerate(midi_files, 1):  # Analyze ALL files
            print(f"Analyzing ({i}/{len(midi_files)}): {midi_file.name}")

            # Extract composer from filename
            composer = self._extract_composer(midi_file.name)

            # Extract chords
            chords = self.extract_chords_from_midi(midi_file)
            if not chords:
                continue

            # Analyze patterns
            analysis = self.analyze_progression_patterns(chords)

            # Store results
            all_results['files_analyzed'] += 1
            all_results['total_chords'] += len(chords)

            # Update global statistics
            for chord_str, count in analysis.get('chord_frequencies', {}).items():
                all_results['global_patterns']['all_chords'][chord_str] += count

            # Store composer-specific data
            if composer not in all_results['composer_analysis']:
                all_results['composer_analysis'][composer] = {
                    'files': 0,
                    'total_chords': 0,
                    'common_chords': Counter(),
                    'common_transitions': defaultdict(Counter)
                }

            comp_data = all_results['composer_analysis'][composer]
            comp_data['files'] += 1
            comp_data['total_chords'] += len(chords)

            for chord, count in analysis.get('chord_frequencies', {}).items():
                comp_data['common_chords'][chord] += count

        return all_results

    def _extract_composer(self, filename: str) -> str:
        """Extract composer name from filename."""
        name_lower = filename.lower()

        if 'beethoven' in name_lower or 'appass' in name_lower or 'waldstein' in name_lower or 'mond' in name_lower or 'pathetique' in name_lower:
            return 'Beethoven'
        elif 'chopin' in name_lower or 'chpn' in name_lower or 'chp_' in name_lower:
            return 'Chopin'
        elif 'mozart' in name_lower or name_lower.startswith('mz_'):
            return 'Mozart'
        elif 'debussy' in name_lower or name_lower.startswith('deb'):
            return 'Debussy'
        elif 'bach' in name_lower or name_lower.startswith('bwv'):
            return 'Bach'
        elif 'schubert' in name_lower or 'schub' in name_lower or name_lower.startswith('scn'):
            return 'Schubert'
        elif 'schumann' in name_lower or 'schum' in name_lower:
            return 'Schumann'
        elif 'brahms' in name_lower or name_lower.startswith('br_'):
            return 'Brahms'
        elif 'liszt' in name_lower:
            return 'Liszt'
        elif 'rachmaninoff' in name_lower or 'rach' in name_lower:
            return 'Rachmaninoff'
        elif 'grieg' in name_lower:
            return 'Grieg'
        elif 'mendelssohn' in name_lower or 'mendel' in name_lower:
            return 'Mendelssohn'
        else:
            return 'Other'

    def generate_report(self, analysis_results: Dict) -> str:
        """Generate a readable report from analysis results."""
        report = []
        report.append("=" * 60)
        report.append("MIDI CHORD PROGRESSION ANALYSIS")
        report.append("=" * 60)
        report.append("")

        report.append(f"Files Analyzed: {analysis_results['files_analyzed']}")
        report.append(f"Total Chords Extracted: {analysis_results['total_chords']}")
        report.append("")

        # Most common chords overall
        report.append("MOST COMMON CHORDS ACROSS ALL FILES:")
        report.append("-" * 40)
        for chord, count in list(analysis_results['global_patterns']['all_chords'].most_common(15)):
            report.append(f"  {chord}: {count} occurrences")
        report.append("")

        # Composer-specific analysis
        report.append("COMPOSER-SPECIFIC ANALYSIS:")
        report.append("-" * 40)

        for composer, data in analysis_results['composer_analysis'].items():
            if data['files'] == 0:
                continue

            report.append(f"\n{composer}:")
            report.append(f"  Files: {data['files']}")
            report.append(f"  Total chords: {data['total_chords']}")
            report.append(f"  Average chords per file: {data['total_chords'] // data['files']}")

            report.append(f"  Most used chords:")
            for chord, count in list(data['common_chords'].most_common(5)):
                percentage = (count / data['total_chords']) * 100
                report.append(f"    {chord}: {count} ({percentage:.1f}%)")

        return "\n".join(report)


def main():
    """Main analysis function."""
    # Create analyzer
    analyzer = MIDIChordAnalyzer()

    # Set data directory
    data_dir = Path("/Users/josephbabcanec/Dropbox/Babcanec Works/Programming/Aurl/data/raw")

    # Analyze all MIDI files
    print("Starting MIDI chord analysis...")
    results = analyzer.analyze_all_midi_files(data_dir)

    # Generate report
    report = analyzer.generate_report(results)

    # Save report
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "midi_chord_analysis.txt"
    report_path.write_text(report)

    # Save raw data as JSON
    json_path = output_dir / "chord_analysis_data.json"

    # Convert Counters to regular dicts for JSON serialization
    json_safe_results = {
        'files_analyzed': results['files_analyzed'],
        'total_chords': results['total_chords'],
        'composer_analysis': {
            composer: {
                'files': data['files'],
                'total_chords': data['total_chords'],
                'common_chords': dict(data['common_chords'].most_common(20))
            }
            for composer, data in results['composer_analysis'].items()
        },
        'global_patterns': {
            'all_chords': dict(results['global_patterns']['all_chords'].most_common(50))
        }
    }

    with open(json_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print(f"\n{report}")
    print(f"\nReport saved to: {report_path}")
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()