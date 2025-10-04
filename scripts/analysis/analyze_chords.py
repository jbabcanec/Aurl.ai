#!/usr/bin/env python3
"""
Main Chord Analysis Runner

This script performs comprehensive chord analysis on MIDI files,
generating publication-quality reports and visualizations.
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from studies.chord_analysis.chord_extractor import (
    AdvancedChordExtractor,
    ChordProgressionAnalyzer
)
from studies.chord_analysis.composer_analysis import (
    ComposerAnalyzer,
    ComposerHarmonicProfile
)
from studies.chord_analysis.transition_matrix import (
    TransitionMatrixBuilder,
    TransitionMatrix
)
from studies.chord_analysis.visualization import ChordAnalysisVisualizer

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ChordAnalysisRunner:
    """
    Orchestrates comprehensive chord analysis workflow.
    """

    def __init__(self, output_dir: str = "studies/chord_analysis/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.chord_extractor = AdvancedChordExtractor()
        self.progression_analyzer = ChordProgressionAnalyzer()
        self.composer_analyzer = ComposerAnalyzer()
        self.matrix_builder = TransitionMatrixBuilder()
        self.visualizer = ChordAnalysisVisualizer(str(self.output_dir / "figures"))

        # Results storage
        self.composer_profiles = {}
        self.transition_matrices = {}
        self.all_chords = []

    def analyze_corpus(self, midi_dir: str):
        """
        Analyze entire corpus of MIDI files.

        Args:
            midi_dir: Directory containing MIDI files
        """
        logger.info(f"Analyzing corpus in {midi_dir}")

        # Organize files by composer
        composer_files = self._organize_by_composer(midi_dir)

        logger.info(f"Found {len(composer_files)} composers")

        # Analyze each composer
        for composer, files in composer_files.items():
            logger.info(f"\nAnalyzing {composer}: {len(files)} pieces")

            # Create composer profile
            profile = self.composer_analyzer.analyze_composer(composer, files)
            self.composer_profiles[composer] = profile

            # Extract all chords for this composer
            composer_chords = []
            for file in files:
                chords = self.chord_extractor.extract_chords_from_midi(file)
                composer_chords.extend(chords)
                self.all_chords.extend(chords)

            # Build transition matrix for composer
            if composer_chords:
                matrix = self.matrix_builder.build_transition_matrix(
                    composer_chords,
                    name=f"{composer} Transitions",
                    normalize=True
                )
                matrix.composer = composer
                matrix.period = profile.period
                self.transition_matrices[composer] = matrix

        # Build overall transition matrix
        if self.all_chords:
            overall_matrix = self.matrix_builder.build_transition_matrix(
                self.all_chords,
                name="Overall Transitions",
                normalize=True
            )
            self.transition_matrices['Overall'] = overall_matrix

        logger.info(f"\nAnalysis complete: {len(self.composer_profiles)} composers analyzed")

    def _organize_by_composer(self, midi_dir: str) -> Dict[str, List[str]]:
        """Organize MIDI files by composer based on filename."""
        composer_files = defaultdict(list)
        midi_path = Path(midi_dir)

        # Common composer name patterns
        composer_keywords = {
            'bach': 'Bach',
            'beethoven': 'Beethoven',
            'mozart': 'Mozart',
            'chopin': 'Chopin',
            'liszt': 'Liszt',
            'brahms': 'Brahms',
            'debussy': 'Debussy',
            'rachmaninoff': 'Rachmaninoff',
            'rachmaninov': 'Rachmaninoff',
            'schubert': 'Schubert',
            'schumann': 'Schumann',
            'mendelssohn': 'Mendelssohn',
            'haydn': 'Haydn',
            'handel': 'Handel',
            'vivaldi': 'Vivaldi',
            'bartok': 'Bartók',
            'bartók': 'Bartók',
            'ravel': 'Ravel',
            'scriabin': 'Scriabin',
            'prokofiev': 'Prokofiev',
            'shostakovich': 'Shostakovich',
            'stravinsky': 'Stravinsky'
        }

        # Scan for MIDI files
        for file_path in midi_path.rglob("*.mid"):
            filename_lower = file_path.stem.lower()

            # Try to identify composer
            composer_found = False
            for keyword, composer_name in composer_keywords.items():
                if keyword in filename_lower:
                    composer_files[composer_name].append(str(file_path))
                    composer_found = True
                    break

            # If no composer identified, use "Unknown"
            if not composer_found:
                # Try to extract composer from path
                parts = str(file_path).lower().split('/')
                for part in parts:
                    for keyword, composer_name in composer_keywords.items():
                        if keyword in part:
                            composer_files[composer_name].append(str(file_path))
                            composer_found = True
                            break
                    if composer_found:
                        break

                if not composer_found:
                    composer_files['Unknown'].append(str(file_path))

        return dict(composer_files)

    def generate_visualizations(self):
        """Generate all visualizations."""
        logger.info("\nGenerating visualizations...")

        # 1. Transition matrix heatmaps for top composers
        for composer, matrix in list(self.transition_matrices.items())[:5]:
            if composer != 'Overall':
                fig = self.visualizer.create_transition_matrix_heatmap(
                    matrix,
                    title=f"{composer} - Harmonic Transition Matrix",
                    save_path=self.output_dir / f"figures/{composer}_matrix.png"
                )
                plt.close(fig)

        # 2. Overall transition network
        if 'Overall' in self.transition_matrices:
            fig = self.visualizer.create_chord_progression_network(
                self.transition_matrices['Overall'],
                title="Overall Chord Progression Network",
                save_path=self.output_dir / "figures/overall_network.png"
            )
            plt.close(fig)

        # 3. Composer comparison chart
        if len(self.composer_profiles) > 1:
            profiles = list(self.composer_profiles.values())[:8]  # Limit to 8 composers
            fig = self.visualizer.create_composer_comparison_chart(
                profiles,
                save_path=self.output_dir / "figures/composer_comparison.png"
            )
            plt.close(fig)

        # 4. Period evolution chart
        if len(self.composer_profiles) > 3:
            profiles = list(self.composer_profiles.values())
            analysis = self.composer_analyzer.compare_composers(profiles)

            fig = self.visualizer.create_period_evolution_chart(
                analysis,
                save_path=self.output_dir / "figures/period_evolution.png"
            )
            plt.close(fig)

        logger.info("Visualizations complete")

    def generate_reports(self):
        """Generate analysis reports."""
        logger.info("\nGenerating reports...")

        # 1. JSON report with all data
        report_data = {
            'analysis_date': str(datetime.now()),
            'num_composers': len(self.composer_profiles),
            'num_pieces': len(set(f for files in self._last_composer_files.values() for f in files)),
            'total_chords': len(self.all_chords),
            'composer_profiles': {},
            'transition_matrices': {},
            'comparative_analysis': {}
        }

        # Add composer profiles
        for composer, profile in self.composer_profiles.items():
            report_data['composer_profiles'][composer] = {
                'period': profile.period,
                'chromatic_usage': profile.chromatic_usage,
                'progression_complexity': profile.progression_complexity,
                'avg_tension': profile.avg_tension,
                'innovation_score': profile.innovation_score,
                'common_progressions': profile.common_progressions[:10],
                'unique_progressions': profile.unique_progressions[:5],
                'harmonic_signature': profile.harmonic_signature
            }

        # Add transition matrix stats
        for name, matrix in self.transition_matrices.items():
            report_data['transition_matrices'][name] = {
                'entropy': matrix.entropy_score,
                'sparsity': matrix.sparsity,
                'total_transitions': matrix.total_transitions,
                'top_transitions': matrix.top_transitions[:10],
                'cycles': matrix.cycles[:5],
                'attractors': matrix.attractors
            }

        # Save JSON report
        json_path = self.output_dir / "chord_analysis_report.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # 2. LaTeX report for publication
        if len(self.composer_profiles) > 1:
            profiles = list(self.composer_profiles.values())
            analysis = self.composer_analyzer.compare_composers(profiles)

            latex_path = self.visualizer.generate_latex_report(
                profiles,
                analysis,
                self.transition_matrices,
                output_file="chord_analysis_academic_report.tex"
            )

        # 3. Summary text report
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CHORD ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Composers Analyzed: {len(self.composer_profiles)}\n")
            f.write(f"Total Chords: {len(self.all_chords)}\n\n")

            f.write("COMPOSER PROFILES:\n")
            f.write("-" * 40 + "\n")
            for composer, profile in self.composer_profiles.items():
                f.write(f"\n{composer} ({profile.period}):\n")
                f.write(f"  Chromatic Usage: {profile.chromatic_usage:.3f}\n")
                f.write(f"  Complexity: {profile.progression_complexity:.3f}\n")
                f.write(f"  Innovation: {profile.innovation_score:.3f}\n")
                if profile.common_progressions:
                    prog, freq = profile.common_progressions[0]
                    f.write(f"  Most Common: {prog} ({freq} times)\n")

            if 'Overall' in self.transition_matrices:
                matrix = self.transition_matrices['Overall']
                f.write(f"\n\nOVERALL STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Transition Entropy: {matrix.entropy_score:.3f}\n")
                f.write(f"Sparsity: {matrix.sparsity:.3f}\n")
                f.write(f"\nTop 5 Transitions:\n")
                for from_c, to_c, prob in matrix.top_transitions[:5]:
                    f.write(f"  {from_c} → {to_c}: {prob:.3f}\n")

        logger.info(f"Reports saved to {self.output_dir}")

    def run_analysis(self, midi_dir: str):
        """
        Run complete chord analysis workflow.

        Args:
            midi_dir: Directory containing MIDI files
        """
        # Store for report generation
        self._last_composer_files = self._organize_by_composer(midi_dir)

        # Run analysis
        self.analyze_corpus(midi_dir)

        # Generate outputs
        self.generate_visualizations()
        self.generate_reports()

        logger.info("\n" + "="*60)
        logger.info("CHORD ANALYSIS COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Chord Analysis for Music Theory Research"
    )
    parser.add_argument(
        '--midi-dir',
        type=str,
        default='data/raw',
        help='Directory containing MIDI files to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='studies/chord_analysis/output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--composers',
        type=str,
        nargs='+',
        help='Specific composers to analyze (optional)'
    )

    args = parser.parse_args()

    # Import matplotlib here to avoid issues if not installed
    global plt
    import matplotlib.pyplot as plt

    # Import datetime
    global datetime
    from datetime import datetime

    # Run analysis
    runner = ChordAnalysisRunner(args.output_dir)
    runner.run_analysis(args.midi_dir)


if __name__ == "__main__":
    main()