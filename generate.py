#!/usr/bin/env python3
"""
Aurl.ai Music Generation with Enhanced Grammar

Production-ready generation script with constrained sampling
to solve token generation issues and produce high-quality music.
"""

import sys
import torch
import torch.nn.functional as F
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler
from src.generation.midi_export import MidiExporter
from src.training.musical_grammar import validate_generated_sequence, get_parameter_quality_report
from src.data.representation import VocabularyConfig, EventType
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

class ConstrainedSampler:
    """Production constrained sampler that solves all three token generation issues."""
    
    def __init__(self, model, device, vocab_config):
        self.model = model
        self.device = device
        self.vocab_config = vocab_config
        
        # Token mappings
        self.start_token = vocab_config.event_to_token_map.get((EventType.START_TOKEN, 0), 0)
        self.end_token = vocab_config.event_to_token_map.get((EventType.END_TOKEN, 0), 1)
        
        # Event ranges
        self.note_on_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                 if event_type == EventType.NOTE_ON])
        self.note_on_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                               if event_type == EventType.NOTE_ON])
        
        self.note_off_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.NOTE_OFF])
        self.note_off_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                if event_type == EventType.NOTE_OFF])
        
        self.velocity_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.VELOCITY_CHANGE])
        self.velocity_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                if event_type == EventType.VELOCITY_CHANGE])
        
        self.time_shift_start = min([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                    if event_type == EventType.TIME_SHIFT])
        self.time_shift_end = max([token for (event_type, _), token in vocab_config.event_to_token_map.items() 
                                  if event_type == EventType.TIME_SHIFT])
    
    def generate_constrained(self, max_length: int = 80, temperature: float = 0.8, 
                           min_notes: int = 3, target_notes: int = None) -> torch.Tensor:
        """Generate tokens with constraints that solve the three main issues."""
        self.model.eval()
        
        # State tracking for constraints
        active_notes: Set[int] = set()
        completed_notes = 0
        sequence_length = 0
        has_velocity = False
        
        # Start with proper initialization
        sequence = [self.start_token]
        
        # Add initial velocity setting (prevents silent notes)
        initial_velocity = self.velocity_start + 16  # Mid-range velocity
        sequence.append(initial_velocity)
        has_velocity = True
        sequence_length = 2
        
        current_input = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        
        for step in range(max_length - len(sequence)):
            with torch.no_grad():
                # Get model predictions  
                # Ensure input doesn't exceed model's max sequence length
                max_seq_len = getattr(self.model, 'max_sequence_length', 256)
                if current_input.size(1) > max_seq_len:
                    current_input = current_input[:, -max_seq_len:]
                
                outputs = self.model(current_input)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                next_logits = logits[0, -1, :] / temperature
                
                # Apply constraints to solve the three issues
                forbidden_tokens = self._get_production_constraints(
                    active_notes, completed_notes, sequence_length, has_velocity, min_notes, target_notes
                )
                
                for token in forbidden_tokens:
                    next_logits[token] = -float('inf')
                
                # Apply musical biases (detect jazz mode from config)
                jazz_mode = 'jazz' in str(target_notes).lower() if target_notes else False
                self._apply_musical_biases(next_logits, sequence, active_notes, completed_notes, target_notes, jazz_mode)
                
                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                sequence.append(next_token)
                sequence_length += 1
                
                # Update state
                if self.note_on_start <= next_token <= self.note_on_end:
                    pitch = next_token - self.note_on_start + self.vocab_config.min_pitch
                    active_notes.add(pitch)
                elif self.note_off_start <= next_token <= self.note_off_end:
                    pitch = next_token - self.note_off_start + self.vocab_config.min_pitch
                    if pitch in active_notes:
                        active_notes.remove(pitch)
                        completed_notes += 1
                elif self.velocity_start <= next_token <= self.velocity_end:
                    has_velocity = True
                
                # Stop conditions
                if next_token == self.end_token and completed_notes >= min_notes:
                    break
                if target_notes and completed_notes >= target_notes:
                    sequence.append(self.end_token)
                    break
                if len(sequence) >= max_length:
                    break
                
                # Update input for next iteration
                current_input = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        
        logger.info(f"Generated sequence: {len(sequence)} tokens, {completed_notes} completed notes")
        return torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _get_production_constraints(self, active_notes: Set[int], completed_notes: int, 
                                   seq_len: int, has_velocity: bool, min_notes: int, 
                                   target_notes: int = None) -> Set[int]:
        """Apply constraints that solve the three token generation issues."""
        forbidden = set()
        
        # Issue 1: Prevent immediate END token
        if seq_len < 8 or completed_notes < min_notes:
            forbidden.add(self.end_token)
        
        # Issue 2: Prevent note_off for inactive notes
        for token in range(self.note_off_start, self.note_off_end + 1):
            pitch = token - self.note_off_start + self.vocab_config.min_pitch
            if pitch not in active_notes:
                forbidden.add(token)
        
        return forbidden
    
    def _apply_musical_biases(self, logits: torch.Tensor, sequence: List[int], 
                            active_notes: Set[int], completed_notes: int, target_notes: int = None,
                            jazz_mode: bool = False):
        """Apply biases for better musical structure and groove."""
        
        # Encourage notes after velocity changes
        if (len(sequence) > 0 and 
            self.velocity_start <= sequence[-1] <= self.velocity_end):
            for token in range(self.note_on_start, self.note_on_end + 1):
                logits[token] += 1.5
        
        # Encourage rhythmic variety in time shifts
        if (len(sequence) > 0 and 
            self.note_on_start <= sequence[-1] <= self.note_on_end):
            # Vary time shift lengths for rhythm
            for i, token in enumerate(range(self.time_shift_start, min(self.time_shift_start + 150, self.time_shift_end + 1))):
                if jazz_mode:
                    # Jazz rhythm: favor shorter and medium shifts, occasional long ones
                    if i < 20:  # Short shifts
                        logits[token] += 1.5
                    elif i < 60:  # Medium shifts  
                        logits[token] += 1.0
                    elif i < 80:  # Occasional longer shifts for variety
                        logits[token] += 0.3
                else:
                    logits[token] += 1.0
        
        # Jazz-style chord encouragement
        if jazz_mode and len(active_notes) > 0 and len(active_notes) < 4:
            # Encourage chord formations (notes close in pitch)
            for existing_pitch in active_notes:
                # Encourage thirds, fifths, and sevenths
                for interval in [3, 4, 7, 10, 11]:  # Major/minor thirds, fifths, sevenths
                    chord_pitch = existing_pitch + interval
                    if self.vocab_config.min_pitch <= chord_pitch <= self.vocab_config.max_pitch:
                        chord_token = self.note_on_start + chord_pitch - self.vocab_config.min_pitch
                        if chord_token <= self.note_on_end:
                            logits[chord_token] += 1.2
        
        # Encourage note completion for long-running notes (but not too aggressively for jazz)
        if len(sequence) > 15:
            for pitch in active_notes:
                note_off_token = self.note_off_start + pitch - self.vocab_config.min_pitch
                if note_off_token <= self.note_off_end:
                    completion_bias = 0.5 if jazz_mode else 0.8  # Gentler for jazz flow
                    logits[note_off_token] += completion_bias
        
        # Strong anti-repetition for groove
        repetition_penalty = 3.0 if jazz_mode else 2.0
        if len(sequence) >= 3:
            recent_tokens = sequence[-3:]
            if len(set(recent_tokens)) == 1:
                logits[recent_tokens[0]] -= repetition_penalty
        
        # Additional anti-drone measures for jazz
        if jazz_mode and len(sequence) >= 6:
            # Penalize too many similar events in a row
            recent_6 = sequence[-6:]
            for token in set(recent_6):
                if recent_6.count(token) >= 4:  # Same token 4+ times in recent history
                    logits[token] -= 2.0
        
        # Velocity variety for musical expression
        if jazz_mode and len(sequence) > 10:
            # Encourage velocity changes every so often
            recent_velocity_changes = sum(1 for token in sequence[-10:] 
                                        if self.velocity_start <= token <= self.velocity_end)
            if recent_velocity_changes == 0:
                for token in range(self.velocity_start, self.velocity_end + 1):
                    logits[token] += 0.8
        
        # Boost note generation if we need more notes
        if target_notes and completed_notes < target_notes * 0.8:
            for token in range(self.note_on_start, self.note_on_end + 1):
                logits[token] += 1.0

def load_generation_config(config_path: str = None) -> Dict:
    """Load generation configuration from YAML file."""
    if config_path is None:
        config_path = "configs/generation_configs/production.yaml"
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {
            'generation': {
                'max_length': 80,
                'temperature': 0.8,
                'use_constrained_sampling': True,
                'min_notes': 5,
                'target_notes': None
            },
            'export': {
                'title_prefix': 'Aurl Generated',
                'tempo': 120
            },
            'output': {
                'base_dir': 'outputs/generated',
                'include_timestamp': True,
                'generate_reports': True
            }
        }
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Production Music Generation")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/generation_configs/production.yaml", help="Generation config file")
    parser.add_argument("--output", "-o", help="Output directory (overrides config)")
    parser.add_argument("--length", "-l", type=int, help="Generation length (overrides config)")
    parser.add_argument("--temperature", "-t", type=float, help="Sampling temperature (overrides config)")
    parser.add_argument("--num-samples", "-n", type=int, default=1, help="Number of samples")
    parser.add_argument("--target-notes", type=int, help="Target number of MIDI notes")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_generation_config(args.config)
    
    # Override config with command line arguments
    gen_config = config['generation']
    if args.length:
        gen_config['max_length'] = args.length
    if args.temperature:
        gen_config['temperature'] = args.temperature
    if args.target_notes:
        gen_config['target_notes'] = args.target_notes
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.output or config['output']['base_dir']
    output_dir = Path(base_dir) / f"generation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🎼 Starting Production Music Generation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target notes: {gen_config.get('target_notes', 'adaptive')}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab_config = VocabularyConfig()
    
    # Model with architecture from checkpoint
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=checkpoint.get('d_model', 128),
        n_layers=checkpoint.get('n_layers', 2),
        n_heads=checkpoint.get('n_heads', 8),
        max_sequence_length=checkpoint.get('max_sequence_length', 256),
        mode="transformer"
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize constrained sampler for production quality
    constrained_sampler = ConstrainedSampler(model, device, vocab_config)
    exporter = MidiExporter()
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Training metrics: {checkpoint.get('metrics', {})}")
    logger.info(f"Device: {device}")
    
    # Generate samples
    successful_generations = []
    total_notes = 0
    
    for i in range(args.num_samples):
        logger.info(f"\n🎵 Generating sample {i+1}/{args.num_samples}...")
        
        try:
            # Use constrained generation for production quality
            if gen_config.get('use_constrained_sampling', True):
                generated_tokens = constrained_sampler.generate_constrained(
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    min_notes=gen_config.get('min_notes', 5),
                    target_notes=gen_config.get('target_notes')
                )
            else:
                # Fallback to standard generation
                from src.generation.sampler import SamplingStrategy, GenerationConfig
                config = GenerationConfig(
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    strategy=SamplingStrategy.TEMPERATURE
                )
                sampler = MusicSampler(model, device)
                generated_tokens = sampler.generate(config=config)
            
            # Validate generation
            sample_tokens = generated_tokens[0].cpu().numpy()
            validation_results = validate_generated_sequence(sample_tokens, vocab_config)
            
            grammar_score = validation_results['grammar_score']
            note_count = validation_results['note_count']
            pairing_score = validation_results.get('note_pairing_score', 0)
            
            logger.info(f"  Generated: {len(sample_tokens)} tokens, {note_count} note events")
            logger.info(f"  Grammar: {grammar_score:.3f}, Pairing: {pairing_score:.3f}")
            
            # Export to MIDI with enhanced naming
            title_prefix = config['export'].get('title_prefix', 'Aurl Generated')
            midi_filename = f"{title_prefix.lower().replace(' ', '_')}_sample_{i+1}_{timestamp}_notes{note_count}_grammar{grammar_score:.2f}.mid"
            midi_path = output_dir / midi_filename
            
            export_stats = exporter.export_tokens_to_midi(
                tokens=sample_tokens,
                output_path=str(midi_path),
                title=f"{title_prefix} Sample {i+1}",
                tempo=config['export'].get('tempo', 120)
            )
            
            # Generate quality report if configured
            quality_report = None
            report_path = None
            if config['output'].get('generate_reports', True):
                quality_report = get_parameter_quality_report(sample_tokens, vocab_config)
                report_path = output_dir / f"quality_report_sample_{i+1}_{timestamp}.txt"
                with open(report_path, 'w') as f:
                    f.write(quality_report)
            
            successful_generations.append({
                'sample': i+1,
                'midi_file': midi_path,
                'quality_report': report_path,
                'grammar_score': grammar_score,
                'note_count': note_count,
                'midi_notes': export_stats.total_notes,
                'duration': export_stats.total_duration,
                'pairing_score': pairing_score
            })
            
            total_notes += export_stats.total_notes
            
            logger.info(f"  ✅ MIDI Export: {export_stats.total_notes} notes, {export_stats.total_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample {i+1}: {e}")
    
    # Production summary
    if successful_generations:
        logger.info(f"\n🎉 Production Generation Complete!")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"✅ Successful generations: {len(successful_generations)}")
        logger.info(f"🎼 Total MIDI notes produced: {total_notes}")
        
        # Find best samples
        best_notes = max(successful_generations, key=lambda x: x['midi_notes'])
        best_grammar = max(successful_generations, key=lambda x: x['grammar_score'])
        
        logger.info(f"\n🏆 Best Results:")
        logger.info(f"  Most notes: {best_notes['midi_file'].name} ({best_notes['midi_notes']} notes)")
        logger.info(f"  Best grammar: {best_grammar['midi_file'].name} ({best_grammar['grammar_score']:.3f})")
        
        # Quality assessment
        avg_grammar = sum(g['grammar_score'] for g in successful_generations) / len(successful_generations)
        avg_notes = sum(g['midi_notes'] for g in successful_generations) / len(successful_generations)
        
        logger.info(f"\n📊 Quality Metrics:")
        logger.info(f"  Average Grammar Score: {avg_grammar:.3f}")
        logger.info(f"  Average MIDI Notes: {avg_notes:.1f}")
        
        # Success assessment
        if avg_grammar > 0.7 and avg_notes > 5:
            logger.info(f"🎯 PRODUCTION QUALITY ACHIEVED! Ready for musical use.")
        elif avg_notes > 2:
            logger.info(f"✅ Good progress - generating substantial musical content")
        else:
            logger.info(f"⚠️  Consider using constrained sampling for better results")
        
        return str(output_dir)
    else:
        logger.error("❌ No successful generations")
        return None

if __name__ == "__main__":
    output_dir = main()