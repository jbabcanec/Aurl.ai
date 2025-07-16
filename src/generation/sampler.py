"""
Music Generation Sampler Module

Implements various sampling strategies for music generation including:
- Temperature-based sampling
- Top-k sampling
- Nucleus (top-p) sampling
- Beam search with musical constraints
- Real-time optimizations for interactive generation

This module provides the core generation infrastructure for the Aurl.ai system.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time

from ..utils.constants import (
    SPECIAL_TOKENS, MAX_SEQUENCE_LENGTH
)
from ..data.representation import EventType

# Token constants
MIDI_PAD_TOKEN = SPECIAL_TOKENS["PAD"]
MIDI_START_TOKEN = SPECIAL_TOKENS["START"] 
MIDI_END_TOKEN = SPECIAL_TOKENS["END"]
MIDI_VOCAB_SIZE = 774  # Based on our analysis
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies for generation."""
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"
    CONSTRAINED_BEAM = "constrained_beam"


@dataclass
class GenerationConfig:
    """Configuration for music generation."""
    # Basic parameters
    max_length: int = 1024
    min_length: int = 32
    temperature: float = 1.0
    
    # Sampling strategy
    strategy: SamplingStrategy = SamplingStrategy.TEMPERATURE
    
    # Top-k parameters
    top_k: int = 50
    
    # Top-p (nucleus) parameters
    top_p: float = 0.9
    
    # Beam search parameters
    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True
    
    # Musical constraints
    use_musical_constraints: bool = True
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    
    # Performance optimization
    use_cache: bool = True
    batch_size: int = 1
    
    # Conditional generation
    conditioning: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be at least 1, got {self.num_beams}")


class MusicSampler:
    """
    Advanced music generation sampler with multiple strategies.
    
    Supports various sampling methods optimized for musical generation,
    including temperature control, top-k/top-p filtering, and beam search
    with musical constraints.
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize the music sampler.
        
        Args:
            model: The trained music generation model
            device: Device to run generation on (CPU/GPU)
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Cache for key-value pairs (for transformer models)
        self._kv_cache = {}
        
        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "average_time": 0.0,
            "tokens_per_second": 0.0
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[torch.Tensor] = None,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate music sequences using the specified configuration.
        
        Args:
            prompt: Optional prompt sequence to condition generation
            config: Generation configuration
            **kwargs: Additional arguments to override config
            
        Returns:
            Generated token sequences
        """
        start_time = time.time()
        
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Validate and prepare inputs
        batch_size = config.batch_size
        if prompt is None:
            # Start with START token
            prompt = torch.full(
                (batch_size, 1), 
                MIDI_START_TOKEN, 
                dtype=torch.long, 
                device=self.device
            )
        else:
            prompt = prompt.to(self.device)
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)
            batch_size = prompt.size(0)
        
        # Clear cache if not using it
        if not config.use_cache:
            self._kv_cache.clear()
        
        # Select generation strategy
        if config.strategy == SamplingStrategy.GREEDY:
            output = self._greedy_search(prompt, config)
        elif config.strategy == SamplingStrategy.TEMPERATURE:
            output = self._temperature_sampling(prompt, config)
        elif config.strategy == SamplingStrategy.TOP_K:
            output = self._top_k_sampling(prompt, config)
        elif config.strategy == SamplingStrategy.TOP_P:
            output = self._top_p_sampling(prompt, config)
        elif config.strategy == SamplingStrategy.BEAM_SEARCH:
            output = self._beam_search(prompt, config)
        elif config.strategy == SamplingStrategy.CONSTRAINED_BEAM:
            output = self._constrained_beam_search(prompt, config)
        else:
            raise ValueError(f"Unknown sampling strategy: {config.strategy}")
        
        # Update statistics
        generation_time = time.time() - start_time
        tokens_generated = output.size(1) - prompt.size(1)
        self._update_stats(generation_time, tokens_generated)
        
        logger.info(
            f"Generated {tokens_generated} tokens in {generation_time:.2f}s "
            f"({tokens_generated/generation_time:.1f} tokens/sec)"
        )
        
        return output
    
    def _greedy_search(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Greedy decoding - always select highest probability token."""
        current_length = input_ids.size(1)
        
        for _ in range(config.max_length - current_length):
            # Get model predictions
            logits = self._get_next_token_logits(input_ids, config)
            
            # Select highest probability token
            next_tokens = logits.argmax(dim=-1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for END token
            if (next_tokens == MIDI_END_TOKEN).all():
                break
        
        return input_ids
    
    def _temperature_sampling(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Sample with temperature scaling."""
        current_length = input_ids.size(1)
        
        for _ in range(config.max_length - current_length):
            # Get model predictions
            logits = self._get_next_token_logits(input_ids, config)
            
            # Apply temperature
            logits = logits / config.temperature
            
            # Apply repetition penalty if enabled
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, input_ids, config.repetition_penalty
                )
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for END token
            if (next_tokens == MIDI_END_TOKEN).all():
                break
        
        return input_ids
    
    def _top_k_sampling(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Top-k sampling - sample from top k tokens only."""
        current_length = input_ids.size(1)
        
        for _ in range(config.max_length - current_length):
            # Get model predictions
            logits = self._get_next_token_logits(input_ids, config)
            
            # Apply temperature
            logits = logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, input_ids, config.repetition_penalty
                )
            
            # Apply top-k filtering
            logits = self._top_k_filtering(logits, config.top_k)
            
            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for END token
            if (next_tokens == MIDI_END_TOKEN).all():
                break
        
        return input_ids
    
    def _top_p_sampling(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Nucleus (top-p) sampling - sample from smallest set with cumulative prob > p."""
        current_length = input_ids.size(1)
        
        for _ in range(config.max_length - current_length):
            # Get model predictions
            logits = self._get_next_token_logits(input_ids, config)
            
            # Apply temperature
            logits = logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, input_ids, config.repetition_penalty
                )
            
            # Apply top-p filtering
            logits = self._top_p_filtering(logits, config.top_p)
            
            # Sample from filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for END token
            if (next_tokens == MIDI_END_TOKEN).all():
                break
        
        return input_ids
    
    def _beam_search(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Beam search decoding.
        
        Maintains multiple hypotheses and selects the best sequence
        based on overall probability.
        """
        batch_size = input_ids.size(0)
        num_beams = config.num_beams
        vocab_size = self.model.config.vocab_size
        
        # Expand input for beam search
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        
        # Initialize beam scores
        beam_scores = torch.zeros(batch_size * num_beams, device=self.device)
        beam_scores[1:num_beams] = -float('inf')  # Only first beam active initially
        
        # Track finished sequences
        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=self.device)
        
        current_length = input_ids.size(1)
        
        for step in range(config.max_length - current_length):
            # Get model predictions
            logits = self._get_next_token_logits(input_ids, config)
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add current beam scores
            log_probs = log_probs + beam_scores.unsqueeze(1)
            
            # Reshape for beam selection
            log_probs = log_probs.view(batch_size, num_beams * vocab_size)
            
            # Select top beams
            beam_scores, beam_indices = torch.topk(
                log_probs, num_beams, dim=-1, largest=True, sorted=True
            )
            beam_scores = beam_scores.view(-1)
            
            # Compute next tokens and beam indices
            next_tokens = beam_indices % vocab_size
            beam_indices = beam_indices // vocab_size
            
            # Reorder beams
            batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(num_beams)
            input_ids = input_ids[batch_indices * num_beams + beam_indices]
            
            # Append next tokens
            next_tokens = next_tokens.view(-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for END tokens
            done = done | (next_tokens == MIDI_END_TOKEN)
            
            # Apply length penalty
            if config.length_penalty != 1.0:
                length_penalty = ((5 + step + 1) / 6) ** config.length_penalty
                beam_scores = beam_scores / length_penalty
            
            # Early stopping
            if config.early_stopping and done.all():
                break
        
        # Select best sequences
        best_sequences = []
        for i in range(batch_size):
            batch_beam_scores = beam_scores[i * num_beams:(i + 1) * num_beams]
            best_idx = batch_beam_scores.argmax()
            best_sequences.append(input_ids[i * num_beams + best_idx])
        
        return torch.stack(best_sequences)
    
    def _constrained_beam_search(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Beam search with musical constraints.
        
        Similar to regular beam search but applies musical constraints
        during generation to ensure musical coherence.
        """
        # Start with regular beam search
        output = self._beam_search(input_ids, config)
        
        # Apply post-generation musical constraints if needed
        if config.use_musical_constraints:
            output = self._apply_musical_constraints(output, config)
        
        return output
    
    def _get_next_token_logits(
        self, 
        input_ids: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """Get logits for next token prediction."""
        # Simple forward pass (our model doesn't support caching yet)
        outputs = self.model(input_ids)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            # Assume outputs is the logits tensor directly
            logits = outputs
        
        return logits[:, -1, :]
    
    def _top_k_filtering(
        self, 
        logits: torch.Tensor, 
        top_k: int
    ) -> torch.Tensor:
        """Filter logits to keep only top k tokens."""
        if top_k > 0:
            # Remove all tokens with a probability less than the top_k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_p: float
    ) -> torch.Tensor:
        """Filter logits using nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create a copy of logits to modify
        filtered_logits = logits.clone()
        
        # Set filtered tokens to -inf
        for i in range(logits.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            filtered_logits[i, indices_to_remove] = -float('inf')
        
        return filtered_logits
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repetitive sequences."""
        for i in range(input_ids.size(0)):
            for token in input_ids[i].unique():
                if token != MIDI_PAD_TOKEN:
                    logits[i, token] /= penalty
        return logits
    
    def _apply_musical_constraints(
        self, 
        sequences: torch.Tensor, 
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Apply musical constraints to generated sequences.
        
        This is a placeholder for more sophisticated musical constraints
        that will be implemented based on music theory rules.
        """
        # TODO: Implement musical constraints based on:
        # - Harmonic progressions
        # - Voice leading rules
        # - Rhythmic patterns
        # - Dynamic consistency
        
        return sequences
    
    def _update_stats(self, generation_time: float, tokens_generated: int):
        """Update generation statistics."""
        self.generation_stats["total_generated"] += tokens_generated
        
        # Update rolling average
        n = self.generation_stats.get("num_generations", 0) + 1
        self.generation_stats["num_generations"] = n
        
        prev_avg = self.generation_stats["average_time"]
        self.generation_stats["average_time"] = (
            prev_avg * (n - 1) + generation_time
        ) / n
        
        self.generation_stats["tokens_per_second"] = (
            self.generation_stats["total_generated"] / 
            (self.generation_stats["average_time"] * n)
        )
    
    def clear_cache(self):
        """Clear the key-value cache."""
        self._kv_cache.clear()
    
    def get_stats(self) -> Dict[str, float]:
        """Get generation statistics."""
        return self.generation_stats.copy()


class BatchMusicSampler(MusicSampler):
    """
    Extended sampler with optimized batch generation capabilities.
    
    Supports generating multiple sequences in parallel for improved
    throughput in production scenarios.
    """
    
    def generate_batch(
        self,
        prompts: Optional[List[torch.Tensor]] = None,
        configs: Optional[List[GenerationConfig]] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate multiple sequences in parallel.
        
        Args:
            prompts: List of prompt sequences
            configs: List of generation configurations
            **kwargs: Common overrides for all generations
            
        Returns:
            List of generated sequences
        """
        if prompts is None:
            prompts = [None]
        
        if configs is None:
            configs = [GenerationConfig() for _ in prompts]
        
        # Group by similar configurations for efficient batching
        config_groups = self._group_by_config(configs)
        
        results = []
        for group_configs, group_indices in config_groups:
            # Get prompts for this group
            group_prompts = [prompts[i] for i in group_indices]
            
            # Pad prompts to same length
            padded_prompts = self._pad_prompts(group_prompts)
            
            # Generate with shared config
            group_output = self.generate(
                padded_prompts,
                group_configs[0],
                **kwargs
            )
            
            # Split results
            for i, idx in enumerate(group_indices):
                results.append((idx, group_output[i]))
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _group_by_config(
        self, 
        configs: List[GenerationConfig]
    ) -> List[Tuple[List[GenerationConfig], List[int]]]:
        """Group configurations by similarity for batching."""
        groups = []
        
        for i, config in enumerate(configs):
            # Find matching group
            matched = False
            for group_configs, indices in groups:
                if self._configs_match(config, group_configs[0]):
                    group_configs.append(config)
                    indices.append(i)
                    matched = True
                    break
            
            if not matched:
                groups.append(([config], [i]))
        
        return groups
    
    def _configs_match(
        self, 
        c1: GenerationConfig, 
        c2: GenerationConfig
    ) -> bool:
        """Check if two configs can be batched together."""
        return (
            c1.strategy == c2.strategy and
            c1.temperature == c2.temperature and
            c1.top_k == c2.top_k and
            c1.top_p == c2.top_p and
            c1.max_length == c2.max_length
        )
    
    def _pad_prompts(
        self, 
        prompts: List[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Pad prompts to same length for batching."""
        # Handle None prompts
        processed = []
        for p in prompts:
            if p is None:
                processed.append(
                    torch.tensor(
                        [MIDI_START_TOKEN], 
                        device=self.device
                    )
                )
            else:
                processed.append(p.to(self.device))
        
        # Find max length
        max_len = max(p.size(-1) for p in processed)
        
        # Pad sequences
        padded = []
        for p in processed:
            if p.size(-1) < max_len:
                padding = torch.full(
                    (max_len - p.size(-1),),
                    MIDI_PAD_TOKEN,
                    device=self.device
                )
                padded.append(torch.cat([p, padding]))
            else:
                padded.append(p)
        
        return torch.stack(padded)