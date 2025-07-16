"""
Conditional Generation Module

Implements conditional generation capabilities for Aurl.ai including:
- Style conditioning (genre, composer, mood)
- Musical attribute control (tempo, key, dynamics)
- Structural conditioning (form, length, complexity)
- Interactive generation with real-time control
- Latent space manipulation for VAE models

This module enables fine-grained control over the generation process.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.base_logger import setup_logger
from ..utils.constants import MAX_SEQUENCE_LENGTH, SPECIAL_TOKENS

# Token constants
MIDI_START_TOKEN = SPECIAL_TOKENS["START"]
MIDI_END_TOKEN = SPECIAL_TOKENS["END"]
from .sampler import GenerationConfig, MusicSampler
from .constraints import ConstraintConfig, MusicalConstraintEngine

logger = setup_logger(__name__)


class ConditioningType(Enum):
    """Types of conditioning available."""
    STYLE = "style"
    TEMPO = "tempo"
    KEY = "key"
    TIME_SIGNATURE = "time_signature"
    DYNAMICS = "dynamics"
    MOOD = "mood"
    STRUCTURE = "structure"
    INSTRUMENTS = "instruments"
    LATENT = "latent"
    TEXT = "text"


@dataclass
class StyleCondition:
    """Style-based conditioning parameters."""
    genre: Optional[str] = None  # "classical", "jazz", "pop", etc.
    composer: Optional[str] = None  # For composer-specific style
    era: Optional[str] = None  # "baroque", "romantic", "modern", etc.
    complexity: float = 0.5  # 0-1, musical complexity level
    
    def to_embedding(self, embedding_dim: int = 128) -> torch.Tensor:
        """Convert style conditions to embedding vector."""
        # This is a placeholder - in practice, you'd use learned embeddings
        embedding = torch.zeros(embedding_dim)
        
        # Simple encoding for demonstration
        if self.genre:
            genre_map = {"classical": 0.0, "jazz": 0.33, "pop": 0.66, "other": 1.0}
            embedding[0] = genre_map.get(self.genre.lower(), 0.5)
        
        embedding[1] = self.complexity
        
        return embedding


@dataclass
class MusicalAttributes:
    """Musical attribute conditioning."""
    tempo: Optional[int] = None  # BPM
    key: Optional[str] = None  # "C major", "A minor", etc.
    time_signature: Optional[Tuple[int, int]] = None  # (4, 4), (3, 4), etc.
    dynamics: Optional[str] = None  # "pp", "mf", "ff", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for constraint engine."""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }


@dataclass
class StructuralCondition:
    """Structural conditioning parameters."""
    form: Optional[str] = None  # "AABA", "verse-chorus", etc.
    total_measures: Optional[int] = None
    phrase_lengths: Optional[List[int]] = None
    sections: Optional[List[str]] = None  # ["intro", "verse", "chorus", ...]
    
    def get_section_boundaries(self) -> List[int]:
        """Get token positions for section boundaries."""
        if not self.phrase_lengths:
            return []
        
        boundaries = [0]
        position = 0
        for length in self.phrase_lengths:
            position += length * 16  # Approximate tokens per phrase
            boundaries.append(position)
        
        return boundaries


@dataclass 
class ConditionalGenerationConfig:
    """Complete configuration for conditional generation."""
    # Base generation config
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Constraint config
    constraint_config: ConstraintConfig = field(default_factory=ConstraintConfig)
    
    # Conditioning parameters
    style: Optional[StyleCondition] = None
    attributes: Optional[MusicalAttributes] = None
    structure: Optional[StructuralCondition] = None
    
    # Advanced conditioning
    latent_vector: Optional[torch.Tensor] = None  # For VAE models
    text_prompt: Optional[str] = None  # For text-conditioned models
    reference_sequence: Optional[torch.Tensor] = None  # For style transfer
    
    # Control parameters
    conditioning_strength: float = 1.0  # How strongly to apply conditioning
    interpolation_alpha: float = 0.0  # For blending conditions
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.conditioning_strength <= 1:
            raise ValueError(
                f"conditioning_strength must be in [0, 1], got {self.conditioning_strength}"
            )
        if not 0 <= self.interpolation_alpha <= 1:
            raise ValueError(
                f"interpolation_alpha must be in [0, 1], got {self.interpolation_alpha}"
            )


class ConditionalMusicGenerator:
    """
    Advanced conditional music generation system.
    
    This class orchestrates conditional generation by combining the base sampler
    with conditioning mechanisms and constraint application.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        condition_encoder: Optional[torch.nn.Module] = None
    ):
        """
        Initialize conditional generator.
        
        Args:
            model: The music generation model
            device: Device to run on
            condition_encoder: Optional encoder for conditioning inputs
        """
        self.model = model
        self.device = device
        self.condition_encoder = condition_encoder
        
        # Initialize components
        self.sampler = MusicSampler(model, device)
        self.constraint_engine = MusicalConstraintEngine()
        
        # Conditioning cache
        self._condition_cache = {}
        
        # Statistics
        self.generation_stats = {
            "total_conditional_generations": 0,
            "conditioning_types_used": {},
            "average_conditioning_strength": 0.0
        }
    
    def generate(
        self,
        config: ConditionalGenerationConfig,
        prompt: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate music with conditional control.
        
        Args:
            config: Conditional generation configuration
            prompt: Optional prompt sequence
            **kwargs: Additional generation arguments
            
        Returns:
            Generated music sequence
        """
        # Update statistics
        self.generation_stats["total_conditional_generations"] += 1
        
        # Prepare conditioning
        conditioning_dict = self._prepare_conditioning(config)
        
        # Update constraint config based on conditioning
        self._update_constraints_from_conditioning(config)
        
        # Set up conditioned generation
        if hasattr(self.model, 'set_conditioning'):
            self.model.set_conditioning(conditioning_dict)
        
        # Custom generation loop with conditioning
        generated = self._conditional_generation_loop(
            prompt=prompt,
            config=config,
            conditioning=conditioning_dict,
            **kwargs
        )
        
        # Clear conditioning
        if hasattr(self.model, 'clear_conditioning'):
            self.model.clear_conditioning()
        
        return generated
    
    def _prepare_conditioning(
        self,
        config: ConditionalGenerationConfig
    ) -> Dict[str, torch.Tensor]:
        """Prepare all conditioning inputs."""
        conditioning = {}
        
        # Style conditioning
        if config.style is not None:
            style_embedding = config.style.to_embedding()
            conditioning["style"] = style_embedding.to(self.device)
            self._update_conditioning_stats(ConditioningType.STYLE)
        
        # Musical attributes
        if config.attributes is not None:
            attr_dict = config.attributes.to_dict()
            if "tempo" in attr_dict:
                conditioning["tempo"] = torch.tensor(
                    [attr_dict["tempo"]], device=self.device
                )
                self._update_conditioning_stats(ConditioningType.TEMPO)
            
            if "key" in attr_dict:
                # Convert key to embedding
                key_embedding = self._encode_key(attr_dict["key"])
                conditioning["key"] = key_embedding
                self._update_conditioning_stats(ConditioningType.KEY)
        
        # Structural conditioning
        if config.structure is not None:
            structure_encoding = self._encode_structure(config.structure)
            conditioning["structure"] = structure_encoding
            self._update_conditioning_stats(ConditioningType.STRUCTURE)
        
        # Latent conditioning (for VAE models)
        if config.latent_vector is not None:
            conditioning["latent"] = config.latent_vector.to(self.device)
            self._update_conditioning_stats(ConditioningType.LATENT)
        
        # Text conditioning
        if config.text_prompt is not None:
            if self.condition_encoder is not None:
                text_encoding = self.condition_encoder(config.text_prompt)
                conditioning["text"] = text_encoding
                self._update_conditioning_stats(ConditioningType.TEXT)
        
        # Apply conditioning strength
        if config.conditioning_strength < 1.0:
            for key in conditioning:
                conditioning[key] = conditioning[key] * config.conditioning_strength
        
        return conditioning
    
    def _conditional_generation_loop(
        self,
        prompt: Optional[torch.Tensor],
        config: ConditionalGenerationConfig,
        conditioning: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Custom generation loop with conditioning and constraints.
        
        This method overrides the standard generation to apply
        conditioning at each step.
        """
        # Initialize sequence
        if prompt is None:
            current_seq = torch.full(
                (1, 1), 
                fill_value=1,  # START token
                device=self.device
            )
        else:
            current_seq = prompt.to(self.device)
        
        # Generation loop
        for step in range(config.generation_config.max_length):
            # Get model predictions with conditioning
            with torch.no_grad():
                if hasattr(self.model, 'forward_with_conditioning'):
                    outputs = self.model.forward_with_conditioning(
                        current_seq,
                        conditioning=conditioning
                    )
                else:
                    outputs = self.model(current_seq)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits'][:, -1, :]
                else:
                    # Assume outputs is the logits tensor directly
                    logits = outputs[:, -1, :]
            
            # Apply temperature
            logits = logits / config.generation_config.temperature
            
            # Apply constraints based on conditioning
            logits = self.constraint_engine.apply_constraints(
                logits=logits,
                generated_sequence=current_seq,
                step=step,
                context=conditioning
            )
            
            # Apply structural conditioning
            if config.structure is not None:
                logits = self._apply_structural_constraints(
                    logits, current_seq, step, config.structure
                )
            
            # Sample next token
            if config.generation_config.strategy.value == "greedy":
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Ensure next_token has correct shape [batch_size, 1]
            if next_token.dim() == 0:  # Scalar
                next_token = next_token.unsqueeze(0).unsqueeze(0)
            elif next_token.dim() == 1:  # [batch_size]
                next_token = next_token.unsqueeze(1)
            # If already [batch_size, 1], keep as is
            
            # Append to sequence
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            # Check stopping conditions
            if self._should_stop_generation(
                current_seq, step, config
            ):
                break
        
        return current_seq
    
    def _update_constraints_from_conditioning(
        self,
        config: ConditionalGenerationConfig
    ):
        """Update constraint configuration based on conditioning."""
        # Update style-specific constraints
        if config.style is not None and config.style.genre is not None:
            self.constraint_engine.config.style = config.style.genre
        
        # Update musical attribute constraints
        if config.attributes is not None:
            if config.attributes.time_signature is not None:
                self.constraint_engine.config.allowed_time_signatures = [
                    config.attributes.time_signature
                ]
            
            if config.attributes.dynamics is not None:
                # Map dynamics to velocity range
                dynamics_map = {
                    "pp": (20, 40),
                    "p": (40, 60),
                    "mp": (50, 70),
                    "mf": (60, 80),
                    "f": (80, 100),
                    "ff": (100, 120)
                }
                if config.attributes.dynamics in dynamics_map:
                    self.constraint_engine.config.dynamic_range = (
                        dynamics_map[config.attributes.dynamics]
                    )
    
    def _apply_structural_constraints(
        self,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        step: int,
        structure: StructuralCondition
    ) -> torch.Tensor:
        """Apply structural conditioning to generation."""
        # Get section boundaries
        boundaries = structure.get_section_boundaries()
        
        # Find current section
        current_length = sequence.size(1)
        current_section = 0
        for i, boundary in enumerate(boundaries[1:]):
            if current_length < boundary:
                current_section = i
                break
        
        # Apply section-specific constraints
        if structure.sections and current_section < len(structure.sections):
            section_name = structure.sections[current_section]
            
            # Adjust generation based on section type
            if section_name == "intro":
                # Encourage simpler patterns
                pass
            elif section_name == "chorus":
                # Encourage more energetic patterns
                pass
            elif section_name == "outro":
                # Encourage resolution
                pass
        
        return logits
    
    def _should_stop_generation(
        self,
        sequence: torch.Tensor,
        step: int,
        config: ConditionalGenerationConfig
    ) -> bool:
        """Determine if generation should stop."""
        # Check maximum length
        if sequence.size(1) >= config.generation_config.max_length:
            return True
        
        # Check structural completion
        if config.structure is not None:
            if config.structure.total_measures is not None:
                # Estimate current measures
                estimated_measures = sequence.size(1) / 64  # Rough estimate
                if estimated_measures >= config.structure.total_measures:
                    return True
        
        # Check for END token
        if sequence.size(1) > 0:
            last_token = sequence[0, -1].item()
            if last_token == 2:  # END token
                return True
        
        return False
    
    def _encode_key(self, key_str: str) -> torch.Tensor:
        """Encode musical key as tensor."""
        # Simple encoding - in practice, use learned embeddings
        key_map = {
            "C major": 0, "G major": 1, "D major": 2, "A major": 3,
            "E major": 4, "B major": 5, "F major": 11, "Bb major": 10,
            "Eb major": 9, "Ab major": 8, "Db major": 7, "Gb major": 6,
            "A minor": 12, "E minor": 13, "B minor": 14, "F# minor": 15,
            "C# minor": 16, "G# minor": 17, "D minor": 18, "G minor": 19,
            "C minor": 20, "F minor": 21, "Bb minor": 22, "Eb minor": 23
        }
        
        key_idx = key_map.get(key_str, 0)
        key_tensor = torch.zeros(24)
        key_tensor[key_idx] = 1.0
        
        return key_tensor.to(self.device)
    
    def _encode_structure(
        self, 
        structure: StructuralCondition
    ) -> torch.Tensor:
        """Encode structural information as tensor."""
        # Simple encoding for demonstration
        encoding = torch.zeros(32)
        
        if structure.total_measures is not None:
            encoding[0] = structure.total_measures / 100.0
        
        if structure.phrase_lengths is not None:
            avg_phrase_length = np.mean(structure.phrase_lengths)
            encoding[1] = avg_phrase_length / 16.0
        
        return encoding.to(self.device)
    
    def _update_conditioning_stats(self, condition_type: ConditioningType):
        """Update conditioning usage statistics."""
        key = condition_type.value
        self.generation_stats["conditioning_types_used"][key] = (
            self.generation_stats["conditioning_types_used"].get(key, 0) + 1
        )
    
    def interpolate_conditions(
        self,
        config1: ConditionalGenerationConfig,
        config2: ConditionalGenerationConfig,
        alpha: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate with interpolated conditions.
        
        Useful for smooth transitions between styles or creating
        hybrid musical outputs.
        """
        # Prepare both conditions
        cond1 = self._prepare_conditioning(config1)
        cond2 = self._prepare_conditioning(config2)
        
        # Interpolate
        interpolated = {}
        for key in set(cond1.keys()) | set(cond2.keys()):
            if key in cond1 and key in cond2:
                interpolated[key] = (
                    (1 - alpha) * cond1[key] + alpha * cond2[key]
                )
            elif key in cond1:
                interpolated[key] = (1 - alpha) * cond1[key]
            else:
                interpolated[key] = alpha * cond2[key]
        
        # Create interpolated config
        interp_config = ConditionalGenerationConfig(
            generation_config=config1.generation_config,
            constraint_config=config1.constraint_config
        )
        
        # Generate with interpolated conditions
        return self._conditional_generation_loop(
            prompt=kwargs.get('prompt'),
            config=interp_config,
            conditioning=interpolated,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.generation_stats.copy()
        stats["sampler_stats"] = self.sampler.get_stats()
        stats["constraint_stats"] = self.constraint_engine.get_stats()
        return stats


class InteractiveGenerator(ConditionalMusicGenerator):
    """
    Interactive music generator for real-time applications.
    
    Supports streaming generation and dynamic condition updates.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize interactive generator."""
        super().__init__(*args, **kwargs)
        self.current_conditions = {}
        self.generation_buffer = []
    
    def start_generation(
        self,
        initial_config: ConditionalGenerationConfig
    ):
        """Start an interactive generation session."""
        self.current_conditions = self._prepare_conditioning(initial_config)
        self.generation_buffer = []
        logger.info("Started interactive generation session")
    
    def generate_next(
        self,
        num_tokens: int = 16
    ) -> torch.Tensor:
        """Generate next batch of tokens."""
        # Use current buffer as prompt
        if self.generation_buffer:
            prompt = torch.cat(self.generation_buffer, dim=1)
        else:
            prompt = None
        
        # Generate short sequence
        config = ConditionalGenerationConfig(
            generation_config=GenerationConfig(max_length=num_tokens)
        )
        
        next_tokens = self._conditional_generation_loop(
            prompt=prompt,
            config=config,
            conditioning=self.current_conditions
        )
        
        # Add to buffer
        self.generation_buffer.append(next_tokens)
        
        return next_tokens
    
    def update_condition(
        self,
        condition_type: str,
        value: Any
    ):
        """Update a specific condition during generation."""
        if condition_type == "tempo":
            self.current_conditions["tempo"] = torch.tensor(
                [value], device=self.device
            )
        elif condition_type == "dynamics":
            # Update dynamic range in constraint engine
            dynamics_map = {
                "pp": (20, 40), "p": (40, 60), "mp": (50, 70),
                "mf": (60, 80), "f": (80, 100), "ff": (100, 120)
            }
            if value in dynamics_map:
                self.constraint_engine.config.dynamic_range = dynamics_map[value]
        
        logger.info(f"Updated condition: {condition_type} = {value}")
    
    def get_full_generation(self) -> torch.Tensor:
        """Get the complete generated sequence."""
        if self.generation_buffer:
            return torch.cat(self.generation_buffer, dim=1)
        return torch.empty(1, 0, device=self.device)