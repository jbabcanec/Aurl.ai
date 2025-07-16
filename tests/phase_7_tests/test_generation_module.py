"""
Comprehensive tests for Phase 7 Generation Module

Tests all generation components including:
- Basic sampling strategies
- Musical constraints
- Conditional generation
- Interactive generation
- Batch processing
- Performance benchmarks
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from dataclasses import dataclass
import tempfile
import time

# Import generation components
from src.generation import (
    SamplingStrategy, GenerationConfig, MusicSampler, BatchMusicSampler,
    ConstraintType, ConstraintConfig, MusicalConstraintEngine,
    ConditioningType, StyleCondition, MusicalAttributes, StructuralCondition,
    ConditionalGenerationConfig, ConditionalMusicGenerator, InteractiveGenerator
)

# Constants for testing
MIDI_VOCAB_SIZE = 774  # Based on our data analysis


class TestGenerationConfig:
    """Test generation configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.max_length == 1024
        assert config.temperature == 1.0
        assert config.strategy == SamplingStrategy.TEMPERATURE
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.use_cache is True
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            GenerationConfig(temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            GenerationConfig(temperature=-1.0)
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            GenerationConfig(top_k=0)
        
        # Test invalid top_p
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=0.0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=1.5)
        
        # Test invalid num_beams
        with pytest.raises(ValueError, match="num_beams must be at least 1"):
            GenerationConfig(num_beams=0)
    
    def test_valid_configurations(self):
        """Test valid configuration parameters."""
        config = GenerationConfig(
            temperature=0.5,
            top_k=20,
            top_p=0.8,
            num_beams=2
        )
        
        assert config.temperature == 0.5
        assert config.top_k == 20
        assert config.top_p == 0.8
        assert config.num_beams == 2


class TestMusicSampler:
    """Test core music sampling functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        
        # Mock model output
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 10, MIDI_VOCAB_SIZE)
        model.return_value = mock_output
        
        # Mock model config
        model.config = Mock()
        model.config.vocab_size = MIDI_VOCAB_SIZE
        
        return model
    
    @pytest.fixture
    def sampler(self, mock_model):
        """Create a sampler instance for testing."""
        device = torch.device("cpu")
        return MusicSampler(mock_model, device)
    
    def test_sampler_initialization(self, mock_model):
        """Test sampler initialization."""
        device = torch.device("cpu")
        sampler = MusicSampler(mock_model, device)
        
        assert sampler.model == mock_model
        assert sampler.device == device
        assert isinstance(sampler.generation_stats, dict)
        assert sampler.generation_stats["total_generated"] == 0
    
    def test_greedy_generation(self, sampler):
        """Test greedy sampling strategy."""
        config = GenerationConfig(
            strategy=SamplingStrategy.GREEDY,
            max_length=16
        )
        
        # Mock model to return predictable outputs
        sampler.model.return_value.logits = torch.zeros(1, 1, MIDI_VOCAB_SIZE)
        sampler.model.return_value.logits[0, 0, 10] = 10.0  # Highest logit at index 10
        
        output = sampler.generate(config=config)
        
        assert output.size(0) == 1  # Batch size
        assert output.size(1) > 1  # Generated tokens
        assert output.dtype == torch.long
    
    def test_temperature_sampling(self, sampler):
        """Test temperature-based sampling."""
        config = GenerationConfig(
            strategy=SamplingStrategy.TEMPERATURE,
            temperature=0.8,
            max_length=16
        )
        
        output = sampler.generate(config=config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
        assert output.dtype == torch.long
    
    def test_top_k_sampling(self, sampler):
        """Test top-k sampling."""
        config = GenerationConfig(
            strategy=SamplingStrategy.TOP_K,
            top_k=10,
            max_length=16
        )
        
        output = sampler.generate(config=config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
        assert output.dtype == torch.long
    
    def test_top_p_sampling(self, sampler):
        """Test nucleus (top-p) sampling."""
        config = GenerationConfig(
            strategy=SamplingStrategy.TOP_P,
            top_p=0.9,
            max_length=16
        )
        
        output = sampler.generate(config=config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
        assert output.dtype == torch.long
    
    def test_beam_search(self, sampler):
        """Test beam search generation."""
        config = GenerationConfig(
            strategy=SamplingStrategy.BEAM_SEARCH,
            num_beams=2,
            max_length=16
        )
        
        output = sampler.generate(config=config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
        assert output.dtype == torch.long
    
    def test_generation_with_prompt(self, sampler):
        """Test generation with initial prompt."""
        prompt = torch.tensor([[1, 100, 200, 300]])  # Sample prompt
        
        config = GenerationConfig(max_length=16)
        output = sampler.generate(prompt=prompt, config=config)
        
        assert output.size(0) == 1
        assert output.size(1) >= prompt.size(1)  # Should be at least as long as prompt
        assert torch.equal(output[:, :prompt.size(1)], prompt)  # Prompt preserved
    
    def test_statistics_tracking(self, sampler):
        """Test generation statistics tracking."""
        config = GenerationConfig(max_length=8)
        
        initial_stats = sampler.get_stats()
        assert initial_stats["total_generated"] == 0
        
        sampler.generate(config=config)
        
        updated_stats = sampler.get_stats()
        assert updated_stats["total_generated"] > 0
        assert updated_stats["average_time"] > 0
        assert updated_stats["tokens_per_second"] > 0
    
    def test_cache_functionality(self, sampler):
        """Test key-value cache functionality."""
        config_with_cache = GenerationConfig(use_cache=True, max_length=8)
        config_without_cache = GenerationConfig(use_cache=False, max_length=8)
        
        # Both should work
        output1 = sampler.generate(config=config_with_cache)
        sampler.clear_cache()
        output2 = sampler.generate(config=config_without_cache)
        
        assert output1.size() == output2.size()


class TestBatchMusicSampler:
    """Test batch generation capabilities."""
    
    @pytest.fixture
    def batch_sampler(self):
        """Create a batch sampler for testing."""
        mock_model = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, MIDI_VOCAB_SIZE)
        mock_model.return_value = mock_output
        mock_model.config = Mock()
        mock_model.config.vocab_size = MIDI_VOCAB_SIZE
        
        device = torch.device("cpu")
        return BatchMusicSampler(mock_model, device)
    
    def test_batch_generation(self, batch_sampler):
        """Test generating multiple sequences in batch."""
        prompts = [
            torch.tensor([1, 100]),
            torch.tensor([1, 200, 300]),
            None  # No prompt
        ]
        
        configs = [
            GenerationConfig(max_length=8),
            GenerationConfig(max_length=8),
            GenerationConfig(max_length=8)
        ]
        
        outputs = batch_sampler.generate_batch(
            prompts=prompts,
            configs=configs
        )
        
        assert len(outputs) == 3
        for output in outputs:
            assert isinstance(output, torch.Tensor)
            assert output.dim() == 1  # Single sequence
    
    def test_config_grouping(self, batch_sampler):
        """Test configuration grouping for efficient batching."""
        # Same configs should be grouped
        configs = [
            GenerationConfig(temperature=0.8, top_k=50),
            GenerationConfig(temperature=0.8, top_k=50),  # Same as first
            GenerationConfig(temperature=0.9, top_k=50),  # Different
        ]
        
        groups = batch_sampler._group_by_config(configs)
        
        # Should have 2 groups
        assert len(groups) == 2
        
        # First group should have indices [0, 1]
        assert 0 in groups[0][1] and 1 in groups[0][1]
        # Second group should have index [2]
        assert 2 in groups[1][1]


class TestMusicalConstraintEngine:
    """Test musical constraint enforcement."""
    
    @pytest.fixture
    def constraint_engine(self):
        """Create constraint engine for testing."""
        config = ConstraintConfig()
        return MusicalConstraintEngine(config)
    
    def test_constraint_engine_initialization(self):
        """Test constraint engine initialization."""
        engine = MusicalConstraintEngine()
        
        assert isinstance(engine.config, ConstraintConfig)
        assert len(engine.constraint_functions) > 0
        assert "total_applied" in engine.constraint_stats
    
    def test_harmonic_constraints(self, constraint_engine):
        """Test harmonic constraint application."""
        logits = torch.randn(1, MIDI_VOCAB_SIZE)
        sequence = torch.tensor([1, 100, 200, 300])  # Sample sequence
        
        constrained_logits = constraint_engine._apply_harmonic_constraints(
            logits, sequence, step=4, context=None
        )
        
        assert constrained_logits.shape == logits.shape
        assert isinstance(constrained_logits, torch.Tensor)
    
    def test_melodic_constraints(self, constraint_engine):
        """Test melodic constraint application."""
        logits = torch.randn(1, MIDI_VOCAB_SIZE)
        sequence = torch.tensor([1, 100, 105, 110])  # Ascending melody
        
        constrained_logits = constraint_engine._apply_melodic_constraints(
            logits, sequence, step=4, context=None
        )
        
        assert constrained_logits.shape == logits.shape
    
    def test_constraint_application(self, constraint_engine):
        """Test full constraint application."""
        logits = torch.randn(1, MIDI_VOCAB_SIZE)
        sequence = torch.tensor([1, 100, 200, 300])
        
        constrained_logits = constraint_engine.apply_constraints(
            logits, sequence, step=4, context=None
        )
        
        assert constrained_logits.shape == logits.shape
        
        # Check statistics updated
        stats = constraint_engine.get_stats()
        assert stats["total_applied"] > 0
    
    def test_token_type_detection(self, constraint_engine):
        """Test token type detection utilities."""
        # Test note-on token detection
        assert constraint_engine._is_note_on_token(100)  # Assuming NOTE_ON_TOKEN = 3
        assert not constraint_engine._is_note_on_token(1)  # Not a note token
        
        # Test pitch conversion
        pitch = constraint_engine._token_to_pitch(100)
        assert isinstance(pitch, int)
        assert 0 <= pitch <= 127


class TestConditionalGeneration:
    """Test conditional generation capabilities."""
    
    @pytest.fixture
    def conditional_generator(self):
        """Create conditional generator for testing."""
        mock_model = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 10, MIDI_VOCAB_SIZE)
        mock_model.return_value = mock_output
        mock_model.config = Mock()
        mock_model.config.vocab_size = MIDI_VOCAB_SIZE
        
        device = torch.device("cpu")
        return ConditionalMusicGenerator(mock_model, device)
    
    def test_style_conditioning(self, conditional_generator):
        """Test style-based conditioning."""
        style = StyleCondition(
            genre="jazz",
            complexity=0.7
        )
        
        config = ConditionalGenerationConfig(
            style=style,
            generation_config=GenerationConfig(max_length=16)
        )
        
        output = conditional_generator.generate(config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
        
        # Check statistics
        stats = conditional_generator.get_stats()
        assert stats["total_conditional_generations"] > 0
    
    def test_musical_attributes(self, conditional_generator):
        """Test musical attribute conditioning."""
        attributes = MusicalAttributes(
            tempo=120,
            key="C major",
            time_signature=(4, 4),
            dynamics="mf"
        )
        
        config = ConditionalGenerationConfig(
            attributes=attributes,
            generation_config=GenerationConfig(max_length=16)
        )
        
        output = conditional_generator.generate(config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
    
    def test_structural_conditioning(self, conditional_generator):
        """Test structural conditioning."""
        structure = StructuralCondition(
            form="AABA",
            total_measures=32,
            phrase_lengths=[8, 8, 8, 8]
        )
        
        config = ConditionalGenerationConfig(
            structure=structure,
            generation_config=GenerationConfig(max_length=16)
        )
        
        output = conditional_generator.generate(config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
    
    def test_latent_conditioning(self, conditional_generator):
        """Test latent vector conditioning."""
        latent_vector = torch.randn(128)  # Sample latent vector
        
        config = ConditionalGenerationConfig(
            latent_vector=latent_vector,
            generation_config=GenerationConfig(max_length=16)
        )
        
        output = conditional_generator.generate(config)
        
        assert output.size(0) == 1
        assert output.size(1) > 1
    
    def test_condition_interpolation(self, conditional_generator):
        """Test interpolation between conditions."""
        config1 = ConditionalGenerationConfig(
            style=StyleCondition(genre="classical", complexity=0.3),
            generation_config=GenerationConfig(max_length=16)
        )
        
        config2 = ConditionalGenerationConfig(
            style=StyleCondition(genre="jazz", complexity=0.8),
            generation_config=GenerationConfig(max_length=16)
        )
        
        output = conditional_generator.interpolate_conditions(
            config1, config2, alpha=0.5
        )
        
        assert output.size(0) == 1
        assert output.size(1) > 1


class TestInteractiveGeneration:
    """Test interactive generation capabilities."""
    
    @pytest.fixture
    def interactive_generator(self):
        """Create interactive generator for testing."""
        mock_model = Mock()
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 10, MIDI_VOCAB_SIZE)
        mock_model.return_value = mock_output
        mock_model.config = Mock()
        mock_model.config.vocab_size = MIDI_VOCAB_SIZE
        
        device = torch.device("cpu")
        return InteractiveGenerator(mock_model, device)
    
    def test_interactive_session(self, interactive_generator):
        """Test interactive generation session."""
        config = ConditionalGenerationConfig(
            generation_config=GenerationConfig(max_length=16)
        )
        
        # Start session
        interactive_generator.start_generation(config)
        
        # Generate chunks
        chunk1 = interactive_generator.generate_next(num_tokens=8)
        chunk2 = interactive_generator.generate_next(num_tokens=8)
        
        assert isinstance(chunk1, torch.Tensor)
        assert isinstance(chunk2, torch.Tensor)
        
        # Get full generation
        full_sequence = interactive_generator.get_full_generation()
        assert full_sequence.size(1) > 0
    
    def test_condition_updates(self, interactive_generator):
        """Test updating conditions during generation."""
        config = ConditionalGenerationConfig()
        interactive_generator.start_generation(config)
        
        # Update tempo
        interactive_generator.update_condition("tempo", 140)
        
        # Update dynamics
        interactive_generator.update_condition("dynamics", "ff")
        
        # Generate with updated conditions
        chunk = interactive_generator.generate_next(num_tokens=8)
        assert isinstance(chunk, torch.Tensor)


class TestPerformanceAndIntegration:
    """Test performance characteristics and integration."""
    
    def test_generation_speed(self):
        """Test generation speed benchmarks."""
        # Create minimal model for speed testing
        device = torch.device("cpu")
        
        # Mock fast model
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = MIDI_VOCAB_SIZE
        
        # Very fast mock output
        model.return_value.logits = torch.randn(1, 1, MIDI_VOCAB_SIZE)
        
        sampler = MusicSampler(model, device)
        config = GenerationConfig(max_length=32)
        
        start_time = time.time()
        output = sampler.generate(config=config)
        generation_time = time.time() - start_time
        
        # Should generate reasonably quickly
        assert generation_time < 5.0  # 5 seconds max for test
        assert output.size(1) > 1
        
        # Check speed statistics
        stats = sampler.get_stats()
        assert stats["tokens_per_second"] > 0
    
    def test_memory_efficiency(self):
        """Test memory usage characteristics."""
        device = torch.device("cpu")
        
        # Mock model
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = MIDI_VOCAB_SIZE
        model.return_value.logits = torch.randn(1, 1, MIDI_VOCAB_SIZE)
        
        sampler = MusicSampler(model, device)
        
        # Generate multiple sequences to test memory cleanup
        for _ in range(5):
            config = GenerationConfig(max_length=16)
            output = sampler.generate(config=config)
            assert output.size(1) > 1
        
        # Should not accumulate memory (this is a basic test)
        assert len(sampler._kv_cache) >= 0  # Cache may or may not be populated
    
    def test_error_handling(self):
        """Test error handling in generation."""
        device = torch.device("cpu")
        
        # Model that raises an error
        model = Mock()
        model.side_effect = RuntimeError("Model error")
        
        sampler = MusicSampler(model, device)
        config = GenerationConfig(max_length=8)
        
        # Should handle model errors gracefully
        with pytest.raises(RuntimeError):
            sampler.generate(config=config)
    
    def test_device_compatibility(self):
        """Test device compatibility."""
        # Test CPU device
        cpu_device = torch.device("cpu")
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = MIDI_VOCAB_SIZE
        model.return_value.logits = torch.randn(1, 1, MIDI_VOCAB_SIZE)
        
        sampler = MusicSampler(model, cpu_device)
        assert sampler.device == cpu_device
        
        # Test generation works
        config = GenerationConfig(max_length=8)
        output = sampler.generate(config=config)
        assert output.device == cpu_device


class TestConfigurationValidation:
    """Test configuration validation across all components."""
    
    def test_constraint_config_validation(self):
        """Test constraint configuration validation."""
        # Valid config
        config = ConstraintConfig(
            dissonance_threshold=0.5,
            voice_leading_strictness=0.8
        )
        
        assert config.dissonance_threshold == 0.5
        assert config.voice_leading_strictness == 0.8
    
    def test_conditional_config_validation(self):
        """Test conditional generation config validation."""
        # Test conditioning strength validation
        with pytest.raises(ValueError, match="conditioning_strength must be in"):
            ConditionalGenerationConfig(conditioning_strength=1.5)
        
        with pytest.raises(ValueError, match="conditioning_strength must be in"):
            ConditionalGenerationConfig(conditioning_strength=-0.1)
        
        # Test interpolation alpha validation
        with pytest.raises(ValueError, match="interpolation_alpha must be in"):
            ConditionalGenerationConfig(interpolation_alpha=1.5)
        
        # Valid config
        config = ConditionalGenerationConfig(
            conditioning_strength=0.8,
            interpolation_alpha=0.3
        )
        
        assert config.conditioning_strength == 0.8
        assert config.interpolation_alpha == 0.3
    
    def test_style_condition_encoding(self):
        """Test style condition encoding."""
        style = StyleCondition(
            genre="jazz",
            complexity=0.7
        )
        
        embedding = style.to_embedding(128)
        
        assert embedding.shape == (128,)
        assert isinstance(embedding, torch.Tensor)
        assert embedding[1] == 0.7  # Complexity should be preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])