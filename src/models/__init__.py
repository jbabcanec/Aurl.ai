"""
Model components for Aurl.ai music generation.
"""

from .music_transformer_vae_gan import MusicTransformerVAEGAN, create_model_from_config
from .components import BaselineTransformer, create_baseline_model
from .attention import (
    HierarchicalAttention, 
    SlidingWindowAttention, 
    MultiScaleAttention,
    MusicalPositionalEncoding,
    create_attention_layer
)
from .discriminator import MultiScaleDiscriminator
from .gan_losses import ComprehensiveGANLoss, FeatureMatchingLoss, SpectralRegularization
from .encoder import EnhancedMusicEncoder
from .decoder import EnhancedMusicDecoder
from .vae_components import MusicalPrior, LatentAnalyzer, AdaptiveBeta

__all__ = [
    'MusicTransformerVAEGAN',
    'create_model_from_config', 
    'BaselineTransformer',
    'create_baseline_model',
    'HierarchicalAttention',
    'SlidingWindowAttention', 
    'MultiScaleAttention',
    'MusicalPositionalEncoding',
    'create_attention_layer',
    'MultiScaleDiscriminator',
    'ComprehensiveGANLoss',
    'FeatureMatchingLoss', 
    'SpectralRegularization',
    'EnhancedMusicEncoder',
    'EnhancedMusicDecoder',
    'MusicalPrior',
    'LatentAnalyzer',
    'AdaptiveBeta'
]