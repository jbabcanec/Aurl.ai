"""
MidiFly: State-of-the-Art Music Generation AI

A comprehensive music generation system using transformer-based architectures
with VAE and GAN components for creating high-quality MIDI compositions.
"""

__version__ = "0.1.0"
__author__ = "MidiFly Team"
__email__ = "contact@midifly.ai"

from src.utils.logger import default_logger as logger

# Package-level exports
__all__ = ["__version__", "__author__", "__email__", "logger"]