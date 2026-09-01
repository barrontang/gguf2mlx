"""
GGUF to MLX Converter — Convert GGUF models to MLX safetensors format
for optimized inference on Apple Silicon devices.
"""

__version__ = "2.0.2"

from .gguf2mlx import build_config, convert, detect_architecture, extract_tokenizer, main

__all__ = ["build_config", "convert", "detect_architecture", "extract_tokenizer", "main"]
