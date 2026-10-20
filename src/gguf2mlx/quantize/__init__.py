"""Quantization helpers that touch GGUF block formats directly.

The public entry point is :mod:`gguf2mlx.quantize.gguf_codes`, which re-expresses
llama.cpp quantized blocks on MLX's affine grid without decoding to float first.
"""

from .gguf_codes import (
    AFFINE_GROUP_SIZE,
    PRESERVE_ARCHES,
    MappingResult,
    UnsupportedQuantType,
    affine_decode,
    map_tensor,
    pack_codes,
    unpack_codes,
)

__all__ = [
    "AFFINE_GROUP_SIZE",
    "PRESERVE_ARCHES",
    "MappingResult",
    "UnsupportedQuantType",
    "affine_decode",
    "map_tensor",
    "pack_codes",
    "unpack_codes",
]
