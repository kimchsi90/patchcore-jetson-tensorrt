"""Minimal local stand-in for the parts of xformers that vLLM 0.2.2 imports (no aarch64 wheel exists).
Prefill attention runs on torch scaled_dot_product_attention. Not the real xformers."""
__version__ = "0.0.0+jetson_shim"
from . import ops  # noqa: F401
