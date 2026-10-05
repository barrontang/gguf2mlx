from __future__ import annotations

try:
    import gguf2mlx_rust as _rust
except ImportError:
    _rust = None


def detect_architecture(
    general_architecture: str | None, general_name: str | None
) -> str | None:
    if _rust is None:
        return None
    try:
        result = _rust.detect_architecture(general_architecture, general_name)
    except Exception:  # noqa: BLE001
        return None
    return result if isinstance(result, str) and result else None


def classify_architecture(architecture: str, tensor_names: list[str]) -> str | None:
    if _rust is None:
        return None
    try:
        result = _rust.classify_architecture(architecture, tensor_names)
    except Exception:  # noqa: BLE001
        return None
    return result if result in {"dense", "moe", "hybrid"} else None
