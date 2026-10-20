# Packaging a model directory into `.mlx`

> Bundling an MLX model directory with a manifest and SHA-256 integrity metadata.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

`gguf2mlx` can now package an MLX model directory into a single `.mlx` bundle,
auto-generate a manifest list, and include SHA-256 integrity metadata.

```bash
# Package model directory into model.mlx (+ model.mlx.sha256)
gguf2mlx package --model-dir ./mlx-model --output ./mlx-model.mlx

# Verify embedded manifest and SHA-256 integrity
gguf2mlx verify --bundle ./mlx-model.mlx
```
