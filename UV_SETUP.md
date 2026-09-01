# UV Installation Instructions for GGUF2MLX

## Quick Setup with UV (Recommended)

### Install UV
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or via pip
pip install uv
```

### Project Setup
```bash
# Clone or navigate to project directory
cd gguf2mlx

# Create virtual environment and install dependencies
uv sync

# Activate the environment (if needed for manual operations)
source .venv/bin/activate
```

### Development Setup
```bash
# Install with development dependencies
uv sync --extra dev

# Install all optional dependencies
uv sync --all-extras
```

### Running Commands
```bash
# Convert GGUF to MLX
uv run gguf2mlx --input ./models/phi3-mini.gguf --output ./models/phi3-mini-mlx

# Run demos
uv run python demo.py --input ./models/phi3-mini.gguf --prompt "Hello world" --keep
```

### Package Management
```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade

# Show installed packages
uv pip list
```

## Removing the Repository

If you no longer need this repository, you can remove it by following these steps:

1. Navigate to the parent directory of the repository:
   ```bash
   cd /path/to/parent/directory
   ```

2. Delete the repository directory:
   ```bash
   rm -rf gguf2mlx
   ```

3. (Optional) If you created a virtual environment, you can remove it as well:
   ```bash
   rm -rf gguf2mlx/.venv
   ```

4. (Optional) If you created a Conda environment for this project, you can remove it:
   ```bash
   conda env remove -n <env_name>
   ```
   Replace `<env_name>` with the name of the Conda environment you used for this repository.

This will completely remove the repository, any associated virtual environments, and Conda environments.

## Why UV?

✅ **Faster**: 10-100x faster than pip
✅ **Deterministic**: Exact reproducible installs via lock file
✅ **Reliable**: Resolves dependency conflicts automatically
✅ **Simple**: Single command setup and management
✅ **Compatible**: Works with existing pip/conda workflows