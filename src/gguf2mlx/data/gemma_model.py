"""Gemma runtime adapter preserving the configured GELU activation."""

from dataclasses import dataclass

from mlx import nn
from mlx_lm.models.gemma import MLP as BaseMLP
from mlx_lm.models.gemma import Model as BaseModel
from mlx_lm.models.gemma import ModelArgs as BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_activation: str = "gelu_pytorch_tanh"


class MLP(BaseMLP):
    def __init__(self, dim: int, hidden_dim: int, activation: str):
        super().__init__(dim, hidden_dim)
        if activation == "gelu_pytorch_tanh":
            self.activation = nn.gelu_approx
        elif activation == "gelu":
            self.activation = nn.gelu
        else:
            raise ValueError(f"Unsupported Gemma activation: {activation}")

    def __call__(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class Model(BaseModel):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        for layer in self.model.layers:
            layer.mlp = MLP(args.hidden_size, args.intermediate_size, args.hidden_activation)
