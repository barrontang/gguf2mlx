"""Fail-closed, full-precision llama.cpp/MLX perplexity quality gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(path: Path) -> dict[str, int]:
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(str(path))
    counts: dict[str, int] = {}
    allowed = {
        GGMLQuantizationType.F32, GGMLQuantizationType.F16,
        GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0,
    }
    for tensor in reader.tensors:
        kind = GGMLQuantizationType(tensor.tensor_type)
        if kind not in allowed:
            raise ValueError(f"MVP benchmark does not support {kind.name}: {tensor.name}")
        counts[kind.name] = counts.get(kind.name, 0) + 1
    if not counts.get("Q4_0"):
        raise ValueError("The reference model must contain Q4_0 tensors")
    return counts


def _shared_tokens(reference, tokenizer, text: str, add_bos: bool) -> list[int]:
    native = reference.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    converted = tokenizer.encode(text, add_special_tokens=False)
    if native != converted:
        raise ValueError("GGUF and MLX tokenizers disagree on the evaluation corpus")
    if add_bos:
        bos = reference.token_bos()
        if bos < 0 or bos != tokenizer.bos_token_id:
            raise ValueError("GGUF and MLX BOS token IDs disagree")
    return native


def _windows(tokens: list[int], length: int, samples: int, bos: int | None) -> np.ndarray:
    if length < 2 or samples < 1:
        raise ValueError("sequence length must be >= 2 and sample count must be >= 1")
    width = length - int(bos is not None)
    needed = width * samples
    if len(tokens) < needed:
        raise ValueError(f"Corpus needs {needed} tokens; only {len(tokens)} are available")
    data = np.asarray(tokens[:needed], dtype=np.int32).reshape(samples, width)
    if bos is not None:
        data = np.concatenate((np.full((samples, 1), bos, dtype=np.int32), data), axis=1)
    return data


def _compare(reference: float, converted: float) -> dict:
    if not all(math.isfinite(value) and value >= 1.0 for value in (reference, converted)):
        raise ValueError("Both perplexities must be finite and >= 1")
    return {
        "llama_cpp": reference,
        "mlx": converted,
        "llama_cpp_hex": reference.hex(),
        "mlx_hex": converted.hex(),
        "absolute_delta": abs(converted - reference),
        "passed": reference == converted,
        "tolerance": 0.0,
    }


class _LlamaLogits:
    """Feed native llama.cpp logits into the same MLX-LM scoring implementation."""

    def __init__(self, reference, mx):
        self.reference = reference
        self.mx = mx
        self.digest = hashlib.sha256()

    def __call__(self, batch):
        rows = []
        for tokens in np.asarray(batch):
            self.reference.reset()
            self.reference.eval(tokens.tolist())
            # logits_all=True retains all positions, rather than only the last token.
            logits = np.asarray(
                self.reference.scores[:len(tokens)], dtype=np.float32
            ).copy()
            if logits.shape != (len(tokens), self.reference.n_vocab()):
                raise ValueError("llama.cpp did not return every token's logits")
            self.digest.update(logits.astype("<f4", copy=False).tobytes())
            rows.append(logits)
        return self.mx.array(np.stack(rows))


class _MlxLogits:
    def __init__(self, model, mx, vocab_size):
        self.model = model
        self.mx = mx
        self.vocab_size = vocab_size
        self.digest = hashlib.sha256()

    def __call__(self, batch):
        logits = self.model(batch).astype(self.mx.float32)
        if logits.shape != (*batch.shape, self.vocab_size):
            raise ValueError("MLX and GGUF logit dimensions disagree")
        self.mx.eval(logits)
        self.digest.update(np.asarray(logits).astype("<f4", copy=False).tobytes())
        return logits


def _evaluate(args) -> dict:
    counts = _validate_source(args.input)
    import mlx.core as mx
    from llama_cpp import Llama, llama_print_system_info
    from mlx_lm.perplexity import eval_ppl
    from mlx_lm.utils import load

    text = args.corpus.read_text(encoding="utf-8")
    model, tokenizer = load(str(args.mlx_model))
    with contextlib.closing(Llama(
        model_path=str(args.input), n_ctx=args.sequence_length,
        n_batch=args.sequence_length, logits_all=True,
        n_gpu_layers=args.n_gpu_layers, seed=0, verbose=False,
    )) as reference:
        tokens = _shared_tokens(reference, tokenizer, text, args.add_bos)
        data = _windows(
            tokens, args.sequence_length, args.num_samples,
            reference.token_bos() if args.add_bos else None,
        )
        if len(tokenizer.get_vocab()) != reference.n_vocab():
            raise ValueError("GGUF and MLX vocabulary sizes disagree")
        native = _LlamaLogits(reference, mx)
        converted = _MlxLogits(model, mx, reference.n_vocab())
        # One sequence per batch keeps scoring and context resets identical.
        native_ppl, native_se = eval_ppl(native, mx.array(data), batch_size=1)
        mlx_ppl, mlx_se = eval_ppl(converted, mx.array(data), batch_size=1)

    comparison = _compare(float(native_ppl), float(mlx_ppl))
    if not all(math.isfinite(float(value)) for value in (native_se, mlx_se)):
        raise ValueError("Perplexity standard errors must be finite; use more scored tokens")
    comparison.update({
        "llama_cpp_std_error": float(native_se),
        "mlx_std_error": float(mlx_se),
        "logits_identical": native.digest.digest() == converted.digest.digest(),
        "llama_cpp_logits_sha256": native.digest.hexdigest(),
        "mlx_logits_sha256": converted.digest.hexdigest(),
    })
    artifacts = [
        path for path in args.mlx_model.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".json", ".model", ".jinja"}
    ]
    if not any(path.suffix == ".safetensors" for path in artifacts):
        raise ValueError("MLX model directory contains no safetensors weights")
    return {
        "success": comparison["passed"],
        "comparison": comparison,
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "source_tensor_types": counts,
        "mlx_model": str(args.mlx_model),
        "mlx_artifacts_sha256": {
            str(path.relative_to(args.mlx_model)): _sha256(path) for path in sorted(artifacts)
        },
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "token_windows_sha256": hashlib.sha256(data.astype("<i4").tobytes()).hexdigest(),
        "sequence_length": args.sequence_length,
        "num_samples": args.num_samples,
        "scored_tokens": args.num_samples * (args.sequence_length - 1),
        "unused_corpus_tokens": len(tokens) - data.size + args.num_samples * args.add_bos,
        "add_bos_per_window": args.add_bos,
        "scoring": "mlx_lm.perplexity.eval_ppl; all next tokens; batch_size=1",
        "n_gpu_layers": args.n_gpu_layers,
        "llama_cpp_system_info": llama_print_system_info().decode("utf-8"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("gguf", "numpy", "mlx", "mlx-lm", "llama-cpp-python")
        },
        "bit_exact_transpacking_proven": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-ppl", action="store_true", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Original Q4_0 GGUF")
    parser.add_argument("--mlx-model", type=Path, required=True, help="Converted local MLX model")
    parser.add_argument("--corpus", type=Path, required=True, help="Fixed UTF-8 text, no downloads")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--add-bos", action="store_true", help="Prepend matching BOS to each window")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="llama.cpp offload; 0 is CPU")
    args = parser.parse_args(argv)
    for name in ("input", "mlx_model", "corpus", "result_json"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        if args.sequence_length < 2 or args.num_samples < 1:
            raise ValueError("sequence length must be >= 2 and sample count must be >= 1")
        if not args.input.is_file() or not args.corpus.is_file() or not args.mlx_model.is_dir():
            raise ValueError("Input, corpus, and local MLX model must exist")
        if args.result_json.exists():
            raise ValueError("Result path must be new; existing files will not be overwritten")
        if (
            args.result_json in {args.input, args.corpus}
            or args.result_json.is_relative_to(args.mlx_model)
        ):
            raise ValueError("Result path must not overwrite input, corpus, or MLX artifacts")
        result = _evaluate(args)
    except Exception as error:  # noqa: BLE001 - runtime failures must fail the gate, never skip
        result = {"success": False, "error": f"{type(error).__name__}: {error}"}
    rendered = json.dumps(result, indent=2, allow_nan=False)
    print(rendered)
    if args.result_json.exists():
        return 1
    if args.result_json not in {args.input, args.corpus} and not args.result_json.is_relative_to(
        args.mlx_model
    ):
        try:
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            with args.result_json.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        except OSError as error:
            print(f"Cannot write report: {error}", file=sys.stderr)
            return 1
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
