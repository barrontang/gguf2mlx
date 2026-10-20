"""Weight-fidelity auditing and the conversion quality gates.

Why this module exists
----------------------
Earlier releases used cross-runtime perplexity as the release gate. That single
number mixes two unrelated questions: *did the converter preserve the weights*,
and *how much accuracy did the source quantization cost*. The first is
deterministic and can be answered exactly; the second is noisy and can never
support a zero-tolerance gate. Using one metric for both is why the gate was
permanently red and why a correct converter looked broken.

The three tiers implemented here keep them apart:

``weight``
    **Primary gate, deterministic.** Compare the safetensors that were actually
    written against an independent float32 dequantization of the source GGUF.
    Runs without a runtime, so CI can enforce it. Only tensors that survived the
    mapping unchanged are compared position-wise; fused/permuted/split tensors
    are counted and reported as ``transformed`` instead of being silently
    dropped.

``numerical``
    **Secondary gate.** Requires a runtime: :func:`load_and_run` loads the
    converted directory with ``mlx_lm`` and checks the forward pass produces
    finite logits. This catches layout and fusion errors that a per-tensor check
    cannot see.

``perplexity``
    **Report only.** llama.cpp-style PPL deltas are recorded for information.
    Nothing in this module fails because of them.

Every :func:`write_report` payload is plain JSON so it can be diffed, archived,
and consumed by CI.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .quantize.gguf_codes import PRESERVE_TOLERANCE

SCHEMA_VERSION = 2
REPORT_NAME = "conversion_report.json"

# A relative L2 error larger than these is not rounding: it means values changed
# shape, order, or content. Half-precision output legitimately loses ~3e-4
# relative to an fp32 reference, so the tolerance is the dtype's unit roundoff
# with comfortable headroom.
DTYPE_EPSILON: dict[str, float] = {
    "float16": 6.0e-4,
    "bfloat16": 5.0e-3,
    "float32": 1.0e-7,
}

# Tiers that must pass before an artifact is considered trustworthy.
GATED_TIERS = ("weight", "numerical")


def _bf16_bits_to_float32(raw: np.ndarray) -> np.ndarray:
    """Reinterpret bfloat16 bit patterns stored as uint16 as float32."""
    bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _raw_as(raw: np.ndarray, dtype: Any) -> np.ndarray:
    """View GGUF raw storage as ``dtype`` without reinterpreting values.

    ``GGUFReader`` exposes most tensors as a typed memmap, but tensors written
    with an explicit ``raw_dtype`` (BF16 in particular) arrive as uint8 bytes.
    Casting those to a wider dtype would misread every value, so view instead.
    """
    flat = np.ascontiguousarray(raw).reshape(-1)
    if flat.dtype == np.dtype(dtype):
        return flat
    if flat.dtype == np.uint8:
        return flat.view(np.dtype(dtype))
    return flat.astype(dtype)


def reference_tensor(tensor: Any) -> np.ndarray:
    """Decode one GGUF tensor to float32 straight from its stored bytes.

    This deliberately does not go through the converter's own decode helper: the
    point of an audit is to be independent of the code path being audited.
    """
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    qtype = int(tensor.tensor_type)
    shape = tuple(reversed(tuple(tensor.shape)))
    raw = tensor.data

    if qtype == 0:  # F32
        return _raw_as(raw, np.float32).reshape(shape)
    if qtype == 1:  # F16
        return _raw_as(raw, np.float16).reshape(shape).astype(np.float32)
    if qtype == 28:  # F64
        return _raw_as(raw, np.float64).reshape(shape).astype(np.float32)
    if qtype == 30:  # BF16
        return _bf16_bits_to_float32(_raw_as(raw, np.uint16)).reshape(shape)

    decoded = dequantize(raw, GGMLQuantizationType(qtype))
    return np.asarray(decoded, dtype=np.float32)


def stored_tensor_to_float32(array: np.ndarray, dtype: str | None = None) -> np.ndarray:
    """Normalise a tensor read back from safetensors to float32.

    NumPy has no bfloat16 type, so BF16 payloads are stored and returned as
    uint16 bit patterns and must be shifted back.
    """
    if array.dtype == np.uint16 and dtype == "bfloat16":
        return _bf16_bits_to_float32(array)
    return np.asarray(array, dtype=np.float32)


def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """Compare two tensors of identical shape and count.

    ``relative_l2`` is the metric the release gate uses: it is scale-free, so one
    tolerance works for embeddings and for tiny norms alike.
    """
    ref = np.ascontiguousarray(reference, dtype=np.float32).reshape(-1)
    act = np.ascontiguousarray(actual, dtype=np.float32).reshape(-1)
    if ref.size != act.size:
        raise ValueError(
            f"cannot compare {ref.size} reference values with {act.size} stored values"
        )

    if ref.size == 0:
        return {
            "elements": 0,
            "relative_l2": 0.0,
            "max_abs_error": 0.0,
            "bit_exact": True,
            "finite": True,
        }

    diff = act - ref
    max_abs = float(np.max(np.abs(diff)))
    norm = float(np.linalg.norm(ref))
    error = float(np.linalg.norm(diff))
    if norm > 0.0:
        relative = error / norm
    else:
        relative = 0.0 if error == 0.0 else math.inf

    return {
        "elements": int(ref.size),
        "relative_l2": relative,
        "max_abs_error": max_abs,
        "bit_exact": bool(np.array_equal(ref, act)),
        "finite": bool(np.isfinite(act).all()),
    }


def empty_tier(status: str = "not_run") -> dict[str, Any]:
    """A tier that has no measurement yet, e.g. numerical before a runtime load."""
    return {"status": status, "gated": status == "pass"}


def evaluate_weight_gate(
    records: Iterable[dict[str, Any]],
    dtype: str,
    *,
    tolerance: float | None = None,
    total_tensors: int | None = None,
    mismatches: Iterable[str] = (),
) -> dict[str, Any]:
    """Turn per-tensor comparisons into a pass/fail decision.

    A non-finite ``relative_l2`` (all-zero reference, non-zero output) and a
    non-finite value on disk both fail: they are corruption, not rounding. A
    tensor recorded as ``identity`` whose shape no longer matches the source is
    also a failure: the artifact and its stated plan disagree.
    """
    tolerance = DTYPE_EPSILON.get(dtype, 1e-5) if tolerance is None else tolerance
    tested = list(records)
    mismatched = list(mismatches)
    failures = [
        record["name"]
        for record in tested
        if not record["finite"] or not math.isfinite(record["relative_l2"])
        or record["relative_l2"] > tolerance
    ]
    failures.extend(mismatched)
    worst = max(tested, key=lambda r: r["relative_l2"], default=None)

    return {
        "status": "fail" if failures else ("pass" if tested else "not_run"),
        "gated": True,
        "tolerance": tolerance,
        "dtype": dtype,
        "tensors_compared": len(tested),
        "tensors_total": total_tensors if total_tensors is not None else len(tested),
        "bit_exact_tensors": sum(1 for record in tested if record["bit_exact"]),
        "max_relative_l2": None if worst is None else worst["relative_l2"],
        "worst_tensor": None if worst is None else worst["name"],
        "failures": failures[:10],
        "failure_count": len(failures),
    }


class _ShardReader:
    """Lazily open output shards and read tensors back by name."""

    def __init__(self, output_dir: Path) -> None:
        from safetensors import safe_open

        index_path = output_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text()).get("weight_map", {})

        self._safe_open = safe_open
        self._dir = output_dir
        self._handles: dict[str, Any] = {}

    def _handle(self, shard: str) -> Any:
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(
                str(self._dir / shard), framework="np"
            )
        return self._handles[shard]

    def get(self, name: str) -> np.ndarray:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name} is not listed in model.safetensors.index.json")
        try:
            return self._handle(shard).get_tensor(name)
        except TypeError:
            # safetensors' numpy backend cannot read BF16 ("data type
            # 'bfloat16' not understood"), so widen the raw bit patterns here.
            return _read_bf16_from_shard(self._dir / shard, name)

    def close(self) -> None:
        self._handles.clear()


def _read_bf16_from_shard(path: Path, name: str) -> np.ndarray:
    """Read one BF16 tensor straight from a safetensors file as float32."""
    with Path(path).open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
        info = header[name]
        if info["dtype"] != "BF16":
            raise TypeError(f"{name} is {info['dtype']}, not BF16")
        start, end = info["data_offsets"]
        handle.seek(8 + header_len + start)
        raw = handle.read(end - start)
    bits = np.frombuffer(raw, dtype=np.uint16).reshape(tuple(info["shape"]))
    return _bf16_bits_to_float32(bits)


def audit_weights(
    output_dir: Path,
    reader: Any,
    layout: dict[str, dict[str, str]],
    dtype: str,
) -> dict[str, Any]:
    """Compare written tensors against an independent decode of the source.

    ``layout`` maps output tensor name to ``{"source": <gguf name>, "layout":
    "identity"|"transformed"}`` as recorded by the conversion pass. Only
    ``identity`` tensors can be compared position-wise; the rest are counted so
    the report never pretends to have checked more than it did.
    """
    by_source = {tensor.name: tensor for tensor in reader.tensors}
    shards = _ShardReader(output_dir)
    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    skipped = {
        "transformed": 0,
        "missing_source": 0,
        "unreadable": 0,
        "layout_mismatch": 0,
    }

    try:
        for name, meta in sorted(layout.items()):
            source = meta.get("source", "")
            if meta.get("layout") != "identity":
                skipped["transformed"] += 1
                continue
            tensor = by_source.get(source)
            if tensor is None:
                skipped["missing_source"] += 1
                mismatches.append(name)
                continue

            try:
                reference = reference_tensor(tensor)
                stored = stored_tensor_to_float32(shards.get(name), dtype)
            except Exception:  # noqa: BLE001 - unreadable means "cannot certify"
                skipped["unreadable"] += 1
                mismatches.append(name)
                continue

            if reference.shape != stored.shape:
                # The plan claims this tensor was written through unchanged, so a
                # shape change is a defect in the artifact, not a transform.
                skipped["layout_mismatch"] += 1
                mismatches.append(name)
                continue

            record = {"name": name, "source": source}
            record.update(compare(reference, stored))
            records.append(record)
    finally:
        shards.close()

    return {
        "tensors": records,
        "mismatches": mismatches,
        "compared": len(records),
        "skipped": skipped,
        "total_output_tensors": len(layout),
    }


def audit_preserved(
    output_dir: Path,
    reader: Any,
    plan: dict[str, dict[str, Any]],
    *,
    tolerance: float = PRESERVE_TOLERANCE,
) -> dict[str, Any]:
    """Compare re-expressed affine tensors against the GGUF blocks they came from.

    This is a representation check, not a numerical one: the stored codes are
    decoded with :func:`gguf2mlx.quantize.gguf_codes.affine_decode` (the same
    ``scale * code + bias`` arithmetic MLX performs) and compared with
    ``gguf.quants.dequantize`` of the source tensor. A mapping error shows up as
    a large relative L2, which is how a block reordered by accident gets caught.
    """
    from .quantize import gguf_codes

    shards = _ShardReader(output_dir)
    by_source = {str(tensor.name): tensor for tensor in reader.tensors}
    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    skipped = {"missing_source": 0, "unreadable": 0, "shape_mismatch": 0}
    error_squares = 0.0
    reference_squares = 0.0

    try:
        for name, meta in sorted(plan.items()):
            tensor = by_source.get(str(meta.get("source", "")))
            if tensor is None:
                skipped["missing_source"] += 1
                mismatches.append(name)
                continue
            base = name.removesuffix(".weight")
            try:
                packed = shards.get(f"{base}.weight")
                scales = shards.get(f"{base}.scales")
                biases = shards.get(f"{base}.biases")
                decoded = gguf_codes.affine_decode(
                    packed, scales, biases, int(meta["bits"]), int(meta["group_size"])
                )
                reference = reference_tensor(tensor)
            except Exception:  # noqa: BLE001 - unreadable means "cannot certify"
                skipped["unreadable"] += 1
                mismatches.append(name)
                continue

            if meta.get("rows") == "rotary":
                reference = gguf_codes.llama_rotary_rows(reference, int(meta["heads"]))
            if reference.shape != decoded.shape:
                skipped["shape_mismatch"] += 1
                mismatches.append(name)
                continue

            record = {
                "name": name,
                "source": meta["source"],
                "qtype": meta["qtype"],
                "bits": int(meta["bits"]),
                "group_size": int(meta["group_size"]),
            }
            record.update(compare(reference, decoded))
            records.append(record)
            diff = (decoded - reference).astype(np.float64)
            error_squares += float(np.sum(diff * diff))
            reference_squares += float(np.sum(reference.astype(np.float64) ** 2))
    finally:
        shards.close()

    if reference_squares > 0.0:
        aggregate = math.sqrt(error_squares / reference_squares)
    else:
        aggregate = 0.0 if error_squares == 0.0 else math.inf
    gate = evaluate_preserve_gate(
        records, aggregate, total_tensors=len(plan), mismatches=mismatches,
        tolerance=tolerance,
    )
    return {
        "tensors": records,
        "mismatches": mismatches,
        "compared": len(records),
        "skipped": skipped,
        "total_output_tensors": len(plan),
        "aggregate_relative_l2": aggregate,
        "gate": gate,
    }


def evaluate_preserve_gate(
    records: Iterable[dict[str, Any]],
    aggregate: float,
    *,
    total_tensors: int,
    mismatches: Iterable[str] = (),
    tolerance: float = PRESERVE_TOLERANCE,
) -> dict[str, Any]:
    """Gate for code-preserved tensors: per-tensor and aggregate relative L2."""
    tested = list(records)
    failures = [
        record["name"]
        for record in tested
        if not record["finite"] or not math.isfinite(record["relative_l2"])
        or record["relative_l2"] > tolerance
    ]
    failures.extend(list(mismatches))
    worst = max(tested, key=lambda record: record["relative_l2"], default=None)
    healthy = not failures and math.isfinite(aggregate) and aggregate < tolerance
    return {
        "status": "pass" if healthy else ("fail" if tested or failures else "not_run"),
        "gated": True,
        "tolerance": tolerance,
        "tensors_compared": len(tested),
        "tensors_total": total_tensors,
        "bit_exact_tensors": sum(1 for record in tested if record["bit_exact"]),
        "aggregate_relative_l2": aggregate,
        "max_relative_l2": None if worst is None else worst["relative_l2"],
        "worst_tensor": None if worst is None else worst["name"],
        "failures": failures[:10],
        "failure_count": len(failures),
    }


def load_and_run(model_dir: str | Path, prompt_ids: Iterable[int] = (1, 2, 3)) -> dict[str, Any]:
    """Load a converted directory with ``mlx_lm`` and check the forward pass.

    This is the numerical tier: a model whose tensors are individually correct
    can still be assembled with the wrong layout. The check is deliberately
    small (one short forward pass, logits must be finite) so it can gate a
    release without a benchmark machine.
    """
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover - depends on the environment
        return {
            "status": "unavailable",
            "gated": False,
            "error": f"mlx_lm is not installed: {exc}",
        }

    try:
        model, tokenizer = load(str(model_dir))
    except Exception as exc:  # noqa: BLE001 - report, do not crash the CLI
        return {"status": "fail", "gated": True, "stage": "load", "error": str(exc)}

    try:
        tokens = mx.array([list(prompt_ids)])
        logits = model(tokens)
        mx.eval(logits)
        finite = bool(mx.isfinite(logits).all())
        payload = {
            "status": "pass" if finite else "fail",
            "gated": True,
            "stage": "forward",
            "logits_shape": [int(dim) for dim in logits.shape],
            "logits_finite": finite,
            "tokenizer_type": type(tokenizer).__name__,
        }
        if not finite:
            payload["error"] = "forward pass produced non-finite logits"
        return payload
    except Exception as exc:  # noqa: BLE001
        return {"status": "fail", "gated": True, "stage": "forward", "error": str(exc)}


def build_report(
    *,
    architecture: str,
    dtype: str,
    audit: dict[str, Any],
    layout: dict[str, dict[str, str]] | None = None,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
    preserve: dict[str, Any] | None = None,
    preserve_plan: dict[str, dict[str, Any]] | None = None,
    kept_float: Iterable[dict[str, str]] = (),
    module_bits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON payload written next to every conversion."""
    weight_gate = evaluate_weight_gate(
        audit["tensors"],
        dtype,
        total_tensors=audit.get("total_output_tensors"),
        mismatches=audit.get("mismatches", ()),
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "dtype": dtype,
        "source": source,
        "conversion": {
            "output_tensors": audit.get("total_output_tensors", 0),
            "compared_tensors": audit.get("compared", 0),
            "skipped": audit.get("skipped", {}),
        },
        "fidelity": {
            "weight": weight_gate,
            "numerical": empty_tier("not_run"),
            "perplexity": {
                "status": "report_only",
                "gated": False,
                "note": (
                    "Cross-runtime perplexity measures quantization noise, not "
                    "conversion correctness; it never fails a conversion."
                ),
            },
        },
        "gates": {"weight": weight_gate["status"]},
    }
    if layout:
        # The audit plan, kept so `gguf2mlx verify --model-dir ... --weights` can
        # re-check the artifact later without decoding the whole GGUF twice.
        report["tensor_layout"] = layout
    if extra:
        report.update(extra)
    if preserve is not None:
        records = preserve.get("tensors", [])
        by_qtype: dict[str, int] = {}
        by_bits: dict[str, int] = {}
        for record in records:
            by_qtype[record["qtype"]] = by_qtype.get(record["qtype"], 0) + 1
            by_bits[str(record["bits"])] = by_bits.get(str(record["bits"]), 0) + 1
        float_reasons: dict[str, int] = {}
        for item in kept_float:
            float_reasons[item["reason"]] = float_reasons.get(item["reason"], 0) + 1
        gate = preserve.get("gate", {})
        report["quant_source"] = "preserve"
        report["preserve"] = {
            "gate": gate.get("status", "not_run"),
            "tolerance": gate.get("tolerance", PRESERVE_TOLERANCE),
            "tensors_compared": gate.get("tensors_compared", 0),
            "tensors_total": gate.get("tensors_total", 0),
            "bit_exact_tensors": gate.get("bit_exact_tensors", 0),
            "aggregate_relative_l2": gate.get("aggregate_relative_l2"),
            "max_relative_l2": gate.get("max_relative_l2"),
            "worst_tensor": gate.get("worst_tensor"),
            "failures": gate.get("failures", []),
            "preserved_by_qtype": dict(sorted(by_qtype.items())),
            "preserved_by_bits": dict(sorted(by_bits.items())),
            "kept_float16": {
                "counts": dict(sorted(float_reasons.items())),
                "tensors": list(kept_float)[:64],
            },
            "group_size": 32,
            "tensors": records,
            "plan": preserve_plan or {},
            "module_bits": module_bits or {},
        }
        report["gates"]["preserve"] = report["preserve"]["gate"]
        if report["preserve"]["gate"] != "pass":
            report["gates"]["weight"] = "fail"
    return report


def verify_weights(
    model_dir: str | Path,
    source_gguf: str | Path,
    dtype: str | None = None,
) -> dict[str, Any]:
    """Re-run the weight audit against an existing converted directory."""
    from gguf import GGUFReader

    model_dir = Path(model_dir)
    report_path = model_dir / REPORT_NAME
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} not found: this directory was not produced by a "
            "gguf2mlx version that records a conversion report"
        )
    report = json.loads(report_path.read_text())
    layout = report.get("tensor_layout")
    if not layout:
        raise ValueError(
            f"{report_path} has no tensor_layout; re-run conversion to record it"
        )

    reader = GGUFReader(str(source_gguf))
    audit = audit_weights(model_dir, reader, layout, dtype or report.get("dtype", "float16"))
    gate = evaluate_weight_gate(
        audit["tensors"],
        dtype or report.get("dtype", "float16"),
        total_tensors=audit["total_output_tensors"],
        mismatches=audit["mismatches"],
    )
    return {
        "status": gate["status"],
        "gate": gate,
        "compared": audit["compared"],
        "skipped": audit["skipped"],
        "tensors": audit["tensors"],
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    path = Path(output_dir) / REPORT_NAME
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def summarize(report: dict[str, Any]) -> str:
    """One-line human summary of the weight gate for CLI output."""
    weight = report.get("fidelity", {}).get("weight", {})
    if weight.get("status") == "not_run":
        return "weight fidelity: not measured"
    if not weight.get("tensors_compared"):
        return (
            f"weight fidelity: {weight.get('status')} "
            "(no tensor could be compared)"
        )
    return (
        f"weight fidelity: {weight.get('status')} "
        f"({weight.get('tensors_compared')}/{weight.get('tensors_total')} tensors, "
        f"max relative L2 {weight['max_relative_l2']:.2e} vs tolerance "
        f"{weight['tolerance']:.1e})"
    )


def summarize_preserve(report: dict[str, Any]) -> str:
    """One-line summary of the code-preservation tier, if it ran."""
    preserve = report.get("preserve")
    if not preserve:
        return ""
    aggregate = preserve.get("aggregate_relative_l2")
    measured = "n/a" if aggregate is None else f"{aggregate:.2e}"
    return (
        f"preserved weights: {preserve['gate']} "
        f"({preserve['tensors_compared']}/{preserve['tensors_total']} tensors, "
        f"aggregate relative L2 {measured} vs tolerance "
        f"{preserve['tolerance']:.1e})"
    )
