"""Offline real Qwen2-VL acceptance checks, not a general visual benchmark.

Run in the isolated environment from tests/requirements-vlm.txt. All model and
photo assets must already be local. A private localhost llama-server subprocess
is stopped on every exit; no external inference service receives the images.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .!\"'")


def correct_answer(text: str, accepted: list[str]) -> bool:
    return normalized_answer(text) in {normalized_answer(answer) for answer in accepted}


def semantic_answer(text: str, case: dict) -> bool:
    """Task-specific label check, separate from the historical exact-answer score."""
    annotation = case.get("semantic")
    if annotation is None:
        return correct_answer(text, case["accepted"])
    if annotation.get("garment") != "space suit" or not annotation.get("color"):
        raise ValueError("Unsupported semantic garment annotation")
    match = re.fullmatch(
        r"(?:(?:the person|she|the astronaut) is wearing )?(?:an? )?"
        r"(?:(?P<color>[a-z]+) )?(?:space ?suit|astronaut suit)",
        normalized_answer(text),
    )
    return match is not None and match["color"] in (None, annotation["color"])


def prepare_cases(root: Path, photo: Path) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont

    root.mkdir()
    cases = []

    def add(name, image, question, accepted, category, pair=None):
        path = root / f"{name}.png"
        image.save(path)
        cases.append({
            "id": name, "image": str(path.resolve()), "image_sha256": sha256(path),
            "question": question, "accepted": accepted, "category": category, "pair": pair,
        })

    color_prompt = "What color is the square? Answer with one color word only."
    for color in ("red", "blue", "green", "yellow"):
        image = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 184, 184), fill=color)
        add("color-" + color, image, color_prompt, [color], "color", "color")
    for count in (1, 2, 3):
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        for i in range(count):
            x = 25 + i * 65
            draw.ellipse((x, 85, x + 40, 125), fill="black")
        add(f"count-{count}", image, "How many black circles are there? Answer with one digit only.",
            [str(count), ("one", "two", "three")[count - 1]], "count", "count")
    for side in ("left", "right"):
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        red, blue = ((20, 80), (144, 204)) if side == "left" else ((144, 204), (20, 80))
        draw.rectangle((red[0], 82, red[1], 142), fill="red")
        draw.ellipse((blue[0], 82, blue[1], 142), fill="blue")
        add(
            "spatial-" + side, image,
            "Is the red square to the left or right of the blue circle? Answer left or right only.",
            [side], "spatial", "spatial",
        )
    for text in ("HELLO", "4827"):
        image = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(image).text((25, 88), text, fill="black", font=ImageFont.load_default(size=44))
        add("ocr-" + text.lower(), image, "Read the text in the image. Output only that text.",
            [text], "ocr")
    with Image.open(photo) as opened:
        image = opened.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    add(
        "astronaut", image, "What is the person wearing? Answer in a short phrase.",
        ["a spacesuit", "spacesuit", "a space suit", "space suit", "an astronaut suit",
         "astronaut suit", "a white spacesuit", "a white space suit"],
        "public-domain-photo",
    )
    cases[-1]["semantic"] = {"garment": "space suit", "color": "orange"}
    return cases


def _request(url: str, body: dict | None = None) -> dict:
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


@contextlib.contextmanager
def native_server(binary: Path, gguf: Path, mmproj: Path, root: Path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), "-m", str(gguf), "--mmproj", str(mmproj),
               "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-ngl", "99",
               "--parallel", "1", "--jinja"]
    with (root / "llama-server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with {process.returncode}; see {log.name}")
                try:
                    ready = _request(url + "/health").get("status") == "ok"
                except (urllib.error.URLError, TimeoutError):
                    # Only retry startup health; inference errors must propagate.
                    ready = False
                if ready:
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError(f"llama-server did not become healthy; see {log.name}")
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def evaluate_native(args, cases: list[dict]) -> dict:
    results = []
    with native_server(args.llama_server, args.gguf, args.mmproj, args.output) as url:
        for case in cases:
            encoded = base64.b64encode(Path(case["image"]).read_bytes()).decode("ascii")
            start = time.perf_counter()
            response = _request(url + "/v1/chat/completions", {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        {"type": "text", "text": case["question"]},
                    ]},
                ],
                "temperature": 0, "seed": 0, "max_tokens": 32,
            })
            text = response["choices"][0]["message"]["content"]
            results.append({
                "id": case["id"], "text": text,
                "correct": correct_answer(text, case["accepted"]),
                "semantic_correct": semantic_answer(text, case),
                "seconds": time.perf_counter() - start, "usage": response.get("usage"),
            })
            print(f"llama.cpp {case['id']}: {text!r}", flush=True)
    return {"cases": results, "correct": sum(case["correct"] for case in results),
            "semantic_correct": sum(case["semantic_correct"] for case in results),
            "count": len(results)}


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    if actual.shape != reference.shape:
        raise ValueError(f"Reference shape mismatch: {actual.shape} != {reference.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite comparison values")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    difference = a - b
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(b), 1e-12)),
    }


def audit_weights(gguf: Path, hf: Path, converted: Path, direct_quant: Path) -> dict:
    """Independent name contract; no converter tensor mapper is used."""
    from gguf import GGMLQuantizationType, GGUFReader
    from gguf.quants import dequantize
    from safetensors import safe_open

    from gguf2mlx.vlm import _read_hf_tensor

    rows = []
    conversion = json.loads((direct_quant / "vlm_conversion_report.json").read_text())
    preserving = conversion["quantization"]["method"] == "preserve_source_q4_k"
    quant_config = json.loads((direct_quant / "config.json").read_text())["quantization"]
    if preserving and quant_config != {"group_size": 32, "bits": 4, "mode": "affine"}:
        raise ValueError("Source-preserving Q4_K requires affine 4-bit groups of 32")
    with contextlib.ExitStack() as stack:
        def open_weights(root):
            weights = {}
            for path in sorted(root.glob("*.safetensors")):
                handle = stack.enter_context(safe_open(str(path), framework="np"))
                for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                    if key in weights:
                        raise ValueError(f"Duplicate tensor {key}")
                    weights[key] = (handle, path)
            return weights

        original, output, quantized = (open_weights(root) for root in (hf, converted, direct_quant))
        checked = set()
        quant_checked = set()
        for name, (handle, path) in original.items():
            if not name.startswith("visual."):
                continue
            target = name.replace("visual.", "vision_tower.", 1)
            expected = _read_hf_tensor(handle, path, name).astype(np.float16)
            for mode, weights in (("converted", output), ("direct_quant", quantized)):
                actual = weights[target][0].get_tensor(target)
                if not np.array_equal(expected, actual):
                    raise ValueError(f"{mode}: original HF vision tensor mismatch at {target}")
            rows.append({"tensor": target, "reference": name, "exact_float16": True})
            checked.add(target)
            quant_checked.add(target)
        blocks = {
            "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
            "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
            "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
        }
        for tensor in GGUFReader(str(gguf)).tensors:
            if tensor.name == "token_embd.weight":
                target = "language_model.model.embed_tokens.weight"
            elif tensor.name == "output_norm.weight":
                target = "language_model.model.norm.weight"
            elif tensor.name == "output.weight":
                target = "language_model.lm_head.weight"
            else:
                match = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.(weight|bias)", tensor.name)
                if match is None or match[2] not in blocks:
                    raise ValueError(f"Unsupported audit source tensor {tensor.name}")
                target = f"language_model.model.layers.{match[1]}.{blocks[match[2]]}.{match[3]}"
            decoded = dequantize(tensor.data, tensor.tensor_type)
            expected = decoded.astype(np.float16)
            actual = output[target][0].get_tensor(target)
            if not np.array_equal(expected, actual):
                raise ValueError(f"Decoded GGUF weight mismatch at {target}")
            row = {"tensor": target, "reference": tensor.name, "exact_float16": True}
            if preserving:
                stem = target.removesuffix(".weight")
                if stem + ".scales" in quantized:
                    if tensor.tensor_type != GGMLQuantizationType.Q4_K:
                        raise ValueError(f"Unexpected source quantization for repacked {target}")
                    packed = quantized[target][0].get_tensor(target)
                    scale = quantized[stem + ".scales"][0].get_tensor(stem + ".scales")
                    bias = quantized[stem + ".biases"][0].get_tensor(stem + ".biases")
                    codes = ((packed[..., None] >> (np.arange(8) * 4)) & 15).reshape(decoded.shape)
                    scales = scale.astype(np.float32).repeat(32, axis=-1)
                    biases = bias.astype(np.float32).repeat(32, axis=-1)
                    restored = codes.astype(np.float32) * scales + biases
                    error = metrics(restored, decoded)
                    eps = np.finfo(scale.dtype).eps
                    bound = eps * (15 * np.abs(scales) + np.abs(biases))
                    bound += 16 * np.finfo(scale.dtype).smallest_subnormal
                    if np.any(np.abs(restored - decoded) > bound):
                        raise ValueError(f"Repacked weights exceed coefficient-rounding bound: {target}")
                    row["source_preserving_error"] = error
                    quant_checked.update((stem + ".scales", stem + ".biases"))
                else:
                    actual = quantized[target][0].get_tensor(target)
                    if not np.array_equal(actual, expected):
                        raise ValueError(f"Preserved floating weight mismatch: {target}")
                    row["source_preserving_exact_float16"] = True
                quant_checked.add(target)
            rows.append(row)
            checked.add(target)
        if checked != set(output):
            raise ValueError(f"Audit coverage mismatch: {checked.symmetric_difference(output)}")
        if preserving and quant_checked != set(quantized):
            raise ValueError("Source-preserving output audit coverage mismatch")
    return {
        "passed": True, "tensor_count": len(rows),
        "vision_reference": "all original HF visual.* tensors cast to float16; both outputs",
        "language_reference": "all GGUF tensors independently dequantized to float16; float output",
        "direct_quant_language": (
            "Q4_K repacked within coefficient-rounding bound; other weights exact float16"
            if preserving else "not bit-identical: additional lossy MLX affine quantization"
        ),
        "tensors": rows,
    }


def evaluate_mlx(path: Path, cases: list[dict], reference=None):
    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image

    mx.reset_peak_memory()
    started = time.perf_counter()
    model, processor = load(str(path), strict=True)
    model.set_dtype(mx.float16)
    load_seconds = time.perf_counter() - started
    config = json.loads((path / "config.json").read_text())
    observations, results = {}, []
    for case in cases:
        prompt = apply_chat_template(processor, config, case["question"], num_images=1)
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        inputs = prepare_inputs(processor, images=[image], prompts=prompt)
        ids, pixels, grid = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]
        mx.eval(ids, pixels, grid)
        token_digest = hashlib.sha256(np.array(ids).tobytes()).hexdigest()
        pixel_digest = hashlib.sha256(np.array(pixels).tobytes()).hexdigest()
        vision = model.vision_tower(
            pixels.astype(model.vision_tower.patch_embed.proj.weight.dtype), grid
        )
        features = model.get_input_embeddings(ids, pixels, image_grid_thw=grid).inputs_embeds
        answer = processor.tokenizer.encode(case["accepted"][0], add_special_tokens=False)
        if not answer:
            raise ValueError("Expected answer tokenized to an empty sequence")
        teacher_ids = mx.concatenate([ids, mx.array([answer])], axis=1)
        output = model(teacher_ids, pixels, image_grid_thw=grid).logits.astype(mx.float32)
        mx.eval(output, vision, features)
        if not bool(mx.all(mx.isfinite(output)).item()):
            raise ValueError(f"Nonfinite logits for {case['id']}")
        answer_logits = output[0, ids.shape[1] - 1:ids.shape[1] + len(answer) - 1]
        log_probs = answer_logits - mx.logsumexp(answer_logits, axis=-1, keepdims=True)
        nll = -mx.mean(mx.take_along_axis(log_probs, mx.array(answer)[:, None], axis=-1)).item()
        snapshot = {
            "vision": np.array(vision), "features": np.array(features),
            "next_logits": np.array(answer_logits[0]), "token_digest": token_digest,
            "pixel_digest": pixel_digest,
        }
        started = time.perf_counter()
        generated = generate(model, processor, prompt, image=case["image"], max_tokens=32,
                             temperature=0, verbose=False)
        row = {
            "id": case["id"], "prompt": prompt, "prompt_token_sha256": token_digest,
            "pixel_sha256": pixel_digest, "grid_thw": np.array(grid).tolist(),
            "answer_tokens": answer, "answer_nll": nll, "text": generated.text,
            "generation_tokens": generated.generation_tokens,
            "correct": correct_answer(generated.text, case["accepted"]),
            "semantic_correct": semantic_answer(generated.text, case),
            "seconds": time.perf_counter() - started,
        }
        if reference is not None:
            ref = reference[case["id"]]
            if token_digest != ref["token_digest"] or pixel_digest != ref["pixel_digest"]:
                raise ValueError("Reference tokenizer/image preprocessing mismatch")
            row["vs_hf"] = {key: metrics(snapshot[key], ref[key])
                            for key in ("vision", "features", "next_logits")}
        observations[case["id"]] = snapshot
        results.append(row)
        print(f"{path.name} {case['id']}: {generated.text!r} NLL={nll:.4f}", flush=True)
    sensitivity = {}
    for pair in ("color", "count", "spatial"):
        members = [case for case in cases if case["pair"] == pair]
        first, second = (observations[case["id"]] for case in members[:2])
        sensitivity[pair] = metrics(first["vision"], second["vision"])
        if sensitivity[pair]["max_abs"] <= 1e-4:
            raise ValueError(f"Image features are insensitive to the {pair} counterfactual")
    result = {
        "cases": results, "correct": sum(row["correct"] for row in results),
        "semantic_correct": sum(row["semantic_correct"] for row in results),
        "count": len(results), "mean_answer_nll": float(np.mean([r["answer_nll"] for r in results])),
        "counterfactual_vision": sensitivity, "load_seconds": load_seconds,
        "peak_mlx_bytes": mx.get_peak_memory(),
    }
    del model, processor, output, features, vision, inputs, pixels, generated
    gc.collect()
    mx.clear_cache()
    return result, observations


def quality_gate(results: dict, require_all_correct: bool = False) -> dict:
    baseline = results["hf"]
    gates = {}
    for label in ("converted", "direct_quant", "llama_cpp"):
        current = results[label]
        gates[label] = {
            "accuracy": current["correct"] / current["count"],
            "minimum_accuracy": 1.0 if require_all_correct else 0.75,
            "maximum_correct_count_drop": 0 if require_all_correct else 1,
            "passed": (
                current["correct"] == current["count"] if require_all_correct else
                current["correct"] / current["count"] >= 0.75
                and current["correct"] >= baseline["correct"] - 1
            ),
        }
        if "semantic_correct" in current:
            semantic_accuracy = current["semantic_correct"] / current["count"]
            gates[label]["semantic_accuracy"] = semantic_accuracy
            gates[label]["passed"] &= semantic_accuracy >= gates[label]["minimum_accuracy"]
    return {"passed": all(gate["passed"] for gate in gates.values()), "checks": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "mmproj", "hf", "converted", "direct-quant", "photo", "llama-server", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--require-all-correct", action="store_true",
                        help="Stricter regression gate: every converted/native answer must pass")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory; previous reports are never overwritten")
    args.output.mkdir(parents=True)
    cases = prepare_cases(args.output / "images", args.photo)
    versions = {name: importlib.metadata.version(name) for name in (
        "mlx", "mlx-vlm", "mlx-lm", "transformers", "huggingface-hub", "gguf", "numpy", "pillow",
    )}
    manifest = {
        "scope": "12-case diagnostic, not broad visual quality certification",
        "scoring": {
            "correct": "historical normalized exact-answer match, not semantic accuracy",
            "semantic_correct": "exact categorical labels; photo garment/color grammar v1",
            "photo_annotation": "NASA portrait visibly shows an orange space suit",
        },
        "platform": platform.platform(), "versions": versions,
        "llama_cpp": subprocess.check_output(
            [str(args.llama_server), "--version"], text=True, stderr=subprocess.STDOUT,
        ).strip(),
        "gguf_sha256": sha256(args.gguf), "mmproj_sha256": sha256(args.mmproj),
        "photo_sha256": sha256(args.photo),
        "models": {label: {
            "path": str(path.resolve()),
            "files": {p.name: sha256(p) for p in sorted(path.iterdir())
                      if p.is_file() and p.suffix in (".json", ".safetensors", ".jinja")},
        } for label, path in (
            ("hf", args.hf), ("converted", args.converted), ("direct_quant", args.direct_quant),
        )},
        "cases": cases,
        "predeclared_gate": {
            "minimum_accuracy": 1.0 if args.require_all_correct else 0.75,
            "maximum_correct_count_drop_vs_hf": 0 if args.require_all_correct else 1,
        },
        "generation": {"temperature": 0, "max_tokens": 32},
    }
    write_json(args.output / "manifest.json", manifest)
    audit = audit_weights(args.gguf, args.hf, args.converted, args.direct_quant)
    write_json(args.output / "weight-audit.json", audit)
    results = {}
    results["llama_cpp"] = evaluate_native(args, cases)
    write_json(args.output / "llama_cpp.json", results["llama_cpp"])
    results["hf"], reference = evaluate_mlx(args.hf, cases)
    write_json(args.output / "hf.json", results["hf"])
    for label, path in (("converted", args.converted), ("direct_quant", args.direct_quant)):
        results[label], observations = evaluate_mlx(path, cases, reference)
        del observations
        write_json(args.output / f"{label}.json", results[label])
    gate = quality_gate(results, args.require_all_correct)
    write_json(args.output / "report.json", {
        "manifest": manifest, "weight_audit": audit, "results": results, "gate": gate,
    })
    print(json.dumps(gate, indent=2))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
