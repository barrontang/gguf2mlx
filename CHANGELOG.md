# Changelog

## Unreleased

### LLaVA-NeXT converts from a real `mmproj`, and the CLIP MLP role comes from shapes

LLaVA-NeXT (anyres) was the largest hole in the VLM adapters: `model_type:
llava_next` was rejected outright, so a 1.6 export had to fall back to the hybrid
path with HF vision weights. `src/gguf2mlx/vision_llava.py` adds a direct adapter,
and running it against a real 624 MB `mmproj` found four things that a name-based
mapping gets silently wrong:

- **`model.image_newline` is not in the tower.** It is a top-level `[hidden]`
  parameter in mlx-vlm and neither `v.*` nor `mm.*`, so a prefix-routed converter
  drops it and ships a model whose anyres tiles are joined by nothing.
- **Real exports store 23 CLIP blocks for a 24-layer tower.** HF's
  `vision_feature_layer = -2` with 24 blocks *is* block 22's output; block 23 only
  feeds the pooler LLaVA discards. The stored count wins, and
  `vision_feature_layer` moves to `-1` with it — keeping `-2` would select block 21
  and produce finite, wrong features.
- **`v.post_ln` is absent.** mlx-vlm loads strictly, so the parameter is written as
  the identity and recorded in `vision_identity_fills`; a runtime test perturbs it
  by +7/−3 and asserts the image features are bit-identical, which is the claim
  that makes the fill legitimate rather than invented.
- **`ffn_up`/`ffn_down` mean opposite things to the two exporters.** llama.cpp's
  LLaVA-era loader reads `ffn_down` as the GELU input and `ffn_up` as the output
  projection; the modern mtmd graph reads `ffn_up` as the expansion. The real file
  is the legacy layout, so the previous mapping would have produced a directory
  whose MLP is transposed in role — and the shapes prove which is which, because a
  CLIP MLP is `width -> feed_forward -> width`. The adapter resolves the roles from
  dimensions and records `clip_ffn_convention`. The generic `llava` path gets the
  same treatment: a decoded tensor that fits the sibling target and not its own is
  re-aimed instead of rejected, which fixes LLaVA-1.5 exports of the same vintage.

Evidence, from `cjpais/llava-v1.6-vicuna-7b-gguf` (`mmproj-model-f16.gguf`, F16)
against `llava-hf/llava-v1.6-vicuna-7b-hf`: all **378** mmproj tensors are
consumed and none is left over; the **380** emitted names (378 plus the two
identity parameters) each exist in the HF index with the same shape; the
**375**-parameter tower loads into `mlx_vlm.models.llava_next.VisionModel` with
`strict=True` and runs, returning 24 hidden states for 23 stored blocks so that
layer `-1` is the block HF calls `-2`. `tests/test_vlm_real_mmproj.py` now pins
those counts and the `llava-tools` convention against the real file.

The text half cannot be reconciled, and is refused with a reason instead of a
generic mismatch. `llava-hf`'s checkpoint declares 32064 rows with
`image_token_index = 32000`; llama.cpp's LLaVA exports declare 32000 tokens with no
`<image>` at all. mlx-vlm embeds the prompt before splicing image features, so the
row has to exist. `_vocabulary_gap_message` now names both counts, the missing IDs
and the image token, and says which checkpoint to supply.

### MiniCPM's SigLIP ffn roles stay declared, and say so

MiniCPM-V shares the tower family with LLaVA but not the problem: only
`tools/mtmd` converts it, so `ffn_up` is unambiguously the expansion and the
declared shapes in `_VISION_BLOCKS` are still asserted against the file. That is
now stated where the mapping lives, because the shape resolution added for
LLaVA-NeXT reads like the correct fix here too — and applying it to a tower that
has no second convention would only invent ambiguity. A test puts the expansion
in `ffn_down` and holds the adapter to the consequence: it stops on `mlp.fc1`'s
shape instead of loading a swapped MLP.

A transposed ffn tensor was the one mismatch whose message pointed the wrong way.
`vlm._clip_mlp_role_hint` now recognises the exact transposition on `mlp.fc1` or
`mlp.fc2` and says the file labels the two roles the other way round, since "got
(4, 8), expected (8, 4)" otherwise reads as corruption.

### Runtime-floor tests skip instead of failing, and the floors are pinned

`tests/test_runtime_registry.py` asserted `mlx-vlm >= 0.7.10` *inside* eight
parametrised tests, so the deliberately older `.venv` (0.7.6) reported eight
failures that said nothing about the converter. The guards now skip with the
installed and required versions named, and a new
`test_runtime_floors_match_the_declared_extras` parses `pyproject.toml` and fails
if the constants drift from the declared extras — so relaxing a floor can no
longer quietly turn the resolution checks into skips.

### One matrix covers every MoE layout's router, shared expert, and fused gate

The MoE families were tested one file at a time, which is the wrong shape for
the bug that actually bites: each family names the same three concepts
differently, and a name borrowed from the wrong family produces an artifact that
converts, lists plausible tensors, and loads without its MoE block. Three module
paths are in use and they are not interchangeable:

| Layout | Router | Routed experts | Shared expert |
|---|---|---|---|
| `qwen2moe` | `mlp.gate` | `mlp.switch_mlp.*` | `mlp.shared_expert` + `mlp.shared_expert_gate` |
| `qwen3moe` | `mlp.gate` | `mlp.switch_mlp.*` | none |
| `deepseek2/3`, `glm4moe`, `glm-dsa` | `mlp.gate` + `e_score_correction_bias` | `mlp.switch_mlp.*` | `mlp.shared_experts` |
| `mixtral` | `block_sparse_moe.gate` | `block_sparse_moe.switch_mlp.*` | none |
| `gemma4` | `router.proj` + `router.scale` | `experts.switch_glu.*` | none |

`tests/test_moe_matrix.py` (99 cases) puts one row per layout in a table and
checks the table three ways: against literals, so the mapping stays reviewable;
against `mlx_lm.models.<family>` parameter trees built from tiny literal
arguments, so a name the runtime does not have cannot pass; and against the set
of layouts the converter classifies as MoE, so a new family has to decide where
its experts live instead of inheriting another family's answer.

Two of the differences are traps rather than cosmetics. The shared-expert module
is spelled `shared_expert` for Qwen2-MoE and `shared_experts` for
DeepSeek/GLM — one character, and the module that reads it is a different one.
Qwen2-MoE's shared-expert *gate* is a separate `shared_expert_gate` linear layer
rather than part of either, while Gemma4's equivalent is a per-layer
`router.scale`; both are asserted at their own destinations. Qwen3-MoE and
Mixtral have no shared expert, so the matrix records `None` and asserts that a
stray `ffn_*_shexp.weight` keeps its GGUF name rather than being renamed into a
module the runtime never reads.

The matrix also refuses to be decorative: reversing a module path, pluralising
`shared_expert`, or dropping a family from the runtime half each fail it, which
was checked by mutating the table before committing it.

### Mixtral converts in both of the layouts llama.cpp has shipped

Mixtral had no adapter, and the failure was silent rather than loud. An 8x7B
file carries `general.architecture = llama` plus `llama.expert_count = 8` —
there is no `mixtral` architecture ID to key on — so `classify_architecture`
called it **dense**, and the tensors were mapped by the Llama rules. Two
different on-disk shapes exist and both were mangled:

- 2023-era exports write **per-expert** tensors: `blk.0.ffn_gate.3.weight`,
  `blk.0.ffn_up.3.weight`, `blk.0.ffn_down.3.weight`, one triple per expert.
  These matched no rule, so they were passed through under their GGUF names —
  `model.layers.0.ffn_gate.3.weight` and so on — leaving the layer with **no
  MLP at all** in the output.
- Since llama.cpp's MoE refactor the same weights arrive **stacked**:
  `blk.0.ffn_gate_exps.weight` with shape `[8, 14336, 4096]`. Those inherited
  the dense `mlp.gate_proj.weight` name, so an 8-expert stack was written where
  a single 4096x4096 matrix belongs.

Neither destination is where mlx-lm looks. `MixtralModel` reads
`model.layers.N.block_sparse_moe.gate.weight` and
`...block_sparse_moe.switch_mlp.{gate,up,down}_proj.weight`, so both layouts
produced a directory that converted with exit code 0, listed plausible tensor
names, and could not load. The evidence for what real files contain came from
HTTP `Range` reads of the GGUF headers rather than from downloading 25 GB:
`TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` (995 tensors, per-expert) and
`mradermacher/dolphin-2.5-mixtral-8x7b-GGUF` (323 tensors, stacked).

What now happens:

- **Architecture resolution is metadata-driven.** `_mixtral_or` upgrades
  `llama`/`mistral` to `mixtral` when `expert_count > 0`, reading the key under
  both the `mixtral.` and `llama.` prefixes, because exporters disagree about
  which to write. `mixtral` is its own member of `CONVERTIBLE_ARCHES`,
  `MOE_ARCHES`, `ROPE_SCALING_ARCHES` and the Llama attention set.
- **Per-expert tensors are stacked, not passed through.** `_ExpertStack`
  buffers each `(layer, projection)` group until all `expert_count` slices have
  arrived, then emits one `[n_experts, out, in]` tensor. It rejects an expert
  index outside the declared count and a duplicate slice, and after the last
  tensor is consumed it names every group that never completed — so a truncated
  GGUF fails with "layer 3 down: 5 of 8 experts" instead of stacking five.
- **A per-expert file without `expert_count` is refused.** Stacking "as many as
  arrived" would produce a loadable model whose expert indices no longer line
  up with the router's output columns, which is worse than not converting.
- **The config is a Mistral-family config.** `build_config` writes
  `model_type: mixtral` with `num_local_experts` / `num_experts_per_tok`
  (mlx-lm's spelling, not the GGUF's), while still reading the shared
  dimensions through the `llama` keys.
- **`--quant-source preserve` now refuses Mixtral explicitly.** Mixtral was
  never a preserve architecture, but while an 8x7B file resolved to `llama` the
  mode accepted it — and re-emitted its experts under names nothing loads. The
  refusal names the architecture and points at `--quant-source requant`; it
  fires before anything is written.

`tests/test_mixtral.py` (24 tests) builds synthetic one-layer GGUFs in both
layouts and asserts conformance against `mlx_lm.models.mixtral.MixtralModel`
itself — every emitted name is checked to exist in the model's flattened
parameter tree, and vice versa — so the target names are read from the runtime
rather than from a copied convention. Expert `e` is filled with `e + 1`, which
makes an ordering error a wrong number instead of an invisible one. Mixtral is
listed in `DECLARED_COVERAGE_GAPS`: the smallest public quant is ~18 GB, over
the per-runner budget, so there is no pinned real-GGUF load case yet.

### The tokenizer is now built by three separate, validated builders

`tokenizer.json` was assembled by one ~300-line `if/elif` ladder in
`gguf2mlx.py` that decided the model block and the four policy blocks
(normalizer, pre-tokenizer, decoder, post-processor) in the same pass, mutating
shared locals as it went. The model block follows `tokenizer.ggml.model`, but
the *policy* is inferred — from the architecture, from `tokenizer.ggml.pre`,
from whether a charsmap is present — and that inference is where a rebuild goes
wrong quietly: a wrong normalizer still loads and still tokenizes, it just
disagrees with the model it came from, and the disagreement shows up later as
degraded generations.

Three things changed:

- **One builder per format.** `src/gguf2mlx/tokenizer.py` now holds
  `build_bpe` (GPT-2 byte-level, and the SentencePiece vocabulary exported as
  merges), `build_unigram`, and `build_wordpiece`, each returning a complete
  document from its own locals. The shared pieces are explicit helpers
  (`_bytelevel_pipeline`, `_sentencepiece_pipeline`, `_wordpiece_pipeline`), not
  branches of a shared ladder, so a decision made for one family cannot leak
  into another. `TokenizerSource` carries the GGUF fields themselves, so a key
  is read in exactly one place.
- **The result is schema-validated.** `validate_document` checks the structure
  `tokenizers` requires and rejects what would otherwise fail at load time or
  not at all: non-contiguous token IDs, a merge naming a piece outside the
  vocabulary, duplicate added-token IDs or contents, a missing block, a
  `Sequence` without its list. `TokenizerSource.validate` checks the metadata
  and names the offending key, so `TokenizerSchemaError.field` distinguishes a
  truncated `tokenizer.ggml.scores` from a malformed `tokenizer.ggml.token_type`
  without matching prose.
- **The architecture sets moved to `families.py`.** The builders need
  `QWEN_FAMILY_ARCHES` and the `tokenizer.ggml.pre` aliases, and importing the
  converter to get them is a cycle; duplicating them would let the two copies
  drift until a Qwen model silently acquired an NFC normalizer. The converter
  re-binds the two names, so `core.QWEN_FAMILY_ARCHES` keeps working.

The rewrite is behaviour-preserving, and that is measured rather than asserted:
13,824 combinations of model, architecture, `tokenizer.ggml.pre`, byte fallback,
space prefix, whitespace flags, charsmap, lowercase and strip-accents produce
**byte-identical** documents before and after. `tests/test_tokenizer_builders.py`
adds 48 tests that build one format at a time, check each format's policy
blocks, prove the isolation (BERT does not inherit a SentencePiece charsmap;
only non-Qwen architectures get NFC), and load every result with `tokenizers` so
"valid" means "loadable", not merely "well-shaped".

### `preserve` covers the legacy 32-value block types, and names what it refuses

`Q4_0`, `Q4_1`, `Q5_0` and `Q5_1` are now re-expressed exactly. They were the
obvious omission: each covers exactly 32 values — MLX's minimum group — so their
codes copy unchanged and the result is **bit-exact**, not merely within
tolerance. `Q4_0`/`Q5_0` need only `scale = d` and `bias = -8d` / `-16d`, which
is a power-of-two scaling of an existing `fp16`; `Q4_1`/`Q5_1` need `bias = m`,
which is already `fp16`. Measured relative L2 is 0 for all four against
`gguf.quants.dequantize`, including on arbitrary code bits rather than only on
quantizer output.

The layout trap is the 5th bit of `Q5_0`/`Q5_1`: `qh` is **one little-endian
`uint32`** with value `i` taking bit `i`, not four independent bytes. Reading it
the second way still produces the right shape and plausible weights, so it is
the kind of error only a decode comparison catches.

Two things are now written down rather than implied:

- **The exactness rule.** A format maps exactly only when its scale granularity
  divides 32. `Q2_K` and `Q3_K` store a scale *and* min per 16 values, so they
  hit the same wall as `Q6_K` — but with 2–3 code bits a least-squares refit
  loses real precision instead of bounded precision, so they are decoded to
  float rather than quietly re-fitted into something worse than the source at
  the source's own byte budget.
- **iMatrix.** An importance-matrix-guided quantization is a recipe, not a
  layout; what a file contains is an `IQ` type, and every `IQ` type decodes as
  `d * kvalues[code]` against a non-uniform table with no `scale*code + bias`
  form. So such a model converts but cannot be preserved. The rejection is now
  reported by *named* reason (`codebook_has_no_affine_form`,
  `sub_block_scale_is_per_16_values`, …) instead of a generic label, an unknown
  format keeps its own distinct label, and the run prints how many tensors were
  affected and which block types caused it — otherwise a 1.9x size ratio reads
  like a mapping bug.

`tests/test_preserve_legacy.py` proves the writer never calls `dequantize` for
these types by replacing the symbol in both modules that could reach it with one
that raises, then requiring the conversion to succeed. The claim is scoped to
the writer: the preserve *audit* decodes by design, because it is the oracle, so
that test runs with the audit off. Doing so exposed a live crash — the summary
line formatted `aggregate_relative_l2` with `.2e` even when the audit had been
skipped and the value was `None` — which is now guarded.

### Qwen2.5-VL and Qwen3-VL convert without any HF vision weights

`ARCHITECTURE_REGISTRY` had three vision layouts, so a Qwen2.5-VL or Qwen3-VL
`mmproj` could only be converted through `--hf-fallback-vision`, which downloads
the original checkpoint's vision tower. Both now have explicit plan adapters and
convert **directly** from the language GGUF plus `mmproj`: the reports record
`vision_hf: 0` for Qwen2.5-VL-3B-Instruct and Qwen3-VL-2B-Instruct.

They needed separate adapters rather than a name alias, because the Qwen2.5-VL
and Qwen2-VL cases are indistinguishable by tensor name — llama.cpp drives both
through one graph builder — while the maths differs:

- Qwen2.5-VL's ViT blocks are RMSNorm with **no** norm bias, and its MLP is the
  gated `ffn_gate`/`ffn_up`/`ffn_down` SiLU form. Declaring the biases Qwen2-VL
  has would have left 65 sources permanently missing on every real export; the
  first draft did exactly that and the real `mmproj` caught it.
- Qwen3-VL fuses `attn_qkv` with bias, uses LayerNorm blocks *with* bias, a
  GELU-tanh `linear_fc1`/`linear_fc2` MLP, a Conv3d patch embedding *with* bias,
  and a learned `pos_embed`. Its deepstack mergers are numbered by **rank** in
  `clip.vision.is_deepstack_layers` (5/11/17 → `deepstack_merger_list.0/1/2`),
  not by layer index, and are normalized at the 4x merged width.

The text halves also diverge: Qwen3-VL carries QK-norm instead of attention bias
and nests its hyperparameters under `text_config`, where Qwen2.5-VL keeps flat
top-level keys that the tested mlx-vlm loader reads. The config writer now
follows `nested_text_config` per adapter instead of always flattening.

Each plan is checked against the real file, not only against its own fixture:
`tests/test_vlm_real_mmproj.py` reads the downloaded `mmproj` and requires the
declared source set, source shapes and assembled target shapes to match, with the
tensor count pinned. That test is opt-in (`GGUF2MLX_RUN_E2E=1`) because the
fixtures are 2 GB. Separately, both converted artifacts were compared against the
original HF checkpoints: 824/824 and 625/625 tensors agree by name and shape, and
the fused QKV, stacked patch embedding and merger weights agree to float16
resolution (max |diff| 3e-8; biases, norms and `ln_q` bit-identical).

`--quant-source preserve` already required a plan adapter and now accepts all
three Qwen-VL families; its size and parity numbers remain measured only on the
Qwen2-VL-2B pair, and the docs say so.

### Emitted `model_type` values are checked against the runtime registry

Conversion could emit a `model_type` the installed runtime has no module for,
which only surfaces as a load failure on the user's machine. The minimum
runtimes are now `mlx-lm>=0.31.3` (Qwen3.5 landed in 0.31.0, `gemma4_text` in
0.31.3) and `mlx-vlm==0.7.10` (`minicpmv4_7`), and `tests/test_runtime_registry.py`
asserts every architecture ID this converter can emit resolves through
`mlx_lm.utils._get_classes()` / `mlx_vlm.utils.get_model_and_args()` — including
the `qwen2_vl_text` / `qwen3_vl_text` text hosts that the VLMs delegate to. A new
`Runtime contract` workflow pins both versions on Apple Silicon and runs that
suite; a portable inventory half fails when an adapter is added without the
registry test being updated.

### `preserve` covers dense Qwen3.5, and its hybrid layout is restored

`--quant-source preserve` now accepts `qwen35` and its aliases (`qwen3_5`,
`qwen3.5`, `qwen36`, `qwen38`). The quantized matrix projections of a
linear-attention block keep their source codes; `A_log`, `dt_bias`, the
depthwise `conv1d` and the gated norm have no affine matrix form and stay float,
recorded as `F32_has_no_affine_reexpression`.

Writing that fixture exposed two layout bugs in the float path, both of which
produced artifacts that could not load or ran with wrong decay:

- `A_log` was only restored for a tensor literally named `ssm_a`. Real GGUFs use
  `blk.N.ssm_a`, so `log(-ssm_a)` never ran and every layer kept the raw
  negative decay rate where MLX expects its logarithm.
- `ssm_norm.weight` was de-shifted by 1. llama.cpp skips its `+1` centering for
  `linear_attn.norm.weight` specifically, so subtracting 1 corrupted the norm.
- The squeezed `(channels, kernel)` kernel is now widened to
  `(channels, kernel, 1)`, which is what MLX's `Conv1d` wants and what keeps
  `mlx_lm`'s `qwen3_5` sanitize from re-applying the HF `+1` norm offset.

The weight audit learned a `reshaped` layout so the widened kernel is still
value-compared instead of being excused as `transformed`.

### Pinning the Qwen3.5 real checkpoint found four more defects

`unsloth/Qwen3.5-0.8B-GGUF` Q4_K_M was supposed to be the easy part of the
matrix. It found:

- `Unsupported qwen35 tensor: blk.0.post_attention_norm.weight` — llama.cpp
  names Qwen3.5's post-attention norm `post_attention_norm`, not `ffn_norm`, and
  only the Llama spelling was mapped, so **every** conversion of a real Qwen3.5
  file failed.
- `Missing 1 parameters: language_model.lm_head.weight` — `build_config` forced
  `tie_word_embeddings: false` for Qwen3.5 while the file ships no
  `output.weight`. Tie/untie now follows the tensor list, as it already did for
  Qwen3.
- The first artifact was **1.446x** the source because 36 `Q5_K` tensors had no
  affine re-expression and fell back to FP16. `Q5_K` is now preserved exactly:
  its 6-bit scale/min is already per-32 values, and its 5-bit code is the `qs`
  low nibble plus one bit from the `qh` plane, so MLX stores the same codes in
  five words per group. Artifact: **1.084x**.
- `tokenizer.ggml.pre='qwen35'` was unlisted, so the tokenizer fell through to
  legacy BPE splitting **and** kept the default NFC normalizer — which no Qwen
  tokenizer has. The model still loaded with finite logits; only a warning in the
  matrix log gave it away. `QWEN_PRE_TOKENIZERS` and `QWEN_FAMILY_ARCHES` now
  route every Qwen ID to the same no-normalizer path, and
  `tests/test_tokenizer_fidelity.py` asserts `normalizer is None` for each of
  them and that an unknown `pre` value still warns.

The case now passes end to end, locally on Apple Silicon (2026-10-26,
`mlx-lm` 0.32.2) and in CI (2026-10-28): convert → weight gate (`115/133`) →
preserve gate (170 `exact`, 17 `inexact`) → `mlx_lm.load` → finite logits. All
twelve pinned entries are green in CI, and re-running the full sweep locally
gives 12/12 as well.

### `publish` can be reviewed before it is visible

- `--dry-run` resolves, downloads, converts, and builds the card and provenance,
  then stops before `create_repo`/`upload_folder`.
- `--verify-load` runs the same check as the release gate
  (`fidelity.load_and_run`: load, one forward pass, finite logits) before the
  destination repository exists, so a layout defect fails the run instead of
  publishing an unloadable model. Loading alone would miss it: a tensor with the
  right name and shape but the wrong layout loads cleanly and computes garbage.
- Shard downloads and license copies stream in fixed-size chunks instead of
  reading each file into memory, and the merged GGUF is written incrementally.
- The model card now carries the source GGUF SHA-256 and byte count, the weight
  and preserve gate results, the artifact disk size, a `base_model_relation`,
  dtype/precision tags, and an exact reproduce command.

### The report explains every source tensor that produced no output

`conversion_report.json` schema 4 adds `source_tensors_without_output`, naming
each dropped tensor with `reproduced_from_config`, `mtp_head_not_consumed`, or
`unmapped`. Silence was the previous behaviour, which made MTP heads and
`rope_freqs` indistinguishable from an adapter gap.

### Three red CI jobs, none of which the converter caused

The first push of this cycle turned four workflows red. Two were mine and two
predated it:

- `tests/requirements-vlm.txt` pinned `mlx==0.32.2` while `mlx-vlm==0.7.10`
  requires `mlx>=0.32.4`, so `Runtime contract` and `Vision conversion` both
  died in `pip install` before a single test ran. Pinned to `0.32.4`, the
  version the Qwen3.5 run actually used, and the set now resolves.
- The nightly `Load matrix` failed on the `gemma2` entries in the forward pass —
  `Caused GPU Timeout Error` one night, `Caused GPU Hang Error` the next — after
  the conversion and the weight gate had passed. The same bytes load locally,
  `mlx` was the same `0.32.4` the previous green run used, and `gemma2-q4_k_m`
  passed on the same commit in the same run, so the runner's GPU is the common
  factor. `fidelity.load_and_run` now separates the two faults: a timeout is
  `retryable: true` and is retried once, while a hang sets
  `environment: gpu_hang` and is not retried at all, because a wedged Metal
  context cannot recover inside the process that wedged it — retrying it only
  spends another ten minutes failing. Non-finite logits and a missing runtime
  are classified as neither, so a real defect still fails immediately.
- `Weekly MoE validation` was failing every run because `GGUF2MLX_MOE_MODELS`
  was never set, and no step can download the 11–20 GB fixtures it expects. It
  now skips with a warning annotation when unconfigured instead of going red,
  and still fails hard when the variable is set but names an unusable file. A
  weekly red that no reader can act on is worse than an honest skip; the
  support-matrix line that promised this coverage now says it is inactive until
  configured, and `tests/test_docs_consistency.py` fails if that promise drifts
  from the workflow guard again.


### Documentation tracks the code again

`llms.txt`, `docs/development.md`, `docs/support-matrix.md`,
`docs/quality-gates.md`, `docs/publishing.md`, `docs/vlm-conversion.md` and the
README no longer describe a Rust build, call direct-quant "proposed", or claim
that source quantization is never preserved.

Fixing the instances would only invite the fourth round, so
`tests/test_docs_consistency.py` now asserts the invariants instead: every
`CONVERTIBLE_ARCHES` and `PRESERVE_ARCHES` entry is named in `llms.txt` (alias
spellings collapsed, word-boundary matched so `qwen3` is not satisfied by
`qwen3.5`), the quoted runtime versions match `pyproject.toml`, every
preservable block type appears in the gate table, no doc mentions the deleted
crate, and `direct-quant-transcoding.md` no longer titles itself as proposed.
Writing it immediately caught two live errors: `llms.txt` undercounted the
canonical architectures by two (`qwen36`/`qwen38` are canonical, not aliases),
and `--verify-load` was documented as running a forward pass while the code only
called `load()`. The check now delegates to `fidelity.load_and_run`, so a
publish-time verification and a CI gate agree on what "loadable" means — a
non-finite logits result and a missing runtime both fail the run rather than
passing silently.

### `preserve` stops expanding Q6_K to FP16

`--quant-source preserve` re-expresses GGUF blocks on MLX's affine grid and
reuses the source integer codes. For Q6_K that only works when a 32-value
group's two 16-value sub-block scales agree — true for 0.7% of groups in a real
model. Every other Q6_K tensor was decoded to FP16, turning 0.82 bytes/value
into 2: **a Q6_K file came out 2.4x the size of the GGUF it came from**, which
is the opposite of the mode's purpose.

Q6_K is now re-fitted at the source width and emitted as 6-bit affine groups:

| Fallback | Q6_K artifact | Relative L2 |
|---|---|---|
| `6bit` (default) | 1.067x the source bytes | 1.7-1.8e-2 |
| `8bit` | 1.371x | 4.4-4.9e-3 |
| `float16` | 2.4x (2.44 bytes/value against 0.82) | 0 |

Measured on Llama-3.2-1B (bartowski). On the **Q4_K_M** file, whose Q6_K
tensors dominate `ffn_down` and `attn_v`, the artifact goes from 1.662x the
GGUF to **1.092x**. All four artifacts load in `mlx_lm` with finite logits.
Fitting is per group — the lower squared error of the source lattice and a
least-squares refit from two starting points — and groups whose pair scales
match still keep their codes bit for bit.

### Audit changes

- The preserve gate is split into tiers. `exact` (codes copied) is held to
  1e-3; `inexact` (re-fitted) is held to the ceiling registered for its width.
  The gate passes only when both do, so a re-fit can neither fail the fp16
  bound nor hide behind a loose one.
- `--preserve-fallback {6bit,8bit,float16}` selects the width.
- `conversion_report.json` records `preserve.tiers`, `preserve.size`
  (source/output bytes and ratio) and `preserve.fallback`; schema 3.
- `preserve` plans now record the *source* block type in `qtype` and the target
  width in `bits`, so `preserved_by_qtype` answers "what came in".

### Fixed

- Preserve conversions of any Llama-3 GGUF wrote `rope_freqs.weight` as a float
  tensor, so the artifact failed to load ("Received 1 parameters not in model").
  `build_config` already reproduces that table as a verified `rope_scaling`
  block; the preserve path now skips the tensor like the float path does.
- The Q6_K mapping is chunked, so an embedding table with a quarter of a
  billion values no longer allocates three full-size float32 temporaries.

### CLI: `--force`, `--json`, and a real logging channel

- `--force` replaces a non-empty `--output`. Without it a non-empty directory is
  still refused — and `--force` will not delete a *file* sitting at `--output`.
- `--json` prints one object (`ok`, `output`, `report`, `gates`, `weight`) read
  back from `conversion_report.json`, so a wrapper never has to parse the human
  summary. `verify --json` already worked this way.
- Diagnostics now go through a `logging` logger on stderr (`-v` for DEBUG, `-q`
  for warnings and errors only). With `--json`, the converter's own progress and
  failure prints are redirected to stderr too, so stdout carries exactly one
  parseable object. There were no `logging` calls before.

### `rust/gguf2mlx-rs` is deleted

Nine tracked files of pyo3 code (architecture detection, metadata, tensor and
tokenizer helpers) sat behind an optional import that never resolved: the cdylib
was never built, never published, and never mentioned in the docs, while the
Python path already computed everything it returned. The MDD left this as
"maturin or delete". A one-shot classification of a few thousand tensor names
does not justify a second toolchain in the release pipeline, so the crate and its
`rust_backend.py` shim are gone. The pre-rewrite backup and the changelog keep
the decision reversible.

### Declared-versus-covered architectures are enforced, not implied

`CONVERTIBLE_ARCHES` claims 30 architecture IDs; the pinned load matrix proves
five. The 17 canonical gaps (25 spellings) are now listed with a reason in
`DECLARED_COVERAGE_GAPS`, and
`test_every_convertible_architecture_is_covered_or_declared` fails when a new
adapter is neither pinned nor declared — and when a declared gap is filled but
its line is left behind. `docs/support-matrix.md` mirrors the table.

`README.md` is back inside its 200-line budget (254 → 200) by pushing the
preserve measurements to the quality-gate page and the `.mlx` walkthrough to its
own page.

## 2.2.0 - 2026-10-21

### Type checking is now a gate, not a configuration file

- `mypy src` runs in CI on Python 3.12 and must report **zero** errors (it
  previously reported 78 across 8 files and was never executed).
- `[tool.mypy]` targets 3.12 because numpy's bundled stubs use PEP 695 `type`
  statements that a 3.10 target rejects as a syntax error. 3.10/3.13 runtime
  compatibility stays covered by the test matrix.
- Fixing the errors removed three real hazards: `mlx_lm.load()` is now indexed
  instead of unpacked into exactly two names, the `streaming.py` writer no
  longer rebinds its `dtype` parameter to `np.dtype`, and MiniCPM-V validation
  checks for a missing preprocessor config instead of assuming one exists.

### Documentation links actually resolve

- Eight relative links in `docs/` pointed at `docs/...` or repo-root paths from
  inside `docs/`, so they 404'd on GitHub (`docs/docs/qwen2-vl-real-validation.md`,
  `docs/.github/workflows/publish-mlx.yml`, …). All internal links now resolve;
  `docs/validation/` entries use `../`.

### Scheduled MoE validation reports its real failure

- `Weekly MoE validation` had been failing for three weeks with a bare
  `JSONDecodeError`: an unset `GGUF2MLX_MOE_MODELS` repository variable arrives
  as an empty string, which is not valid JSON. It now parses `""` as "nothing
  configured" and exits with the variable name and expected format.

### What the first full load-matrix sweep caught

The matrix had never executed. The first sweep of all thirteen pinned cases
passed 4 of 11 end-to-end conversions and exposed four defects that no unit
test could see:

- **llama3 rope scaling was being dropped.** llama.cpp ships llama3 scaling as
  a `rope_freqs.weight` table with no `llama.rope.scaling.*` metadata. The
  mapper passed the tensor through unchanged, so `mlx_lm` rejected the model
  ("Received 1 parameters not in model"), and the config had no scaling at all.
  The table is now read, the equivalent `rope_scaling` block is inferred and
  **verified by regenerating the table**, and the tensor is not emitted. A
  table that cannot be reproduced fails the conversion instead of rotating
  positions differently.
- **Qwen3-0.6B got an `lm_head` it never had.** `build_config` forced
  `tie_word_embeddings=False` for qwen3 while the file ships no `output.weight`.
  Tie/untie is now derived from the tensor list for every architecture
  ("Missing 1 parameters: lm_head.weight").
- **Gemma 2/3 lost two of their four block norms.** The Llama mapper maps
  `ffn_norm` to `post_attention_layernorm`, which is the Gemma-1 layout;
  Gemma 2/3 needs `post_attention_norm` and `post_ffw_norm` as well, and its
  `ffn_norm` is the *pre*-FFN norm. Both post norms also need the same `-1`
  de-shift as the others (llama.cpp stores `1 + w`).
- **The job ran out of disk, not out of correctness.** A 4 GB GGUF becomes an
  8 GB FP16 tree, and one `macos-14` runner has ~14 GB. The workflow now fans
  out one job per (architecture, qtype) and the test drops the converted tree
  after each case while keeping the report.

`docs/quality-gates.md` records the symptom → cause → fix table, and the
workflow is a real matrix (`plan` job → `strategy.matrix`) rather than a loop.

### Version

- `2.2.0` replaces `2.1.5.dev0`. `publish-mlx.yml`'s `converter_version`
  default moves to `2.2.0`, the first published release that ships the
  `publish` subcommand.

### Preserve the GGUF quantization instead of requantizing

- New `--quant-source preserve|requant` on `convert`. `requant` remains the
  default, so existing conversions are unchanged; `preserve` re-expresses the
  source GGUF blocks as MLX affine weights on a 32-value grid instead of
  decoding to FP16 and quantizing again.
- `src/gguf2mlx/quantize/gguf_codes.py` decodes Q4_K, Q6_K and Q8_0 blocks and
  re-encodes the *same* integer codes with an fp16 scale and bias. Measured
  per-tensor relative L2 against `gguf.quants.dequantize`: Q4_K 2.0-2.4e-4,
  Q8_0 0.0 (bit-exact), Q6_K ~2.2e-4 when a group's two sub-block scales agree.
- Q6_K scales 16 values at a time, so a 32-value affine group cannot represent
  it exactly when those scales disagree (1.1-1.5e-2). Those tensors report
  `exact: false` and stay in FP16 instead of being silently requantized.
- Preserve fails closed: `llama`, `qwen2`, `qwen3` and `mistral` only. Any other
  architecture errors out naming the arch rather than falling back to
  requantization.
- `conversion_report.json` gains a `preserve` block: per-tensor relative L2 and
  max abs error, `preserved_by_qtype`/`preserved_by_bits`, the decisions to keep
  a tensor in FP16, and an aggregate gate at 1e-3 (fp16 scale rounding only).
  `docs/conversion-report.schema.json` describes it.
- Mixed 4/6/8-bit module widths are emitted as per-module `quantization`
  overrides in `config.json`; `mlx-lm` loads them as `QuantizedLinear`.
- `--preserve-source-quant` keeps working as a deprecated alias that warns and
  resolves to `--quant-source preserve`. On multimodal GGUFs the Qwen2-VL
  preserve path still requires `--quantize --direct-quant` and keeps its
  affine 4-bit / group-64 layout.

### `gguf2mlx publish` and split GGUF support

- New `publish` subcommand: `gguf2mlx publish --source <repo> --gguf <file>
  --dest <repo> [--source-revision] [--quantize] [--quant-source]`. The pipeline
  moved out of the workflow's heredoc payload into `src/gguf2mlx/publish.py`;
  `.github/workflows/publish-mlx.yml` is now a thin shell around the same code.
- Split GGUFs (`-00001-of-0000N`) are discovered from any shard, merged back
  into one file before conversion, and a partial shard set fails with an error
  naming the missing indices instead of publishing a truncated model.
- Safety properties are unchanged and now unit-tested: repository-id and path
  validation before the first network call, source ≠ destination, immutable
  source revision, provenance file, source license/notice files copied, and the
  destination repository created only after conversion produced weights.
- Float16 conversion needs only `numpy`/`gguf`/`safetensors`, so the workflow's
  default runner is now `ubuntu-latest`; only `quantize: true` requires a macOS
  Apple Silicon runner.
- New `publish` extra (`huggingface-hub`), and the provenance record now carries
  the GGUF SHA-256, byte size, shard list, quant source and converter version.

### Quality gates that can actually fail and mean it

- Every conversion now writes `conversion_report.json` with a **weight-fidelity
  gate** (schema 2). Each written tensor is compared position-wise against an
  independent `gguf.quants.dequantize` decode of the source, with a per-dtype
  relative-L2 tolerance (fp16 6e-4, bf16 5e-3, fp32 1e-7). A tensor that moved
  or was mis-mapped fails the conversion instead of shipping silently.
- Fused, permuted and split tensors are recorded as `transformed` and reported
  in `conversion.skipped` rather than being counted as verified. A tensor marked
  `identity` whose stored shape no longer matches the source fails as a
  `layout_mismatch`.
- `gguf2mlx verify --model-dir DIR --weights --source model.gguf` replays the
  audit from the recorded `tensor_layout` without converting again.
  `gguf2mlx verify --model-dir DIR --load` is the numerical tier: it loads with
  `mlx_lm`, runs one forward pass and requires finite logits. Both exit non-zero
  on failure.
- Perplexity is downgraded to a **report-only** tier. Cross-runtime PPL mixes
  source quantization noise with conversion error, so it can no longer fail a
  build; see `docs/quality-gates.md`.
- `--weight-audit/--no-weight-audit` (default on) controls the extra pass, which
  roughly doubles conversion time.
- Quantized output carries the report forward with
  `artifact.stage = "mlx_lm_quantized"` and states explicitly that the 4-bit
  weights are not covered by the FP16 weight audit.

### Load matrix: "declared supported" is now testable

- `tests/load_matrix.json` pins real public GGUFs by repo, filename and SHA-256:
  Llama-3.2-1B, Qwen2.5-0.5B, Qwen3-0.6B, Gemma-2-2B and Phi-3-mini across
  Q4_0 / Q4_K_M / Q8_0 where published.
- `tests/test_load_matrix.py` (opt-in via `GGUF2MLX_LOAD_MATRIX=1`) downloads,
  hash-checks, converts, gates weights, then loads and runs each one.
- New `Load matrix` workflow runs the Q4_K_M subset nightly on Apple Silicon and
  the full sweep on manual dispatch.

### Stage 0: repository hygiene

- README reduced from 1,231 to 199 lines; the long-form material moved verbatim
  into `docs/` (support matrix, VLM, text encoder, publishing, benchmarks, FAQ,
  development, roadmap, validation logs).
- Repository traffic tracking (GPL-3.0-or-later workflow, dashboard, badge art
  and license text) moved to `barrontang/gguf2mlx-traffic`, removing the
  copyleft surface and the long-lived gist token from this Apache-2.0 repo.
- Publishing refuses PEP 440 pre-release versions on PyPI and restricts manual
  `workflow_dispatch` runs to `main`, so a development build cannot be uploaded.
- GitHub repository description and topics now describe the actual package.

### Trust and repository hygiene

- Conversion no longer guesses model dimensions: `build_config(..., strict=True)` raises
  with the missing `llama.*` / `<arch>.*` keys instead of silently falling back to
  4096/32/32/32000. `inspect` keeps the lenient behaviour.
- Missing `gguf` / `safetensors` now raise `ImportError` at import time instead of calling
  `sys.exit(1)`, so notebooks, CI wrappers and the publish pipeline can catch it;
  `python -m gguf2mlx` still exits 1 with a single-line message.
- GLM-5.2 `generation_config.json` is derived from the GGUF tokenizer metadata
  (`tokenizer.ggml.eos_token_id` / `eos_token_ids`) instead of hardcoded EOS ids and a
  fabricated `transformers_version`; the file is skipped when the metadata is absent.
- Affine 4-bit quantization rounds half away from zero (`std::round` semantics used by
  MLX) instead of `np.round`'s half-to-even, and rejects groups containing NaN/Inf with the
  offending tensor name instead of writing silent garbage.
- Version bumped to `2.1.5.dev0` so local builds no longer collide with the published
  2.1.4 wheel.
- Removed 1,211 tracked `rust/gguf2mlx-rs/target/` build artifacts from the repository.
- CI now exercises Python 3.10/3.12/3.13 and enforces an 80% coverage floor.

### Vision support

- Add a strict Qwen2-VL adapter for combined or split GGUFs: reconstruct fused
  vision QKV and temporal Conv3D kernels, retain the patch merger from GGUF,
  normalize HF configuration and validate M-RoPE, special tokens and processors.
- Extend direct affine 4-bit quantization to LLaVA and Qwen2-VL language
  projections while keeping vision/projector tensors floating-point. Reuse
  the existing quantizer; this is re-quantization, not lossless transpacking.
- Detect additional vision tensor/metadata prefixes and newer multimodal
  architecture IDs without enabling unsupported architecture fallbacks.
- Reject stale HF quantization configs, inconsistent image normalization and
  CLIP activation/normalization settings not honored by the tested MLX runtime.
- Add isolated pinned VLM test requirements and Apple Silicon CI covering actual
  MLX-VLM loading, image processing, feature/logit parity and generation with
  offline tiny F32/Q4_0 checkpoints; add an opt-in local-model image smoke test.
- Document the upstream LLaVA processor incompatibility between MLX-VLM 0.7.6
  and Transformers 5.19; isolate native tests rather than downgrading dev requirements.
- Support the historical `qwen2vl_merger` visual MLP naming and explicitly
  validated zero position placeholder. Merge HF `added_tokens_decoder` token
  definitions without relaxing conflict or vocabulary checks.
- Complete pinned Qwen2-VL-2B real-checkpoint acceptance: all 729 floating tensors
  audited, exact projected visual features against HF, and 12/12 labeled image
  checks for HF, GGUF-to-FP16 and llama.cpp. Additional language-only MLX 4-bit
  scores 11/12 exact matches; retain its longer photograph answer. Publish the offline
  four-path harness and scoped report, not a broad visual-quality certification.
- Correct the photograph failure diagnosis: the NASA suit is visibly orange;
  the uniform 4-bit answer was semantically correct, not a color hallucination.
  Report semantic garment/color checks separately from historical exact matches.
- Add explicit Qwen2-VL `--preserve-source-quant` with direct quantization:
  repack Q4_K's original codes/groups of 32 and keep other projections floating
  instead of lowering Q6_K to four bits. The pinned model passes all 12 original
  exact-answer checks; all-projection relative L2 error against decoded GGUF
  falls from 8.51% to 0.0614%. This is mixed precision (~2.7 GiB), not all-four-bit
  or bit-exact FP16 transpacking. Keep uniform quantization behavior unchanged.

### Fixes

- Keep dev-only CI portable: skip HF precision-control tests when optional
  MLX is absent, and run the Qwen BPE regression with `tokenizers` independently
  of the optional `transformers` loader integration.
- Rebuild first-generation Gemma SentencePiece tokenizers as score-ranked BPE,
  preserving GGUF whitespace-prefix behavior, Unicode, token IDs and BOS/EOS flags.
- Use the public safetensors `keys()` API when finalizing shard indexes, with a
  real multi-shard regression test.
- Preserve NumPy row-major dense GGUF layouts instead of reshaping to GGML
  axis order and transposing, which scrambled dense matrices.
- Ship a local Gemma runtime adapter honoring `gelu_pytorch_tanh`; stock
  MLX-LM 0.31.3 uses erf GELU regardless of that configuration.
- Pack direct affine 4-bit weights as MLX-compatible uint32 words instead of
  uint8 byte arrays, including stacked expert matrices.
- Restore Qwen2MoE shared-expert tensor mappings, reshape its scalar gate to
  a linear matrix, and infer legacy GGUF expert dimensions from tensor shapes
  when explicit metadata is absent; reject conflicting metadata.
- Permit IQ4_NL source weights used inside mixed Q2_K MoE GGUF checkpoints.
- Preserve Qwen GGUF decomposed Unicode and omit a fabricated unknown token
  when GPT-2-style metadata has none, avoiding special-token registration of `!`.

### Validation

- Add portable Gemma tokenizer regressions and an opt-in native llama.cpp
  comparison (`GGUF2MLX_GEMMA_GGUF` with `GGUF2MLX_RUN_E2E=1`).
- Complete public Gemma Q4_0 conversion and zero-tolerance PPL evaluation on Mac.
  Tokenizer parity passes; PPL parity fails (absolute delta 0.41424293554212).
- Add native Q4_0 decode auditing and explicitly labeled unquantized controls.
  Full Q4 decode audit has zero bit differences; aligned dense FP32 delta is
  0.0007508356276275663, HF-mirror-derived FP16 delta is 0.08708103047850102.
  Original Q4_0 CPU versus repaired default FP16 MLX delta is
  0.4417528951986185. All strict PPL gates remain failed.
- Add synthetic Qwen2MoE standard/direct load regressions and extend real
  MoE validation to Qwen1.5. Measure router logits during the actual forward
  pass rather than applying the gate directly to raw embeddings. Optional
  environment variables persist converted models and JSON validation reports.
- Validate public Qwen1.5-MoE-A2.7B-Chat Q2_K on Apple Silicon: mixed policy
  validation, uniform MLX load, finite logits, actual routing entropy,
  chat-template generation and native tokenizer parity all pass.

## 2.1.1 - 2026-09-21

### Fixes

- Fix `'builtins.safe_open' object is not iterable` crash during weight extraction on safetensors 0.8.0+. The `_finalize_safetensor_shards` function iterated directly over a `safe_open` context manager, which is not iterable in safetensors 0.8.0; changed to `f.keys()`. This bug prevented any real GGUF conversion from completing — all shard-output paths failed at the index-building step. Affects all architectures including stablelm (reported in #14).

### Tests

- Add opt-in real-GGUF end-to-end load and finite-logit validation for gemma and phi3 fixtures (`GGUF2MLX_RUN_E2E=1`). These tests exercise the full `core.convert()` path without monkeypatching and would have caught the safe_open bug on first run.

### Features

- Widen `--direct-quant` architecture support: llama, gemma, mistral, qwen2, stablelm (was llama, gemma only).
- Widen `--direct-quant` source quantization types to 12 qtypes: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, BF16 (was Q4_0, Q5_1 only).
- Add StableLM architecture support (`stablelm`): norm_eps, partial_rotary_factor, qk_layernorm, use_parallel_residual.
- Add `--eval-ppl` perplexity quality guardrail to benchmark harness, reusing `mlx_lm.perplexity`; reports PPL delta between standard and `--direct-quant` paths. New flags: `--ppl-dataset`, `--ppl-num-samples`, `--ppl-seq-len`.
- Add `--compare` mode to benchmark harness: runs both standard and `--direct-quant` paths, prints peak RSS / output size / elapsed delta table, emits JSON.

### Documentation

- Update README roadmap to reflect completed work and research-backed next priorities: sensitivity-aware bit retention, per-arch e2e coverage, chat-template fallback, non-text model evaluation.

---

## 2.1.0 - (initial PyPI release)

- Initial public release.
