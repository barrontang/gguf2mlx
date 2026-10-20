# Publishing: MLX alongside GGUF, and PyPI releases

> The reusable `publish-mlx` workflow and the release process for this package.

> Split out of the top-level README on 2026-10-20.

## The `gguf2mlx publish` command

The workflow is a thin shell around this command; anything the workflow does can
be reproduced locally with a token.

```bash
pip install "gguf2mlx[publish]"
export HF_TOKEN=...            # source read, destination write/create

gguf2mlx publish \
  --source your-org/your-model-GGUF \
  --gguf your-model-Q4_K_M.gguf \
  --dest your-org/your-model-MLX \
  --source-revision <upload-commit-sha>
```

What it does, in order (all validation happens before the first network call):

1. validates `--source`/`--dest` (repository ids, no `..`/`--`, source ≠ dest)
   and `--gguf` (repository-relative, normalised, `.gguf`);
2. resolves `--source-revision` to an immutable commit;
3. lists the repository and expands a split GGUF into its full shard set —
   **a partial set is an error naming the missing indices**, because split
   GGUFs are byte slices of one file and a missing shard would publish a
   truncated model;
4. downloads every shard, concatenates them back into one GGUF, converts it;
5. writes `gguf2mlx-provenance.json` (source repo, resolved commit, shard list,
   GGUF SHA-256 and size, precision, converter version, quant source);
6. rewrites the model card (`library_name: mlx`, `base_model: <source>`, `mlx`
   and `gguf2mlx` tags) and copies `LICENSE`/`NOTICE`/`COPYING` files;
7. creates the destination repository and uploads the folder — but only after
   conversion produced `config.json` and at least one safetensors shard.

`--quantize` enables lossy affine 4-bit (requires MLX, so Apple Silicon).
`--quant-source preserve` keeps the source GGUF blocks instead of requantizing
(see the [quality gates](quality-gates.md#preserve-tier)); it still needs the
result to be loadable by `mlx-lm`, so it is limited to `llama`, `qwen2`,
`qwen3` and `mistral`. `--keep-work-dir` leaves the temporary tree in place for
debugging.

Environment: `HF_TOKEN` is read when `--token` is absent. `--work-dir` moves the
temporary download/convert tree (defaults to the system temp directory).

## Publish MLX alongside GGUF on Hugging Face

The reusable [publish-mlx workflow](../.github/workflows/publish-mlx.yml) lets model
publishers chain GGUF uploads into MLX publication, including repositories in
`mlx-community` **when the publisher has write access to that organization**.
It does not grant organization membership or monitor Hugging Face uploads.

Add a caller like this to your publishing repository's `.github/workflows/`:

```yaml
name: Publish GGUF and MLX
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  upload-gguf:
    runs-on: ubuntu-latest
    outputs:
      revision: ${{ steps.upload.outputs.revision }}
    steps:
      # Insert your existing GGUF upload steps here.
      # The step with id: upload must write revision=<HF upload commit SHA>
      # to "$GITHUB_OUTPUT", after all GGUF files and README.md are uploaded.
      - id: upload
        run: |
          echo "Replace this step with your existing HF upload job."
          exit 1
  publish-mlx:
    needs: upload-gguf
    uses: barrontang/gguf2mlx/.github/workflows/publish-mlx.yml@main
    with:
      source_repo: your-org/your-model-GGUF
      gguf_file: your-model-Q4_K_M.gguf
      source_revision: ${{ needs.upload-gguf.outputs.revision }}
      destination_repo: your-org/your-model-MLX
      private: true
      quantize: false
    secrets:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

Replace the placeholder upload job with your existing pipeline and keep the
`needs` dependency; use its actual output name for the HF commit SHA. Pin the
reusable workflow to a reviewed **full commit SHA** instead of `@main` for
production. For uploads outside GitHub, have the trusted uploader trigger a
`workflow_dispatch` caller **after** the HF commit completes, passing the source
repository, file, and revision as inputs. Uploading to HF alone does not trigger
a GitHub workflow.

- Configure `HF_TOKEN` as a GitHub Actions secret: source read access (including
  any gated-model approval) and destination write/create access are required.
  Use a fine-grained token and pass only this secret, not `secrets: inherit`.
  Never run publishing from untrusted pull requests or accept unreviewed runner,
  converter-version, or destination inputs. No caller code is checked out.
- Float16 conversion needs only `numpy`/`gguf`/`safetensors`, so the default
  runner is **`ubuntu-latest`**. Only `quantize: true` uses `runner`
  (default `macos-15`) and checks macOS/ARM64 before installing anything; MLX
  ships Apple Silicon wheels only. Size RAM and disk for the GGUF, HF download
  cache, converted weights, and temporary files; FP16 output may be much larger
  than a quantized source. Self-hosted runners must be trusted and their HF
  caches cleaned according to your private-model retention policy.
- `converter_version` defaults to `2.2.0`, the first release that ships the
  `publish` subcommand; select a reviewed published version supporting your
  model. The workflow installs `gguf2mlx[vlm]` to obtain the existing Hugging
  Face client extra, but converts **supported text LLMs only**, not
  VLM/projector or text-encoder models. Pass any shard of a split GGUF: the
  whole set is discovered, missing shards fail the run before anything is
  downloaded, and the shards are merged before conversion.
- `source_revision` defaults to `main`, but a successful upload's commit SHA is
  recommended. Downloads, the original model card, and license/notice files
  are pinned to the resolved HF commit. The resulting model card retains source
  license metadata and includes a provenance JSON with source, version, and
  precision. Check redistribution rights and the generated card before making
  a model public; conversion does not change its license.
- FP16 is the default. `quantize: true` explicitly enables **lossy** affine 4-bit
  direct quantization with group size 64; it is not bit-exact preservation of
  GGUF quantization. (The CLI also exposes `--quant-source preserve`; the
  workflow does not, because mixing preserved widths into a public artifact is
  a deliberate per-model decision.) Failed downloads or conversions do not publish artifacts.
- New HF repositories are private by default. `private` does **not** change the
  visibility of an existing repository. Each upload commits the converted
  folder together; reruns update matching files but do not delete stale remote
  shards. Prefer a fresh destination for a different model/layout, and never
  use the source GGUF repository as the destination. Publications to the same
  destination are serialized within the calling GitHub repository.

Keep the versions in `pyproject.toml` and `src/gguf2mlx/__init__.py` identical.
Merge the release changes, then run **Actions → Publish to PyPI → Run workflow**
on `main` with target `pypi`. Alternatively, push a matching `vX.Y.Z` tag.
The workflow checks the versions, builds and validates the distributions, publishes
to PyPI, and creates a GitHub Release (and its tag if missing) at the built commit.
The `testpypi` target does not create a GitHub Release.

The old `v2.1.2` and `v2.1.3` tags contain package metadata for `2.1.1`; do not
rerun them or move them. Version `2.1.4` is the fresh release prepared to recover
from those failed uploads. PyPI cannot overwrite an existing distribution.
If publishing succeeds but GitHub Release creation fails, rerun only the failed
jobs from that same run rather than rebuilding or uploading the package again.
