# Publishing: MLX alongside GGUF, and PyPI releases

> The reusable `publish-mlx` workflow and the release process for this package.

> Split out of the top-level README on 2026-10-20; content is unchanged apart from heading levels.

### Publish MLX alongside GGUF on Hugging Face

The reusable [publish-mlx workflow](.github/workflows/publish-mlx.yml) lets model
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
- The default runner is `macos-15`; it **must resolve to Apple Silicon**.
  The workflow checks macOS/ARM64 before installing anything. Override `runner`
  with an available ARM64 macOS label if necessary. Size RAM and disk for the
  GGUF, HF download cache, converted weights, and temporary files; FP16 output
  may be much larger than a quantized source. Self-hosted runners must be trusted
  and their HF caches cleaned according to your private-model retention policy.
- `converter_version` defaults to the published `2.1.4` release; select a
  reviewed published version supporting your model. The workflow installs
  `gguf2mlx[vlm]` to obtain the existing Hugging Face client extra, but converts
  **supported text LLMs only**, not VLM/projector or text-encoder models.
  Supply one complete, unsplit GGUF; split GGUF sets must be merged first.
- `source_revision` defaults to `main`, but a successful upload's commit SHA is
  recommended. Downloads, the original model card, and license/notice files
  are pinned to the resolved HF commit. The resulting model card retains source
  license metadata and includes a provenance JSON with source, version, and
  precision. Check redistribution rights and the generated card before making
  a model public; conversion does not change its license.
- FP16 is the default. `quantize: true` explicitly enables **lossy** affine 4-bit
  direct quantization with group size 64; it is not bit-exact preservation of
  GGUF quantization. Failed downloads or conversions do not publish artifacts.
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
