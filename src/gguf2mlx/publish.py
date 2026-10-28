"""Publish an MLX conversion of a Hugging Face GGUF repository.

This module is the implementation behind ``gguf2mlx publish``. It used to live
inline in ``.github/workflows/publish-mlx.yml`` as a heredoc Python payload,
which meant the pipeline could only be exercised by running a workflow. The
workflow is now a thin shell around this module so the same code path is
unit-testable without a runner, a token, or a real repository.

The pipeline is deliberately boring and fail-closed:

1. validate every caller-supplied identifier *before* touching the network
   (repository ids, the GGUF path, source != destination);
2. resolve the source branch/tag to an immutable commit so the published
   artifact can be traced back to exact bytes;
3. discover the shard set for split GGUFs and refuse partial sets;
4. download, merge, and convert;
5. write a provenance file and carry the source license files;
6. create the destination repository and upload only after conversion
   produced a loadable-looking directory.

Float16 conversion needs only numpy/gguf/safetensors, so the pipeline runs on
any runner. Affine quantization needs MLX, which is why the workflow keeps a
macOS runner for the ``quantize`` path only.

Everything that talks to the network goes through :class:`HuggingFaceHub`,
which is constructed lazily so importing this module never requires
``huggingface_hub``. Tests inject a fake hub object with the same methods.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shlex
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "CARD_NAME",
    "LICENSE_PREFIXES",
    "PROVENANCE_NAME",
    "PublishError",
    "PublishResult",
    "build_provenance",
    "copy_file_stream",
    "copy_license_files",
    "discover_shards",
    "merge_shards",
    "publish",
    "resolve_converter_version",
    "sha256_file",
    "update_model_card",
    "validate_gguf_file",
    "validate_repo_id",
]

PROVENANCE_NAME = "gguf2mlx-provenance.json"
PROVENANCE_FORMAT = "gguf2mlx.provenance.v1"
CARD_NAME = "README.md"
LICENSE_PREFIXES = ("LICENSE", "NOTICE", "COPYING")
MAX_REPO_ID_LENGTH = 96
COPY_CHUNK = 8 << 20

_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHARD = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)


class PublishError(RuntimeError):
    """A publish request that cannot be honoured safely."""


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_repo_id(repo_id: str) -> str:
    """Reject repository ids that could escape a URL or a filesystem path.

    Mirrors the Hugging Face rules closely enough to catch the cases that
    matter here: one optional namespace, no ``..``/``--`` runs, no leading or
    trailing punctuation. We do not call ``huggingface_hub.utils`` so the
    validation itself stays dependency-free and testable offline.
    """
    if not isinstance(repo_id, str) or not repo_id:
        raise PublishError("repository id is empty")
    if repo_id != repo_id.strip():
        raise PublishError(f"repository id has surrounding whitespace: {repo_id!r}")
    if len(repo_id) > MAX_REPO_ID_LENGTH:
        raise PublishError(
            f"repository id is longer than {MAX_REPO_ID_LENGTH} characters: {repo_id!r}"
        )
    parts = repo_id.split("/")
    if len(parts) > 2:
        raise PublishError(f"repository id must be 'name' or 'owner/name': {repo_id!r}")
    for part in parts:
        if not _REPO_COMPONENT.match(part):
            raise PublishError(f"invalid repository id component: {part!r}")
        if ".." in part or "--" in part:
            raise PublishError(f"repository id cannot contain '..' or '--': {part!r}")
        if part.endswith((".", "-")):
            raise PublishError(f"repository id cannot end with '.' or '-': {part!r}")
    return repo_id


def validate_gguf_file(filename: str) -> PurePosixPath:
    """Validate a repository-relative GGUF path against traversal."""
    if not isinstance(filename, str) or not filename:
        raise PublishError("gguf file is empty")
    if "\\" in filename:
        raise PublishError(f"gguf path must use '/' separators: {filename!r}")
    path = PurePosixPath(filename)
    if path.is_absolute():
        raise PublishError(f"gguf path must be repository-relative: {filename!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PublishError(f"gguf path cannot contain '.' or '..': {filename!r}")
    if str(path) != filename:
        # ``PurePosixPath`` normalises ``a//b`` and ``a/./b``; a request that
        # does not survive normalisation untouched is a request we cannot map
        # back to exactly one repository entry.
        raise PublishError(f"gguf path is not normalised: {filename!r}")
    if path.suffix.lower() != ".gguf":
        raise PublishError(f"gguf path must end in .gguf: {filename!r}")
    return path


# --------------------------------------------------------------------------
# split GGUF handling
# --------------------------------------------------------------------------


def _shard_name(directory: str, stem: str, index: int, total: int) -> str:
    name = f"{stem}-{index:05d}-of-{total:05d}.gguf"
    return f"{directory}/{name}" if directory else name


def discover_shards(
    gguf_file: str, files: Sequence[str] | None = None
) -> list[str]:
    """Return the full shard list for ``gguf_file`` in ascending index order.

    A plain (non-split) file returns a single-element list. For a split name
    like ``model-00001-of-00004.gguf`` every sibling is required: llama.cpp
    writes split GGUFs as byte slices of one logical file, so a missing shard
    silently produces a truncated model. When ``files`` (the repository file
    listing) is supplied, missing indices are reported by number.
    """
    path = validate_gguf_file(gguf_file)
    match = _SHARD.match(path.name)
    if match is None:
        return [gguf_file]

    index = int(match.group("index"))
    total = int(match.group("total"))
    if total < 2:
        raise PublishError(
            f"split GGUF must declare at least two shards: {path.name!r}"
        )
    if not 1 <= index <= total:
        raise PublishError(
            f"split GGUF shard index {index} is outside 1-{total}: {path.name!r}"
        )

    directory = str(path.parent) if str(path.parent) != "." else ""
    expected = [_shard_name(directory, match.group("stem"), i, total) for i in range(1, total + 1)]
    if files is not None:
        available = set(files)
        missing = [
            (i, name) for i, name in enumerate(expected, start=1) if name not in available
        ]
        if missing:
            listed = ", ".join(f"{i} of {total} ({name})" for i, name in missing)
            raise PublishError(
                f"split GGUF is incomplete: missing shard {listed}. "
                f"Publishing a partial shard set would produce a truncated model."
            )
    return expected


def copy_file_stream(source: Path, target: Path) -> int:
    """Copy a file with bounded memory and return the byte count written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as reader, target.open("wb") as writer:
        shutil.copyfileobj(reader, writer, COPY_CHUNK)
    return target.stat().st_size


def merge_shards(paths: Sequence[Path], target: Path) -> int:
    """Concatenate shard files into one GGUF and return its size in bytes.

    Split GGUFs are byte slices of a single file, so concatenating them in
    index order reconstructs the original byte stream.
    """
    if not paths:
        raise PublishError("no GGUF shards to merge")
    expected_size = sum(path.stat().st_size for path in paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as sink:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, sink, COPY_CHUNK)
    written = target.stat().st_size
    if written != expected_size:
        raise PublishError(
            f"merged GGUF is {written} bytes but the shards total {expected_size}"
        )
    return written


# --------------------------------------------------------------------------
# provenance, card, licenses
# --------------------------------------------------------------------------


def resolve_converter_version() -> str:
    """Return the version recorded in provenance, never a guessed constant."""
    for package in ("gguf2mlx", "gguf2mlx-nightly"):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    from . import __version__

    return __version__


def build_provenance(
    *,
    source_repo: str,
    source_revision: str,
    gguf_file: str,
    shards: Sequence[str],
    precision: str,
    gguf_sha256: str | None = None,
    gguf_bytes: int | None = None,
    converter_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe exactly which bytes produced the published artifact."""
    provenance: dict[str, Any] = {
        "format": PROVENANCE_FORMAT,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "gguf_file": gguf_file,
        "gguf_shards": list(shards),
        "precision": precision,
        "gguf2mlx_version": converter_version or resolve_converter_version(),
    }
    if gguf_sha256 is not None:
        provenance["gguf_sha256"] = gguf_sha256
    if gguf_bytes is not None:
        provenance["gguf_bytes"] = gguf_bytes
    if extra:
        provenance.update(extra)
    return provenance


def precision_label(
    *,
    quantize: bool,
    quant_source: str,
    dtype: str,
    q_bits: int,
    q_group_size: int | None,
) -> str:
    """Name the artifact precision the way the published card does."""
    if quant_source == "preserve":
        return f"gguf-preserved-{dtype}"
    if not quantize:
        return dtype
    group = 64 if q_group_size is None else q_group_size
    return f"affine-{q_bits}bit-group{group}"


def update_model_card(
    card: Any,
    *,
    source_repo: str,
    source_revision: str,
    precision: str,
    tags: Iterable[str] = (),
    base_model_relation: str | None = None,
    disk_size: int | None = None,
    conversion_command: str | None = None,
    destination: str | None = None,
    source_gguf: str | None = None,
    gguf_sha256: str | None = None,
    gguf_bytes: int | None = None,
    conversion_report: dict[str, Any] | None = None,
) -> Any:
    """Tag the upstream model card as an MLX conversion and prepend a notice."""
    card.data.library_name = "mlx"
    card.data.base_model = source_repo
    card.data.source_sha = source_revision
    if base_model_relation is not None:
        card.data.base_model_relation = base_model_relation
    if disk_size is not None:
        card.data.disk_size = disk_size
    card.data.tags = sorted(
        (set(card.data.tags or []) - {"gguf", "GGUF"})
        | {"mlx", "gguf2mlx"}
        | set(tags)
    )
    details = ""
    rows = ["| Property | Value |", "|---|---|", f"| Precision | `{precision}` |"]
    if source_gguf is not None:
        rows.append(f"| Source GGUF | `{source_gguf}` |")
    if gguf_bytes is not None:
        rows.append(f"| GGUF bytes | {gguf_bytes:,} |")
    if gguf_sha256 is not None:
        rows.append(f"| GGUF SHA-256 | `{gguf_sha256}` |")
    if disk_size is not None:
        rows.append(f"| Converted bytes | {disk_size:,} |")
    if conversion_report:
        weight = (conversion_report.get("fidelity") or {}).get("weight") or {}
        preserve = conversion_report.get("preserve") or {}
        if weight.get("status"):
            rows.append(
                f"| Weight gate | `{weight['status']}` "
                f"({weight.get('tensors_compared', 0)} tensors, max relative L2 "
                f"{weight.get('max_relative_l2')}) |"
            )
        if preserve.get("gate"):
            rows.append(
                f"| Preserve gate | `{preserve['gate']}` "
                f"({preserve.get('tensors_compared', 0)} tensors) |"
            )
    details += "\n## Conversion details\n\n" + "\n".join(rows) + "\n"
    if conversion_command is not None:
        details += (
            "\n## Reproduce this conversion\n\n"
            f"```bash\n{conversion_command}\n```\n"
        )
    if destination is not None:
        details += (
            "\n## Load with MLX-LM\n\n"
            "```python\nfrom mlx_lm import generate, load\n\n"
            f"model, tokenizer = load({destination!r})\n"
            'print(generate(model, tokenizer, prompt="Hello", max_tokens=32))\n```\n'
        )
    card.text = (
        "# MLX conversion\n\n"
        f"Converted from `{source_repo}@{source_revision}` using gguf2mlx "
        f"({precision}). See `{PROVENANCE_NAME}` for the exact source commit "
        "and precision. Conversion cannot recover precision lost in the GGUF "
        f"source.\n{details}\n"
        "## Original source model card\n\n"
        + (card.text or "")
    )
    return card


def copy_license_files(
    *,
    files: Iterable[str],
    download: Callable[[str], Path],
    output_dir: Path,
) -> list[str]:
    """Copy the source repository's license files next to the converted model."""
    copied: list[str] = []
    for filename in files:
        relative = PurePosixPath(filename)
        if not relative.parts:
            continue
        if not relative.name.upper().startswith(LICENSE_PREFIXES):
            continue
        if relative.is_absolute() or ".." in relative.parts:
            continue
        source = Path(download(filename))
        copy_file_stream(source, output_dir / relative)
        copied.append(str(relative))
    return sorted(copied)


# --------------------------------------------------------------------------
# hub adapter
# --------------------------------------------------------------------------


class HuggingFaceHub:
    """Thin adapter over ``huggingface_hub`` so tests can substitute a fake."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self._module = None

    def _load(self):
        if self._module is None:
            try:
                import huggingface_hub
            except ImportError as error:  # pragma: no cover - depends on extras
                raise PublishError(
                    "publishing needs the 'huggingface_hub' package: "
                    "pip install gguf2mlx[publish]"
                ) from error
            self._module = huggingface_hub
        return self._module

    @property
    def api(self):
        hub = self._load()
        return hub.HfApi(token=self.token)

    def resolve_revision(self, repo: str, revision: str) -> str:
        sha = self.api.model_info(repo, revision=revision).sha
        if not sha:
            raise PublishError(f"could not resolve {repo}@{revision} to a commit")
        return str(sha)

    def list_files(self, repo: str, revision: str) -> list[str]:
        return list(self.api.list_repo_files(repo, revision=revision))

    def download(self, repo: str, filename: str, revision: str) -> Path:
        return Path(
            self._load().hf_hub_download(repo, filename, revision=revision)
        )

    def load_model_card(self, path: Path):
        return self._load().ModelCard.load(str(path))

    def create_repo(self, repo: str, *, private: bool) -> None:
        self.api.create_repo(repo, repo_type="model", private=private, exist_ok=True)

    def upload_folder(self, repo: str, folder: Path, message: str) -> str:
        commit = self.api.upload_folder(
            repo_id=repo, repo_type="model", folder_path=str(folder),
            commit_message=message,
        )
        return getattr(commit, "commit_url", "") or ""


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


@dataclass
class PublishResult:
    """What a successful publish produced."""

    destination: str
    commit_url: str
    provenance: dict[str, Any]
    shards: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)
    commit_message: str = ""
    dry_run: bool = False
    verified_load: bool = False


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_cleanup(
    staging: tempfile.TemporaryDirectory, keep: bool
) -> Callable[[], None]:
    if not keep:
        return staging.cleanup

    def keep_staging() -> None:
        # ``keep_work_dir`` is for debugging a failed run: detach the finalizer
        # so the tree survives interpreter exit instead of being deleted.
        finalizer = getattr(staging, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()

    return keep_staging


def _default_convert() -> Callable[..., bool]:
    def convert(*args: Any, **kwargs: Any) -> bool:
        from .gguf2mlx import convert as convert_model

        return convert_model(*args, **kwargs)

    return convert


def _default_load() -> Callable[[str], Any]:
    """Verify with the same load-and-forward check the release gate uses.

    Loading alone catches missing or unexpected tensors, but a tensor with the
    wrong layout can load cleanly and still produce garbage. Reusing
    ``fidelity.load_and_run`` means a publish-time check and a CI gate agree on
    what "loadable" means, down to the finite-logits requirement.
    """

    def load(path: str) -> Any:
        from gguf2mlx import fidelity

        return fidelity.load_and_run(path)

    return load


def publish(
    source: str,
    gguf: str,
    dest: str,
    *,
    source_revision: str = "main",
    private: bool = True,
    quantize: bool = False,
    quant_source: str = "requant",
    q_bits: int = 4,
    q_group_size: int | None = None,
    q_mode: str = "affine",
    dtype: str = "float16",
    token: str | None = None,
    work_dir: str | Path | None = None,
    hub: Any = None,
    convert_fn: Callable[..., bool] | None = None,
    keep_work_dir: bool = False,
    dry_run: bool = False,
    verify_load: bool = False,
    load_fn: Callable[[str], Any] | None = None,
) -> PublishResult:
    """Convert ``gguf`` from ``source`` and publish the result to ``dest``.

    All validation happens before the first network call. The destination is
    only created once conversion produced ``config.json`` and at least one
    safetensors shard, so a failed run never leaves a half-published repo.
    ``dry_run`` prepares the same local artifact but skips both destination
    writes. ``verify_load`` additionally loads that artifact with ``load_fn``
    (or ``mlx_lm.load``) before any destination write.
    """
    validate_repo_id(source)
    validate_repo_id(dest)
    if source.casefold() == dest.casefold():
        raise PublishError(
            f"source and destination must be different repositories (both {source})"
        )
    validate_gguf_file(gguf)
    if quant_source not in {"preserve", "requant"}:
        raise PublishError(f"unsupported quant source: {quant_source}")

    client = hub if hub is not None else HuggingFaceHub(token=token)
    convert_impl = convert_fn or _default_convert()

    revision = client.resolve_revision(source, source_revision)
    shards = discover_shards(gguf, client.list_files(source, revision))
    precision = precision_label(
        quantize=quantize or quant_source == "preserve",
        quant_source=quant_source,
        dtype=dtype,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )

    parent = Path(work_dir) if work_dir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        prefix="gguf2mlx-publish-", dir=None if parent is None else str(parent)
    )
    cleanup = _make_cleanup(staging, keep_work_dir)
    try:
        root = Path(staging.name)
        downloads = root / "source"
        downloads.mkdir()
        local_shards: list[Path] = []
        for shard in shards:
            local = downloads / PurePosixPath(shard).name
            downloaded = client.download(source, shard, revision)
            copy_file_stream(Path(downloaded), local)
            local_shards.append(local)

        if len(local_shards) == 1:
            merged = local_shards[0]
        else:
            merged = root / "source.gguf"
            merge_shards(local_shards, merged)

        wants_quant = quantize or quant_source == "preserve"
        output = root / "mlx"
        converted = convert_impl(
            str(merged),
            str(output),
            dtype=dtype,
            quantize=wants_quant,
            q_bits=q_bits,
            q_group_size=q_group_size,
            q_mode=q_mode,
            direct_quant=wants_quant,
            quant_source=quant_source,
            model_type="llm",
        )
        if not converted:
            raise PublishError(f"conversion of {source}@{revision} GGUF did not succeed")
        produced = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        )
        if "config.json" not in produced or not any(
            name.endswith(".safetensors") for name in produced
        ):
            raise PublishError("conversion did not produce an MLX model directory")

        disk_size = sum((output / name).stat().st_size for name in produced)
        command = [
            "gguf2mlx", "publish",
            "--source", source,
            "--gguf", gguf,
            "--dest", dest,
            "--source-revision", source_revision,
            "--quant-source", quant_source,
            "--q-mode", q_mode,
            "--dtype", dtype,
        ]
        if not private:
            command.append("--no-private")
        if quantize:
            command.extend(["--quantize", "--q-bits", str(q_bits)])
        if q_group_size is not None:
            command.extend(["--q-group-size", str(q_group_size)])
        conversion_command = shlex.join(command)

        provenance = build_provenance(
            source_repo=source,
            source_revision=revision,
            gguf_file=gguf,
            shards=shards,
            precision=precision,
            gguf_sha256=sha256_file(merged),
            gguf_bytes=merged.stat().st_size,
            extra={
                "source_revision_requested": source_revision,
                "quant_source": quant_source,
                "dtype": dtype,
                "output_bytes": disk_size,
                "conversion_command": conversion_command,
            },
        )
        (output / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

        card_tags = {dtype, precision}
        if quant_source == "preserve":
            # Preserved artifacts keep several source widths at once, so a
            # single "N-bit" tag would be wrong.
            card_tags.update({"quantized", "gguf-preserved"})
        elif wants_quant:
            card_tags.update({"quantized", f"{q_bits}-bit"})
        card = client.load_model_card(client.download(source, CARD_NAME, revision))
        update_model_card(
            card,
            source_repo=source,
            source_revision=revision,
            precision=precision,
            tags=card_tags,
            base_model_relation="quantized" if wants_quant else None,
            disk_size=disk_size,
            conversion_command=conversion_command,
            destination=dest,
            source_gguf=gguf,
            gguf_sha256=provenance["gguf_sha256"],
            gguf_bytes=provenance["gguf_bytes"],
            conversion_report=json.loads(
                (output / "conversion_report.json").read_text(encoding="utf-8")
            )
            if (output / "conversion_report.json").is_file()
            else None,
        )
        card.save(output / CARD_NAME)

        licenses = copy_license_files(
            files=client.list_files(source, revision),
            download=lambda name: client.download(source, name, revision),
            output_dir=output,
        )

        if verify_load:
            loader = load_fn or _default_load()
            try:
                loaded = loader(str(output))
            except Exception as error:
                raise PublishError(f"load verification failed for {dest}: {error}") from error
            if isinstance(loaded, dict):
                # ``fidelity.load_and_run`` reports instead of raising, so a
                # missing runtime and a bad layout are told apart by ``stage``.
                if loaded.get("status") != "pass":
                    stage = loaded.get("stage")
                    detail = loaded.get("error") or loaded.get("status")
                    where = f" ({stage})" if stage else ""
                    raise PublishError(
                        f"load verification failed for {dest}{where}: {detail}"
                    )
            elif loaded is None:
                raise PublishError(f"load verification failed for {dest}: loader returned no model")

        message = f"Convert {source}@{revision} GGUF to MLX"
        commit_url = ""
        if not dry_run:
            client.create_repo(dest, private=private)
            commit_url = client.upload_folder(dest, output, message)
        files = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        )
        return PublishResult(
            destination=dest,
            commit_url=commit_url,
            provenance=provenance,
            shards=list(shards),
            files=files,
            licenses=licenses,
            commit_message=message,
            dry_run=dry_run,
            verified_load=verify_load,
        )
    finally:
        cleanup()
