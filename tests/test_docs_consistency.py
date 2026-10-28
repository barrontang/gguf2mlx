"""Keep the prose in step with the code.

Every stale claim fixed in the 2.3.0 cycle was the same failure: a doc asserted
something the modules had moved away from (a Rust build that did not exist, a
"proposed" feature that shipped, a preserve mode documented as impossible). A
reviewer cannot catch that by reading one file, so the cheap invariants are
asserted here instead.

These checks are deliberately coarse — "the identifier appears somewhere in the
document" — because a precise wording assertion would fail on every edit and get
deleted. They catch removals and renames, which is what actually goes stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gguf2mlx import __version__, publish
from gguf2mlx import gguf2mlx as core
from gguf2mlx.quantize import gguf_codes

ROOT = Path(__file__).resolve().parent.parent
LLMS = ROOT / "llms.txt"
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
QUALITY = ROOT / "docs" / "quality-gates.md"
SUPPORT = ROOT / "docs" / "support-matrix.md"
PUBLISHING = ROOT / "docs" / "publishing.md"
DIRECT_QUANT = ROOT / "docs" / "direct-quant-transcoding.md"

# Spellings that exist only so a GGUF's metadata value resolves; naming every
# one of them in the prose is noise, so the docs name the canonical form and
# the alias group instead.
ALIASES = {
    "qwen3_5": "qwen35",
    "qwen3.5": "qwen35",
    "qwen3_6": "qwen36",
    "qwen3.6": "qwen36",
    "qwen3_8": "qwen38",
    "qwen3.8": "qwen38",
    "qwen2vl": "qwen2_vl",
    "qwen3vl": "qwen3_vl",
}


def _read(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} is missing"
    return path.read_text(encoding="utf-8")


def _canonical(arch: str) -> str:
    return ALIASES.get(arch, arch)


def _unnamed(text: str, identifiers: set[str]) -> list[str]:
    """Identifiers from ``identifiers`` that ``text`` never mentions as a whole word.

    Dots and underscores count as part of the name on both sides, so `qwen3`
    is not satisfied by `qwen3.5`, `qwen35`, or `qwen3_vl`.
    """
    return sorted(
        name
        for name in identifiers
        if not re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text)
    )


def test_every_convertible_architecture_is_named_in_llms_txt():
    canonical = {_canonical(arch) for arch in core.CONVERTIBLE_ARCHES}
    missing = _unnamed(_read(LLMS), canonical)
    assert not missing, (
        "llms.txt is the machine-readable summary; it must name these "
        f"convertible architectures: {missing}"
    )


def test_every_preservable_architecture_is_documented():
    """`PRESERVE_ARCHES` is a promise; the docs have to say which models it covers."""
    canonical = {_canonical(arch) for arch in gguf_codes.PRESERVE_ARCHES}
    missing = _unnamed(_read(LLMS) + _read(QUALITY), canonical)
    assert not missing, f"preserve mode accepts these but no doc names them: {missing}"


def test_weekly_moe_claim_matches_its_workflow_guard():
    """The support matrix promised MoE coverage from a job that cannot run.

    `weekly-moe.yml` needs 11-20 GB of fixtures that no step downloads, so it
    only works on a runner that already has them. Claiming it "covers two
    families" was false while the repository variable was unset, and the job
    spent weeks going red. The claim is allowed only while the workflow keeps
    the guard that turns "unconfigured" into `skipped`.
    """
    if "weekly-moe" not in _read(SUPPORT):
        return
    workflow = _read(ROOT / ".github" / "workflows" / "weekly-moe.yml")
    assert "vars.GGUF2MLX_MOE_MODELS != ''" in workflow, (
        "docs/support-matrix.md claims weekly MoE coverage, but the workflow no "
        "longer guards on GGUF2MLX_MOE_MODELS, so an unconfigured runner fails "
        "the job every week instead of skipping it"
    )
    assert "inactive until `GGUF2MLX_MOE_MODELS`" in _read(SUPPORT), (
        "the MoE coverage claim must say it is inactive until configured"
    )


def test_llms_txt_architecture_counts_are_right():
    """The stated counts are the kind of number that silently rots.

    Deriving them from the code instead of hardcoding means adding an adapter
    fails here until the sentence is updated.
    """
    every = core.CONVERTIBLE_ARCHES
    canonical = {_canonical(arch) for arch in every}
    aliases = len(every) - len(canonical)
    text = _read(LLMS)
    assert f"{len(canonical)} canonical" in text, (
        f"llms.txt should say {len(canonical)} canonical architectures; "
        f"CONVERTIBLE_ARCHES has {len(every)} IDs collapsing to {len(canonical)}"
    )
    assert f"{aliases} alias" in text, (
        f"llms.txt should say {aliases} alias spellings, got "
        f"{len(every)} IDs - {len(canonical)} canonical"
    )


def test_llms_txt_runtime_minimums_match_pyproject():
    """The version an agent quotes must be the version pip installs.

    Only the *number* is compared. The operator is not, because the two
    documents deliberately differ: pyproject declares the oldest runtime that
    might work, while `llms.txt` and the `Runtime contract` workflow name the
    exact build the suite was verified against.
    """
    text = _read(LLMS).lower()
    pyproject = _read(PYPROJECT)
    for package in ("mlx-lm", "mlx-vlm"):
        pinned = sorted(set(re.findall(rf'"{re.escape(package)}>=([\d.]+)', pyproject)))
        assert pinned, f"pyproject no longer pins a minimum for {package}"
        assert len(pinned) == 1, f"pyproject disagrees with itself on {package}: {pinned}"
        assert package in text, f"llms.txt never mentions {package}"
        assert pinned[0] in text, (
            f"llms.txt quotes no {package} version; pyproject requires >={pinned[0]}"
        )


def test_docs_do_not_describe_a_rust_build():
    """The crate was deleted; a doc promising `maturin` sends readers to a dead path."""
    offenders = [
        path.name
        for path in (LLMS, README, QUALITY, SUPPORT, PUBLISHING)
        if re.search(r"\bmaturin\b|gguf2mlx-rs|rust_backend", _read(path))
    ]
    assert not offenders, f"these docs still reference the deleted Rust crate: {offenders}"


def test_direct_quant_is_not_described_as_proposed():
    """It shipped; `proposed` in the title makes readers think it is unavailable."""
    text = _read(DIRECT_QUANT)
    assert not re.search(r"(?im)^#+\s*.*\bproposed\b", text), (
        "docs/direct-quant-transcoding.md still titles itself as proposed"
    )
    assert "direct_quant" in text or "direct-quant" in text


def test_quality_gates_documents_every_preservable_block_type():
    """A preserved width that no gate table mentions is unverifiable to a reader."""
    text = _read(QUALITY)
    missing = [
        qtype for qtype in gguf_codes.QTYPE_IDS if qtype not in text
    ]
    assert not missing, f"the preserve gate table omits these block types: {missing}"


def test_publishing_documents_every_public_publish_flag():
    """`--dry-run`/`--verify-load` exist to be reviewed; undocumented is unusable."""
    text = _read(PUBLISHING)
    source = Path(publish.__file__).read_text(encoding="utf-8")
    assert "dry_run" in source and "verify_load" in source
    for flag in ("--dry-run", "--verify-load"):
        assert flag in text, f"docs/publishing.md never mentions {flag}"


@pytest.mark.parametrize("path", [LLMS, README])
def test_version_in_docs_matches_the_package(path):
    """Only assert this when the doc volunteers a version, so the test stays useful."""
    text = _read(path)
    stated = set(re.findall(r"gguf2mlx==(\d+\.\d+\.\d+)", text))
    if stated:
        assert stated == {__version__}, (
            f"{path.name} pins gguf2mlx=={sorted(stated)} but the package is {__version__}"
        )
