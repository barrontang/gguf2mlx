"""Tests for the publish pipeline: validation, split GGUFs, and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gguf2mlx import publish as publishing
from tests.quant_fixtures import write_llama_gguf


class FakeHub:
    """Stand-in for :class:`gguf2mlx.publish.HuggingFaceHub` with no network."""

    def __init__(self, files, blobs, revision="immutable-source-commit", cache=None):
        self.files = list(files)
        self.blobs = dict(blobs)
        self.revision = revision
        self.cache = Path(cache) if cache is not None else Path("/tmp/gguf2mlx-fakehub")
        self.events = []
        self.creates = []
        self.uploads = []
        self.downloads = []
        self.card = SimpleNamespace(
            data=SimpleNamespace(
                tags=["gguf", "text-generation"], license="apache-2.0", library_name=None,
                base_model=None,
            ),
            text="Original model card and license information.",
        )
        self.card.save = lambda path: Path(path).write_text(self.card.text)

    def resolve_revision(self, repo, revision):
        self.events.append("resolve")
        return self.revision

    def list_files(self, repo, revision):
        self.events.append("list")
        return list(self.files)

    def download(self, repo, filename, revision):
        self.events.append("download")
        assert revision == self.revision, "every download must be pinned to the commit"
        self.downloads.append(filename)
        target = self.cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.blobs[filename])
        return target

    def load_model_card(self, path):
        self.events.append("card")
        return self.card

    def create_repo(self, repo, *, private):
        self.events.append("create")
        self.creates.append((repo, private))

    def upload_folder(self, repo, folder, message):
        self.events.append("upload")
        folder = Path(folder)
        self.uploads.append(
            {
                "repo": repo,
                "message": message,
                "folder": str(folder),
                "files": {
                    path.relative_to(folder).as_posix(): path.read_bytes()
                    for path in folder.rglob("*")
                    if path.is_file()
                },
            }
        )
        return "https://huggingface.co/commit/example"


def _gguf_blobs(tmp_path, *, shards=1):
    """Write a tiny llama GGUF and slice its bytes into ``shards`` parts."""
    source = write_llama_gguf(tmp_path / "tiny.gguf", seed=5)
    data = source.read_bytes()
    if shards == 1:
        return {"weights/model.gguf": data}, "weights/model.gguf"
    step = -(-len(data) // shards)
    parts = [data[offset : offset + step] for offset in range(0, len(data), step)]
    blobs = {
        f"weights/model-{index:05d}-of-{len(parts):05d}.gguf": part
        for index, part in enumerate(parts, start=1)
    }
    return blobs, f"weights/model-{1:05d}-of-{len(parts):05d}.gguf"


@pytest.fixture
def tiny_gguf(tmp_path):
    path = write_llama_gguf(tmp_path / "tiny.gguf", seed=7)
    return path


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_id", ["model", "owner/model", "owner/model-MLX_2.1", "a/b-c.d"]
)
def test_valid_repo_ids_pass(repo_id):
    assert publishing.validate_repo_id(repo_id) == repo_id


@pytest.mark.parametrize(
    "repo_id",
    [
        "",
        "  ",
        "owner/",
        "/model",
        "owner/model/extra",
        "owner/..",
        "owner/./model",
        "owner/mo..del",
        "owner/mo--del",
        "owner/model.",
        "owner/model-",
        "owner/model name",
        "owner\\model",
        "x" * 100,
    ],
)
def test_invalid_repo_ids_are_rejected(repo_id):
    with pytest.raises(publishing.PublishError):
        publishing.validate_repo_id(repo_id)


@pytest.mark.parametrize(
    "filename",
    ["../model.gguf", "/model.gguf", "model.bin", "a/../b.gguf", "a\\b.gguf", "", "a/./b.gguf"],
)
def test_invalid_gguf_paths_are_rejected(filename):
    with pytest.raises(publishing.PublishError):
        publishing.validate_gguf_file(filename)


def test_valid_gguf_paths_are_accepted():
    assert str(publishing.validate_gguf_file("weights/model.gguf")) == "weights/model.gguf"
    assert str(publishing.validate_gguf_file("MODEL.GGUF")) == "MODEL.GGUF"


# --------------------------------------------------------------------------
# split GGUF discovery
# --------------------------------------------------------------------------


def test_plain_gguf_is_its_own_shard_list():
    assert publishing.discover_shards("weights/model.gguf") == ["weights/model.gguf"]


def test_split_gguf_expands_to_the_whole_set_from_any_entry_point():
    files = [f"m/model-{index:05d}-of-00003.gguf" for index in (1, 2, 3)]
    expected = list(files)
    for entry in files:
        assert publishing.discover_shards(entry, files) == expected


def test_missing_shard_is_reported_by_index():
    files = ["model-00001-of-00004.gguf", "model-00003-of-00004.gguf", "model-00004-of-00004.gguf"]
    with pytest.raises(publishing.PublishError) as error:
        publishing.discover_shards("model-00001-of-00004.gguf", files)
    message = str(error.value)
    assert "missing shard" in message
    assert "2 of 4" in message
    assert "model-00002-of-00004.gguf" in message


def test_shard_index_outside_the_declared_total_is_rejected():
    with pytest.raises(publishing.PublishError, match="outside 1-3"):
        publishing.discover_shards("model-00004-of-00003.gguf")


def test_single_shard_split_set_is_rejected():
    with pytest.raises(publishing.PublishError, match="at least two shards"):
        publishing.discover_shards("model-00001-of-00001.gguf")


def test_merge_shards_reproduces_the_original_bytes(tmp_path):
    data = bytes(range(256)) * 40
    parts = [tmp_path / f"part{index}" for index in range(3)]
    step = -(-len(data) // 3)
    for index, part in enumerate(parts):
        part.write_bytes(data[index * step : (index + 1) * step])
    target = tmp_path / "merged.gguf"

    size = publishing.merge_shards(parts, target)

    assert size == len(data)
    assert target.read_bytes() == data


def test_merge_shards_without_input_is_rejected(tmp_path):
    with pytest.raises(publishing.PublishError, match="no GGUF shards"):
        publishing.merge_shards([], tmp_path / "merged.gguf")


# --------------------------------------------------------------------------
# provenance, card, licenses
# --------------------------------------------------------------------------


def test_provenance_records_identity_and_precision(tiny_gguf):
    digest = publishing.sha256_file(tiny_gguf)
    record = publishing.build_provenance(
        source_repo="owner/model-GGUF",
        source_revision="abc123",
        gguf_file="weights/model.gguf",
        shards=["weights/model.gguf"],
        precision="float16",
        gguf_sha256=digest,
        gguf_bytes=tiny_gguf.stat().st_size,
        converter_version="9.9.9",
    )
    assert record["format"] == "gguf2mlx.provenance.v1"
    assert record["source_revision"] == "abc123"
    assert record["gguf_sha256"] == digest == hashlib.sha256(tiny_gguf.read_bytes()).hexdigest()
    assert record["gguf_bytes"] == tiny_gguf.stat().st_size
    assert record["gguf2mlx_version"] == "9.9.9"
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    ("quantize", "quant_source", "dtype", "bits", "group", "expected"),
    [
        (False, "requant", "float16", 4, None, "float16"),
        (True, "requant", "float16", 4, None, "affine-4bit-group64"),
        (True, "requant", "float16", 8, 32, "affine-8bit-group32"),
        (False, "preserve", "float16", 4, None, "gguf-preserved-float16"),
    ],
)
def test_precision_labels(quantize, quant_source, dtype, bits, group, expected):
    assert (
        publishing.precision_label(
            quantize=quantize,
            quant_source=quant_source,
            dtype=dtype,
            q_bits=bits,
            q_group_size=group,
        )
        == expected
    )


def test_model_card_is_retagged_and_credits_the_source():
    card = SimpleNamespace(
        data=SimpleNamespace(tags=["gguf", "text-generation"], library_name=None, base_model=None),
        text="Upstream card body.",
    )
    publishing.update_model_card(
        card, source_repo="owner/model-GGUF", source_revision="abc123", precision="float16"
    )
    assert card.data.library_name == "mlx"
    assert card.data.base_model == "owner/model-GGUF"
    assert card.data.tags == ["gguf2mlx", "mlx", "text-generation"]
    assert "owner/model-GGUF@abc123" in card.text
    assert card.text.endswith("Upstream card body.")


def test_license_files_are_copied_and_others_ignored(tmp_path):
    blobs = {
        "LICENSE": b"apache",
        "legal/NOTICE.txt": b"notice",
        "model.safetensors": b"weights",
    }

    def download(name):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blobs[name])
        return path

    copied = publishing.copy_license_files(
        files=list(blobs), download=download, output_dir=tmp_path / "out"
    )

    assert copied == ["LICENSE", "legal/NOTICE.txt"]
    assert (tmp_path / "out" / "LICENSE").read_bytes() == b"apache"
    assert not (tmp_path / "out" / "model.safetensors").exists()


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def test_publish_validates_before_touching_the_network():
    hub = FakeHub(files=[], blobs={})
    with pytest.raises(publishing.PublishError):
        publishing.publish("../escape", "model.gguf", "owner/dest", hub=hub)
    with pytest.raises(publishing.PublishError, match="different repositories"):
        publishing.publish("owner/model", "model.gguf", "OWNER/model", hub=hub)
    assert hub.events == []


def test_publish_end_to_end_merges_shards_and_records_provenance(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path, shards=3)
    hub = FakeHub(
        files=[*blobs, "README.md", "LICENSE", "notices/NOTICE.txt"],
        blobs={**blobs, "README.md": b"# Upstream\n", "LICENSE": b"apache",
               "notices/NOTICE.txt": b"notice"},
        cache=tmp_path / "cache",
    )

    result = publishing.publish(
        "owner/model-GGUF", gguf_file, "owner/model-MLX", hub=hub, work_dir=tmp_path / "work"
    )

    assert result.destination == "owner/model-MLX"
    assert result.commit_url == "https://huggingface.co/commit/example"
    assert result.shards == sorted(blobs)
    assert hub.events[0] == "resolve"
    assert hub.events[-2:] == ["create", "upload"]
    assert hub.creates == [("owner/model-MLX", True)]

    uploaded = hub.uploads[0]["files"]
    assert "config.json" in uploaded
    assert any(name.endswith(".safetensors") for name in uploaded)
    assert "LICENSE" in uploaded and "notices/NOTICE.txt" in uploaded

    provenance = json.loads(uploaded[publishing.PROVENANCE_NAME])
    assert provenance["gguf_shards"] == sorted(blobs)
    assert provenance["source_revision"] == "immutable-source-commit"
    assert provenance["precision"] == "float16"
    assert provenance["quant_source"] == "requant"
    expected_bytes = b"".join(blobs[name] for name in sorted(blobs))
    assert provenance["gguf_bytes"] == len(expected_bytes)
    assert provenance["gguf_sha256"] == hashlib.sha256(expected_bytes).hexdigest()

    card = uploaded["README.md"].decode()
    assert card.startswith("# MLX conversion")
    assert "owner/model-GGUF@immutable-source-commit" in card
    assert hub.card.data.tags == ["gguf2mlx", "mlx", "text-generation"]

    # The staging tree is temporary: nothing may survive a successful publish.
    leftovers = [path for path in (tmp_path / "work").iterdir()] if (tmp_path / "work").exists() else []
    assert leftovers == []


def test_publish_forwards_quantization_options(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path)
    hub = FakeHub(
        files=[*blobs, "README.md"], blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )
    seen = {}

    def fake_convert(source, output, **kwargs):
        seen.update(kwargs)
        Path(output).mkdir(parents=True)
        (Path(output) / "config.json").write_text("{}")
        (Path(output) / "model.safetensors").write_bytes(b"weights")
        return True

    result = publishing.publish(
        "owner/model-GGUF", gguf_file, "owner/model-MLX", quantize=True, q_bits=8,
        q_group_size=32, hub=hub, convert_fn=fake_convert,
    )

    assert seen["quantize"] is True
    assert seen["direct_quant"] is True
    assert seen["q_bits"] == 8
    assert seen["q_group_size"] == 32
    assert seen["quant_source"] == "requant"
    assert seen["dtype"] == "float16"
    assert seen["model_type"] == "llm"
    assert result.provenance["precision"] == "affine-8bit-group32"


def test_publish_preserve_implies_direct_quant_without_requantizing(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path)
    hub = FakeHub(
        files=[*blobs, "README.md"], blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )
    seen = {}

    def fake_convert(source, output, **kwargs):
        seen.update(kwargs)
        Path(output).mkdir(parents=True)
        (Path(output) / "config.json").write_text("{}")
        (Path(output) / "model.safetensors").write_bytes(b"weights")
        return True

    result = publishing.publish(
        "owner/model-GGUF", gguf_file, "owner/model-MLX", quant_source="preserve",
        hub=hub, convert_fn=fake_convert,
    )

    assert seen["quant_source"] == "preserve"
    assert seen["quantize"] is True and seen["direct_quant"] is True
    assert result.provenance["precision"] == "gguf-preserved-float16"


def test_publish_never_creates_a_repo_when_conversion_fails(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path)
    hub = FakeHub(
        files=[*blobs, "README.md"], blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )

    with pytest.raises(publishing.PublishError, match="did not succeed"):
        publishing.publish(
            "owner/model-GGUF", gguf_file, "owner/model-MLX", hub=hub,
            convert_fn=lambda *args, **kwargs: False,
        )
    assert not hub.creates
    assert not hub.uploads


def test_publish_rejects_a_conversion_without_weights(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path)
    hub = FakeHub(
        files=[*blobs, "README.md"], blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )

    def empty_convert(source, output, **kwargs):
        Path(output).mkdir(parents=True)
        (Path(output) / "config.json").write_text("{}")
        return True

    with pytest.raises(publishing.PublishError, match="did not produce"):
        publishing.publish(
            "owner/model-GGUF", gguf_file, "owner/model-MLX", hub=hub,
            convert_fn=empty_convert,
        )
    assert not hub.creates


def test_publish_refuses_a_partial_shard_set_before_downloading(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path, shards=3)
    missing = sorted(blobs)[1]
    hub = FakeHub(
        files=[*[name for name in blobs if name != missing], "README.md"],
        blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )

    with pytest.raises(publishing.PublishError, match="missing shard"):
        publishing.publish("owner/model-GGUF", gguf_file, "owner/model-MLX", hub=hub)
    assert "download" not in hub.events


def test_publish_rejects_an_unsupported_quant_source(tmp_path):
    hub = FakeHub(files=[], blobs={})
    with pytest.raises(publishing.PublishError, match="quant source"):
        publishing.publish(
            "owner/model-GGUF", "model.gguf", "owner/model-MLX", quant_source="mxfp4",
            hub=hub,
        )
    assert hub.events == []


def test_keep_work_dir_leaves_the_tree_for_debugging(tmp_path):
    blobs, gguf_file = _gguf_blobs(tmp_path)
    hub = FakeHub(
        files=[*blobs, "README.md"], blobs={**blobs, "README.md": b"# Upstream\n"},
        cache=tmp_path / "cache",
    )
    work = tmp_path / "work"

    def fake_convert(source, output, **kwargs):
        Path(output).mkdir(parents=True)
        (Path(output) / "config.json").write_text("{}")
        (Path(output) / "model.safetensors").write_bytes(b"weights")
        return True

    publishing.publish(
        "owner/model-GGUF", gguf_file, "owner/model-MLX", hub=hub,
        convert_fn=fake_convert, work_dir=work, keep_work_dir=True,
    )
    assert any(work.iterdir())


def test_resolve_converter_version_falls_back_to_the_package_version():
    version = publishing.resolve_converter_version()
    assert isinstance(version, str) and version


def test_hugging_face_hub_requires_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(publishing.PublishError, match="huggingface_hub"):
        publishing.HuggingFaceHub(token="x")._load()


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_cli_publish_forwards_options(monkeypatch, capsys):
    from gguf2mlx import gguf2mlx as core

    monkeypatch.delenv("HF_TOKEN", raising=False)
    recorded = {}

    def fake_publish(source, gguf, dest, **kwargs):
        recorded.update({"source": source, "gguf": gguf, "dest": dest, **kwargs})
        return SimpleNamespace(
            destination=dest,
            commit_url="https://huggingface.co/commit/example",
            files=["config.json", "model.safetensors"],
            shards=["weights/model.gguf"],
            provenance={"gguf2mlx_version": "9.9.9"},
        )

    monkeypatch.setattr(publishing, "publish", fake_publish)
    core.main([
        "publish", "--source", "owner/src", "--gguf", "weights/model.gguf",
        "--dest", "owner/dst", "--quantize", "--q-bits", "8", "--no-private",
    ])

    assert recorded["source"] == "owner/src"
    assert recorded["dest"] == "owner/dst"
    assert recorded["quantize"] is True
    assert recorded["q_bits"] == 8
    assert recorded["private"] is False
    assert recorded["quant_source"] == "requant"
    assert recorded["dtype"] == "float16"
    assert recorded["token"] is None
    out = capsys.readouterr().out
    assert "Published owner/dst" in out
    assert "https://huggingface.co/commit/example" in out


def test_cli_publish_falls_back_to_the_hub_token(monkeypatch):
    from gguf2mlx import gguf2mlx as core

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    recorded = {}

    def fake_publish(source, gguf, dest, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(
            destination=dest, commit_url="", files=[], shards=[],
            provenance={"gguf2mlx_version": "x"},
        )

    monkeypatch.setattr(publishing, "publish", fake_publish)
    core.main(["publish", "--source", "owner/src", "--gguf", "m.gguf", "--dest", "owner/dst"])
    assert recorded["token"] == "hf_from_env"


def test_cli_publish_exits_nonzero_on_failure(monkeypatch, capsys):
    from gguf2mlx import gguf2mlx as core

    def failing(*args, **kwargs):
        raise publishing.PublishError("split GGUF is incomplete: missing shard 2 of 3")

    monkeypatch.setattr(publishing, "publish", failing)
    with pytest.raises(SystemExit) as error:
        core.main([
            "publish", "--source", "owner/src", "--gguf", "m.gguf", "--dest", "owner/dst"
        ])
    assert error.value.code == 1
    assert "missing shard 2 of 3" in capsys.readouterr().out
