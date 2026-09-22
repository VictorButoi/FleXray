"""Regression tests for restricted ThunderDB decoding.

The MessagePack probes contain an ordinary pickled dictionary. A spy raises
before deserialization, so these tests never execute code from a payload.
"""

import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

thunderpack = pytest.importorskip("thunderpack")
lmdb = pytest.importorskip("lmdb")

from fxr.datasets import (
    SplitThunderDBStorageBackend,
    inspect_thunderdb,
    validate_packed_dataset,
)
from fxr.datasets._thunderdb import open_thunderdb
from fxr.datasets.cli import main as dataset_main


def _write_raw_attrs(path: Path, encoded: bytes) -> Path:
    """Write encoded metadata to a temporary database; return its path."""
    with lmdb.open(str(path), map_size=2**20) as database:
        with database.begin(write=True) as transaction:
            transaction.put(b"_attrs", encoded)
    return path


@pytest.mark.parametrize(
    "extension", [".pkl", ".pickle", ".pkl.lz4", ".pickle.gz", ".PKL", "value.PiCkLe"]
)
def test_explicit_pickle_tags_are_refused(tmp_path: Path, extension: str) -> None:
    """Verify the original guard blocks plain and compressed pickle tags."""
    encoded = thunderpack.autopackb({"dataset_name": "Review"}, ext=extension)
    path = _write_raw_attrs(tmp_path / "database", encoded)
    with pytest.raises(ValueError, match="Refusing to unpickle"):
        inspect_thunderdb(path)


@pytest.mark.parametrize("extension", [".msgpack", ".msgpack.lz4"])
@pytest.mark.parametrize("nested", [False, True])
def test_msgpack_metadata_never_calls_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str, nested: bool
) -> None:
    """Reject object-array pickle metadata, including nested metadata values."""
    envelope = {
        b"nd": True,
        b"kind": b"O",
        b"data": pickle.dumps({"dataset_name": "Review"}),
    }
    value = {"ignored_metadata": envelope} if nested else envelope
    encoded = thunderpack.autopackb(value, ext=extension)
    path = _write_raw_attrs(tmp_path / "database", encoded)

    def refuse_pickle(*args, **kwargs):
        """Fail before the upstream decoder unpickles even the benign fixture."""
        raise AssertionError("MessagePack reached unsafe pickle.loads")

    monkeypatch.setattr(pickle, "loads", refuse_pickle)
    with pytest.raises(ValueError):
        inspect_thunderdb(path)


@pytest.mark.parametrize("reader", ["check", "training"])
def test_other_readers_never_unpickle_msgpack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    """Require the shared guard to protect validation and training metadata."""
    encoded = thunderpack.autopackb(
        {b"nd": True, b"kind": b"O", b"data": pickle.dumps({})},
        ext=".msgpack.lz4",
    )
    path = _write_raw_attrs(tmp_path / "database", encoded)

    def refuse_pickle(*args, **kwargs):
        """Fail before the upstream decoder unpickles the benign fixture."""
        raise AssertionError("MessagePack reached unsafe pickle.loads")

    monkeypatch.setattr(pickle, "loads", refuse_pickle)
    with pytest.raises(ValueError):
        if reader == "check":
            validate_packed_dataset(path)
        else:
            SplitThunderDBStorageBackend(
                path, dataset_name="Review", modality="xray", split="train"
            )


@pytest.mark.parametrize("extension", [".pt", ".pt.lz4"])
def test_torch_values_request_restricted_cpu_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str
) -> None:
    """Make tensor decoding independent of torch defaults and saved devices."""
    encoded = thunderpack.autopackb({"dataset_name": "Review"}, ext=extension)
    path = _write_raw_attrs(tmp_path / "database", encoded)
    calls = []

    def record_load(*args, **kwargs):
        """Record deserialization options and return harmless metadata."""
        calls.append(kwargs)
        return {"dataset_name": "Review"}

    monkeypatch.setattr(torch, "load", record_load)
    inspect_thunderdb(path)
    assert calls and calls[0].get("weights_only") is True
    assert calls[0].get("map_location") == "cpu"


def test_numeric_msgpack_round_trip(tmp_path: Path) -> None:
    """Retain ordinary NumPy arrays and canonical label keys in compressed data."""
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    with thunderpack.ThunderDB.open(str(tmp_path / "database"), "c") as database:
        database["sample"] = {"img": array, "labels": {"0": "background", "1": "bone"}}
    with open_thunderdb(tmp_path / "database") as database:
        sample = database["sample"]
    np.testing.assert_array_equal(sample["img"], array)
    assert sample["labels"] == {"0": "background", "1": "bone"}


@pytest.mark.parametrize("command", ["inspect", "check"])
def test_cli_refuses_explicit_pickle_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """Reject explicit pickle metadata with an actionable CLI error."""
    path = _write_raw_attrs(
        tmp_path / "database", thunderpack.autopackb({}, ext=".pkl.lz4")
    )
    with pytest.raises(SystemExit) as error:
        dataset_main([command, str(path)])
    assert error.value.code == 2
    assert "Refusing to unpickle" in capsys.readouterr().err


@pytest.mark.parametrize("descriptor", [
    "O",
    [["object_field", "O"]],
    [["nested", [["object_field", "O"]]]],
])
@pytest.mark.parametrize("is_array", [False, True])
def test_numpy_object_dtype_cannot_hide_behind_a_numeric_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, descriptor, is_array: bool
) -> None:
    """Reject disguised object dtypes before NumPy reads any object pointers."""
    encoded = thunderpack.autopackb({
        b"nd": is_array, b"kind": b"", b"type": descriptor,
        b"shape": [1], b"data": bytes(64),
    }, ext=".msgpack.lz4")
    path = _write_raw_attrs(tmp_path / "database", encoded)

    def refuse_construction(*args, **kwargs):
        """Prove dtype validation precedes all array construction."""
        raise AssertionError("An object dtype reached NumPy construction")

    monkeypatch.setattr(np, "ndarray", refuse_construction)
    monkeypatch.setattr(np, "frombuffer", refuse_construction)
    with open_thunderdb(path) as database, pytest.raises(ValueError, match="object"):
        database["_attrs"]


@pytest.mark.parametrize("suffix", ["", ".lz4", ".gz", ".bz2", ".snappy", ".zst"])
@pytest.mark.parametrize("format_name", ["msgpack", "pt"])
def test_compressed_safe_values_round_trip(
    tmp_path: Path, suffix: str, format_name: str
) -> None:
    """Keep every existing compression codec usable for numeric data."""
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = array if format_name == "msgpack" else torch.from_numpy(array)
    encoded = thunderpack.autopackb(
        {"nested": [expected, {"name": "case", "missing": None}]},
        ext=f".{format_name}{suffix}",
    )
    path = _write_raw_attrs(tmp_path / "database", encoded)
    with open_thunderdb(path) as database:
        result = database["_attrs"]["nested"]
    if format_name == "msgpack":
        np.testing.assert_array_equal(result[0], expected)
        assert result[0].dtype == array.dtype
    else:
        torch.testing.assert_close(result[0], expected)
        assert result[0].device.type == "cpu"
    assert result[1] == {"name": "case", "missing": None}


def test_numpy_values_keep_their_dtypes_and_shapes(tmp_path: Path) -> None:
    """Preserve scalar, empty, string, endian, and structured NumPy values."""
    values = [
        np.float32(1.5), np.int16(-4), np.bool_(True), complex(1, 2),
        np.arange(6, dtype=">i2").reshape(2, 3),
        np.ones((2, 3), dtype=np.uint16)[:, ::-1],
        np.zeros((0, 2), dtype=np.float16), np.array(5, dtype=np.int32),
        np.array(["ab", "cd"]), np.array([b"ab", b"cd"]),
        np.zeros(2, dtype=[("number", "<i2"), ("coords", "<f4", (2,))]),
        np.zeros(2, dtype=[("nested", [("number", "<i2")])]),
    ]
    path = _write_raw_attrs(
        tmp_path / "database", thunderpack.autopackb(values, ext=".msgpack.lz4")
    )
    with open_thunderdb(path) as database:
        results = database["_attrs"]
    for expected, result in zip(values, results, strict=True):
        np.testing.assert_array_equal(result, expected)
        if isinstance(expected, (np.ndarray, np.generic)):
            assert result.dtype == expected.dtype
            assert result.shape == expected.shape


def _record_unpickling(marker: str) -> dict:
    """Write a temporary marker if restricted loading regresses; return metadata."""
    Path(marker).write_text("unsafe decoding ran", encoding="utf-8")
    return {"dataset_name": "Probe"}


class _PickleProbe:
    """Fixture that records unintended pickle execution.

    Attributes:
        marker: Temporary file path written only if its reducer is executed.
    """

    def __init__(self, marker: Path) -> None:
        """Store the marker path argument; return None."""
        self.marker = str(marker)

    def __reduce__(self):
        """Return a marker-writing reducer without executing it during save."""
        return _record_unpickling, (self.marker,)


@pytest.mark.parametrize("extension", [".pt", ".pt.lz4"])
@pytest.mark.parametrize("force_unsafe_default", [False, True])
def test_torch_rejects_executable_pickle_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    extension: str, force_unsafe_default: bool,
) -> None:
    """Reject executable reducers even when the environment requests unsafe defaults."""
    marker = tmp_path / "unsafe-marker"
    encoded = thunderpack.autopackb(_PickleProbe(marker), ext=extension)
    path = _write_raw_attrs(tmp_path / "database", encoded)
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    if force_unsafe_default:
        monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    else:
        monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    with pytest.raises(ValueError, match="weights-only"):
        inspect_thunderdb(path)
    assert not marker.exists()


def test_torch_values_saved_on_cuda_load_on_cpu(tmp_path: Path) -> None:
    """Read a CUDA-tagged tensor without requiring a GPU or CUDA allocation."""
    expected = torch.arange(4, dtype=torch.float32)
    original = io.BytesIO()
    torch.save(expected, original)
    modified = io.BytesIO()
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(modified, "w") as target:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename.endswith("/data.pkl"):
                old = b"X\x03\x00\x00\x00cpu"
                assert old in data
                data = data.replace(old, b"X\x06\x00\x00\x00cuda:0")
            target.writestr(entry, data)
    path = _write_raw_attrs(tmp_path / "database", b".pt\x00" + modified.getvalue())
    with open_thunderdb(path) as database:
        result = database["_attrs"]
    assert result.device.type == "cpu"
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("legacy", [False, True])
def test_torch_save_formats_remain_readable(tmp_path: Path, legacy: bool) -> None:
    """Read regular tensor archives and the older torch.save pickle stream."""
    expected = {"image": torch.arange(8).reshape(2, 4), "metadata": ("xray", 7)}
    buffer = io.BytesIO()
    torch.save(expected, buffer, _use_new_zipfile_serialization=not legacy)
    path = _write_raw_attrs(tmp_path / "database", b".pt\x00" + buffer.getvalue())
    with open_thunderdb(path) as database:
        result = database["_attrs"]
    torch.testing.assert_close(result["image"], expected["image"])
    assert result["metadata"] == expected["metadata"]


def test_torchscript_is_rejected_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent older torch versions from dispatching archives to jit.load."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/constants.pkl", b"not executable")
    path = _write_raw_attrs(tmp_path / "database", b".pt\x00" + buffer.getvalue())

    def refuse_load(*args, **kwargs):
        """Fail if a TorchScript archive reaches any PyTorch loader."""
        raise AssertionError("TorchScript reached torch.load")

    monkeypatch.setattr(torch, "load", refuse_load)
    with pytest.raises(ValueError, match="TorchScript"):
        inspect_thunderdb(path)


@pytest.mark.parametrize("tag", [".bin.pt", ".bin.msgpack", ".msgpack.pt", ".pt.msgpack"])
def test_value_decoders_cannot_masquerade_as_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str
) -> None:
    """Reject compound tags that hide executable decoding in a codec suffix."""
    path = _write_raw_attrs(tmp_path / "database", tag.encode() + b"\x00payload")

    def refuse_decode(*args, **kwargs):
        """Fail if an invalid codec tag reaches an unsafe loader."""
        raise AssertionError("Invalid codec reached a decoder")

    monkeypatch.setattr(torch, "load", refuse_decode)
    monkeypatch.setattr(pickle, "loads", refuse_decode)
    with pytest.raises(ValueError, match="compression"):
        inspect_thunderdb(path)


def test_msgpack_python_backend_also_refuses_object_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the unpatched fallback backend used without the C extension."""
    monkeypatch.setitem(sys.modules, "msgpack._cmsgpack", None)
    encoded = thunderpack.autopackb(
        {b"nd": True, b"kind": b"O", b"data": pickle.dumps({})},
        ext=".msgpack.lz4",
    )
    path = _write_raw_attrs(tmp_path / "database", encoded)
    with pytest.raises(ValueError, match="Refusing to unpickle"):
        inspect_thunderdb(path)
