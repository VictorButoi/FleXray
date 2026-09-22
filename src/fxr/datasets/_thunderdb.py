"""ThunderDB readers that reject executable pickle payloads.

MessagePack NumPy values use a local decoder without object arrays. Tensor
values use explicit weights-only CPU loading. Passed-in database objects are
owned by callers and are not changed by this module.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PICKLE_EXTENSIONS = frozenset({"pkl", "pickle"})


@lru_cache(maxsize=1)
def _safe_thunderdb_type() -> type:
    """Return the guarded ThunderDB subclass, importing the optional backend.

    Returns:
        ThunderDB subclass that overrides value decoding.

    Raises:
        ImportError: If the training dependency thunderpack is unavailable.
    """

    from .packaging import _import_thunderpack

    thunderpack = _import_thunderpack()

    class SafeThunderDB(thunderpack.ThunderDB):
        """ThunderDB with restricted value decoding.

        Attributes:
            env: Inherited LMDB environment used for database reads.
            autogrow: Inherited map-growth setting, unused for read-only opens.
            map_size: Inherited property reporting the LMDB mapping size.
        """

        def _post_value(self, value: bytes) -> Any:
            """Decode a stored value while refusing executable formats.

            Args:
                value: Stored bytes shaped extension, NUL, then payload.

            Returns:
                Decoded data with NumPy arrays or CPU tensors when present.

            Raises:
                ValueError: If the tag or payload is unsafe or malformed.
            """

            tag, separator, payload = value.partition(b"\x00")
            if not separator:
                raise ValueError("ThunderDB value is missing its format separator.")
            extension = tag.decode("utf-8")
            # Match thunderpack's filename and extension normalization.
            parts = extension.partition(".")[2].strip(".").lower().split(".")
            if _PICKLE_EXTENSIONS.intersection(parts):
                raise ValueError(
                    f"Refusing to unpickle a ThunderDB value tagged {extension!r}."
                )
            if not parts[0] or len(parts) > 2:
                raise ValueError(f"Invalid ThunderDB format tag {extension!r}.")

            if len(parts) == 2:
                from thunderpack.compression import CompressionFormat
                from thunderpack.supported import SupportedFormats

                compression = SupportedFormats.get_format(parts[1])
                # A decoder such as .pt must never serve as a compression codec.
                if not isinstance(compression, type) or not issubclass(
                    compression, CompressionFormat
                ):
                    raise ValueError(
                        f"Invalid ThunderDB compression tag {extension!r}."
                    )
                if parts[0] in {"msgpack", "pt"}:
                    payload = compression.decode(payload)

            if parts[0] == "msgpack":
                return _decode_msgpack(payload)
            if parts[0] == "pt":
                return _decode_tensor(payload)
            return super()._post_value(value)

    return SafeThunderDB


def _decode_msgpack(payload: bytes) -> Any:
    """Decode MessagePack with a local, non-pickling NumPy hook.

    Args:
        payload: Uncompressed MessagePack bytes.

    Returns:
        Decoded data, including non-object NumPy arrays and scalars.

    Raises:
        ValueError: If the payload is malformed or contains object arrays.
    """

    # thunderpack patches msgpack.unpackb to invoke msgpack_numpy.decode before
    # caller hooks. Use the underlying backend so no upstream hook runs first.
    try:
        from msgpack._cmsgpack import unpackb
    except ImportError:
        from msgpack.fallback import unpackb

    return unpackb(payload, raw=False, object_hook=_decode_numpy_value)


def _decode_numpy_value(value: dict) -> Any:
    """Decode one MessagePack mapping without allowing Python object dtypes.

    Args:
        value: Mapping supplied by the MessagePack object hook.

    Returns:
        The mapping itself, or its NumPy array/scalar or complex value.

    Raises:
        ValueError: If a NumPy representation is unsafe or malformed.
    """

    if b"nd" not in value:
        if b"complex" in value:
            data = value[b"data"]
            return complex(data.decode("utf-8") if isinstance(data, bytes) else data)
        return value
    if value.get(b"kind") == b"O":
        raise ValueError("Refusing to unpickle a ThunderDB NumPy object value.")
    try:
        dtype = _numpy_dtype(value[b"type"])
        if dtype.hasobject:
            raise ValueError("ThunderDB NumPy object dtypes are not supported.")
        data = value[b"data"]
        if not isinstance(data, bytes):
            raise TypeError("NumPy data must be bytes.")
        if value[b"nd"] is True:
            return np.ndarray(shape=value[b"shape"], dtype=dtype, buffer=data)
        return np.frombuffer(data, dtype=dtype)[0]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid ThunderDB NumPy value: {exc}") from exc


def _numpy_dtype(descriptor: Any) -> np.dtype:
    """Restore a dtype descriptor, including nested structured fields.

    Args:
        descriptor: NumPy dtype string or field descriptions from MessagePack.

    Returns:
        NumPy dtype; the caller must reject any dtype containing objects.
    """

    if isinstance(descriptor, bytes):
        descriptor = descriptor.decode("utf-8")
    if isinstance(descriptor, list):
        fields = []
        for name, field_type, *shape in descriptor:
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            fields.append((name, _numpy_dtype(field_type), *shape))
        descriptor = fields
    return np.dtype(descriptor)


def _decode_tensor(payload: bytes) -> Any:
    """Load a tensor value on CPU using the restricted PyTorch unpickler.

    Args:
        payload: Uncompressed torch.save bytes.

    Returns:
        Data accepted by PyTorch weights-only loading.

    Raises:
        ValueError: If the value is a TorchScript archive or requires unsafe
            pickle loading. No unrestricted retry is attempted.
    """

    # Older supported torch versions dispatch TorchScript ZIPs to jit.load
    # before checking weights_only. Reject them before entering torch.load.
    if payload.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if any(
                name.rsplit("/", 1)[-1] == "constants.pkl"
                for name in archive.namelist()
            ):
                raise ValueError(
                    "ThunderDB tensor values cannot be TorchScript archives."
                )
    try:
        # BytesIO also avoids the legacy tar loader that older torch versions
        # use only for files with a directly readable file descriptor.
        return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Unsafe or unsupported ThunderDB tensor value; "
            "values must support weights-only loading."
        ) from exc


def open_thunderdb(path: str | Path) -> Any:
    """Open a ThunderDB path read-only with restricted value decoding.

    Args:
        path: Existing ThunderDB directory.

    Returns:
        Open database context manager with non-pickling MessagePack decoding
        and weights-only CPU tensor loading.

    Raises:
        ImportError: If the training dependency thunderpack is unavailable.
    """

    return _safe_thunderdb_type().open(str(path), "r")
