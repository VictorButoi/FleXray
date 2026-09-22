"""Filesystem-backed experiment foundation.

``BaseExperiment`` owns the run directory: it loads the immutable ``config.yml``,
maintains a small JSON ``properties`` file used to detect resumable runs, and
provides ``from_config`` which stamps a new run directory with a time-ordered
unique id. It deliberately carries no training logic.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fxr.config import (
    Config,
    check_missing,
    config_digest,
    generate_tuid,
    validate_run_id,
)


def fix_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs.

    Args:
        seed: Seed value applied to all RNGs.

    Returns:
        ``None``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_persisted_run_identity(path: Path, config: Config) -> None:
    """Validate persisted run metadata and digest when they are available.

    Older hand-authored run directories may lack ``metadata.json`` and remain
    readable. Runs created by :meth:`BaseExperiment.from_config` bind the three
    metadata components to the directory name; generated 32-hex digests are
    additionally checked against the immutable config.

    Args:
        path: Existing run directory.
        config: Config loaded from that run.

    Returns:
        ``None`` when the stored identity is internally consistent.

    Raises:
        TypeError: If ``metadata.json`` is not a mapping.
        ValueError: If metadata, directory identity, or a generated digest does
            not match the persisted config.
    """

    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Run metadata must be a mapping: {metadata_path}.")
    parts = validate_run_id(path.name)
    actual_digest = config_digest(config)
    stored_run_id = metadata.get("run_id")
    if stored_run_id is not None and stored_run_id != path.name:
        raise ValueError(
            f"Run metadata field 'run_id' does not match directory {path.name!r}."
        )
    stored_config_digest = metadata.get("config_digest")
    if stored_config_digest is not None and stored_config_digest != actual_digest:
        raise ValueError(
            f"Run config digest does not match metadata for {path}; "
            "config.yml may have been modified."
        )
    for key, expected in zip(("create_time", "nonce", "digest"), parts):
        if metadata.get(key) != expected:
            raise ValueError(
                f"Run metadata field {key!r} does not match directory {path.name!r}."
            )
    digest = parts[2]
    is_generated_digest = len(digest) == 32 and all(
        character in "0123456789abcdef" for character in digest.casefold()
    )
    if is_generated_digest and actual_digest != digest.casefold():
        raise ValueError(
            f"Run config digest does not match metadata for {path}; "
            "config.yml may have been modified."
        )


class _Properties:
    """Tiny JSON-backed mutable mapping for run properties.

    Attributes:
        path: JSON file the properties are persisted to.
        _data: In-memory property values awaiting persistence.
    """

    def __init__(self, path: Path) -> None:
        """Load existing properties from ``path`` or start empty.

        Args:
            path: JSON file used to load and persist property values.

        Returns:
            ``None``.
        """
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the stored value for ``key`` or ``default``.

        Args:
            key: Property key to look up.
            default: Value returned when ``key`` is absent.

        Returns:
            Stored property value or ``default``.
        """
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Return the stored value for ``key``.

        Args:
            key: Property key to look up.

        Returns:
            Stored property value.
        """
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` in memory.

        Args:
            key: Property key to update.
            value: Value to associate with ``key``.

        Returns:
            ``None``.
        """
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is currently stored.

        Args:
            key: Candidate property key.

        Returns:
            Whether the property mapping contains ``key``.
        """
        return key in self._data

    def save(self) -> None:
        """Atomically persist the current properties mapping.

        Returns:
            ``None``.
        """

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent, delete=False, encoding="utf-8"
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(self._data, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class BaseExperiment:
    """Run directory holding an immutable config and resumable properties.

    Attributes:
        path: Run directory under ``log.root``.
        name: Directory name, equal to the run uuid.
        config: Immutable ``Config`` loaded from ``config.yml``.
        properties: JSON-backed mutable run properties.
    """

    def __init__(self, path: str | Path, set_seed: bool = True) -> None:
        """Open an existing run directory.

        Args:
            path: Run directory containing ``config.yml``.
            set_seed: Whether to seed RNGs from ``experiment.seed`` if present.

        Returns:
            ``None``.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Experiment path does not exist: {self.path}.")
        self.name = self.path.name
        self.config = Config.from_file(self.path / "config.yml")
        _validate_persisted_run_identity(self.path, self.config)
        self.properties = _Properties(self.path / "properties.json")
        self.properties["experiment.class"] = type(self).__name__

        if set_seed and "experiment.seed" in self.config:
            fix_seed(int(self.config["experiment.seed"]))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Config,
        uuid: str | None = None,
        **kwargs: Any,
    ) -> "BaseExperiment":
        """Create and open a run directory for a config.

        Args:
            config: Experiment config mapping or ``Config``.
            uuid: Optional safe three-part run-directory id; generated when omitted.
            **kwargs: Forwarded to the constructor.

        Returns:
            An opened experiment instance for the new run directory.
        """

        if isinstance(config, Config):
            config = config.to_dict()
        config = dict(config)
        check_missing(config)

        root = Path(config.get("log", {}).get("root", "."))
        if uuid is None:
            create_time, nonce = generate_tuid()
            digest = config_digest(config)
            uuid = f"{create_time}-{nonce}-{digest}"
        else:
            create_time, nonce, digest = validate_run_id(uuid)

        path = root / uuid
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Run directory already exists: {path}. Open it explicitly "
                "to resume instead of creating over it."
            ) from exc
        metadata = {
            "create_time": create_time,
            "nonce": nonce,
            "digest": digest,
            "run_id": uuid,
            "config_digest": config_digest(config),
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        with (path / "config.yml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, indent=2, sort_keys=False)
        return cls(str(path.absolute()), **kwargs)

    def __repr__(self) -> str:
        """Return a concise constructor-style experiment representation.

        Returns:
            Debug representation containing this experiment path.
        """
        return f'{type(self).__name__}("{self.path}")'
