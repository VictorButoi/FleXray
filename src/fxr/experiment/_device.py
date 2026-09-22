"""Single-device selection for local and scheduled training runs."""

from __future__ import annotations

import os
import re

import torch

_VALID_DEVICE_POLICIES = frozenset({"auto", "cpu", "cuda"})
_ARCHITECTURE_RE = re.compile(r"^(sm|compute)_(\d+)$")


def cuda_compatibility_error() -> str | None:
    """Return why the visible CUDA device cannot run this Torch build.

    Returns:
        ``None`` when CUDA is available and the visible device is compatible,
        otherwise a concise diagnostic suitable for a CLI error.
    """

    if not torch.cuda.is_available():
        return "CUDA is unavailable to this PyTorch process"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        architecture_names = torch.cuda.get_arch_list()
    except (AssertionError, RuntimeError) as exc:
        return f"CUDA device inspection failed: {exc}"

    native_architectures: list[tuple[int, int]] = []
    ptx_architectures: list[tuple[int, int]] = []
    for name in architecture_names:
        match = _ARCHITECTURE_RE.fullmatch(name)
        if match is None:
            continue
        digits = match.group(2)
        if len(digits) < 2:
            continue
        capability = (int(digits[:-1]), int(digits[-1]))
        target = (
            ptx_architectures
            if match.group(1) == "compute"
            else native_architectures
        )
        target.append(capability)

    same_generation = [
        compiled_minor
        for compiled_major, compiled_minor in native_architectures
        if compiled_major == major
    ]
    ptx_compatible = any(
        compiled_capability <= (major, minor)
        for compiled_capability in ptx_architectures
    )
    if same_generation and minor < min(same_generation) and not ptx_compatible:
        supported = ", ".join(architecture_names) or "none reported"
        return (
            f"{device_name} has compute capability {major}.{minor}, but "
            f"PyTorch {torch.__version__} was built for {supported}; install "
            "the repository's CUDA 12.6 environment for Volta/V100 GPUs"
        )
    if architecture_names and not same_generation and not ptx_compatible:
        supported = ", ".join(architecture_names)
        return (
            f"{device_name} has compute capability {major}.{minor}, but "
            f"PyTorch {torch.__version__} was built for {supported}"
        )
    return None


def resolve_training_device(requested: str | None = None) -> torch.device:
    """Resolve one CPU or CUDA device from the launch policy.

    Args:
        requested: ``"auto"``, ``"cpu"``, or ``"cuda"``. When omitted,
            ``FXR_DEVICE`` is read and defaults to ``"auto"``.

    Returns:
        The single device on which this training process should run.

    Raises:
        ValueError: If the requested policy is unknown.
        RuntimeError: If explicit CUDA was requested but is unavailable or
            incompatible with the installed PyTorch build.
    """

    policy = os.environ.get("FXR_DEVICE", "auto") if requested is None else requested
    if policy not in _VALID_DEVICE_POLICIES:
        choices = ", ".join(sorted(_VALID_DEVICE_POLICIES))
        raise ValueError(f"Training device must be one of {choices}; got {policy!r}.")
    if policy == "cpu":
        return torch.device("cpu")

    error = cuda_compatibility_error()
    if error is None:
        return torch.device("cuda")
    if policy == "cuda":
        raise RuntimeError(f"--device cuda was requested, but {error}.")
    return torch.device("cpu")
