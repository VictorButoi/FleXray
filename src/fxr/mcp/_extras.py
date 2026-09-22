"""Shared optional-dependency guards for MCP server commands."""

from __future__ import annotations

import importlib.util

MCP_EXTRA_MESSAGE = (
    'Install the MCP extra to use this command: python -m pip install '
    '"flexray[mcp]"  (from a checkout: python -m pip install ".[mcp]").'
)


def require_mcp_extra(import_name: str = "mcp") -> None:
    """Ensure the MCP SDK module is importable.

    Args:
        import_name: Top-level module name used as a representative dependency.

    Returns:
        None.

    Raises:
        SystemExit: If the requested module cannot be found.
    """

    if importlib.util.find_spec(import_name) is None:
        raise SystemExit(MCP_EXTRA_MESSAGE)


__all__ = ["MCP_EXTRA_MESSAGE", "require_mcp_extra"]
