"""Protocol and label-space introspection tools for the FleXray MCP server.

The implementations live in :mod:`fxr.protocols.introspect` and are shared with
the ``fxr-protocol`` command; they are re-exported here so ``fxr.mcp.server``
registers the same callables (and docstrings) as MCP tools.
"""

from __future__ import annotations

from fxr.protocols.introspect import (
    describe_dataset,
    describe_protocol,
    explain_mapping,
    list_datasets,
    list_protocols,
)

__all__ = [
    "describe_dataset",
    "describe_protocol",
    "explain_mapping",
    "list_datasets",
    "list_protocols",
]
