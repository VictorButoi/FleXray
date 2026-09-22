"""FastMCP server assembly for the FleXray MCP surface.

This is the only `fxr.mcp` module that imports the MCP SDK; it requires the
`mcp` optional extra. Tool behavior lives in `fxr.mcp.inference_tools` and
`fxr.mcp.protocol_tools` as plain functions.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .inference_tools import describe_model, list_models, segment_image
from .protocol_tools import (
    describe_dataset,
    describe_protocol,
    explain_mapping,
    list_datasets,
    list_protocols,
)

_INSTRUCTIONS = (
    "FleXray segments anatomical structures in X-ray images with pretrained "
    "models. Use segment_image to run a model on image files and read the "
    "returned per-label statistics; use list_models/describe_model to pick a "
    "model, and the protocol tools to inspect label spaces and dataset-to-"
    "protocol label mappings. Models load on CPU; the first use of a model "
    "downloads its artifacts from Hugging Face."
)

_TOOLS = (
    segment_image,
    list_models,
    describe_model,
    list_protocols,
    describe_protocol,
    list_datasets,
    describe_dataset,
    explain_mapping,
)


def create_server() -> FastMCP:
    """Build the FleXray FastMCP server with every public tool registered.

    Returns:
        A `FastMCP` instance named `flexray` exposing the FleXray inference
        and protocol-introspection tools over the MCP protocol.
    """

    server = FastMCP("flexray", instructions=_INSTRUCTIONS)
    for tool in _TOOLS:
        server.tool()(tool)
    return server


__all__ = ["create_server"]
