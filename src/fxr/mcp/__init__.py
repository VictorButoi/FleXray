"""FleXray MCP server surface.

Re-exports the MCP tool functions, the static model registry, and the
`fxr-mcp` console entry point. Importing this package requires neither the
MCP SDK (`.[mcp]` extra) nor torch; `fxr.mcp.server.create_server` is the
only surface that needs the SDK installed.
"""

from .cli import main
from .inference_tools import describe_model, list_models, segment_image
from .models import DEFAULT_MODEL_ID, MODEL_REGISTRY, KnownModel
from .protocol_tools import (
    describe_dataset,
    describe_protocol,
    explain_mapping,
    list_datasets,
    list_protocols,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "KnownModel",
    "MODEL_REGISTRY",
    "describe_dataset",
    "describe_model",
    "describe_protocol",
    "explain_mapping",
    "list_datasets",
    "list_models",
    "list_protocols",
    "main",
    "segment_image",
]
