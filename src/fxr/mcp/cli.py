"""Console entry point for the FleXray MCP server.

Importing this module never imports the MCP SDK, so the `fxr-mcp` console
script resolves on base installs and reports the missing `mcp` extra with a
clean one-line message instead of a traceback.
"""

from __future__ import annotations

import argparse

from ._extras import require_mcp_extra


def _build_parser() -> argparse.ArgumentParser:
    """Build the `fxr-mcp` argument parser.

    Returns:
        Configured argument parser for the stdio MCP server command.
    """

    parser = argparse.ArgumentParser(
        prog="fxr-mcp",
        description=(
            "Run the FleXray MCP server on stdio, exposing pretrained "
            "X-ray segmentation and protocol introspection tools to MCP "
            "clients such as Claude Code and Claude Desktop."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device for the models this server loads, e.g. cuda:1 or "
            "cpu. Defaults to auto: CUDA when available, otherwise CPU."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the FleXray MCP server over stdio until the client disconnects.

    Args:
        argv: Optional argument list. Defaults to `sys.argv[1:]`.

    Returns:
        Process exit status; `0` on clean shutdown.

    Raises:
        SystemExit: If the `mcp` optional extra is not installed.
    """

    args = _build_parser().parse_args(argv)
    require_mcp_extra("mcp")
    from .inference_tools import set_server_device
    from .server import create_server

    set_server_device(args.device)
    create_server().run()
    return 0


__all__ = ["main"]
