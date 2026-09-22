"""Lazy public entry points that keep ``python -m`` execution warning-free."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Run the local training command.

    Args:
        argv: Optional command arguments excluding the program name.

    Returns:
        Process exit status from the training CLI.
    """

    from .cli import main as cli_main

    return cli_main(argv)


def submit_main(argv: list[str] | None = None) -> int:
    """Run the cluster submission command.

    Args:
        argv: Optional command arguments excluding the program name.

    Returns:
        Process exit status from the submission CLI.
    """

    from .submit_cli import main as cli_main

    return cli_main(argv)


__all__ = ["main", "submit_main"]
