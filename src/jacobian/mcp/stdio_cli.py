"""Local stdio entry point for one Jacobian runtime."""

from __future__ import annotations

import argparse

from jacobian import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jacobian-mcp",
        description="Run one local Jacobian MCP server over stdio.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main() -> None:
    _parser().parse_args()

    from jacobian.mcp.server import create_server

    create_server().run("stdio")


if __name__ == "__main__":
    main()
