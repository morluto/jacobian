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
    parser.add_argument(
        "--inline-result-bytes",
        type=int,
        metavar="BYTES",
        help="Opt into bounded local delivery (4096-1048576 bytes); larger complete results use temporary MCP resources.",
    )
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()

    from jacobian.mcp.delivery import ResultDelivery
    from jacobian.mcp.server import create_server

    delivery = None
    if arguments.inline_result_bytes is not None:
        try:
            delivery = ResultDelivery(arguments.inline_result_bytes)
        except ValueError as exc:
            parser.error(str(exc))

    create_server(result_delivery=delivery).run("stdio")


if __name__ == "__main__":
    main()
