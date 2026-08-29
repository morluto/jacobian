"""Serve a local direct-tool surface for frozen adoption evaluations."""

from __future__ import annotations

import argparse

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument(
        "--with-math-find",
        action="store_true",
        help="add the mathematical-vocabulary control to the direct surface",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = args.path if args.path.startswith("/") else f"/{args.path}"
    server = _build_server(
        state=AppState(operation_catalog=Catalog.open()),
        include_math_find=args.with_math_find,
    )
    server.run(
        "streamable-http",
        host=args.host,
        port=args.port,
        streamable_http_path=path,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
