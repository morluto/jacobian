"""Lightweight command-line entry point for the Jacobian MCP server."""

from __future__ import annotations

import argparse
from pathlib import Path

from jacobian import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jacobian-mcp",
        description="Run the Jacobian MCP server locally or over remote HTTP.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse"),
        default="stdio",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="state root; defaults to JACOBIAN_STATE_DIR or .jacobian",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        help="use stateless Streamable HTTP sessions",
    )
    parser.add_argument(
        "--auth-tokens-file",
        type=Path,
        help="JSON secret mapping opaque bearer tokens to tenant IDs",
    )
    parser.add_argument(
        "--public-base-url",
        help="public issuer/resource base URL advertised to remote clients",
    )
    parser.add_argument(
        "--allow-anonymous",
        action="store_true",
        help="development only: permit unauthenticated remote requests",
    )
    parser.add_argument(
        "--anonymous-tenant-id",
        default="anonymous",
        help=(
            "fixed operator-chosen tenant namespace for anonymous mode; use a "
            "different value for each isolated test endpoint"
        ),
    )
    parser.add_argument(
        "--capability-adapter",
        action="append",
        default=[],
        help="operator-approved package.module:factory entrypoint; repeatable",
    )
    parser.add_argument(
        "--capability-policy-profile",
        choices=("DEFAULT", "COMPUTE_VERIFY_NO_RETRIEVAL"),
        default="DEFAULT",
        help=(
            "operator capability policy profile; the no-retrieval profile is "
            "intended for compute/verify evaluation isolation"
        ),
    )
    for option, destination, help_text in (
        (
            "--allow-capability",
            "allowed_capability_ids",
            "allow only this capability ID; repeatable",
        ),
        (
            "--deny-capability",
            "denied_capability_ids",
            "deny this capability ID; repeatable",
        ),
        ("--allow-domain", "allowed_domains", "allow only this domain; repeatable"),
        ("--deny-domain", "denied_domains", "deny this domain; repeatable"),
        ("--allow-tag", "allowed_tags", "allow only capabilities with this tag"),
        ("--deny-tag", "denied_tags", "deny capabilities with this tag"),
    ):
        parser.add_argument(
            option,
            dest=destination,
            action="append",
            default=[],
            help=help_text,
        )
    parser.add_argument(
        "--max-tenant-runtimes",
        type=int,
        default=32,
        help="maximum in-memory tenant runtimes for remote transports",
    )
    parser.add_argument(
        "--tenant-idle-timeout-seconds",
        type=float,
        default=900.0,
        help=(
            "inactive tenant runtime TTL in seconds; expiry is checked on the "
            "next acquisition"
        ),
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.max_tenant_runtimes < 1:
        parser.error("--max-tenant-runtimes must be positive")
    if args.tenant_idle_timeout_seconds <= 0:
        parser.error("--tenant-idle-timeout-seconds must be positive")
    if args.allow_anonymous and args.auth_tokens_file is not None:
        parser.error("--allow-anonymous and --auth-tokens-file are mutually exclusive")
    if args.anonymous_tenant_id != "anonymous" and not args.allow_anonymous:
        parser.error("--anonymous-tenant-id requires --allow-anonymous")
    args.path = args.path if args.path.startswith("/") else f"/{args.path}"

    from jacobian.adapters.mcp.server import create_server
    from jacobian.capability_service import CapabilityPolicy

    capability_policy = CapabilityPolicy(
        profile=args.capability_policy_profile,
        allowed_capability_ids=frozenset(args.allowed_capability_ids),
        denied_capability_ids=frozenset(args.denied_capability_ids),
        allowed_domains=frozenset(args.allowed_domains),
        denied_domains=frozenset(args.denied_domains),
        allowed_tags=frozenset(args.allowed_tags),
        denied_tags=frozenset(args.denied_tags),
    )
    if args.transport == "stdio":
        if (
            args.auth_tokens_file is not None
            or args.allow_anonymous
            or args.anonymous_tenant_id != "anonymous"
        ):
            parser.error("remote authentication options cannot be used with stdio")
        create_server(
            state_dir=args.state_dir,
            capability_adapter_entrypoints=tuple(args.capability_adapter),
            capability_policy=capability_policy,
        ).run("stdio")
        return

    if args.auth_tokens_file is None and not args.allow_anonymous:
        parser.error(
            "remote transports require --auth-tokens-file or explicit --allow-anonymous"
        )
    token_verifier = None
    auth = None
    if args.auth_tokens_file is not None:
        from mcp.server.auth.settings import AuthSettings
        from pydantic import AnyHttpUrl

        from jacobian.adapters.mcp.remote import (
            StaticTokenVerifier,
            load_static_token_file,
        )

        public_base_url = str(
            args.public_base_url or f"http://{args.host}:{args.port}"
        ).rstrip("/")
        token_verifier = StaticTokenVerifier(
            load_static_token_file(args.auth_tokens_file)
        )
        auth = AuthSettings(
            issuer_url=AnyHttpUrl(public_base_url),
            resource_server_url=AnyHttpUrl(f"{public_base_url}{args.path}"),
            required_scopes=["jacobian:use"],
        )
    server = create_server(
        state_dir=args.state_dir,
        tenant_isolation=True,
        allow_anonymous=args.allow_anonymous,
        anonymous_tenant_id=args.anonymous_tenant_id,
        token_verifier=token_verifier,
        auth=auth,
        capability_adapter_entrypoints=tuple(args.capability_adapter),
        capability_policy=capability_policy,
        max_tenant_runtimes=args.max_tenant_runtimes,
        tenant_idle_timeout_seconds=args.tenant_idle_timeout_seconds,
    )
    if args.transport == "streamable-http":
        server.run(
            "streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
            stateless_http=args.stateless_http,
        )
    else:
        server.run(
            "sse",
            host=args.host,
            port=args.port,
            sse_path=args.path,
            message_path="/messages/",
        )


if __name__ == "__main__":
    main()
