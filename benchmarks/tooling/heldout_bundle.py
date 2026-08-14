"""Command-line entry point for held-out Harbor bundle maintenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--manifest-uri", required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--manifest", type=Path, required=True)
    render_parser.add_argument("--bundle-root", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--stage", choices=("pilot", "decision"), required=True)
    render_parser.add_argument("--max-tokens", type=int, required=True)
    render_parser.add_argument("--max-cost-usd", type=float, required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument("--mcp-url", default="")
    preflight_parser.add_argument("--probe-timeout-seconds", type=float, default=120.0)
    preflight_parser.add_argument("--readiness-retries", type=int, default=3)
    preflight_parser.add_argument(
        "--readiness-retry-delay-seconds", type=float, default=5.0
    )
    control_parser = subparsers.add_parser("control-routing-status")
    control_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        from benchmarks.tooling.heldout_manifest import validate_manifest

        validate_manifest(args.manifest)
        print(args.manifest)
    elif args.command == "fetch":
        from benchmarks.tooling.heldout_integrity import fetch_bundle

        print(fetch_bundle(args.manifest_uri, args.output))
    elif args.command == "render":
        from benchmarks.tooling.heldout_plan import render_plan

        print(
            render_plan(
                args.manifest,
                args.bundle_root,
                args.output,
                args.stage,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            )
        )
    elif args.command == "preflight":
        from benchmarks.tooling.heldout_routing import treatment_readiness_preflight

        contract = treatment_readiness_preflight(
            args.manifest,
            mcp_url=args.mcp_url,
            probe_timeout_seconds=args.probe_timeout_seconds,
            readiness_retries=args.readiness_retries,
            readiness_retry_delay_seconds=args.readiness_retry_delay_seconds,
        )
        print(json.dumps(contract, indent=2, sort_keys=True))
    else:
        from benchmarks.tooling.heldout_routing import control_routing_status

        contract = control_routing_status(args.manifest)
        print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
