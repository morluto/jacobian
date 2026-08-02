"""Create, validate, and publish immutable Harbor benchmark snapshot locks.

Subcommands:

* ``create --dataset <id>``: build a content-addressed snapshot lock for one
  registered dataset and write it under
  ``benchmarks/snapshots/<suite>/<digest>.lock.json`` (or ``--output``).
  Requires a clean source tree; use ``--source-tree <sha>`` for controlled
  generation.
* ``validate --lock <path>``: historically validate an existing lock (schema,
  content address, internal ordering).  Use ``--reproduce`` for the
  prospective check that re-builds from the current tree.
* ``publish --lock <path>``: historically validate a lock and regenerate the
  publication ``dataset.toml`` (and copy the lock) into
  ``dist/harbor/<suite>/<snapshot>/``.

Harbor's native task digests are used by default; ``--harbor-version`` pins the
recorded Harbor version (default ``0.20.0`` to match the Makefile pin).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.benchmark_snapshots import (  # noqa: E402
    DEFAULT_HARBOR_VERSION,
    HarborSuiteError,
    build_lock,
    generate_publication,
    publication_dir,
    validate_lock,
)


def _create(args: argparse.Namespace) -> int:
    lock = build_lock(
        args.dataset,
        harbor_version=args.harbor_version,
        source_tree=args.source_tree,
    )
    rendered = json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output = args.output
    if output is None:
        digest = str(lock["snapshot_id"]).removeprefix("sha256:")
        output = (
            ROOT
            / "benchmarks"
            / "snapshots"
            / str(lock["suite"]["id"])
            / f"{digest}.lock.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)
    print(
        f"snapshot {lock['snapshot_id']} for {lock['suite']['name']} "
        f"(harbor {lock['harbor_version']}, tree {lock['source']['tree_sha']})",
        file=sys.stderr,
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    lock = validate_lock(
        args.lock,
        reproduce=args.reproduce,
        source_tree=args.source_tree,
    )
    mode = "reproduces from current tree" if args.reproduce else "historically valid"
    print(
        f"snapshot {lock['snapshot_id']} for {lock['suite']['name']} is {mode}",
        file=sys.stderr,
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    lock = validate_lock(args.lock, reproduce=args.reproduce)
    dataset_path = generate_publication(lock, dest_root=args.dest)
    print(dataset_path)
    print(
        f"published snapshot {lock['snapshot_id']} to "
        f"{publication_dir(lock, args.dest)}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser(
        "create", help="build a content-addressed snapshot lock for one dataset"
    )
    create.add_argument("--dataset", required=True, help="registered dataset id")
    create.add_argument(
        "--harbor-version",
        default=DEFAULT_HARBOR_VERSION,
        help=f"pinned Harbor version (default {DEFAULT_HARBOR_VERSION})",
    )
    create.add_argument(
        "--source-tree",
        help="explicit 40-char git tree SHA for controlled generation (bypasses "
        "the dirty-tree fail-closed check)",
    )
    create.add_argument(
        "--output",
        type=Path,
        help="override the canonical benchmarks/snapshots output path",
    )
    create.set_defaults(func=_create)

    validate = sub.add_parser(
        "validate", help="validate an existing immutable snapshot lock"
    )
    validate.add_argument("--lock", type=Path, required=True, help="lock file path")
    validate.add_argument(
        "--reproduce",
        action="store_true",
        help="prospective check: re-build from the current tree and compare",
    )
    validate.add_argument(
        "--source-tree",
        help="explicit git tree SHA for the prospective reproduction check",
    )
    validate.set_defaults(func=_validate)

    publish = sub.add_parser(
        "publish", help="regenerate the publication dataset.toml from a lock"
    )
    publish.add_argument("--lock", type=Path, required=True, help="lock file path")
    publish.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "dist" / "harbor",
        help="publication root (default dist/harbor)",
    )
    publish.add_argument(
        "--reproduce",
        action="store_true",
        help="require the current tree to reproduce the lock before publishing",
    )
    publish.set_defaults(func=_publish)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except HarborSuiteError as exc:
        print(f"benchmark snapshot error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
