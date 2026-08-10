"""Inventory complete-runtime fixture usage and resource escapes per test module.

Fails closed when ``authorized_complete_runtime`` appears without a verify /
authority signal. Collection-time enforcement lives in
``tests.support.resource_closure_plugin``; this inventory is the static audit.

Usage::

    uv run python tools/inventory_test_runtime.py
    uv run python tools/inventory_test_runtime.py --unjustified-only
    make test-runtime-inventory
"""

from __future__ import annotations

import argparse
import ast
import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.test_plan.authority_signals import has_verify_authority_signal

_COMPLETE_RUNTIME_FIXTURES = frozenset(
    {
        "fresh_complete_runtime",
        "attached_complete_runtime",
        "authorized_complete_runtime",
        "complete_portfolio_template",
        "authorized_portfolio_template",
    }
)
_RESOURCE_IMPORTS = frozenset({"sqlite3", "subprocess", "multiprocessing"})


@dataclass(frozen=True, slots=True)
class ModuleInventory:
    path: str
    semantic_owner: str | None
    complete_runtime_fixtures: tuple[str, ...]
    has_verify_signal: bool
    resource_imports: tuple[str, ...]
    unjustified_authorized: bool


def _load_topology(root: Path) -> Mapping[str, tuple[str, ...]]:
    path = root / "tests" / "topology.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        str(lane["name"]): tuple(str(item) for item in lane.get("paths", ()))
        for lane in payload.get("lanes", ())
    }


def _owner_for(relative: str, lanes: Mapping[str, tuple[str, ...]]) -> str | None:
    owners = [
        name
        for name, patterns in lanes.items()
        if any(
            relative == pattern or relative.startswith(pattern.rstrip("/") + "/")
            for pattern in patterns
        )
    ]
    return owners[0] if len(owners) == 1 else None


def _collect_resource_imports(node: ast.AST, resources: set[str]) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in _RESOURCE_IMPORTS:
                resources.add(root)
    elif isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".", 1)[0]
        if root in _RESOURCE_IMPORTS:
            resources.add(root)


def _collect_runtime_fixtures(node: ast.AST, fixtures: set[str]) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (*node.args.args, *node.args.kwonlyargs):
            if arg.arg in _COMPLETE_RUNTIME_FIXTURES:
                fixtures.add(arg.arg)
    elif isinstance(node, ast.Name) and node.id in _COMPLETE_RUNTIME_FIXTURES:
        fixtures.add(node.id)


def _source_signals(tree: ast.AST, source: str) -> tuple[set[str], bool, set[str]]:
    fixtures: set[str] = set()
    resources: set[str] = set()
    for node in ast.walk(tree):
        _collect_resource_imports(node, resources)
        _collect_runtime_fixtures(node, fixtures)
    has_verify = has_verify_authority_signal(source)
    return fixtures, has_verify, resources


def inventory_modules(root: Path) -> tuple[ModuleInventory, ...]:
    lanes = _load_topology(root)
    rows: list[ModuleInventory] = []
    tests_root = root / "tests"
    for path in sorted(tests_root.rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        fixtures, has_verify, resources = _source_signals(tree, source)
        ordered = tuple(sorted(fixtures))
        unjustified = "authorized_complete_runtime" in fixtures and not has_verify
        rows.append(
            ModuleInventory(
                path=relative,
                semantic_owner=_owner_for(relative, lanes),
                complete_runtime_fixtures=ordered,
                has_verify_signal=has_verify,
                resource_imports=tuple(sorted(resources)),
                unjustified_authorized=unjustified,
            )
        )
    return tuple(rows)


def _render_text(rows: Iterable[ModuleInventory]) -> str:
    lines = [
        "path\towner\tfixtures\tverify_signal\tresources\tunjustified_authorized",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.path,
                    row.semantic_owner or "-",
                    ",".join(row.complete_runtime_fixtures) or "-",
                    "yes" if row.has_verify_signal else "no",
                    ",".join(row.resource_imports) or "-",
                    "yes" if row.unjustified_authorized else "no",
                ]
            )
        )
    interesting = [
        row for row in rows if row.complete_runtime_fixtures or row.resource_imports
    ]
    unjustified = [row for row in rows if row.unjustified_authorized]
    lines.append("")
    lines.append(
        f"# modules={sum(1 for _ in rows)} "
        f"with_complete_runtime={len(interesting)} "
        f"unjustified_authorized={len(unjustified)}"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="fmt",
    )
    parser.add_argument(
        "--unjustified-only",
        action="store_true",
        help="Only print modules that request authorized_complete_runtime without verify signals.",
    )
    parser.add_argument(
        "--fail-on-unjustified",
        action="store_true",
        help="Exit non-zero when any unjustified authorized_complete_runtime use remains.",
    )
    args = parser.parse_args(argv)
    rows = inventory_modules(args.root.resolve())
    display = rows
    if args.unjustified_only:
        display = tuple(row for row in rows if row.unjustified_authorized)
    if args.fmt == "json":
        print(json.dumps([asdict(row) for row in display], indent=2, sort_keys=True))
    else:
        print(_render_text(display), end="")
    if args.fail_on_unjustified and any(row.unjustified_authorized for row in rows):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
