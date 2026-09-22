#!/usr/bin/env python3
"""List the narrow test files that explicitly own one semantic pytest marker."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

SEMANTIC_MARKERS = ("property", "exhaustive", "scale")


def _uses_marker(path: Path, marker: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == marker
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        for node in ast.walk(tree)
    )


def marker_test_roots(tests_root: Path, marker: str) -> tuple[Path, ...]:
    """Return sorted test files containing an explicit ``pytest.mark`` owner."""

    if marker not in SEMANTIC_MARKERS:
        raise ValueError(f"unsupported semantic marker: {marker}")
    return tuple(
        path
        for path in sorted(tests_root.rglob("test_*.py"))
        if _uses_marker(path, marker)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker", choices=SEMANTIC_MARKERS)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    arguments = parser.parse_args()
    roots = marker_test_roots(arguments.tests_root, arguments.marker)
    if not roots:
        parser.error(f"no explicit pytest.mark.{arguments.marker} owners found")
    for path in roots:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
