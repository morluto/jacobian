#!/usr/bin/env python3
"""List the narrow test files that explicitly own one semantic pytest marker."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

SEMANTIC_MARKERS = ("property", "exhaustive", "scale")


@dataclass(frozen=True, slots=True)
class MarkerTestRootIndex:
    """One scan of test files and their explicitly owned semantic markers."""

    test_files: tuple[Path, ...]
    roots_by_marker: dict[str, tuple[Path, ...]]


def _markers_used(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in SEMANTIC_MARKERS
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def marker_test_root_index(tests_root: Path) -> MarkerTestRootIndex:
    """Parse each test module once and index every semantic marker owner."""

    test_files = tuple(sorted(tests_root.rglob("test_*.py")))
    roots_by_marker: dict[str, list[Path]] = {marker: [] for marker in SEMANTIC_MARKERS}
    for path in test_files:
        for marker in _markers_used(path):
            roots_by_marker[marker].append(path)
    return MarkerTestRootIndex(
        test_files=test_files,
        roots_by_marker={
            marker: tuple(paths) for marker, paths in roots_by_marker.items()
        },
    )


def marker_test_roots(tests_root: Path, marker: str) -> tuple[Path, ...]:
    """Return sorted test files containing an explicit ``pytest.mark`` owner."""

    if marker not in SEMANTIC_MARKERS:
        raise ValueError(f"unsupported semantic marker: {marker}")
    return marker_test_root_index(tests_root).roots_by_marker[marker]


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
