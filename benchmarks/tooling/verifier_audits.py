"""AST audits for Harbor task verifiers.

These checks flag representation contracts that the public protocol no longer
permits: comparing a normalized ``Fraction`` back to the submitted numerator
and denominator pair.
"""

from __future__ import annotations

import ast
from itertools import pairwise
from pathlib import Path

from benchmarks.tooling.harbor_suite import ROOT

_COMPONENT_NAMES = frozenset({"numerator", "denominator"})


def _identifier(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _coprimality_message(node: ast.Compare) -> str | None:
    if not all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
        return None
    terms = [node.left, *node.comparators]
    for left, right in pairwise(terms):
        left_name = _identifier(left)
        right_name = _identifier(right)
        if left_name not in _COMPONENT_NAMES or left_name != right_name:
            continue
        if isinstance(left, ast.Attribute) and isinstance(right, ast.Name):
            return f"do not compare Fraction.{left_name} to the submitted {right_name}"
        if isinstance(left, ast.Name) and isinstance(right, ast.Attribute):
            return f"do not compare submitted {left_name} to Fraction.{right_name}"
    return None


def fraction_coprimality_failures(verifier_path: Path) -> list[str]:
    """Return failures when a verifier re-imposes lowest terms after ``Fraction``."""

    if not verifier_path.is_file():
        return []
    try:
        relative = verifier_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = verifier_path.as_posix()
    try:
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return [f"{relative}: cannot parse verifier: {exc}"]
    failures: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        message = _coprimality_message(node)
        if message is not None:
            failures.append(f"{relative}:{node.lineno}: {message}")
    return failures


__all__ = ["fraction_coprimality_failures"]
