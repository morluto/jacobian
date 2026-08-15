"""AST and schema audits for Harbor task verifiers.

These checks flag representation contracts that the public protocol no longer
permits: comparing a normalized ``Fraction`` back to the submitted numerator
and denominator pair, or encoding exact rationals as canonical strings.
"""

from __future__ import annotations

import ast
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

from benchmarks.tooling.harbor_suite import ROOT

_COMPONENT_NAMES = frozenset({"numerator", "denominator"})
STRUCTURED_RATIONAL_SCHEMA = {
    "additionalProperties": False,
    "properties": {
        "denominator": {"minimum": 1, "type": "integer"},
        "numerator": {"type": "integer"},
    },
    "required": ["numerator", "denominator"],
    "type": "object",
}
_RATIONAL_STRING_PATTERNS = frozenset(
    {
        r"^-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?$",
        r"^-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?$",
        r"^-?(?:0|[1-9][0-9]{0,5})(?:/[2-9]|/[1-9][0-9]{1,5})?$",
        r"^(0|1|[1-9][0-9]*/[1-9][0-9]*)$",
        r"^(?:0|-?[1-9][0-9]*)(?:/[1-9][0-9]*)?$",
        r"^-?[0-9]+(?:/[1-9][0-9]*)?$",
    }
)


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


def is_canonical_rational_string_schema(node: object) -> bool:
    """Return whether a JSON Schema fragment is a canonical rational string."""

    if not isinstance(node, dict) or node.get("type") != "string":
        return False
    pattern = node.get("pattern")
    if not isinstance(pattern, str):
        return False
    if pattern in _RATIONAL_STRING_PATTERNS:
        return True
    return "/[1-9" in pattern and "[0-9]" in pattern


def _schema_paths(node: Any, prefix: str) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = [(prefix, node)]
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_schema_paths(value, f"{prefix}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_schema_paths(value, f"{prefix}[{index}]"))
    return found


def canonical_string_rational_schema_failures(schema_path: Path) -> list[str]:
    """Return failures when a public schema encodes rationals as strings."""

    if not schema_path.is_file():
        return []
    try:
        relative = schema_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = schema_path.as_posix()
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{relative}: cannot parse schema: {exc}"]
    failures: list[str] = []
    for path, node in _schema_paths(payload, relative):
        if is_canonical_rational_string_schema(node):
            failures.append(f"{path}: encode exact rationals as structured objects")
    return failures


__all__ = [
    "STRUCTURED_RATIONAL_SCHEMA",
    "canonical_string_rational_schema_failures",
    "fraction_coprimality_failures",
    "is_canonical_rational_string_schema",
]
