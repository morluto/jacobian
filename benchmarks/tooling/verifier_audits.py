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


_FORMULA_FIELD_NAMES = frozenset(
    {
        "count_formula",
        "determinant_identity",
        "event_mass_formula",
        "identity_scope",
        "limit_function",
        "maximum_formula",
        "remainder_coefficient",
        "singleton_cap",
        "schur_weight",
        "tangent_weight",
    }
)
_FORMULA_NAME_MARKERS = ("formula",)
_ENUM_CONST = frozenset(
    {
        "AT_LEAST_ONE",
        "ZERO",
        "SUM_OF_SQUARED_CLASS_SIZES",
        "NOT_REQUIRED_FOR_POLYNOMIAL_IDENTITY",
    }
)
_FORMULA_CONST_MARKERS = ("^", "floor(", "det(", "->")
_FORMULA_SKIP_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "claim_id",
        "frozen_answer",
        "frozen_inference",
        "constraint",
        "invertibility_assumption",
        "pair_count_formula",
        "frozen_formula_holds",
    }
)


def _is_enum_label(value: object) -> bool:
    return (
        isinstance(value, str) and value.replace("_", "").isalnum() and value.isupper()
    )


def _formula_like_const(value: object) -> bool:
    if not isinstance(value, str) or _is_enum_label(value) or value in _ENUM_CONST:
        return False
    return any(marker in value for marker in _FORMULA_CONST_MARKERS)


def _formula_field_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _is_formula_field(name: str) -> bool:
    if name in _FORMULA_SKIP_FIELDS:
        return False
    return name in _FORMULA_FIELD_NAMES or any(
        marker in name for marker in _FORMULA_NAME_MARKERS
    )


def _enum_values_are_labels(enum: object) -> bool:
    return (
        isinstance(enum, list)
        and bool(enum)
        and all(_is_enum_label(value) for value in enum)
    )


def _privileged_formula_string(node: dict[str, Any]) -> bool:
    const = node.get("const")
    if _is_enum_label(const) or const in _ENUM_CONST:
        return False
    if _enum_values_are_labels(node.get("enum")):
        return False
    return node.get("type") == "string" or _formula_like_const(const)


def formula_string_schema_failures(schema_path: Path) -> list[str]:
    """Return failures when a scored formula field is a privileged string."""

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
        if not isinstance(node, dict) or not _is_formula_field(
            _formula_field_name(path)
        ):
            continue
        if _privileged_formula_string(node):
            failures.append(
                f"{path}: encode mathematical formulas as structured objects, not strings"
            )
    return failures


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parse(path: Path) -> ast.AST | list[str]:
    relative = _relative(path)
    if not path.is_file():
        return []
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return [f"{relative}: cannot parse verifier: {exc}"]


def mirror_witness_failures(verifier_path: Path) -> list[str]:
    """Return failures when evidence is only an equality copy of ``result``."""

    parsed = _parse(verifier_path)
    if isinstance(parsed, list):
        return parsed
    relative = _relative(verifier_path)
    failures: list[str] = []
    for node in ast.walk(parsed):
        keys: set[str] = set()
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    keys.add(elt.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        if keys >= {"schema_version", "task_id", "result"} and keys <= {
            "schema_version",
            "task_id",
            "result",
            "limitations",
            "claimed_assurance",
            "scope",
        }:
            failures.append(
                f"{relative}:{node.lineno}: evidence must not mirror submission.result"
            )
    return failures


_SEMANTIC_READERS = frozenset(
    {"read_evidence_json", "resolve_evidence", "read_evidence_text"}
)


def unread_hash_witness_failures(verifier_path: Path) -> list[str]:
    """Return failures when a bound witness file is never semantically read."""

    parsed = _parse(verifier_path)
    if isinstance(parsed, list):
        return parsed
    relative = _relative(verifier_path)
    names = {node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(parsed) if isinstance(node, ast.Attribute)}
    called = names | attrs
    if "witness_list_is_bound" not in called:
        return []
    if called & _SEMANTIC_READERS:
        return []
    failures = [
        f"{relative}: witness_list_is_bound must be followed by a semantic evidence read"
    ]
    return failures


def _function_uses(node: ast.FunctionDef) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _math_functions(tree: ast.AST) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_math"
    ]


def _scores_hidden_expected_without_input(tree: ast.AST, verifier_path: Path) -> bool:
    source = (
        verifier_path.read_text(encoding="utf-8") if verifier_path.is_file() else ""
    )
    if "expected.json" not in source:
        return False
    if "e['expected_" not in source.replace('"', "'"):
        return False
    return not any("x" in _function_uses(node) for node in _math_functions(tree))


def hidden_expected_scoring_failures(verifier_path: Path) -> list[str]:
    """Return failures when correctness is hidden-expected equality without input."""

    parsed = _parse(verifier_path)
    if isinstance(parsed, list):
        return parsed
    relative = _relative(verifier_path)
    failures: list[str] = []
    for node in _math_functions(parsed):
        arg_names = [arg.arg for arg in node.args.args]
        if "e" not in arg_names or "x" not in arg_names:
            continue
        used = _function_uses(node)
        if "e" in used and "x" not in used:
            failures.append(
                f"{relative}:{node.lineno}: derive the scored predicate from frozen input, not expected.json"
            )
    if _scores_hidden_expected_without_input(parsed, verifier_path):
        failures.append(
            f"{relative}: do not score equality with hidden expected.json while ignoring frozen input"
        )
    return failures


_PROSE_KEYWORDS = ("RESULT_JSON:", "result_sha256", "BOUNDARY_FAMILY_JSON:")


def prose_witness_failures(verifier_path: Path) -> list[str]:
    """Return failures when evidence is a digest, result copy, or keyword gate."""

    if not verifier_path.is_file():
        return []
    relative = _relative(verifier_path)
    source = verifier_path.read_text(encoding="utf-8")
    failures: list[str] = []
    for marker in _PROSE_KEYWORDS:
        if marker in source:
            failures.append(
                f"{relative}: do not score result hashes, RESULT_JSON copies, or keyword prose as witnesses"
            )
            break
    parsed = _parse(verifier_path)
    if isinstance(parsed, list):
        return [*failures, *parsed]
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "strip":
            parent = func.value
            if (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Attribute)
                and parent.func.attr == "read_text"
            ):
                failures.append(
                    f"{relative}:{node.lineno}: nonempty text is not a mathematical witness"
                )
    return failures


__all__ = [
    "STRUCTURED_RATIONAL_SCHEMA",
    "canonical_string_rational_schema_failures",
    "formula_string_schema_failures",
    "fraction_coprimality_failures",
    "hidden_expected_scoring_failures",
    "is_canonical_rational_string_schema",
    "mirror_witness_failures",
    "prose_witness_failures",
    "unread_hash_witness_failures",
]
