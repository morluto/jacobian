"""Guard strict-JSON container handling in owner-local preflight validators."""

from __future__ import annotations

import ast
from pathlib import Path


def _is_before_validator(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "model_validator"
        and any(
            keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "before"
            for keyword in decorator.keywords
        )
    )


def _uses_canonical_container_projection(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "canonicalize_json_containers"
        for node in ast.walk(function)
    )


def test_math_before_validators_project_json_arrays_to_canonical_tuples() -> None:
    """Preflight validators cannot hand raw JSON arrays back to Pydantic.

    ``model_validator(mode="before")`` switches downstream tuple validation
    to Python semantics.  Each owner therefore projects strict-JSON arrays
    to its canonical tuple values before returning to Pydantic.
    """

    source_root = Path(__file__).parents[2] / "src" / "jacobian" / "math"
    missing = [
        f"{path.relative_to(source_root)}:{function.lineno}"
        for path in sorted(source_root.rglob("*.py"))
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(function, ast.FunctionDef)
        and any(_is_before_validator(item) for item in function.decorator_list)
        and not _uses_canonical_container_projection(function)
    ]

    assert not missing, (
        "mode='before' validators must call canonicalize_json_containers: "
        + ", ".join(missing)
    )
