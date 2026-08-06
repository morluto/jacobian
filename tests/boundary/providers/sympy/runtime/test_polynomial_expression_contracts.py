"""Typed input and normalization contract tests for the SymPy provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.support.capabilities import invoke_capability as _invoke
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import CapabilityAssuranceLevel, CapabilityMode
from jacobian.contracts.results import ExecutionStatus
from jacobian.runtime import create_runtime


def _variable(name: str) -> dict[str, Any]:
    return {"kind": "variable", "name": name}


def _expression(
    node: dict[str, Any], *, variables: list[str] | None = None
) -> dict[str, Any]:
    return {"variables": variables or ["x", "y"], "expression": node}


def _difference_of_squares_plus_half_x() -> dict[str, Any]:
    return _expression(
        {
            "kind": "add",
            "operands": [
                {
                    "kind": "multiply",
                    "operands": [
                        {"kind": "add", "operands": [_variable("x"), _variable("y")]},
                        {
                            "kind": "add",
                            "operands": [
                                _variable("x"),
                                {"kind": "negate", "operand": _variable("y")},
                            ],
                        },
                    ],
                },
                {
                    "kind": "multiply",
                    "operands": [
                        {"kind": "rational", "value": _q(1, 2)},
                        _variable("x"),
                    ],
                },
            ],
        }
    )


def test_expansion_term_budget_failure_is_specific_and_non_retryable(
    attached_complete_runtime,
) -> None:
    result = _invoke(
        attached_complete_runtime,
        "polynomial.expression.normalize",
        {
            "expression": _expression(
                {
                    "kind": "power",
                    "base": {
                        "kind": "add",
                        "operands": [_variable("x") for _ in range(16)],
                    },
                    "exponent": 4,
                }
            )
        },
        mode=CapabilityMode.EXPLORE,
    )
    assert result.execution.status is ExecutionStatus.ERROR
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "EXPANSION_TERM_BUDGET_EXCEEDED"
    assert diagnostic.stage == "bounded_normalization"
    assert diagnostic.details == {
        "limit": 1024,
        "estimated_expanded_terms_upper_bound": 1025,
        "bound_kind": "CONSERVATIVE_UPPER_BOUND",
        "requested_exponent": 4,
        "retryable_with_same_input": False,
        "mathematical_scope": "ONE_CONCRETE_TYPED_EXPRESSION",
        "supports_universal_claim": False,
        "larger_same_family_full_expansions_expected_to_help": False,
        "alternatives": [
            "use a factored symbolic operation",
            "split the expression before normalization",
            "use a domain capability with bounded coefficient access",
        ],
        "normalization_uri": None,
        "checker_input_available": False,
    }
    assert result.artifact_uris == ()
    assert "Finite normalizations cannot prove an all-orders claim" in (
        diagnostic.hint or ""
    )


def test_sympy_normalizes_typed_multivariate_expression(
    attached_complete_runtime,
) -> None:
    result = _invoke(
        attached_complete_runtime,
        "polynomial.expression.normalize",
        {
            "expression": _difference_of_squares_plus_half_x(),
            "resource_budget": {"wall_seconds": 5},
        },
        mode=CapabilityMode.EXPLORE,
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "NORMALIZATION_PRODUCED"
    assert result.output["conclusion"] == "UNKNOWN"
    assert result.output["verification"] == "UNVERIFIED"
    assert result.output["normalized"] == {
        "terms": [
            {"coefficient": _q(1), "exponents": [2, 0]},
            {"coefficient": _q(1, 2), "exponents": [1, 0]},
            {"coefficient": _q(-1), "exponents": [0, 2]},
        ]
    }
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert (
        result.relationships[0].relation_id
        == "polynomial.relation.expression-normalization-of"
    )
    resolved = (
        attached_complete_runtime.core.polynomial_expressions.resolve_normalization(
            result.output["normalization_uri"]
        )
    )
    assert (
        resolved.candidate.source.expression_artifact_uri
        == result.output["expression_uri"]
    )
    assert result.output["expression_uri"] in resolved.artifact.manifest.parents


def test_sympy_normalization_preserves_exact_zero(attached_complete_runtime) -> None:
    result = _invoke(
        attached_complete_runtime,
        "polynomial.expression.normalize",
        {
            "expression": _expression(
                {
                    "kind": "add",
                    "operands": [
                        _variable("x"),
                        {"kind": "negate", "operand": _variable("x")},
                    ],
                },
                variables=["x"],
            )
        },
        mode=CapabilityMode.EXPLORE,
    )
    assert result.output["normalized"] == {"terms": []}
    assert result.output["status"] == "NORMALIZATION_PRODUCED"


@pytest.mark.parametrize(
    "expression",
    [
        {"variables": ["x"], "expression": {"kind": "variable", "name": "undeclared"}},
        {
            "variables": ["x"],
            "expression": {"kind": "formula", "value": "__import__('os').system('id')"},
        },
        {
            "variables": ["x"],
            "expression": {
                "kind": "power",
                "base": {
                    "kind": "add",
                    "operands": [_variable("x") for _ in range(16)],
                },
                "exponent": 4,
            },
        },
    ],
    ids=("undeclared_variable", "formula_string", "expansion_blowup"),
)
def test_normalization_rejects_inputs_outside_typed_ast_contract(
    tmp_path: Path,
    expression: dict[str, Any],
) -> None:
    result = _invoke(
        create_runtime(tmp_path),
        "polynomial.expression.normalize",
        {"expression": expression},
        mode=CapabilityMode.EXPLORE,
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["stage"] in {
        "capability_input_validation",
        "input_validation",
        "bounded_normalization",
    }
