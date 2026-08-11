"""Typed input and normalization contract tests for the SymPy provider."""

from __future__ import annotations

from typing import Any

from tests.support.capabilities import invoke_capability as _invoke
from tests.support.rationals import rational_payload as _q

from jacobian.contracts.capabilities import CapabilityAssuranceLevel
from jacobian.contracts.results import ExecutionStatus


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
    polynomial_normalization_services,
) -> None:
    result = _invoke(
        polynomial_normalization_services,
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
    polynomial_normalization_services,
) -> None:
    result = _invoke(
        polynomial_normalization_services,
        "polynomial.expression.normalize",
        {
            "expression": _difference_of_squares_plus_half_x(),
            "resource_budget": {"wall_seconds": 5},
        },
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
    resolved = polynomial_normalization_services.core.polynomial_expressions.resolve_normalization(
        result.output["normalization_uri"]
    )
    assert (
        resolved.candidate.source.expression_artifact_uri
        == result.output["expression_uri"]
    )
    assert result.output["expression_uri"] in resolved.artifact.manifest.parents


def test_sympy_normalization_preserves_exact_zero(
    polynomial_normalization_services,
) -> None:
    result = _invoke(
        polynomial_normalization_services,
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
    )
    assert result.output["normalized"] == {"terms": []}
    assert result.output["status"] == "NORMALIZATION_PRODUCED"


_INVALID_EXPRESSIONS = (
    (
        "undeclared_variable",
        {"variables": ["x"], "expression": {"kind": "variable", "name": "undeclared"}},
    ),
    (
        "formula_string",
        {
            "variables": ["x"],
            "expression": {"kind": "formula", "value": "__import__('os').system('id')"},
        },
    ),
    (
        "expansion_blowup",
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
    ),
)


def test_normalization_rejects_inputs_outside_typed_ast_contract(
    polynomial_normalization_services,
) -> None:
    for case, expression in _INVALID_EXPRESSIONS:
        result = _invoke(
            polynomial_normalization_services,
            "polynomial.expression.normalize",
            {"expression": expression},
        )
        assert result.execution.status is ExecutionStatus.ERROR, case
        assert result.output["error"]["stage"] in {
            "capability_input_validation",
            "input_validation",
            "bounded_normalization",
        }, case
