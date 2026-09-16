"""Shift Ore-operator operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras._models import (
    ShiftOperatorMultiplyRequest,
    ShiftOperatorMultiplyResult,
)
from jacobian.math.ore_algebras.operations import shift_operator_multiply


def _run_shift_operator_multiply(
    request: ShiftOperatorMultiplyRequest,
) -> ShiftOperatorMultiplyResult:
    return shift_operator_multiply(request.left, request.right)


def _rf_num_den(
    numerator: list[tuple[int, list[int]]], denominator: list[tuple[int, list[int]]]
) -> dict[str, Any]:
    def _part(terms: list[tuple[int, list[int]]]) -> dict[str, Any]:
        return {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": exponents,
                }
                for coefficient, exponents in terms
            ]
        }

    return {
        "domain": "QQ",
        "variables": ["n"],
        "numerator": _part(numerator),
        "denominator": _part(denominator),
    }


def _one() -> dict[str, Any]:
    return _rf_num_den([(1, [0])], [(1, [0])])


def _n() -> dict[str, Any]:
    return _rf_num_den([(1, [1])], [(1, [0])])


def _operator(terms: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "variable": "n",
        "terms": [
            {"exponent": exponent, "coefficient": coefficient}
            for exponent, coefficient in terms
        ],
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="ore.shift.operator.multiply.compute",
        title="Multiply exact shift operators over QQ(n)",
        description=(
            "Multiply two shift operators in left-coefficient normal form "
            "over QQ(n)<S; S a(n) = a(n+1) S>, returning the exact canonical "
            "product with one ledger row per nonzero term pair carrying the "
            "shifted right coefficient. Operator orders, term counts, "
            "coefficient degrees and digits, term-pair rows, and result "
            "growth are preflighted before noncommutative expansion. The "
            "differential Ore type is out of scope."
        ),
        request_type=ShiftOperatorMultiplyRequest,
        result_type=ShiftOperatorMultiplyResult,
        run=_run_shift_operator_multiply,
        tags=("ore-algebra", "shift-operator", "holonomic", "exact"),
        discovery_terms=(
            "shift operator product",
            "Ore polynomial multiplication",
            "S n = (n+1) S",
            "recurrence operator composition",
        ),
        examples=(
            OperationExample(
                name="shift_times_coordinate",
                description=(
                    "Compute S * n = (n+1) * S in the shift Ore algebra; "
                    "both operators must use the QQ(n) coefficient axis with "
                    "reduced presentations."
                ),
                input={
                    "left": _operator([(1, _one())]),
                    "right": _operator([(0, _n())]),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
