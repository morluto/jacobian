"""Shift Ore-operator operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras._models import (
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
    DifferentialOperatorMultiplyRequest,
    DifferentialOperatorMultiplyResult,
    ShiftOperatorMultiplyRequest,
    ShiftOperatorMultiplyResult,
)
from jacobian.math.ore_algebras.operations import (
    differential_operator_apply,
    differential_operator_multiply,
    shift_operator_multiply,
)


def _run_shift_operator_multiply(
    request: ShiftOperatorMultiplyRequest,
) -> ShiftOperatorMultiplyResult:
    return shift_operator_multiply(request.left, request.right)


def _rf_num_den(
    numerator: list[tuple[int, list[int]]],
    denominator: list[tuple[int, list[int]]],
    variable: str = "n",
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
        "variables": [variable],
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


_DIFF_ONE = _rf_num_den([(1, [0])], [(1, [0])], "x")
_DIFF_X = _rf_num_den([(1, [1])], [(1, [0])], "x")

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
            "growth are preflighted before noncommutative expansion."
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
    MathTool(
        operation_id="ore.operator.apply.compute",
        title="Apply a differential Ore operator exactly",
        description="Apply a left-coefficient differential Ore operator to a rational function in QQ(x), using the exact differential action and preserving the variable axis.",
        request_type=DifferentialOperatorApplyRequest,
        result_type=DifferentialOperatorApplyResult,
        run=lambda request: differential_operator_apply(
            request.operator, request.function
        ),
        tags=("ore-algebra", "differential-operator", "action", "exact"),
        examples=(
            OperationExample(
                name="derivative_of_coordinate",
                description="Apply D to x and return 1; the operator and function must use the QQ(x) coefficient axis.",
                input={
                    "operator": {
                        "variable": "x",
                        "terms": [{"order": 1, "coefficient": _DIFF_ONE}],
                    },
                    "function": _DIFF_X,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.differential.operator.multiply.compute",
        title="Multiply exact differential Ore operators over QQ(x)",
        description="Multiply left-coefficient differential Ore operators using D a(x)=a(x) D+a'(x), preserving exact QQ(x) coefficients and operator order.",
        request_type=DifferentialOperatorMultiplyRequest,
        result_type=DifferentialOperatorMultiplyResult,
        run=lambda request: differential_operator_multiply(request.left, request.right),
        tags=("ore-algebra", "differential-operator", "exact"),
        examples=(
            OperationExample(
                name="derivative_coordinate_commutation",
                description="Compute D*x = x*D + 1 in the differential Ore algebra; both coefficients must use the QQ(x) axis.",
                input={
                    "left": {
                        "variable": "x",
                        "terms": [{"order": 1, "coefficient": _DIFF_ONE}],
                    },
                    "right": {
                        "variable": "x",
                        "terms": [{"order": 0, "coefficient": _DIFF_X}],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
