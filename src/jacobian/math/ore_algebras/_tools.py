"""Shift Ore-operator operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.ore_algebras._models import (
    DifferentialOperatorAddRequest,
    DifferentialOperatorAddResult,
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
    DifferentialOperatorMultiplyRequest,
    DifferentialOperatorMultiplyResult,
    DifferentialOperatorNormalizeRequest,
    DifferentialOperatorNormalizeResult,
    PolynomialRecurrencePrefix,
    PolynomialRecurrencePrefixRequest,
    ShiftOperatorAddRequest,
    ShiftOperatorAddResult,
    ShiftOperatorMultiplyRequest,
    ShiftOperatorMultiplyResult,
    ShiftOperatorNormalizeRequest,
    ShiftOperatorNormalizeResult,
    ShiftOperatorPowerRequest,
    ShiftOperatorPowerResult,
    ShiftOperatorPrefixRequest,
    ShiftOperatorPrefixResult,
    ShiftOperatorScalarMultiplyRequest,
    ShiftOperatorScalarMultiplyResult,
)
from jacobian.math.ore_algebras.operations import (
    differential_operator_add,
    differential_operator_apply,
    differential_operator_multiply,
    differential_operator_normalize_polynomial_coefficients,
    polynomial_recurrence_generate_prefix,
    shift_operator_add,
    shift_operator_apply_to_sequence_prefix,
    shift_operator_multiply,
    shift_operator_normalize_polynomial_coefficients,
    shift_operator_power,
    shift_operator_scalar_left_multiply,
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
        operation_id="ore.shift.operator.add.compute",
        title="Add polynomial-coefficient shift operators",
        description=(
            "Add two exact shift operators in left-coefficient normal form "
            "over the closed polynomial-coefficient subalgebra QQ[n]<S>, "
            "where S a(n)=(a(n+1))S. Coefficients must be polynomials in "
            "QQ[n]; rational-function denominators are not accepted by this "
            "bounded operation. The sparse result removes exact zero terms."
        ),
        request_type=ShiftOperatorAddRequest,
        result_type=ShiftOperatorAddResult,
        run=lambda request: shift_operator_add(request.left, request.right),
        tags=("ore-algebra", "shift-operator", "addition", "exact"),
        discovery_terms=(
            "add shift operators",
            "recurrence operator sum",
            "polynomial coefficient Ore addition",
        ),
        examples=(
            OperationExample(
                name="add_coordinate_and_shift",
                description=(
                    "Compute n+S in QQ[n]<S>; both operators must have "
                    "polynomial coefficients."
                ),
                input={
                    "left": _operator([(0, _n())]),
                    "right": _operator([(1, _one())]),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.operator.scalar_left_multiply.compute",
        title="Left-scale a polynomial-coefficient shift operator",
        description=(
            "Multiply a shift operator on the left by an exact rational "
            "constant in QQ. This bounded operation accepts polynomial "
            "operator coefficients in QQ[n] and returns the exact sparse "
            "operator; it does not accept a nonconstant QQ(n) scalar."
        ),
        request_type=ShiftOperatorScalarMultiplyRequest,
        result_type=ShiftOperatorScalarMultiplyResult,
        run=lambda request: shift_operator_scalar_left_multiply(
            request.scalar, request.operator
        ),
        tags=("ore-algebra", "shift-operator", "scalar-multiplication", "exact"),
        discovery_terms=(
            "left scalar multiply shift operator",
            "rational scalar recurrence operator",
            "QQ-linear operator scaling",
        ),
        examples=(
            OperationExample(
                name="double_coordinate_plus_shift",
                description=(
                    "Compute 2(n+S) in QQ[n]<S>; the scalar must be rational "
                    "constant and operator coefficients polynomial."
                ),
                input={
                    "scalar": _rf_num_den([(2, [0])], [(1, [0])]),
                    "operator": _operator([(0, _n()), (1, _one())]),
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.shift.operator.normalize.compute",
        title="Extract rational content from a polynomial-coefficient shift operator",
        description=(
            "Return a polynomial-coefficient shift operator with integer "
            "polynomial coefficients of joint gcd one and positive leading "
            "coefficient (greatest shift exponent, then greatest polynomial "
            "degree), together with the exact rational scale c satisfying "
            "P=c*P_primitive. This normalization is defined on QQ[n]<S>; "
            "rational-function coefficients are not accepted."
        ),
        request_type=ShiftOperatorNormalizeRequest,
        result_type=ShiftOperatorNormalizeResult,
        run=lambda request: shift_operator_normalize_polynomial_coefficients(
            request.operator
        ),
        tags=("ore-algebra", "shift-operator", "normalization", "exact"),
        discovery_terms=(
            "primitive shift operator",
            "normalize rational content Ore operator",
            "operator scalar content",
        ),
        examples=(
            OperationExample(
                name="primitive_operator_with_scale",
                description=(
                    "Write 3*(n*S+2) with primitive integer polynomial "
                    "coefficients; the operator must have coefficients in QQ[n]."
                ),
                input={
                    "operator": _operator(
                        [
                            (0, _rf_num_den([(6, [0])], [(1, [0])])),
                            (1, _rf_num_den([(3, [1])], [(1, [0])])),
                        ]
                    )
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.shift.operator.apply_to_sequence_prefix.compute",
        title="Apply a shift operator on a finite rational sequence prefix",
        description=(
            "Evaluate each exact coefficient contribution and residual of a "
            "QQ(n) shift operator on a finite rational sequence whose first "
            "value is assigned the explicit start_index. Coefficient poles "
            "and right-boundary indices lacking shifted values are returned "
            "as exclusions. Zero residuals on a finite prefix do not establish "
            "global annihilation."
        ),
        request_type=ShiftOperatorPrefixRequest,
        result_type=ShiftOperatorPrefixResult,
        run=lambda request: shift_operator_apply_to_sequence_prefix(
            request.operator, request.start_index, request.sequence
        ),
        tags=("ore-algebra", "shift-operator", "sequence", "exact"),
        discovery_terms=(
            "apply recurrence operator to sequence prefix",
            "shift operator residual",
            "finite recurrence residual",
        ),
        examples=(
            OperationExample(
                name="fibonacci_recurrence_residuals",
                description=(
                    "Evaluate (S^2-S-1)F on a finite Fibonacci prefix. The "
                    "result is finite residual evidence, not a proof of a "
                    "global recurrence."
                ),
                input={
                    "operator": _operator(
                        [
                            (0, _rf_num_den([(-1, [0])], [(1, [0])])),
                            (1, _rf_num_den([(-1, [0])], [(1, [0])])),
                            (2, _one()),
                        ]
                    ),
                    "start_index": 0,
                    "sequence": {"values": ["0", "1", "1", "2", "3", "5"]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.shift.recurrence.generate_finite_prefix.compute",
        title="Generate a finite prefix from a polynomial recurrence",
        description=(
            "Generate an exact rational sequence prefix from polynomial "
            "coefficients in QQ[n], consecutive initial values, and a finite "
            "index interval. The leading coefficient must be nonzero at each "
            "generated index. The result represents only those finite "
            "recurrence equations and does not assert an infinite sequence."
        ),
        request_type=PolynomialRecurrencePrefixRequest,
        result_type=PolynomialRecurrencePrefix,
        run=lambda request: polynomial_recurrence_generate_prefix(
            request.operator,
            request.start_index,
            request.initial_values,
            request.steps,
        ),
        tags=("ore-algebra", "recurrence", "finite-prefix", "exact"),
        discovery_terms=(
            "generate sequence prefix from recurrence",
            "polynomial recurrence finite initial value problem",
            "P-recursive finite prefix",
        ),
        examples=(
            OperationExample(
                name="fibonacci_finite_recurrence_prefix",
                description=(
                    "Generate five further Fibonacci values from S^2-S-1; "
                    "the recurrence is asserted only at the five generated indices."
                ),
                input={
                    "operator": _operator(
                        [
                            (0, _rf_num_den([(-1, [0])], [(1, [0])])),
                            (1, _rf_num_den([(-1, [0])], [(1, [0])])),
                            (2, _one()),
                        ]
                    ),
                    "start_index": 0,
                    "initial_values": {"values": ["0", "1"]},
                    "steps": 5,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="ore.shift.operator.power.compute",
        title="Compute a bounded nonnegative shift operator power",
        description=(
            "Compute P^m in QQ(n)<S; S a(n)=a(n+1)S> for a nonnegative "
            "integer m at most 16. Exponent zero returns the identity; the "
            "result retains left-coefficient normal form. Admission bounds "
            "the total product-cell work and rejects powers whose order "
            "exceeds the canonical shift-operator carrier. Powers above one "
            "currently require coefficients in ZZ[n], enabling whole-power "
            "degree, support, and integer-height bounds before expansion."
        ),
        request_type=ShiftOperatorPowerRequest,
        result_type=ShiftOperatorPowerResult,
        run=lambda request: shift_operator_power(request.operator, request.exponent),
        tags=("ore-algebra", "shift-operator", "power", "exact"),
        discovery_terms=(
            "shift Ore operator power",
            "recurrence operator exponentiation",
            "noncommutative shift operator power",
        ),
        examples=(
            OperationExample(
                name="cube_one_plus_shift",
                description="Compute (1+S)^3 over constant coefficients in QQ(n)<S>.",
                input={
                    "operator": _operator([(0, _one()), (1, _one())]),
                    "exponent": 3,
                },
            ),
        ),
    ),
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
        operation_id="ore.differential.operator.add.compute",
        title="Add exact differential Ore operators over QQ(x)",
        description=(
            "Add left-coefficient differential operators over QQ(x) by exact "
            "coefficientwise addition at each derivative order. Equal orders "
            "are combined, exact zero coefficients are removed, and the empty "
            "operator is the canonical zero. Inputs, rational-function growth, "
            "sparse work, and serialized output are admitted before expansion; "
            "the operation does not normalize by a scalar."
        ),
        request_type=DifferentialOperatorAddRequest,
        result_type=DifferentialOperatorAddResult,
        run=lambda request: differential_operator_add(request.left, request.right),
        tags=("ore-algebra", "differential-operator", "addition", "exact"),
        discovery_terms=(
            "add differential operators",
            "differential Ore operator sum",
            "coefficientwise operator addition",
        ),
        examples=(
            OperationExample(
                name="add_derivative_and_coordinate",
                description="Compute D + x over QQ(x), retaining left-coefficient form.",
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
    MathTool(
        operation_id="ore.differential.operator.normalize.compute",
        title="Normalize a polynomial-coefficient differential Ore operator",
        description=(
            "Extract the exact rational content of a differential operator in "
            "QQ[x]<D> whose coefficients lie in QQ[x]. The result has integer "
            "polynomial coefficients with joint gcd one and positive leading "
            "coefficient, plus the exact rational scale reconstructing the "
            "source. Rational-function coefficients are rejected."
        ),
        request_type=DifferentialOperatorNormalizeRequest,
        result_type=DifferentialOperatorNormalizeResult,
        run=lambda request: differential_operator_normalize_polynomial_coefficients(
            request.operator
        ),
        tags=("ore-algebra", "differential-operator", "normalization", "exact"),
        discovery_terms=(
            "primitive differential Ore operator",
            "normalize rational content differential operator",
            "differential operator scalar content",
        ),
        examples=(
            OperationExample(
                name="primitive_differential_operator_with_scale",
                description=(
                    "Write (2/3)x + (1/3)D as a primitive integer-polynomial "
                    "operator and an exact rational scale."
                ),
                input={
                    "operator": {
                        "variable": "x",
                        "terms": [
                            {
                                "order": 0,
                                "coefficient": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "numerator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "2", "den": "3"},
                                                "exponents": [1],
                                            }
                                        ]
                                    },
                                    "denominator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                },
                            },
                            {
                                "order": 1,
                                "coefficient": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "numerator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "3"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                    "denominator": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                },
                            },
                        ],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
