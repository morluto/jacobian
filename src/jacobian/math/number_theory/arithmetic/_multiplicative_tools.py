"""Integer multiplicative normal-form operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.arithmetic._multiplicative_forms import (
    IntegerKRequest,
    IntegerRequest,
    KFreeDecompositionResult,
    NonnegativeIntegerRequest,
    NormalizedQuadraticRadicalResult,
    PerfectPowerProfileResult,
    SquarefreeDecompositionResult,
)
from jacobian.math.number_theory.arithmetic.operations import (
    k_free_decomposition,
    normalized_quadratic_radical,
    perfect_power_profile,
    squarefree_decomposition,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def _compute_perfect_power_profile(
    request: IntegerRequest,
) -> PerfectPowerProfileResult:
    """Project the wire request onto the native integer-value operation."""

    return perfect_power_profile(IntegerValue(value=request.value))


def _compute_k_free_decomposition(request: IntegerKRequest) -> KFreeDecompositionResult:
    """Project the wire request onto the native integer-value operation."""

    return k_free_decomposition(IntegerValue(value=request.value), request.k)


def _compute_squarefree_decomposition(
    request: IntegerRequest,
) -> SquarefreeDecompositionResult:
    """Project the wire request onto the native integer-value operation."""

    return squarefree_decomposition(IntegerValue(value=request.value))


def _compute_normalized_quadratic_radical(
    request: NonnegativeIntegerRequest,
) -> NormalizedQuadraticRadicalResult:
    """Project the wire request onto the native integer-value operation."""

    return normalized_quadratic_radical(IntegerValue(value=request.value))


def _mf_op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


MULTIPLICATIVE_FORM_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _mf_op(
        "integer.perfect_power.profile.compute",
        "Compute maximal perfect-power profile",
        "Compute the maximal integer exponent e and base b such that n = b^e. "
        "For negative n, e is the largest odd divisor of the gcd of prime "
        "exponents of |n|. Zero and units use closed structural variants.",
        IntegerRequest,
        PerfectPowerProfileResult,
        _compute_perfect_power_profile,
        "integer",
        "multiplicative",
        "exact",
        examples=(
            example(
                "perfect_power_64",
                "Compute the maximal perfect-power profile of 64; "
                "n must be one canonical integer.",
                {"value": "64"},
            ),
        ),
    ),
    _mf_op(
        "integer.k_free_decomposition.compute",
        "Compute k-free decomposition",
        "Compute the unique decomposition n = a^k * c where a >= 1 and "
        "|c| is k-th-power-free. Zero returns a ZERO variant; otherwise "
        "the result carries the extracted base, signed cofactor, per-prime "
        "exponent rows, and exact reconstruction.",
        IntegerKRequest,
        KFreeDecompositionResult,
        _compute_k_free_decomposition,
        "integer",
        "multiplicative",
        "exact",
        examples=(
            example(
                "k_free_72_k3",
                "Compute the 3-free decomposition of 72; "
                "n must be one canonical integer and k >= 2.",
                {"value": "72", "k": 3},
            ),
        ),
    ),
    _mf_op(
        "integer.squarefree_decomposition.compute",
        "Compute squarefree decomposition",
        "Compute the unique decomposition n = s^2 * d where s >= 1 and "
        "|d| is squarefree. Zero returns a ZERO variant; otherwise the "
        "result carries the square factor, signed squarefree part, "
        "per-prime exponent rows, and exact reconstruction.",
        IntegerRequest,
        SquarefreeDecompositionResult,
        _compute_squarefree_decomposition,
        "integer",
        "multiplicative",
        "exact",
        examples=(
            example(
                "squarefree_72",
                "Compute the squarefree decomposition of 72; "
                "n must be one canonical integer.",
                {"value": "72"},
            ),
        ),
    ),
    _mf_op(
        "integer.squarefree_part.compute",
        "Compute signed squarefree part",
        "Compute the signed squarefree part d and extracted square factor s "
        "such that n = s^2 * d with |d| squarefree. This is the compact "
        "projection of the squarefree decomposition carrying only the "
        "squarefree part and square factor.",
        IntegerRequest,
        SquarefreeDecompositionResult,
        _compute_squarefree_decomposition,
        "integer",
        "multiplicative",
        "exact",
        examples=(
            example(
                "squarefree_part_72",
                "Compute the signed squarefree part of 72; "
                "n must be one canonical integer.",
                {"value": "72"},
            ),
        ),
    ),
    _mf_op(
        "quadratic_radical.positive_integer.normalize.compute",
        "Normalize positive integer square root",
        "Compute the canonical positive square root sqrt(n) = s * sqrt(d) "
        "with s >= 0, d >= 1 squarefree, and s^2 * d = n. "
        "Classifies as ZERO, RATIONAL_INTEGER, or IRRATIONAL_QUADRATIC. "
        "The radicand n must be a nonnegative integer.",
        NonnegativeIntegerRequest,
        NormalizedQuadraticRadicalResult,
        _compute_normalized_quadratic_radical,
        "integer",
        "multiplicative",
        "exact",
        examples=(
            example(
                "radical_72",
                "Normalize sqrt(72) = 6*sqrt(2); n must be a nonnegative integer.",
                {"value": "72"},
            ),
        ),
    ),
)
