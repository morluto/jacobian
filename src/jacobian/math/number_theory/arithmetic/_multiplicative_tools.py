"""Integer multiplicative normal-form operation declarations."""

from typing import Any

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


MULTIPLICATIVE_FORM_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="integer.perfect_power.profile.compute",
        title="Compute maximal perfect-power profile",
        description="Compute the maximal integer exponent e and base b such that n = b^e. For negative n, e is the largest odd divisor of the gcd of prime exponents of |n|. Zero and units use closed structural variants.",
        request_type=IntegerRequest,
        result_type=PerfectPowerProfileResult,
        run=_compute_perfect_power_profile,
        tags=("integer", "multiplicative", "exact"),
        examples=(
            OperationExample(
                name="perfect_power_64",
                description="Compute the maximal perfect-power profile of 64; n must be one canonical integer.",
                input={"value": "64"},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.k_free_decomposition.compute",
        title="Compute k-free decomposition",
        description="Compute the unique decomposition n = a^k * c where a >= 1 and |c| is k-th-power-free. Zero returns a ZERO variant; otherwise the result carries the extracted base, signed cofactor, per-prime exponent rows, and exact reconstruction.",
        request_type=IntegerKRequest,
        result_type=KFreeDecompositionResult,
        run=_compute_k_free_decomposition,
        tags=("integer", "multiplicative", "exact"),
        examples=(
            OperationExample(
                name="k_free_72_k3",
                description="Compute the 3-free decomposition of 72; n must be one canonical integer and k >= 2.",
                input={"value": "72", "k": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="integer.squarefree_decomposition.compute",
        title="Compute squarefree decomposition",
        description=(
            "Compute the unique decomposition n = s^2 * d where s >= 1 and |d| is "
            "squarefree, returning the square factor and the signed squarefree "
            "part together with per-prime exponent rows and exact "
            "reconstruction."
        ),
        request_type=IntegerRequest,
        result_type=SquarefreeDecompositionResult,
        run=_compute_squarefree_decomposition,
        tags=("integer", "multiplicative", "exact"),
        discovery_terms=(
            "signed squarefree part",
            "extract square factor",
        ),
        examples=(
            OperationExample(
                name="squarefree_72",
                description="Compute the squarefree decomposition of 72; n must be one canonical integer.",
                input={"value": "72"},
            ),
        ),
    ),
)
