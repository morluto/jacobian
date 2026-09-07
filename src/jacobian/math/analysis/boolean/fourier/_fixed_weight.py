"""Exact integration of sparse Boolean characters on a Hamming sphere."""

from fractions import Fraction
from math import comb, lcm
from typing import Literal

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.analysis.boolean.fourier.values import RationalWalshPolynomial


class FixedWeightMomentRequest(StrictModel):
    polynomial: RationalWalshPolynomial
    weight: int = Field(
        ge=0,
        description="Number of one bits, between zero and the retained ambient variable_count.",
    )
    order: Literal[1, 2] = 1


def fixed_weight_moment(
    polynomial: RationalWalshPolynomial, weight: int, order: Literal[1, 2] = 1
) -> CanonicalRational:
    """Return E[f(X)^order] for uniform X of the specified Hamming weight.

    If S_d is the signed number of sphere points for a degree-d character,
    (N-d) S_(d+1) = (N-2r) S_d - d S_(d-1), with S_0=C(N,r).
    Thus |S_d| <= C(N,r) <= 2^N. Products of characters use symmetric
    difference; no sphere points or dense truth table are materialized.
    """
    request_checkpoint("before fixed-weight admission")
    n = polynomial.variable_count
    if (
        type(weight) is not int
        or not 0 <= weight <= n
        or type(order) is not int
        or order not in (1, 2)
    ):
        raise OperationDomainValidationError(
            location=("weight", "order"),
            code="boolean.fixed_weight_domain",
            message="weight must lie in 0..variable_count and order must be 1 or 2",
        )
    terms = polynomial.terms
    pairs = len(terms) if order == 1 else len(terms) ** 2
    denominator = 1
    for term in terms:
        denominator = lcm(denominator, term.coefficient.den)
        if denominator.bit_length() > 24_000:
            raise OperationDomainValidationError(
                location=("polynomial",),
                code="boolean.fixed_weight_growth",
                message="common denominator exceeds the 8000-digit working envelope",
            )
    lifted_bits = max(
        (
            abs(t.coefficient.num).bit_length()
            + (denominator // t.coefficient.den).bit_length()
            for t in terms
        ),
        default=1,
    )
    norm_bits = lifted_bits + max(1, len(terms)).bit_length()
    result_bits = order * max(norm_bits, denominator.bit_length()) + n + 2
    work = pairs * max(1, (n + 63) // 64) * max(1, (result_bits + 63) // 64)
    if result_bits > 96_000 or work > 20_000_000:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="boolean.fixed_weight_work",
            message="fixed-weight moment exceeds 20,000,000 weighted character products or the rational height envelope",
        )
    if not terms:
        return CanonicalRational(num=0, den=1)
    sphere_size = comb(n, weight)
    signed = [sphere_size]
    if n:
        signed.append((n - 2 * weight) * sphere_size // n)
    for degree in range(1, n):
        signed.append(
            ((n - 2 * weight) * signed[-1] - degree * signed[-2]) // (n - degree)
        )
    masks = [sum(1 << i for i in term.character) for term in terms]
    coefficients = [
        term.coefficient.num * (denominator // term.coefficient.den) for term in terms
    ]
    if order == 1:
        numerator = sum(
            c * signed[mask.bit_count()]
            for c, mask in zip(coefficients, masks, strict=True)
        )
    else:
        numerator = sum(
            a * b * signed[(x ^ y).bit_count()]
            for a, x in zip(coefficients, masks, strict=True)
            for b, y in zip(coefficients, masks, strict=True)
        )
    request_checkpoint("before fixed-weight result construction")
    return CanonicalRational.from_fraction(
        Fraction(numerator, denominator**order * sphere_size)
    )


def _run(request: FixedWeightMomentRequest) -> CanonicalRational:
    return fixed_weight_moment(request.polynomial, request.weight, request.order)


FIXED_WEIGHT_MOMENT = MathTool(
    operation_id="boolean.walsh_polynomial.fixed_weight_moment.compute",
    title="Compute a sparse Walsh polynomial moment at fixed Hamming weight",
    description="Return the exact first or second raw moment under the uniform distribution on Boolean vectors with exactly weight ones. Retain the polynomial's ambient dimension, including unused coordinates. Uses character contraction without enumerating the sphere; admits at most 20,000,000 height-weighted character-product units and bounded rational growth. Variance is the second moment minus the squared first moment.",
    request_type=FixedWeightMomentRequest,
    result_type=CanonicalRational,
    run=_run,
    tags=(
        "boolean",
        "walsh",
        "expectation",
        "hamming-sphere",
        "fixed-cardinality",
        "moment",
    ),
    examples=(
        OperationExample(
            name="fixed_weight_character",
            description="Average a degree-two character over four-bit vectors of weight two; weight lies within the retained ambient dimension.",
            input={
                "polynomial": {
                    "variable_count": 4,
                    "terms": [
                        {"character": [0, 1], "coefficient": {"num": "1", "den": "1"}}
                    ],
                },
                "weight": 2,
                "order": 1,
            },
        ),
    ),
)
