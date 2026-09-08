"""Exact root counts from Schur and real Hermite forms.

Schur's signature is inside minus outside even for singular forms; its
nullity is deliberately unused.  Cayley transformation and a real gcd
identify the boundary separately.  See Heinig--Rost, *Bezoutians* (2008),
Theorems 10.4 and 10.9. All matrix signatures use real-rooted Descartes
variation on a division-free Berkowitz characteristic polynomial.
"""

from __future__ import annotations

import time
from itertools import pairwise
from math import gcd, lcm

from pydantic import Field, StrictInt

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import RationalPolynomial


class UnitDiskProfileRequest(StrictModel):
    polynomial: RationalPolynomial


class UnitDiskProfile(StrictModel):
    """Root multiplicities in the three regions relative to the unit circle."""

    polynomial: RationalPolynomial
    degree: StrictInt = Field(ge=0)
    inside: StrictInt = Field(ge=0)
    on: StrictInt = Field(ge=0)
    outside: StrictInt = Field(ge=0)


def _resource(message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("polynomial",),
        code="polynomial.unit_disk.resource_admission",
        message=message,
    )


def _admit(polynomial: RationalPolynomial) -> tuple[int, int, list[int]]:
    """Inspect once, then expand only the exponent-compressed integer source.

    Let h bound the cleared source coefficients and n its reduced degree.
    Mignotte bounds each primitive squarefree factor by 2**n ||p||_2.
    Cayley substitution adds n+log2(n+1) bits; its real gcd gains the same
    factor bound. Derivatives and Bezoutian sums add at most 2log2(n+1)
    to twice that height. Berkowitz intermediates are sums of products of
    at most n matrix entries, bounded by 2n(b+2log2(n+1)+2) bits.

    In FLINT's squarefree loop, v is a primitive integer divisor of p.
    Its w/s auxiliaries are weighted derivative sums, with weights <=n:
    sum_j m_j f_j' product_{k!=j} f_k. Their coefficients are bounded by
    n**2 * 2**n * M(v), and M(v)<=M(p)<=sqrt(n+1)*2**h. Thus the
    factor height plus 2log2(n+1)+2 bounds every such auxiliary. Also the
    sum of active v degrees across the loop is <=n (each factor appears
    once per multiplicity), bounding the total modular gcd work by cubic
    degree times operand height. The gcd dispatcher uses
    bounded heuristic attempts then subresultant/modular gcd. The modular
    termination bound is (n+3)*max(norm_squared_bits)+(n+1), including
    unlucky primes; subresultant pseudo-division temporaries are bounded
    by 4(n+1)**2 times operand height. This larger ceiling also covers
    exact-division temporaries. No root separation or tolerance is used.

    Work is a coefficient-operation/word-height proxy, not a bit-operation
    claim. Sum of squarefree factor degrees is <=n, so their matrix work
    is bounded by two order-n Berkowitz calls. Integer payload and dense
    matrix allocations are admitted separately from transport policy.
    """
    terms = polynomial.polynomial.terms
    if len(polynomial.variables) != 1 or not terms:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.unit_disk.nonzero_univariate_required",
            message="a nonzero univariate rational polynomial is required",
        )
    if len(terms) > 4096:
        raise _resource("source support exceeds 4096 terms")
    valuation = terms[-1].exponents[0]
    if len(terms) == 1:
        # Canonical terms are nonzero. The coefficient is immaterial to
        # the root multiset, including arbitrarily large rational scalars.
        return valuation, 1, [1]
    values = [term.coefficient.as_fraction() for term in terms]
    denominator_bits = sum(value.denominator.bit_length() for value in values)
    numerator_bits = max(abs(value.numerator).bit_length() for value in values)
    if denominator_bits + numerator_bits > 65_536:
        raise _resource("cleared coefficient height exceeds 65,536 bits")
    stride = 0
    for term in terms:
        stride = gcd(stride, term.exponents[0] - valuation)
    stride = stride or 1
    degree = (terms[0].exponents[0] - valuation) // stride
    if degree:
        log = (degree + 1).bit_length()
        height = denominator_bits + numerator_bits
        factor_height = height + degree + log
        cayley_height = factor_height + degree + log
        boundary_height = cayley_height + degree + log
        matrix_height = 2 * boundary_height + 2 * log + 2
        characteristic_height = 2 * degree * (matrix_height + 2 * log + 2)
        auxiliary_height = factor_height + 2 * log + 2
        gcd_height = max(auxiliary_height, cayley_height)
        intermediate_height = 4 * (degree + 1) ** 2 * (gcd_height + log + 64)
        work = 16 * (degree + 1) ** 4 + 16 * (degree + 1) ** 3 * (
            (gcd_height + 63) // 64
        )
        if work > 100_000_000 or intermediate_height > 16_777_216:
            raise _resource(
                "derived exact-arithmetic work or intermediate height exceeds the envelope"
            )
        # Recursive source submatrices coexist, but each Toeplitz product
        # workspace is allocated only after its recursive child has returned.
        allocation = (degree + 1) ** 3 * matrix_height + 4 * (
            degree + 1
        ) ** 2 * characteristic_height
        if allocation > 268_435_456:
            raise _resource(
                "derived characteristic-polynomial allocation exceeds the envelope"
            )
    denominator = 1
    for value in values:
        denominator = lcm(denominator, value.denominator)
    coefficients = [0] * (degree + 1)
    for term, value in zip(terms, values, strict=True):
        coefficients[(term.exponents[0] - valuation) // stride] = value.numerator * (
            denominator // value.denominator
        )
    content = gcd(*coefficients)
    return valuation, stride, [value // content for value in coefficients]


def _signature(matrix: list[list[int]]) -> int:
    from sympy.polys.domains import ZZ
    from sympy.polys.matrices.ddm import DDM

    order = len(matrix)
    if not order:
        return 0
    # DDM explicitly selects the maintained division-free dense Berkowitz
    # algorithm, avoiding a backend-dependent modular charpoly dispatcher.
    characteristic = DDM(
        [[ZZ(value) for value in row] for row in matrix], (order, order), ZZ
    ).charpoly()
    positive = [1 if value > 0 else -1 for value in characteristic if value]
    negative = [
        (1 if value > 0 else -1) * (-1 if index % 2 else 1)
        for index, value in enumerate(characteristic)
        if value
    ]
    return sum(a != b for a, b in pairwise(positive)) - sum(
        a != b for a, b in pairwise(negative)
    )


def _schur_signature(coefficients: list[int]) -> int:
    n = len(coefficients) - 1
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = sum(
                coefficients[n - i + k] * coefficients[n - j + k]
                - coefficients[i - k] * coefficients[j - k]
                for k in range(min(i, j) + 1)
            )
    return _signature(matrix)


def _real_root_count(coefficients: list[int]) -> int:
    """Hermite's Bezoutian signature counts distinct real roots."""
    n = len(coefficients) - 1
    derivative = [(i + 1) * coefficients[i + 1] for i in range(n)] + [0]
    matrix = [[0] * n for _ in range(n)]
    for a in range(1, n + 1):
        for b in range(a):
            coefficient = (
                coefficients[a] * derivative[b] - derivative[a] * coefficients[b]
            )
            for k in range(a - b):
                matrix[a - 1 - k][b + k] += coefficient
    return _signature(matrix)


def unit_disk_profile(polynomial: RationalPolynomial) -> UnitDiskProfile:
    """Count all roots inside, on and outside |z|=1, including multiplicity."""
    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    deadline = started + 60.0
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)

    def checkpoint() -> None:
        request_checkpoint("computing unit-disk root profile")
        if time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(
                "unit-disk root profile deadline expired"
            )

    bind_request_deadline(deadline)
    checkpoint()
    valuation, stride, coefficients = _admit(polynomial)
    from flint import fmpz_poly

    source = fmpz_poly(coefficients)
    inside, boundary, outside = valuation, 0, 0
    _, factors = source.factor_squarefree()
    for factor, multiplicity in factors:
        checkpoint()
        a = [int(value) for value in factor.coeffs()]
        n = len(a) - 1
        difference = _schur_signature(a)
        # Homogeneous Horner evaluation of (1-s)^n p((1+s)/(1-s)).
        plus, minus = fmpz_poly([1, 1]), fmpz_poly([1, -1])
        transformed = fmpz_poly([a[-1]])
        power = fmpz_poly([1])
        for coefficient in reversed(a[:-1]):
            power *= minus
            transformed = transformed * plus + coefficient * power
        q = [int(value) for value in transformed.coeffs()]
        real = fmpz_poly(
            [value * (-1) ** (i // 2) if i % 2 == 0 else 0 for i, value in enumerate(q)]
        )
        imaginary = fmpz_poly(
            [value * (-1) ** (i // 2) if i % 2 else 0 for i, value in enumerate(q)]
        )
        common = real.gcd(imaginary)
        # A squarefree factor stays squarefree under the Mobius map;
        # hence the real gcd is squarefree as well. Missing degree is -1.
        on = (
            n
            - int(transformed.degree())
            + _real_root_count([int(value) for value in common.coeffs()])
        )
        inside += stride * multiplicity * ((n - on + difference) // 2)
        boundary += stride * multiplicity * on
        outside += stride * multiplicity * ((n - on - difference) // 2)
    checkpoint()
    return UnitDiskProfile(
        polynomial=polynomial,
        degree=polynomial.polynomial.terms[0].exponents[0],
        inside=inside,
        on=boundary,
        outside=outside,
    )
