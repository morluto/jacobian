"""Source-derived bounds before characteristic expansion or root isolation."""

from collections import Counter
from dataclasses import dataclass
from math import lcm

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.matrices.inertia_cells._normalization import (
    normalize_affine,
    specialize,
)
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.polynomials.values import RationalPolynomial

Block = tuple[tuple[RationalPolynomial, ...], ...]


@dataclass(frozen=True)
class BlockPlan:
    entries: Block
    multiplicity: int
    degree: int
    coefficient_bits: int
    denominator: int


def reject_budget(message: str) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=("matrix",), code="matrix.inertia_cells.budget", message=message
    )


def _blocks(matrix: RationalPolynomialMatrix) -> Counter[Block]:
    n = matrix.row_count
    unseen = set(range(n))
    blocks: Counter[Block] = Counter()
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        axes: list[int] = []
        while pending:
            v = pending.pop()
            axes.append(v)
            adjacent = [
                w for w in sorted(unseen) if matrix.entries[v][w].polynomial.terms
            ]
            unseen.difference_update(adjacent)
            pending.extend(adjacent)
        axes.sort()
        block = tuple(tuple(matrix.entries[i][j] for j in axes) for i in axes)
        blocks[block] += 1
    return blocks


def _block_plan(
    block: Block, multiplicity: int, interval: ClosedRationalInterval
) -> BlockPlan:
    if len(block) == 1 and all(
        t.exponents[0] <= 1 for t in block[0][0].polynomial.terms
    ):
        block = ((normalize_affine(block[0][0], interval),),)
    coefficients = [
        term.coefficient
        for row in block
        for entry in row
        for term in entry.polynomial.terms
    ]
    degree = max(
        (
            term.exponents[0]
            for row in block
            for entry in row
            for term in entry.polynomial.terms
        ),
        default=0,
    )
    n = len(block)
    if n * degree > 16:
        raise reject_budget(
            "a connected block's characteristic coefficient degree exceeds the degree-16 algebraic carrier"
        )
    denominator = 1
    for q in coefficients:
        denominator = lcm(denominator, q.den)
    height = max(
        (
            abs(q.num).bit_length() + (denominator // q.den).bit_length()
            for q in coefficients
        ),
        default=1,
    )
    # The kernel computes charpoly(D*A), with positive D. This preserves
    # every specialization inertia and removes all rational denominators.
    # Principal k-minor coefficients have at most k!*(degree+1)^k
    # products; the characteristic coefficient sums at most 2^n minors.
    # The same bound with slack bounds division-free Berkowitz products.
    bits = n * (height + n.bit_length() + (degree + 1).bit_length() + 2)
    if bits > (100000 if n == 1 and degree <= 1 else 65536):
        raise reject_budget(
            "characteristic expansion exceeds the coefficient bit bound"
        )
    if (
        degree
        and not (n == 1 and degree == 1)
        and bits + n * degree + (n * degree + 1).bit_length() > 3300
    ):
        raise reject_budget(
            "primitive factor coefficients would exceed the canonical algebraic carrier"
        )
    return BlockPlan(block, multiplicity, n * degree, bits, denominator)


def admit(
    matrix: RationalPolynomialMatrix, interval: ClosedRationalInterval
) -> tuple[BlockPlan, ...]:
    n = matrix.row_count
    if matrix.column_count != n:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="matrix.inertia_cells.shape",
            message="matrix must be square",
        )
    if n > 128:
        raise reject_budget("source matrix exceeds 16,384 ordered entries")
    total_bits = 0
    for i, row in enumerate(matrix.entries):
        request_checkpoint("during inertia-cell source admission")
        for j, entry in enumerate(row):
            if entry != matrix.entries[j][i]:
                raise OperationDomainValidationError(
                    location=("matrix",),
                    code="matrix.inertia_cells.symmetry",
                    message="matrix must be symmetric over QQ[t]",
                )
            total_bits += sum(
                abs(term.coefficient.num).bit_length()
                + term.coefficient.den.bit_length()
                for term in entry.polynomial.terms
            )
    if total_bits > 1_000_000:
        raise reject_budget("source coefficients exceed one million component bits")
    total_terms = sum(
        len(entry.polynomial.terms) for row in matrix.entries for entry in row
    )
    if total_terms > 65_536:
        raise reject_budget(
            "source polynomial terms exceed the echoed-matrix allocation envelope"
        )
    working = (
        specialize(matrix, interval.lower.as_fraction())
        if interval.lower == interval.upper
        else matrix
    )
    plans = tuple(
        _block_plan(block, multiplicity, interval)
        for block, multiplicity in _blocks(working).items()
    )
    characteristic_work = sum(
        len(p.entries) ** 4 * (p.degree + 1) ** 2 * p.coefficient_bits**2 for p in plans
    )
    if characteristic_work > 1_000_000_000_000:
        raise reject_budget(
            "division-free characteristic coefficient work exceeds the admitted budget"
        )
    _admit_isolation(plans, interval)
    return plans


def _admit_isolation(
    plans: tuple[BlockPlan, ...], interval: ClosedRationalInterval
) -> None:
    all_active = tuple(p for p in plans if p.degree)
    nonconstant = tuple(
        p for p in all_active if not (len(p.entries) == 1 and p.degree == 1)
    )
    linear = tuple(p for p in all_active if len(p.entries) == 1 and p.degree == 1)
    # Affine roots are rational: sorting exact ratios requires no factorization.
    if not nonconstant:
        # Exact sorting and affine signs use rational cross-products only.
        if (
            4 * (len(linear) + 2) * sum(p.coefficient_bits**2 for p in linear)
            > 100_000_000_000_000
        ):
            raise reject_budget(
                "rational affine comparisons exceed the arithmetic budget"
            )
        if 12 * sum(p.coefficient_bits for p in linear) > 64_000_000:
            raise reject_budget("rational cell boundaries exceed the output bit budget")
        return
    degree = sum(p.degree for p in nonconstant) + len(linear) + 2
    if degree > 130:
        raise reject_budget("transition union exceeds 128 candidate roots")
    endpoint_bits = max(
        abs(q.num).bit_length() + q.den.bit_length()
        for q in (interval.lower, interval.upper)
    )
    # A primitive squarefree divisor obeys Landau-Mignotte's factor bound.
    # Product coefficient height adds a convolution factor per source.
    height = (
        sum(
            p.coefficient_bits + p.degree + 2 * (p.degree + 1).bit_length()
            for p in nonconstant
        )
        + sum(p.coefficient_bits + 2 for p in linear)
        + 2 * endpoint_bits
        + 4
    )
    # Cauchy and Mignotte bounds admit separating rational endpoints with
    # this bit length, also separating the two rational domain endpoints.
    isolation_bits = 4 * degree * (height + degree.bit_length() + 2) + height + 8
    if isolation_bits > 100_000:
        raise reject_budget(
            "root separation exceeds the canonical rational endpoint envelope"
        )
    if degree**4 * height**2 > 1_000_000_000_000:
        raise reject_budget(
            "transition root isolation exceeds the admitted arithmetic budget"
        )
    sign_work = 0
    factor_degree = max(p.degree for p in nonconstant)
    factor_bits = max(
        p.coefficient_bits + p.degree + (p.degree + 1).bit_length() for p in nonconstant
    )
    # Each boundary appears in a point and at most two adjacent open cells:
    # twelve rational numerator/denominator components plus three minimal
    # polynomials. Counts and fixed schema fields have bounded constant size.
    output_bits = degree * (12 * isolation_bits + 3 * (factor_degree + 1) * factor_bits)
    if output_bits > 64_000_000:
        raise reject_budget("exact cell boundary output exceeds the bit budget")
    for p in all_active:
        # Remainders modulo a factor have common denominator dividing the
        # source denominator times a power of the factor leading coefficient.
        # Isolating factor*remainder needs degree at most 2*d-1.
        remainder_bits = 4 * (p.degree + 1) * (p.coefficient_bits + factor_bits + 2)
        sign_work += (
            (len(p.entries) + 1)
            * degree
            * (factor_degree + min(p.degree, factor_degree - 1) + 1) ** 4
            * remainder_bits**2
        )
    if sign_work > 100_000_000_000_000:
        raise reject_budget(
            "algebraic coefficient sign work exceeds the admitted budget"
        )
    if (
        degree * len(plans) * max(p.degree for p in nonconstant) * isolation_bits
        > 100_000_000
    ):
        raise reject_budget("rational cell sampling exceeds the admitted work budget")
