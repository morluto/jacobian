"""Exact polynomial inertia through real-rooted characteristic coefficients.

Schweighofer, Real Algebraic Geometry (2016/17), Theorem 1.5.14 and
Example 1.5.15: Descartes sign variation counts positive eigenvalues exactly
for the characteristic polynomial of a real symmetric matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, TypedDict

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math._root_isolation import strict_root_count
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.matrices.inertia_cells._admission import BlockPlan, admit
from jacobian.math.matrices.inertia_cells._models import (
    InertiaCell,
    InertiaCellsResult,
    InertiaOpenCell,
    InertiaParameterBoundary,
    InertiaPointCell,
)
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
    RealAlgebraicValue,
)
from jacobian.math.polynomials._conversions import rational_polynomial_to_sympy


class _Counts(TypedDict):
    n_positive: int
    n_negative: int
    n_zero: int


@dataclass(frozen=True)
class _Boundary:
    public: InertiaParameterBoundary
    lower: Any
    upper: Any
    factor: Any | None = None
    root_index: int = 0


def _rational(q: Any) -> CanonicalRational:
    return CanonicalRational(num=int(q.p), den=int(q.q))


def _rational_boundary(q: Any) -> _Boundary:
    value = _rational(q)
    return _Boundary(
        InertiaParameterBoundary(
            value=value,
            isolating_interval=RationalIsolatingInterval(
                lower=value, upper=value, interval_type="SINGLETON"
            ),
        ),
        q,
        q,
    )


def _coefficients(plan: BlockPlan) -> list[Any]:
    from sympy import QQ, Poly, Symbol
    from sympy.polys.matrices import DomainMatrix

    t = Symbol(plan.entries[0][0].variables[0])
    ring = QQ.poly_ring(t)
    rows = {
        i: {
            j: ring.from_sympy(
                rational_polynomial_to_sympy(entry).as_expr() * plan.denominator
            )
            for j, entry in enumerate(row)
            if entry.polynomial.terms
        }
        for i, row in enumerate(plan.entries)
    }
    charpoly = DomainMatrix(rows, (len(rows), len(rows)), ring).charpoly()
    return [Poly(ring.to_sympy(coefficient), t, domain=QQ) for coefficient in charpoly]


def _primitive(poly: Any) -> Any:
    _, integer = poly.clear_denoms(convert=True)
    _, primitive = integer.primitive()
    return -primitive if primitive.LC() < 0 else primitive


def _boundaries(
    coefficients: list[list[Any]], lower: Any, upper: Any
) -> list[_Boundary]:
    from sympy import Poly

    if lower == upper:
        return [_rational_boundary(lower)]
    transitions = [
        next(c for c in reversed(block) if not c.is_zero) for block in coefficients
    ]
    active = [p for p in transitions if p.degree() > 0]
    if not active:
        return [_rational_boundary(lower), _rational_boundary(upper)]
    if all(p.degree() == 1 for p in active):
        roots = {-p.TC() / p.LC() for p in active}
        return [
            _rational_boundary(q)
            for q in sorted({lower, upper, *(q for q in roots if lower <= q <= upper)})
        ]
    t = active[0].gens[0]
    factors: dict[tuple[int, ...], Any] = {}
    for polynomial in (*active, Poly(t - lower, t), Poly(t - upper, t)):
        request_checkpoint("during transition factorization")
        for factor, _ in _primitive(polynomial).factor_list()[1]:
            factor = _primitive(factor)
            factors[tuple(int(c) for c in factor.all_coeffs())] = factor
    product = Poly(1, t)
    for factor in factors.values():
        product *= factor
    root_indices = dict.fromkeys(factors, 0)
    answer: list[_Boundary] = []
    request_checkpoint("before transition root isolation")
    for (lo, hi), _ in product.intervals():
        request_checkpoint("during transition root isolation")
        key = next(key for key, f in factors.items() if strict_root_count(f, lo, hi))
        factor = factors[key]
        index = root_indices[key]
        root_indices[key] += 1
        if factor.degree() == 1:
            value = -factor.TC() / factor.LC()
            if lower <= value <= upper:
                answer.append(_rational_boundary(value))
        elif lo >= lower and hi <= upper:
            # Isolating open intervals may touch a different rational root.
            # Refine away from every product root so midpoint samples are
            # strictly inside their parameter cells, never at a point cell.
            while product.eval(lo) == 0 or product.eval(hi) == 0:
                request_checkpoint("during boundary separation")
                lo, hi = factor.refine_root(lo, hi, steps=1)
            algebraic = RealAlgebraicValue._from_admitted_polynomial(
                polynomial=key, real_root_index=index
            )
            boundary = InertiaParameterBoundary(
                value=algebraic,
                isolating_interval=RationalIsolatingInterval(
                    lower=_rational(lo), upper=_rational(hi), interval_type="OPEN"
                ),
            )
            answer.append(_Boundary(boundary, lo, hi, factor, index))
    return answer


def _sign_at_root(polynomial: Any, boundary: _Boundary) -> int:
    if boundary.factor is None or polynomial.degree() <= 0:
        value = polynomial.eval(boundary.lower)
        return 1 if value > 0 else -1 if value < 0 else 0
    representative = polynomial.rem(boundary.factor)
    if representative.is_zero:
        return 0
    # Same coprime-product isolation relation used by the exact real field
    # owner: a reduced nonzero representative shares no root with the factor.
    seen = 0
    for (lo, hi), _ in (boundary.factor * representative).intervals():
        if not strict_root_count(boundary.factor, lo, hi):
            continue
        if seen == boundary.root_index:
            sample = lo if lo == hi else (lo + hi) / 2
            value = representative.eval(sample)
            return 1 if value > 0 else -1 if value < 0 else 0
        seen += 1
    raise ArithmeticError("exact coefficient sign isolation lost its defining root")


def _inertia(
    coefficients: list[list[Any]], plans: tuple[BlockPlan, ...], boundary: _Boundary
) -> _Counts:
    counts = [0, 0, 0]
    for block, plan in zip(coefficients, plans, strict=True):
        request_checkpoint("during exact inertia specialization")
        signs = [_sign_at_root(c, boundary) for c in block]
        positive = [s for s in signs if s]
        negative = [s * (-1) ** i for i, s in enumerate(signs) if s]
        p = sum(a != b for a, b in pairwise(positive))
        m = sum(a != b for a, b in pairwise(negative))
        counts[0] += plan.multiplicity * p
        counts[1] += plan.multiplicity * m
        counts[2] += plan.multiplicity * (len(plan.entries) - p - m)
    return {"n_positive": counts[0], "n_negative": counts[1], "n_zero": counts[2]}


def _require_inertia_inputs(
    matrix: object, interval: object
) -> tuple[RationalPolynomialMatrix, ClosedRationalInterval]:
    if not isinstance(matrix, RationalPolynomialMatrix) or not isinstance(
        interval, ClosedRationalInterval
    ):
        raise TypeError("expected RationalPolynomialMatrix and ClosedRationalInterval")
    try:
        validated_matrix = RationalPolynomialMatrix.model_validate(
            matrix.model_dump(warnings="none")
        )
        validated_interval = ClosedRationalInterval.model_validate(
            interval.model_dump(warnings="none")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(),
            code="matrix.inertia_cells_domain",
            message="inertia cell carriers failed structural validation",
        ) from exc
    return validated_matrix, validated_interval


def compute_inertia_cells(
    matrix: RationalPolynomialMatrix, interval: ClosedRationalInterval
) -> InertiaCellsResult:
    """Partition a closed parameter interval into exact constant-inertia cells.

    Rank is constant away from zeros of the lowest nonzero characteristic
    coefficient. Eigenvalue continuity then makes inertia constant on each
    intervening interval. Point cells are evaluated exactly, including nullity
    increases without sign changes and identically singular matrix families.
    """
    validated_matrix, validated_interval = _require_inertia_inputs(matrix, interval)
    if current_request_execution() is None:
        with request_execution(time.monotonic()):
            return _compute(validated_matrix, validated_interval)
    return _compute(validated_matrix, validated_interval)


def _compute(
    matrix: RationalPolynomialMatrix, interval: ClosedRationalInterval
) -> InertiaCellsResult:
    from sympy import Rational

    execution = current_request_execution()
    assert execution is not None
    deadline = execution.started_at + 60.0
    bind_request_deadline(
        min(deadline, execution.deadline)
        if execution.deadline is not None
        else deadline
    )
    plans = admit(matrix, interval)
    coefficients = []
    for plan in plans:
        request_checkpoint("before polynomial characteristic expansion")
        coefficients.append(_coefficients(plan))
    lower, upper = (Rational(q.num, q.den) for q in (interval.lower, interval.upper))
    boundaries = _boundaries(coefficients, lower, upper)
    cells: list[InertiaCell] = []
    for i, boundary in enumerate(boundaries):
        if i:
            previous = boundaries[i - 1]
            q = (previous.upper + boundary.lower) / 2
            sample = _Boundary(previous.public, q, q)
            cells.append(
                InertiaOpenCell(
                    lower=previous.public,
                    upper=boundary.public,
                    **_inertia(coefficients, plans, sample),
                )
            )
        cells.append(
            InertiaPointCell(
                parameter=boundary.public, **_inertia(coefficients, plans, boundary)
            )
        )
    result = InertiaCellsResult(matrix=matrix, interval=interval, cells=tuple(cells))
    request_checkpoint("after inertia-cell result construction")
    return result
