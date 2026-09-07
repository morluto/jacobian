"""Smith decomposition over QQ[t] through the existing maintained PID kernel."""

import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Self

from pydantic import model_validator

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.certified_snf._polynomial_bounds import (
    polynomial_smith_bound,
)
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_WALL_SECONDS = 60.0
_MAX_SERIALIZED_BYTES = 8 * 1024 * 1024


class PolynomialSmithDecomposition(StrictModel):
    """D=UAV with monic divisibility diagonal and QQ[t]-unimodular U,V.

    Parsing establishes shapes and rings. It does not establish an authored
    transformation relation or replay the producer's mathematics.
    """

    diagonal: RationalPolynomialMatrix
    left_transformation: RationalPolynomialMatrix
    right_transformation: RationalPolynomialMatrix

    @model_validator(mode="after")
    def require_shapes_and_ring(self) -> Self:
        rows, columns = self.diagonal.row_count, self.diagonal.column_count
        for matrix, size in (
            (self.left_transformation, rows),
            (self.right_transformation, columns),
        ):
            if (matrix.row_count, matrix.column_count) != (size, size):
                raise ValueError(
                    "Smith transformations must be square on the respective axes"
                )
            if matrix.variables != self.diagonal.variables:
                raise ValueError(
                    "Smith matrices must belong to the same polynomial ring"
                )
        return self


def _reject(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("matrix",), code="matrix.polynomial_smith_budget", message=message
    )


type _AffinePivot = tuple[bool, int, int, Fraction]


@dataclass(frozen=True)
class _SmithPlan:
    row_order: tuple[int, ...]
    column_order: tuple[int, ...]
    affine: _AffinePivot | None = None
    shift: int = 0
    common_factor: RationalPolynomial | None = None


def _proportional_factor(
    matrix: RationalPolynomialMatrix,
    nonzero: list[tuple[int, int, int]],
) -> RationalPolynomial | None:
    """Recognize A=p(t)B with B rational, retaining monic p unchanged.

    Support and coefficient heights are admitted before these bounded
    coefficient comparisons. No polynomial gcd or expanded arithmetic is used.
    """
    _, i, j = nonzero[0]
    reference = matrix.entries[i][j].polynomial.terms
    support = tuple(term.exponents for term in reference)
    if any(
        tuple(term.exponents for term in matrix.entries[i][j].polynomial.terms)
        != support
        for _, i, j in nonzero
    ):
        return None
    leading = reference[0].coefficient.as_fraction()
    coefficients = tuple(term.coefficient.as_fraction() / leading for term in reference)
    for _, i, j in nonzero:
        terms = matrix.entries[i][j].polynomial.terms
        leading = terms[0].coefficient.as_fraction()
        if any(
            term.coefficient.as_fraction() != coefficient * leading
            for term, coefficient in zip(terms, coefficients, strict=True)
        ):
            return None
    if (
        max(
            max(abs(q.numerator).bit_length(), q.denominator.bit_length())
            for q in coefficients
        )
        > 100_000
    ):
        _reject(
            "proportional polynomial normalization exceeds the rational height envelope"
        )
    return RationalPolynomial(
        variables=matrix.variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in zip(support, coefficients, strict=True)
            )
        ),
    )


def _affine_pivot(
    matrix: RationalPolynomialMatrix,
) -> tuple[_AffinePivot, int, int] | None:
    """Retain one bounded rational elementary map exposing an affine unit.

    The caller first bounds source support and coefficient height. This
    two-by-two coefficient determinant comparison is not repeated at execution.
    """
    for by_rows in (True, False):
        outer = matrix.column_count if by_rows else matrix.row_count
        inner = matrix.row_count if by_rows else matrix.column_count
        for fixed in range(outer):
            first: tuple[int, Fraction, Fraction] | None = None
            for index in range(inner):
                i, j = (index, fixed) if by_rows else (fixed, index)
                terms = matrix.entries[i][j].polynomial.terms
                if not terms:
                    continue
                slope = terms[0].coefficient.as_fraction()
                constant = (
                    terms[1].coefficient.as_fraction()
                    if len(terms) == 2
                    else Fraction()
                )
                if first is None:
                    first = index, slope, constant
                    continue
                source, source_slope, source_constant = first
                if constant * source_slope != slope * source_constant:
                    return (by_rows, source, index, slope / source_slope), i, j
    return None


def _admit_sparse_diagonal(
    matrix: RationalPolynomialMatrix,
    nonzero: list[tuple[int, int, int]],
    cells: int,
) -> _SmithPlan:
    # A monomial weighted partial permutation is already Smith after
    # ordering its exponents and scaling rational units. The maintained
    # kernel performs no polynomial Euclid or divisibility repair here.
    # Retain sparse high-degree entries without predicting dense support.
    if 64 * len(nonzero) ** 4 > 20_000_000:
        _reject("embedded monomial identity products exceed the work envelope")
    coefficient_bits = sum(
        abs(term.coefficient.num).bit_length() + term.coefficient.den.bit_length()
        for _, i, j in nonzero
        for term in matrix.entries[i][j].polynomial.terms
    )
    extra_terms = 0
    if len(nonzero) == 1:
        _, i, j = nonzero[0]
        terms = matrix.entries[i][j].polynomial.terms
        extra_terms = len(terms) - 1
        if extra_terms:
            lead = terms[0].coefficient
            components = [
                (
                    abs(term.coefficient.num).bit_length() + lead.den.bit_length(),
                    term.coefficient.den.bit_length() + abs(lead.num).bit_length(),
                )
                for term in terms
            ]
            if max(max(pair) for pair in components) > 100_000:
                _reject(
                    "sparse scalar monic normalization exceeds the rational height envelope"
                )
            coefficient_bits = (
                sum(sum(pair) for pair in components)
                + abs(lead.num).bit_length()
                + lead.den.bit_length()
            )
    if (
        cells + extra_terms
    ) * 512 + 2 * coefficient_bits + 1024 > _MAX_SERIALIZED_BYTES:
        _reject("sparse Smith output exceeds the serialization envelope")
    ordered = sorted(nonzero)
    return _SmithPlan(tuple(i for _, i, _ in ordered), tuple(j for _, _, j in ordered))


def _admit(
    matrix: RationalPolynomialMatrix,
) -> _SmithPlan:
    rows, columns = matrix.row_count, matrix.column_count
    cells = rows * columns + rows * rows + columns * columns
    if cells > 16_384 or cells * 512 + 1024 > _MAX_SERIALIZED_BYTES:
        _reject("dense source and full transformation output exceed the cell envelope")
    row_order = [
        i
        for i, row in enumerate(matrix.entries)
        if any(p.polynomial.terms for p in row)
    ]
    column_order = [
        j
        for j in range(columns)
        if any(matrix.entries[i][j].polynomial.terms for i in row_order)
    ]
    nonzero = [
        (p.polynomial.terms[0].exponents[0], i, j)
        for i, row in enumerate(matrix.entries)
        for j, p in enumerate(row)
        if p.polynomial.terms
    ]
    if not nonzero:
        return _SmithPlan(tuple(row_order), tuple(column_order))
    if len(nonzero) == len(row_order) == len(column_order) and (
        len(nonzero) == 1
        or all(len(matrix.entries[i][j].polynomial.terms) == 1 for _, i, j in nonzero)
    ):
        return _admit_sparse_diagonal(matrix, nonzero, cells)
    pivot_degree, pivot_row, pivot_column = min(nonzero)
    # Permutations put a smallest-degree source entry first. This preserves
    # Smith semantics and allows constant pivots to use scalar-division bounds.
    row_order.remove(pivot_row)
    row_order.insert(0, pivot_row)
    column_order.remove(pivot_column)
    column_order.insert(0, pivot_column)
    diagonal = len(nonzero) == len(row_order) == len(column_order)
    if diagonal:
        ordered = sorted(nonzero)
        row_order = [i for _, i, _ in ordered]
        column_order = [j for _, _, j in ordered]
    terms = [t for row in matrix.entries for p in row for t in p.polynomial.terms]
    if len(terms) > 16_384:
        _reject("source polynomial support exceeds the term envelope")
    if (
        sum(
            abs(term.coefficient.num).bit_length() + term.coefficient.den.bit_length()
            for term in terms
        )
        > _MAX_SERIALIZED_BYTES
    ):
        _reject("source coefficient inspection exceeds the rational work envelope")
    denominator_bits = sum(
        (q - 1).bit_length() for q in {t.coefficient.den for t in terms} if q != 1
    )
    height = denominator_bits + max(abs(t.coefficient.num).bit_length() for t in terms)
    if height > 100_000:
        _reject("polynomial coefficient comparisons exceed the input height envelope")
    original_degree = max(d for d, _, _ in nonzero)
    common_factor = _proportional_factor(matrix, nonzero) if original_degree else None
    # A common monomial factor translates support without changing either
    # reconstruction map. Restore it on D after Smith reduction, preserving
    # module torsion; it is never inverted in the public coefficient ring.
    shift = min(term.exponents[0] for term in terms) if common_factor is None else 0
    degree = original_degree - shift if common_factor is None else 0
    pivot_degree = pivot_degree - shift if common_factor is None else 0
    affine = None
    input_height = height
    if degree == pivot_degree == 1:
        exposed = _affine_pivot(matrix)
        if exposed is not None:
            affine, pivot_row, pivot_column = exposed
            row_order.remove(pivot_row)
            row_order.insert(0, pivot_row)
            column_order.remove(pivot_column)
            column_order.insert(0, pivot_column)
            pivot_degree = 0
            # The scalar is a ratio of two slopes. Bound elementary
            # subtraction, its retained map, and unreduced coefficient work.
            height = 6 * height + 8
    bound = polynomial_smith_bound(
        len(row_order),
        len(column_order),
        degree,
        height,
        pivot_degree=pivot_degree,
        diagonal=diagonal,
    )
    if bound.degree + shift > 32_768:
        _reject("restored Smith diagonal exceeds the polynomial exponent envelope")
    # Final monic normalization divides a D and U row by the same rational
    # leading coefficient. The shared scalar type allows 32768 decimal digits.
    output_bits = (
        4 * bound.height + 4 * input_height + 4
        if affine is not None
        else 2 * bound.height + 1
    )
    if output_bits > 100_000:
        _reject("monic normalization exceeds the rational coefficient envelope")
    output_bytes = cells * (bound.degree + 1) * (2 * output_bits + 512) + 1024
    if common_factor is not None:
        factor_bytes = sum(
            512
            + abs(term.coefficient.num).bit_length()
            + term.coefficient.den.bit_length()
            for term in common_factor.polynomial.terms
        )
        output_bytes = (
            (rows * rows + columns * columns) * (2 * output_bits + 512)
            + rows * columns * 512
            + min(rows, columns) * factor_bytes
            + 1024
        )
    if output_bytes > _MAX_SERIALIZED_BYTES:
        _reject("serialized Smith matrices exceed the exact output envelope")
    return _SmithPlan(
        tuple(row_order), tuple(column_order), affine, shift, common_factor
    )


def _apply_affine_presolve(
    entries: list[list[Any]],
    row_order: tuple[int, ...],
    column_order: tuple[int, ...],
    affine: _AffinePivot | None,
) -> tuple[bool, int, int, Any] | None:
    if affine is None:
        return None
    from sympy import QQ

    by_rows, original_source, original_target, scalar = affine
    order = row_order if by_rows else column_order
    source, target = order.index(original_source), order.index(original_target)
    q = QQ(scalar.numerator, scalar.denominator)
    if by_rows:
        entries[target] = [
            a - b.mul_ground(q)
            for a, b in zip(entries[target], entries[source], strict=True)
        ]
    else:
        for row in entries:
            row[target] -= row[source].mul_ground(q)
    return by_rows, source, target, q


def _compose_affine_maps(
    left: list[list[Any]],
    right: list[list[Any]],
    affine: tuple[bool, int, int, Any] | None,
) -> None:
    if affine is None:
        return
    by_rows, source, target, q = affine
    if by_rows:
        for row in left:
            row[source] -= row[target].mul_ground(q)
    else:
        right[source] = [
            a - b.mul_ground(q)
            for a, b in zip(right[source], right[target], strict=True)
        ]


def _encode_polynomial(value: RationalPolynomial, plan: _SmithPlan, ring: Any) -> Any:
    from sympy import QQ

    if plan.common_factor is not None:
        if not value.polynomial.terms:
            return ring.zero
        leading = value.polynomial.terms[0].coefficient
        return ring(QQ(leading.num, leading.den))
    return ring.ring.from_dict(
        {
            (term.exponents[0] - plan.shift,): QQ(
                term.coefficient.num, term.coefficient.den
            )
            for term in value.polynomial.terms
        }
    )


def polynomial_smith_decomposition(
    matrix: RationalPolynomialMatrix,
) -> PolynomialSmithDecomposition:
    """Return the monic Smith diagonal and polynomial reconstruction maps."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return polynomial_smith_decomposition(matrix)
    deadline = execution.started_at + _WALL_SECONDS
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before polynomial Smith admission")
    plan = _admit(matrix)
    row_order, column_order, affine, shift = (
        plan.row_order,
        plan.column_order,
        plan.affine,
        plan.shift,
    )
    request_checkpoint("before polynomial Smith conversion")

    from sympy import QQ, Symbol
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp

    ring = QQ.poly_ring(Symbol(matrix.variables[0]))

    active_rows, active_columns = len(row_order), len(column_order)
    source_entries = [
        [_encode_polynomial(matrix.entries[i][j], plan, ring) for j in column_order]
        for i in row_order
    ]
    affine_indices = _apply_affine_presolve(
        source_entries, row_order, column_order, affine
    )
    source = DomainMatrix(source_entries, (active_rows, active_columns), ring)
    diagonal, left, right = smith_normal_decomp(source)
    request_checkpoint("after polynomial Smith kernel")
    d, u, v = diagonal.to_list(), left.to_list(), right.to_list()
    _compose_affine_maps(u, v, affine_indices)
    common_factor_backend = (
        ring.ring.from_dict(
            {
                term.exponents: QQ(term.coefficient.num, term.coefficient.den)
                for term in plan.common_factor.polynomial.terms
            }
        )
        if plan.common_factor is not None
        else ring.one
    )
    for i in range(min(active_rows, active_columns)):
        if d[i][i]:
            scale = QQ.one / d[i][i].LC
            d[i][i] = d[i][i].mul_ground(scale)
            d[i][i] *= common_factor_backend
            if shift:
                d[i][i] *= ring.gens[0] ** shift
            u[i] = [entry.mul_ground(scale) for entry in u[i]]

    rows, columns = matrix.row_count, matrix.column_count
    full_d = [[ring.zero for _ in range(columns)] for _ in range(rows)]
    full_u = [[ring.zero for _ in range(rows)] for _ in range(rows)]
    full_v = [[ring.zero for _ in range(columns)] for _ in range(columns)]
    for i in range(active_rows):
        for j in range(active_columns):
            full_d[i][j] = d[i][j]
        for j, original in enumerate(row_order):
            full_u[i][original] = u[i][j]
    for i, original in enumerate(column_order):
        for j in range(active_columns):
            full_v[original][j] = v[i][j]
    for i, original in enumerate(j for j in range(rows) if j not in row_order):
        full_u[active_rows + i][original] = ring.one
    for i, original in enumerate(j for j in range(columns) if j not in column_order):
        full_v[original][active_columns + i] = ring.one

    def decode(entries: list[list[Any]], n: int, m: int) -> RationalPolynomialMatrix:
        return RationalPolynomialMatrix(
            variables=matrix.variables,
            row_count=n,
            column_count=m,
            entries=tuple(
                tuple(
                    RationalPolynomial(
                        variables=matrix.variables,
                        polynomial=SparseRationalPolynomial(
                            terms=tuple(
                                RationalPolynomialTerm(
                                    coefficient=CanonicalRational(
                                        num=int(coefficient.numerator),
                                        den=int(coefficient.denominator),
                                    ),
                                    exponents=exponents,
                                )
                                for exponents, coefficient in sorted(
                                    value.items(), reverse=True
                                )
                            )
                        ),
                    )
                    for value in row
                )
                for row in entries
            ),
        )

    result = PolynomialSmithDecomposition(
        diagonal=decode(full_d, rows, columns),
        left_transformation=decode(full_u, rows, rows),
        right_transformation=decode(full_v, columns, columns),
    )
    request_checkpoint("after polynomial Smith result construction")
    return result


__all__ = ["PolynomialSmithDecomposition", "polynomial_smith_decomposition"]
