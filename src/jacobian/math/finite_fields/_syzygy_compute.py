"""Producer-side kernels for finite-field Jacobian syzygy leaves (#964).

Formal characteristic-p gradients, quotient normal forms with replayable
reduction ledgers, and bounded-degree syzygy-slice bases.  Every kernel
reuses the checker's exact term arithmetic, reduction regime, and envelope;
every produced syzygy row replays to a zero remainder by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.finite_fields._admission import require_field
from jacobian.math.finite_fields._algebraic_sets import AlgebraicPolynomial
from jacobian.math.finite_fields._jacobian_syzygy import (
    _add_into,
    _admit_reduction_regime,
    _differentiate,
    _from_terms,
    _multiply,
    _normal_form,
    _require_canonical,
    _require_shared_parent,
    _Terms,
)
from jacobian.math.finite_fields._jacobian_syzygy_models import (
    MAX_JACOBIAN_SYZYGY_DEGREE,
    MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS,
    MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS,
    MAX_JACOBIAN_SYZYGY_VARS,
    MAX_JACOBIAN_SYZYGY_WORK,
)
from jacobian.math.finite_fields._syzygy_compute_models import (
    MAX_SYZYGY_EQUATIONS,
    MAX_SYZYGY_RREF_UPDATES,
    MAX_SYZYGY_SLICE_DEGREE,
    MAX_SYZYGY_UNKNOWNS,
    FiniteFieldJacobianResult,
    QuotientReduceResult,
    ReduceStep,
    SyzygyGeneratorsResult,
)

__all__ = [
    "jacobian_compute",
    "quotient_reduce",
    "syzygy_generators",
    "verify_jacobian",
    "verify_quotient_reduction",
    "verify_syzygy_generators",
]


def _domain_error(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationDomainValidationError:
    return OperationDomainValidationError(location=location, code=code, message=message)


def _resource_error(
    location: tuple[str | int, ...], code: str, message: str
) -> OperationResourceAdmissionError:
    return OperationResourceAdmissionError(
        location=location, code=code, message=message
    )


def _admit_compute_source(
    polynomial: AlgebraicPolynomial,
    ideal_generators: Sequence[AlgebraicPolynomial],
    location: str = "polynomial",
) -> int:
    """Admit one source polynomial with the checker envelope; return the prime."""

    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    nvars = len(axis.labels)
    if not 1 <= nvars <= MAX_JACOBIAN_SYZYGY_VARS:
        raise _resource_error(
            (location, "variable_axis"),
            "finite_field.jacobian_syzygy_variable_bound",
            "the variable axis exceeds the Jacobian-syzygy variable bound",
        )
    if len(set(axis.labels)) != nvars:
        raise _domain_error(
            (location, "variable_axis"),
            "finite_field.jacobian_syzygy_axis_labels_unique",
            "variable axis labels must be unique",
        )
    if len(ideal_generators) > MAX_JACOBIAN_SYZYGY_IDEAL_GENERATORS:
        raise _resource_error(
            ("ideal_generators",),
            "finite_field.jacobian_syzygy_ideal_generator_bound",
            "the admitted reduction regime carries at most one principal generator",
        )
    if presentation.degree != 1:
        raise _domain_error(
            (location, "presentation"),
            "finite_field.jacobian_syzygy_prime_field_only",
            "Jacobian syzygies are bound to a prime field GF(p) presentation",
        )
    require_field(presentation)
    _require_canonical(polynomial, location)
    for index, generator in enumerate(ideal_generators):
        _require_shared_parent(
            generator, presentation, axis, ("ideal_generators", index)
        )
        _require_canonical(generator, f"ideal_generators.{index}")
    return presentation.characteristic


def jacobian_compute(polynomial: AlgebraicPolynomial) -> FiniteFieldJacobianResult:
    """Derive the formal characteristic-p gradient of one polynomial."""

    prime = _admit_compute_source(polynomial, ())
    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    source_terms = {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }
    return FiniteFieldJacobianResult._from_kernel(
        polynomial=polynomial,
        partials=tuple(
            _from_terms(_differentiate(source_terms, index, prime), presentation, axis)
            for index in range(len(axis.labels))
        ),
        characteristic=prime,
    )


def verify_jacobian(claim: FiniteFieldJacobianResult) -> bool:
    """Check a claimed gradient by re-deriving it from the source."""
    return jacobian_compute(claim.polynomial) == claim


def _reduce_with_ledger(
    terms: _Terms,
    *,
    tail: _Terms | None,
    reduction_degree: int,
    reduction_index: int,
    prime: int,
) -> tuple[_Terms, list[tuple[tuple[int, ...], int]]]:
    """Reduce by z^d == -h, recording each rewritten monomial choice."""

    if tail is None:
        return dict(terms), []
    working = dict(terms)
    steps: list[tuple[tuple[int, ...], int]] = []
    while True:
        candidates = [
            exponents
            for exponents in working
            if exponents[reduction_index] >= reduction_degree
        ]
        if not candidates:
            return working, steps
        exponents = max(candidates)
        coefficient = working.pop(exponents)
        steps.append((exponents, coefficient))
        if len(steps) > MAX_JACOBIAN_SYZYGY_REDUCTION_STEPS:
            raise _resource_error(
                ("polynomial",),
                "finite_field.jacobian_syzygy_reduction_step_bound",
                "quotient reduction exceeds the admitted step bound",
            )
        base = list(exponents)
        base[reduction_index] -= reduction_degree
        for tail_exponents, tail_coefficient in tail.items():
            shifted = tuple(
                first + second
                for first, second in zip(base, tail_exponents, strict=True)
            )
            accumulated = (
                working.get(shifted, 0) - coefficient * tail_coefficient
            ) % prime
            if accumulated:
                working[shifted] = accumulated
            else:
                working.pop(shifted, None)


def quotient_reduce(
    polynomial: AlgebraicPolynomial,
    ideal_generators: Sequence[AlgebraicPolynomial] = (),
    reduction_variable: str | None = None,
) -> QuotientReduceResult:
    """Reduce one polynomial to normal form with a replayable step ledger."""

    prime = _admit_compute_source(polynomial, ideal_generators)
    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    reduction_index, tail, reduction_degree = _admit_reduction_regime(
        axis, ideal_generators, reduction_variable
    )
    source_terms = {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }
    if tail is not None:
        tail_terms = max(1, len(tail))
        work = (
            max(1, len(source_terms))
            * (1 + MAX_JACOBIAN_SYZYGY_DEGREE // max(1, reduction_degree))
            * tail_terms
        )
        if work > MAX_JACOBIAN_SYZYGY_WORK:
            raise _resource_error(
                ("polynomial",),
                "finite_field.jacobian_syzygy_work_bound",
                "quotient reduction exceeds the admitted work bound",
            )
    remainder_terms, steps = _reduce_with_ledger(
        source_terms,
        tail=tail,
        reduction_degree=reduction_degree,
        reduction_index=reduction_index,
        prime=prime,
    )
    return QuotientReduceResult._from_kernel(
        polynomial=polynomial,
        ideal_generators=tuple(ideal_generators),
        reduction_variable=reduction_variable,
        remainder=_from_terms(remainder_terms, presentation, axis),
        steps=tuple(
            ReduceStep._from_kernel(target=target, coefficient=coefficient)
            for target, coefficient in steps
        ),
        characteristic=prime,
    )


def verify_quotient_reduction(claim: QuotientReduceResult) -> bool:
    """Check a claimed reduction by replaying its recorded monomial choices."""

    try:
        prime = _admit_compute_source(claim.polynomial, claim.ideal_generators)
        _require_shared_parent(
            claim.remainder,
            claim.polynomial.presentation,
            claim.polynomial.variable_axis,
            ("remainder",),
        )
        _require_canonical(claim.remainder, "remainder")
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    presentation = claim.polynomial.presentation
    axis = claim.polynomial.variable_axis
    if claim.characteristic != presentation.characteristic:
        return False
    try:
        _reduction_index, tail, reduction_degree = _admit_reduction_regime(
            axis, claim.ideal_generators, claim.reduction_variable
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
    if tail is None:
        return (
            not claim.steps
            and claim.reduction_variable is None
            and {
                term.exponents: int(term.coefficient.coordinates[0])
                for term in claim.polynomial.terms
                if not term.coefficient.is_zero
            }
            == {
                term.exponents: int(term.coefficient.coordinates[0])
                for term in claim.remainder.terms
                if not term.coefficient.is_zero
            }
        )
    working = {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in claim.polynomial.terms
        if not term.coefficient.is_zero
    }
    reduction_index = axis.labels.index(claim.reduction_variable or "")
    for step in claim.steps:
        if (
            len(step.target) != len(axis.labels)
            or any(
                type(exponent) is not int or exponent < 0 for exponent in step.target
            )
            or not 1 <= step.coefficient < prime
            or step.target[reduction_index] < reduction_degree
            or working.get(step.target, 0) != step.coefficient
        ):
            return False
        coefficient = working.pop(step.target)
        base = list(step.target)
        base[reduction_index] -= reduction_degree
        for tail_exponents, tail_coefficient in tail.items():
            shifted = tuple(
                first + second
                for first, second in zip(base, tail_exponents, strict=True)
            )
            accumulated = (
                working.get(shifted, 0) - coefficient * tail_coefficient
            ) % prime
            if accumulated:
                working[shifted] = accumulated
            else:
                working.pop(shifted, None)
    if any(exponents[reduction_index] >= reduction_degree for exponents in working):
        return False
    return working == {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in claim.remainder.terms
        if not term.coefficient.is_zero
    }


def _bounded_monomials(nvars: int, degree: int) -> list[tuple[int, ...]]:
    """List exponent vectors of total degree at most ``degree`` in order."""

    if nvars == 0:
        return [()]
    if nvars == 1:
        return [(total,) for total in range(degree + 1)]
    out: list[tuple[int, ...]] = []
    for first in range(degree + 1):
        for rest in _bounded_monomials(nvars - 1, degree - first):
            out.append((first, *rest))
    return sorted(out)


def _nullspace_gfp(
    matrix: list[list[int]], prime: int, unknowns: int
) -> tuple[list[list[int]], int]:
    """Return a kernel basis and rank of an exact GF(p) matrix.

    ``unknowns`` is supplied by the caller so a zero-equation slice still spans
    the full coordinate space; the matrix shape alone cannot recover it.
    Reduced row-echelon form with deterministic pivot order; free columns
    in order give the basis.  Raises a resource error past the admitted
    elimination-update budget.
    """

    rows = [row[:] for row in matrix]
    equations = len(rows)
    pivots: list[int] = []
    updates = 0
    pivot_row = 0
    for column in range(unknowns):
        pivot = next(
            (
                candidate
                for candidate in range(pivot_row, equations)
                if rows[candidate][column] % prime
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        inverse = pow(rows[pivot_row][column], -1, prime)
        rows[pivot_row] = [(value * inverse) % prime for value in rows[pivot_row]]
        updates += unknowns
        for other in range(equations):
            if other != pivot_row and rows[other][column]:
                factor = rows[other][column]
                rows[other] = [
                    (current - factor * pivot_value) % prime
                    for current, pivot_value in zip(
                        rows[other], rows[pivot_row], strict=True
                    )
                ]
                updates += unknowns
                if updates > MAX_SYZYGY_RREF_UPDATES:
                    raise _resource_error(
                        ("max_syzygy_degree",),
                        "finite_field.syzygy_slice_work_bound",
                        "syzygy-slice elimination exceeds the admitted work bound",
                    )
        pivots.append(column)
        pivot_row += 1
    pivot_set = set(pivots)
    basis: list[list[int]] = []
    for free in range(unknowns):
        if free in pivot_set:
            continue
        vector = [0] * unknowns
        vector[free] = 1
        for row_index, pivot_column in enumerate(pivots):
            vector[pivot_column] = (-rows[row_index][free]) % prime
        basis.append(vector)
    return basis, len(pivots)


def syzygy_generators(
    polynomial: AlgebraicPolynomial,
    ideal_generators: Sequence[AlgebraicPolynomial] = (),
    reduction_variable: str | None = None,
    max_syzygy_degree: int = 4,
) -> SyzygyGeneratorsResult:
    """Compute a basis of the bounded-degree Jacobian syzygy slice.

    Unknowns are the coefficients of monomials of total degree at most
    ``max_syzygy_degree`` in each of the ``nvars`` row components; the
    linear map sends a row to the normal form of its dot product with the
    characteristic-p Jacobian.  The exact kernel basis over GF(p) is the
    slice basis: every returned row replays to a zero remainder, and the
    dimension theorem binds ``rows == unknowns - rank``.  An empty family
    is exact (the slice is the zero space), never a claim beyond the slice.
    """

    if (
        type(max_syzygy_degree) is not int
        or max_syzygy_degree < 0
        or max_syzygy_degree > MAX_SYZYGY_SLICE_DEGREE
    ):
        raise _domain_error(
            ("max_syzygy_degree",),
            "finite_field.syzygy_slice_degree_bound",
            "the syzygy slice degree stays within the admitted slice bound",
        )
    prime = _admit_compute_source(polynomial, ideal_generators)
    presentation = polynomial.presentation
    axis = polynomial.variable_axis
    nvars = len(axis.labels)
    reduction_index, tail, reduction_degree = _admit_reduction_regime(
        axis, ideal_generators, reduction_variable
    )
    source_terms = {
        term.exponents: int(term.coefficient.coordinates[0])
        for term in polynomial.terms
        if not term.coefficient.is_zero
    }
    jacobian_terms = tuple(
        _differentiate(source_terms, index, prime) for index in range(nvars)
    )
    monomials = _bounded_monomials(nvars, max_syzygy_degree)
    unknowns = nvars * len(monomials)
    if unknowns > MAX_SYZYGY_UNKNOWNS:
        raise _resource_error(
            ("max_syzygy_degree",),
            "finite_field.syzygy_slice_unknown_bound",
            "the syzygy-slice unknown count exceeds the admitted bound",
        )
    equation_of: dict[tuple[int, ...], int] = {}
    columns: list[dict[int, int]] = []
    for component in range(nvars):
        for monomial in monomials:
            image = _multiply({monomial: 1}, jacobian_terms[component], prime)
            normal, _ = _normal_form(
                image,
                tail=tail,
                reduction_degree=reduction_degree,
                reduction_index=reduction_index,
                prime=prime,
            )
            column: dict[int, int] = {}
            for exponents, coefficient in normal.items():
                row = equation_of.setdefault(exponents, len(equation_of))
                column[row] = coefficient
            columns.append(column)
    equations = len(equation_of)
    if equations > MAX_SYZYGY_EQUATIONS:
        raise _resource_error(
            ("max_syzygy_degree",),
            "finite_field.syzygy_slice_equation_bound",
            "the syzygy-slice equation count exceeds the admitted bound",
        )
    matrix = [[0] * unknowns for _ in range(equations)]
    for column_index, column in enumerate(columns):
        for row_index, coefficient in column.items():
            matrix[row_index][column_index] = coefficient
    basis, rank = _nullspace_gfp(matrix, prime, unknowns)
    rows: list[tuple[AlgebraicPolynomial, ...]] = []
    for vector in basis:
        entries: list[AlgebraicPolynomial] = []
        for component in range(nvars):
            terms = {
                monomial: vector[component * len(monomials) + position]
                for position, monomial in enumerate(monomials)
                if vector[component * len(monomials) + position]
            }
            entries.append(_from_terms(terms, presentation, axis))
        combined: _Terms = {}
        for entry, partial in zip(entries, jacobian_terms, strict=True):
            _add_into(
                combined,
                _multiply(
                    {
                        term.exponents: int(term.coefficient.coordinates[0])
                        for term in entry.terms
                        if not term.coefficient.is_zero
                    },
                    partial,
                    prime,
                ),
                prime,
            )
        remainder, _ = _normal_form(
            combined,
            tail=tail,
            reduction_degree=reduction_degree,
            reduction_index=reduction_index,
            prime=prime,
        )
        if remainder:
            raise RuntimeError("a produced syzygy row must replay to zero")
        rows.append(tuple(entries))
    return SyzygyGeneratorsResult._from_kernel(
        polynomial=polynomial,
        ideal_generators=tuple(ideal_generators),
        reduction_variable=reduction_variable,
        max_syzygy_degree=max_syzygy_degree,
        jacobian=tuple(
            _from_terms(partial, presentation, axis) for partial in jacobian_terms
        ),
        rows=tuple(rows),
        unknowns=unknowns,
        equations=equations,
        rank=rank,
        characteristic=prime,
    )


def verify_syzygy_generators(claim: SyzygyGeneratorsResult) -> bool:
    """Check a claimed slice basis by recomputing it within its bounds."""
    return (
        syzygy_generators(
            claim.polynomial,
            claim.ideal_generators,
            claim.reduction_variable,
            claim.max_syzygy_degree,
        )
        == claim
    )
