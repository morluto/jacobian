"""Native exact Lie-bracket operations over QQ."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    MAX_BRACKET_RESULT_COEFFICIENT_DIGITS,
    MAX_BRACKET_WORK,
    MAX_ELEMENT_COEFFICIENT_DIGITS,
    MAX_LIE_DIMENSION,
    MAX_STRUCTURE_COEFFICIENT_DIGITS,
    MAX_STRUCTURE_NONZEROS,
    BracketPairContribution,
    FiniteDimensionalLieAlgebra,
    IdealViolationWitness,
    LieAlgebraElement,
    LieBracketResult,
    LieCenterResult,
    LieDerivedSeriesResult,
    LieIdealCheckResult,
    LieKillingResult,
    LieLowerCentralSeriesResult,
    LieQuotientResult,
    LieSubspace,
    StructureConstant,
)
from jacobian.math.matrices.values import (
    RationalMatrix,
    rational_matrix_from_fractions,
)


@dataclass(frozen=True, slots=True)
class _BracketPairPlan:
    """One admitted nonzero basis-pair contribution."""

    i: int
    j: int
    pair_coefficient: Fraction
    terms: tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True, slots=True)
class _BracketPlan:
    """Exact pair products retained between admission and result construction."""

    pairs: tuple[_BracketPairPlan, ...]
    totals: tuple[Fraction, ...]
    work: int


def _as_algebra(
    value: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> FiniteDimensionalLieAlgebra:
    return (
        value
        if isinstance(value, FiniteDimensionalLieAlgebra)
        else FiniteDimensionalLieAlgebra.model_validate(value)
    )


def _as_element(value: LieAlgebraElement | Mapping[str, Any]) -> LieAlgebraElement:
    return (
        value
        if isinstance(value, LieAlgebraElement)
        else LieAlgebraElement.model_validate(value)
    )


def _run_admission(admission: Any, *, location: tuple[str | int, ...]) -> None:
    try:
        admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="lie_algebra.admission",
            message=str(exc),
        ) from exc


def _bracket_table(
    algebra: FiniteDimensionalLieAlgebra,
) -> dict[tuple[int, int], dict[int, Fraction]]:
    """Expand the ordered constants into the full antisymmetric bracket table."""
    table: dict[tuple[int, int], dict[int, Fraction]] = {}
    for constant in algebra.structure_constants:
        value = constant.coefficient.as_fraction()
        table.setdefault((constant.i, constant.j), {})[constant.k] = value
        table.setdefault((constant.j, constant.i), {})[constant.k] = -value
    return table


def _admit_lie_algebra(
    algebra: FiniteDimensionalLieAlgebra,
) -> dict[tuple[int, int], dict[int, Fraction]]:
    """Establish antisymmetry and every basis-triple Jacobi identity."""
    if not 1 <= len(algebra.basis) <= MAX_LIE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("algebra", "basis"),
            code="lie_algebra.dimension_bound",
            message=(
                "the Lie algebra dimension must stay within the admitted "
                f"1..{MAX_LIE_DIMENSION} basis bound"
            ),
        )
    if len(algebra.structure_constants) > MAX_STRUCTURE_NONZEROS:
        raise OperationResourceAdmissionError(
            location=("algebra", "structure_constants"),
            code="lie_algebra.structure_constant_bound",
            message=(
                "the Lie algebra exceeds the admitted sparse structure-constant "
                f"bound of {MAX_STRUCTURE_NONZEROS}"
            ),
        )
    for index, constant in enumerate(algebra.structure_constants):
        _run_admission(
            lambda constant=constant: require_bounded_rational(
                constant.coefficient,
                max_digits=MAX_STRUCTURE_COEFFICIENT_DIGITS,
                label="structure constant",
            ),
            location=("algebra", "structure_constants", index),
        )
    dimension = len(algebra.basis)
    table = _bracket_table(algebra)

    def _pair(first: int, second: int) -> dict[int, Fraction]:
        if first == second:
            return {}
        return table.get((first, second), {})

    for first in range(dimension):
        for second in range(dimension):
            for third in range(dimension):
                accumulator: dict[int, Fraction] = {}
                for outer_first, outer_second, inner in (
                    (first, second, third),
                    (second, third, first),
                    (third, first, second),
                ):
                    for middle, outer_value in _pair(outer_first, outer_second).items():
                        for target, inner_value in _pair(middle, inner).items():
                            accumulator[target] = (
                                accumulator.get(target, Fraction(0))
                                + outer_value * inner_value
                            )
                if any(value != 0 for value in accumulator.values()):
                    raise OperationDomainValidationError(
                        location=("algebra", "structure_constants"),
                        code="lie_algebra.jacobi_identity",
                        message="structure constants must satisfy the Jacobi identity",
                    )
    return table


def _require_fraction_bound(
    value: Fraction,
    *,
    label: str,
    location: tuple[str | int, ...],
) -> None:
    """Admit one derived rational before canonical result allocation."""

    if (
        decimal_digit_width(value.numerator) > MAX_BRACKET_RESULT_COEFFICIENT_DIGITS
        or decimal_digit_width(value.denominator)
        > MAX_BRACKET_RESULT_COEFFICIENT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=location,
            code="lie_algebra.bracket.result_height_bound",
            message=(
                f"{label} exceeds the admitted "
                f"{MAX_BRACKET_RESULT_COEFFICIENT_DIGITS}-digit canonical bound"
            ),
        )


def _admit_bracket_operands(
    algebra: FiniteDimensionalLieAlgebra,
    left: LieAlgebraElement,
    right: LieAlgebraElement,
) -> _BracketPlan:
    table = _admit_lie_algebra(algebra)
    if left.basis != algebra.basis or right.basis != algebra.basis:
        raise OperationDomainValidationError(
            location=("left",),
            code="lie_algebra.element_basis",
            message="bracket elements must use the algebra's ordered basis",
        )
    for label, element in (("left", left), ("right", right)):
        for index, coordinate in enumerate(element.coordinates):
            _run_admission(
                lambda coordinate=coordinate, label=label: require_bounded_rational(
                    coordinate,
                    max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                    label=f"{label} coordinate",
                ),
                location=(label, "coordinates", index),
            )

    dimension = len(algebra.basis)
    left_coords = tuple(coordinate.as_fraction() for coordinate in left.coordinates)
    right_coords = tuple(coordinate.as_fraction() for coordinate in right.coordinates)
    totals = [Fraction(0)] * dimension
    pairs: list[_BracketPairPlan] = []
    ledger_terms = 0
    for first in range(dimension):
        for second in range(first + 1, dimension):
            pair = (
                left_coords[first] * right_coords[second]
                - left_coords[second] * right_coords[first]
            )
            if pair == 0:
                continue
            _require_fraction_bound(
                pair,
                label="bracket pair coefficient",
                location=("left", "right", "pair_coefficient"),
            )
            row_terms: list[tuple[int, Fraction]] = []
            for target, value in sorted(table.get((first, second), {}).items()):
                scaled = pair * value
                _require_fraction_bound(
                    scaled,
                    label="scaled structure constant",
                    location=("ledger", len(pairs), "terms", len(row_terms)),
                )
                totals[target] += scaled
                row_terms.append((target, scaled))
            if not row_terms:
                continue
            ledger_terms += len(row_terms)
            pairs.append(
                _BracketPairPlan(
                    i=first,
                    j=second,
                    pair_coefficient=pair,
                    terms=tuple(row_terms),
                )
            )
            if len(pairs) > len(algebra.basis) * (len(algebra.basis) - 1) // 2:
                raise OperationResourceAdmissionError(
                    location=("ledger",),
                    code="lie_algebra.bracket.ledger_rows_bound",
                    message="the complete bracket ledger exceeds its row bound",
                )

    for index, total in enumerate(totals):
        _require_fraction_bound(
            total,
            label="bracket coordinate",
            location=("bracket", "coordinates", index),
        )
    work = 2 * (dimension * (dimension - 1) // 2) + 2 * ledger_terms
    if work > MAX_BRACKET_WORK:
        raise OperationResourceAdmissionError(
            location=("ledger",),
            code="lie_algebra.bracket.work_bound",
            message="bracket expansion exceeds its derived arithmetic-work bound",
        )
    return _BracketPlan(pairs=tuple(pairs), totals=tuple(totals), work=work)


def lie_bracket(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    left: LieAlgebraElement | Mapping[str, Any],
    right: LieAlgebraElement | Mapping[str, Any],
) -> LieBracketResult:
    """Compute the exact bracket of two Lie-algebra elements with a ledger."""
    algebra_value = _as_algebra(algebra)
    left_value = _as_element(left)
    right_value = _as_element(right)
    plan = _admit_bracket_operands(algebra_value, left_value, right_value)
    ledger: list[BracketPairContribution] = []
    for pair_plan in plan.pairs:
        row_terms: list[StructureConstant] = []
        for target, scaled in pair_plan.terms:
            row_terms.append(
                StructureConstant.model_construct(
                    i=pair_plan.i,
                    j=pair_plan.j,
                    k=target,
                    coefficient=CanonicalRational.from_fraction(scaled),
                )
            )
        ledger.append(
            BracketPairContribution.model_construct(
                i=pair_plan.i,
                j=pair_plan.j,
                pair_coefficient=CanonicalRational.from_fraction(
                    pair_plan.pair_coefficient
                ),
                terms=tuple(row_terms),
            )
        )
    bracket = LieAlgebraElement.model_construct(
        basis=algebra_value.basis,
        coordinates=tuple(
            CanonicalRational.from_fraction(value) for value in plan.totals
        ),
    )
    return LieBracketResult._from_kernel(
        algebra_value,
        left_value,
        right_value,
        bracket=bracket,
        ledger=tuple(ledger),
    )


def _adjoint_matrices(
    algebra: FiniteDimensionalLieAlgebra,
) -> list[list[list[Fraction]]]:
    """Return the adjoint matrices with ``ad_i`` acting on column vectors.

    Column ``j`` of ``ad_i`` holds the coordinates of ``[b_i, b_j]`` in
    the ordered basis, expanded from the canonical bracket table.
    """

    dimension = len(algebra.basis)
    table = _bracket_table(algebra)
    matrices = []
    for first in range(dimension):
        matrix = [[Fraction(0)] * dimension for _ in range(dimension)]
        for second in range(dimension):
            for target, value in table.get((first, second), {}).items():
                matrix[target][second] = value
        matrices.append(matrix)
    return matrices


def lie_killing_form(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieKillingResult:
    """Compute the exact Killing-form Gram matrix ``tr(ad_i ad_j)``.

    Operation admission establishes every Jacobi identity first, so the
    trace form is the Killing form of a Lie algebra, not of an arbitrary
    bilinear bracket table. At dimension at most 8 the quartic trace
    work is a small constant.
    """

    algebra_value = _as_algebra(algebra)
    _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    adjoints = _adjoint_matrices(algebra_value)

    def _trace_product(first: int, second: int) -> Fraction:
        return sum(
            (
                adjoints[first][row][column] * adjoints[second][column][row]
                for row in range(dimension)
                for column in range(dimension)
            ),
            start=Fraction(0),
        )

    killing = rational_matrix_from_fractions(
        tuple(
            tuple(_trace_product(first, second) for second in range(dimension))
            for first in range(dimension)
        )
    )
    return LieKillingResult._from_kernel(algebra_value, killing)


def lie_center(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieCenterResult:
    """Compute the exact center as a canonical RREF subspace.

    Centrality ``[x, b_j] = 0`` for every basis element is one exact
    rational linear system in the coordinates of ``x``; its solution
    space is the center. Operation admission establishes every Jacobi
    identity first. At dimension at most 8 the nullspace and RREF work
    through maintained exact kernels is a small constant.
    """

    from jacobian.math.matrices.operations import nullspace_result, rref_result

    algebra_value = _as_algebra(algebra)
    _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    table = _bracket_table(algebra_value)
    constraint = rational_matrix_from_fractions(
        tuple(
            tuple(
                table.get((coordinate, basis), {}).get(target, Fraction(0))
                for coordinate in range(dimension)
            )
            for basis in range(dimension)
            for target in range(dimension)
        ),
        column_count=dimension,
    )
    nullspace = nullspace_result(constraint)
    if nullspace.nullity == 0:
        generators = RationalMatrix(row_count=0, column_count=dimension, entries=())
    else:
        generators = rref_result(nullspace.basis_matrix).reduced_matrix
    center = LieSubspace(basis=algebra_value.basis, generators=generators)
    return LieCenterResult._from_kernel(algebra_value, center)


def _subspace_bracket_rows(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    table: dict[tuple[int, int], dict[int, Fraction]],
    dimension: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return RREF rows spanning ``[span(left), span(right)]``.

    Pairwise brackets of the spanning rows are collected exactly and
    reduced through the maintained RREF kernel; an all-zero bracket
    family yields the empty row family.
    """

    from jacobian.math.matrices.operations import rref_result

    vectors: list[tuple[Fraction, ...]] = []
    for first in left:
        for second in right:
            coordinates = [Fraction(0)] * dimension
            for i, first_value in enumerate(first):
                if first_value == 0:
                    continue
                for j, second_value in enumerate(second):
                    if second_value == 0:
                        continue
                    for target, constant in table.get((i, j), {}).items():
                        coordinates[target] += first_value * second_value * constant
            if any(coordinates):
                vectors.append(tuple(coordinates))
    if not vectors:
        return ()
    reduced = rref_result(
        rational_matrix_from_fractions(tuple(vectors), column_count=dimension)
    )
    return tuple(
        tuple(value.as_fraction() for value in row)
        for row in reduced.reduced_matrix.entries[: reduced.rank]
    )


def _subspace_value(
    algebra: FiniteDimensionalLieAlgebra,
    rows: tuple[tuple[Fraction, ...], ...],
) -> LieSubspace:
    return LieSubspace(
        basis=algebra.basis,
        generators=RationalMatrix(
            row_count=len(rows),
            column_count=len(algebra.basis),
            entries=tuple(
                tuple(CanonicalRational.from_fraction(value) for value in row)
                for row in rows
            ),
        ),
    )


def _identity_rows(dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(int(column == row)) for column in range(dimension))
        for row in range(dimension)
    )


def lie_derived_series(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieDerivedSeriesResult:
    """Compute the derived series with its solvability decision.

    ``terms[k+1] = [terms[k], terms[k]]`` from the whole algebra,
    stopping at zero or the first fixed term. Operation admission
    establishes every Jacobi identity first, so each bracket family
    spans an ideal and the series descends.
    """

    algebra_value = _as_algebra(algebra)
    _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    table = _bracket_table(algebra_value)
    rows = [_identity_rows(dimension)]
    while True:
        bracketed = _subspace_bracket_rows(rows[-1], rows[-1], table, dimension)
        if bracketed == rows[-1] or not bracketed:
            if bracketed != rows[-1]:
                rows.append(bracketed)
            break
        rows.append(bracketed)
    terms = tuple(_subspace_value(algebra_value, family) for family in rows)
    return LieDerivedSeriesResult._from_kernel(
        algebra_value, terms, solvable=not rows[-1]
    )


def lie_lower_central_series(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieLowerCentralSeriesResult:
    """Compute the lower central series with its nilpotency decision.

    ``terms[k+1] = [algebra, terms[k]]`` from the whole algebra,
    stopping at zero or the first fixed term. Operation admission
    establishes every Jacobi identity first, so each bracket family
    spans an ideal and the series descends.
    """

    algebra_value = _as_algebra(algebra)
    _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    table = _bracket_table(algebra_value)
    whole = _identity_rows(dimension)
    rows = [whole]
    while True:
        bracketed = _subspace_bracket_rows(whole, rows[-1], table, dimension)
        if bracketed == rows[-1] or not bracketed:
            if bracketed != rows[-1]:
                rows.append(bracketed)
            break
        rows.append(bracketed)
    terms = tuple(_subspace_value(algebra_value, family) for family in rows)
    return LieLowerCentralSeriesResult._from_kernel(
        algebra_value, terms, nilpotent=not rows[-1]
    )


def _rref_pivots(rows: tuple[tuple[Fraction, ...], ...]) -> tuple[int, ...]:
    """Return the pivot column of each RREF row in row order."""

    return tuple(
        next(column for column, value in enumerate(row) if value != 0) for row in rows
    )


def _reduce_by_subspace(
    rows: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Reduce a vector by RREF rows, eliminating pivot entries in order."""

    pivots = _rref_pivots(rows)
    reduced = list(vector)
    for row, pivot in zip(rows, pivots, strict=True):
        factor = reduced[pivot]
        if factor != 0:
            reduced = [
                value - factor * entry
                for value, entry in zip(reduced, row, strict=True)
            ]
    return tuple(reduced)


def _subspace_rows(subspace: LieSubspace) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(entry.as_fraction() for entry in row)
        for row in subspace.generators.entries
    )


def check_ideal(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    candidate: LieSubspace | Mapping[str, Any],
) -> LieIdealCheckResult:
    """Decide whether a subspace absorbs every algebra bracket.

    For every basis element and generator row the bracket is reduced
    against the candidate rows; the first nonzero remainder witnesses
    non-ideal status in deterministic order. Operation admission
    establishes every Jacobi identity first.
    """

    algebra_value = _as_algebra(algebra)
    candidate_value = (
        candidate
        if isinstance(candidate, LieSubspace)
        else LieSubspace.model_validate(candidate)
    )
    _admit_lie_algebra(algebra_value)
    if candidate_value.basis != algebra_value.basis:
        raise OperationDomainValidationError(
            location=("candidate",),
            code="lie_algebra.candidate_basis",
            message="the candidate subspace must use the algebra's ordered basis",
        )
    dimension = len(algebra_value.basis)
    table = _bracket_table(algebra_value)
    rows = _subspace_rows(candidate_value)
    for basis_index in range(dimension):
        for row_index, generator in enumerate(rows):
            coordinates = [Fraction(0)] * dimension
            for j, value in enumerate(generator):
                if value == 0:
                    continue
                for target, constant in table.get((basis_index, j), {}).items():
                    coordinates[target] += value * constant
            if any(_reduce_by_subspace(rows, tuple(coordinates))):
                element = LieAlgebraElement.model_construct(
                    basis=algebra_value.basis,
                    coordinates=tuple(
                        CanonicalRational.from_fraction(value) for value in coordinates
                    ),
                )
                witness = IdealViolationWitness.model_construct(
                    basis_index=basis_index,
                    subspace_row=row_index,
                    bracket=element,
                )
                return LieIdealCheckResult._from_kernel(
                    algebra_value, candidate_value, False, witness
                )
    return LieIdealCheckResult._from_kernel(algebra_value, candidate_value, True, None)


def _require_ideal(
    algebra: FiniteDimensionalLieAlgebra, candidate: LieSubspace
) -> None:
    """Reject a non-ideal subspace before quotient construction."""

    decision = check_ideal(algebra, candidate)
    if not decision.is_ideal:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="lie_algebra.not_an_ideal",
            message="quotient construction requires an ideal subspace",
        )


def lie_quotient(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    ideal: LieSubspace | Mapping[str, Any],
    quotient_basis: tuple[str, ...] | list[str],
) -> LieQuotientResult:
    """Form the quotient algebra on cosets of an ideal subspace.

    Quotient basis labels attach in increasing free-column order of the
    ideal RREF; bracket constants are the ideal-reduced brackets of the
    corresponding basis vectors. Operation admission establishes every
    Jacobi identity and verifies ideal absorption first, so the coset
    bracket is well defined.
    """

    algebra_value = _as_algebra(algebra)
    ideal_value = (
        ideal if isinstance(ideal, LieSubspace) else LieSubspace.model_validate(ideal)
    )
    _admit_lie_algebra(algebra_value)
    if ideal_value.basis != algebra_value.basis:
        raise OperationDomainValidationError(
            location=("ideal",),
            code="lie_algebra.candidate_basis",
            message="the ideal subspace must use the algebra's ordered basis",
        )
    _require_ideal(algebra_value, ideal_value)
    dimension = len(algebra_value.basis)
    rows = _subspace_rows(ideal_value)
    pivots = set(_rref_pivots(rows))
    free = tuple(column for column in range(dimension) if column not in pivots)
    if not isinstance(quotient_basis, (tuple, list)) or any(
        not isinstance(label, str) for label in quotient_basis
    ):
        raise OperationDomainValidationError(
            location=("quotient_basis",),
            code="lie_algebra.quotient_labels",
            message="quotient basis labels must be unique strings",
        )
    labels = tuple(quotient_basis)
    if len(set(labels)) != len(labels):
        raise OperationDomainValidationError(
            location=("quotient_basis",),
            code="lie_algebra.quotient_labels",
            message="quotient basis labels must be unique strings",
        )
    if len(labels) != len(free) or not labels:
        raise OperationDomainValidationError(
            location=("quotient_basis",),
            code="lie_algebra.quotient_dimension",
            message="quotient labels must number dimension minus ideal dimension",
        )
    table = _bracket_table(algebra_value)
    rank = {column: position for position, column in enumerate(free)}
    constants: list[StructureConstant] = []
    for left in range(len(free)):
        for right in range(left + 1, len(free)):
            coordinates = [Fraction(0)] * dimension
            for target, constant in table.get((free[left], free[right]), {}).items():
                coordinates[target] = constant
            reduced = _reduce_by_subspace(rows, tuple(coordinates))
            for target in free:
                value = reduced[target]
                if value != 0:
                    constants.append(
                        StructureConstant.model_construct(
                            i=left,
                            j=right,
                            k=rank[target],
                            coefficient=CanonicalRational.from_fraction(value),
                        )
                    )
    quotient = FiniteDimensionalLieAlgebra(
        basis=labels,
        structure_constants=tuple(
            sorted(constants, key=lambda constant: (constant.i, constant.j, constant.k))
        ),
    )
    return LieQuotientResult._from_kernel(algebra_value, ideal_value, quotient)


def lie_direct_sum(
    left: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    right: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    basis: tuple[str, ...] | list[str],
) -> FiniteDimensionalLieAlgebra:
    """Assemble the direct sum on concatenated bases (native-only).

    The left block keeps its indices and the right block shifts past
    the left dimension; brackets     across blocks vanish. Both summands
    pass Jacobi admission, so the sum is a Lie algebra.
    """

    left_value = _as_algebra(left)
    right_value = _as_algebra(right)
    _admit_lie_algebra(left_value)
    _admit_lie_algebra(right_value)
    left_dimension = len(left_value.basis)
    total = left_dimension + len(right_value.basis)
    if not isinstance(basis, (tuple, list)):
        raise OperationDomainValidationError(
            location=("basis",),
            code="lie_algebra.direct_sum_basis",
            message="direct-sum labels must be unique across both dimensions",
        )
    labels = tuple(basis)
    if (
        any(not isinstance(label, str) for label in labels)
        or len(labels) != total
        or len(set(labels)) != total
        or total > MAX_LIE_DIMENSION
    ):
        raise OperationDomainValidationError(
            location=("basis",),
            code="lie_algebra.direct_sum_basis",
            message="direct-sum labels must be unique across both dimensions",
        )
    constants = tuple(left_value.structure_constants) + tuple(
        StructureConstant.model_construct(
            i=constant.i + left_dimension,
            j=constant.j + left_dimension,
            k=constant.k + left_dimension,
            coefficient=constant.coefficient,
        )
        for constant in right_value.structure_constants
    )
    return FiniteDimensionalLieAlgebra(
        basis=labels,
        structure_constants=tuple(
            sorted(constants, key=lambda constant: (constant.i, constant.j, constant.k))
        ),
    )


def lie_adjoint_matrices(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> tuple[RationalMatrix, ...]:
    """Return the adjoint matrices ``ad_{b_i}`` on column vectors.

    Native-only projection of the structure constants: column ``j`` of
    ``ad_i`` holds the coordinates of ``[b_i, b_j]``. Operation
    admission establishes every Jacobi identity first, so the matrices
    represent a Lie algebra adjoint action.
    """

    algebra_value = _as_algebra(algebra)
    _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    return tuple(
        RationalMatrix(
            row_count=dimension,
            column_count=dimension,
            entries=tuple(
                tuple(CanonicalRational.from_fraction(value) for value in row)
                for row in matrix
            ),
        )
        for matrix in _adjoint_matrices(algebra_value)
    )


__all__ = [
    "check_ideal",
    "lie_adjoint_matrices",
    "lie_bracket",
    "lie_center",
    "lie_derived_series",
    "lie_direct_sum",
    "lie_killing_form",
    "lie_lower_central_series",
    "lie_quotient",
]
