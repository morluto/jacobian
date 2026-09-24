"""Native exact Lie-bracket operations over QQ."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian.canonical import decimal_digit_width
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lie_algebras._models import (
    MAX_BRACKET_RESULT_COEFFICIENT_DIGITS,
    MAX_BRACKET_WORK,
    MAX_CENTRALIZER_WORK,
    MAX_ELEMENT_COEFFICIENT_DIGITS,
    MAX_GENERATED_MATRIX_DECIMAL_DIGITS,
    MAX_LIE_DIMENSION,
    MAX_STRUCTURE_COEFFICIENT_DIGITS,
    MAX_STRUCTURE_NONZEROS,
    MAX_SUBALGEBRA_CHECK_WORK,
    MAX_UPPER_CENTRAL_RESULT_COEFFICIENTS,
    MAX_UPPER_CENTRAL_WORK,
    BracketPairContribution,
    FiniteDimensionalLieAlgebra,
    IdealViolationWitness,
    LieAdjointRepresentationResult,
    LieAdjointResult,
    LieAlgebraElement,
    LieBracketResult,
    LieCenterResult,
    LieCentralizerResult,
    LieDerivedSeriesResult,
    LieIdeal,
    LieIdealCheckResult,
    LieKillingRadicalResult,
    LieKillingResult,
    LieLowerCentralSeriesResult,
    LieQuotientResult,
    LieSubalgebra,
    LieSubalgebraCheckResult,
    LieSubalgebraViolationWitness,
    LieSubspace,
    LieUpperCentralSeriesResult,
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


def _rref_height_bound(dimension: int, entry_digits: int) -> int:
    """Bound rational RREF entries through row denominators and minors."""
    return dimension**2 * entry_digits + len(str(factorial(dimension)))


def _bracket_height_bound(
    term_count: int,
    structure_digits: int,
    entry_digits: int,
) -> int:
    """Bound one bracket-coordinate sum of rational products."""
    if term_count == 0:
        return 1
    return term_count * (2 * entry_digits + structure_digits) + len(str(term_count))


def _check_generated_matrix_output_bound(
    *, entries: int, entry_digits: int, location: tuple[str | int, ...]
) -> None:
    """Admit canonical matrix size before constructing exact matrices."""
    if entries * entry_digits > MAX_GENERATED_MATRIX_DECIMAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=location,
            code="lie_algebra.generated_subalgebra_output_bound",
            message="generated-subalgebra intermediate matrix exceeds the admitted canonical entry-count times digit-width budget",
        )


def _admit_generator_family(
    algebra_value: FiniteDimensionalLieAlgebra,
    generators_value: tuple[LieAlgebraElement, ...],
    *,
    coordinate_label: str,
    family_bound_code: str,
    family_bound_message: str,
    basis_code: str,
    basis_message: str,
) -> None:
    """Admit a generated closure's generator family, axes, and digit heights."""
    if len(generators_value) > MAX_LIE_DIMENSION:
        raise OperationDomainValidationError(
            location=("generators",),
            code=family_bound_code,
            message=family_bound_message,
        )
    for generator_index, generator in enumerate(generators_value):
        if generator.basis != algebra_value.basis:
            raise OperationDomainValidationError(
                location=("generators", generator_index),
                code=basis_code,
                message=basis_message,
            )
        for coordinate_index, coordinate in enumerate(generator.coordinates):
            _run_admission(
                lambda coordinate=coordinate: require_bounded_rational(
                    coordinate,
                    max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                    label=coordinate_label,
                ),
                location=(
                    "generators",
                    generator_index,
                    "coordinates",
                    coordinate_index,
                ),
            )


def _admit_generated_closure_height(
    *,
    dimension: int,
    rank_lower_bound: int,
    initial_height: int,
    term_count: int,
    structure_digits: int,
) -> None:
    """Admit every bracket/RREF pass, including the terminal closure pass."""
    max_digits = MAX_BRACKET_RESULT_COEFFICIENT_DIGITS
    entry_height = initial_height
    if term_count and rank_lower_bound > 1:
        for _ in range(dimension - rank_lower_bound + 1):
            bracket_height = _bracket_height_bound(
                term_count, structure_digits, entry_height
            )
            _check_generated_matrix_output_bound(
                entries=dimension**3,
                entry_digits=bracket_height,
                location=("generators", "brackets"),
            )
            combined_height = _rref_height_bound(
                dimension, max(entry_height, bracket_height)
            )
            if max(bracket_height, combined_height) > max_digits:
                raise OperationResourceAdmissionError(
                    location=("generators",),
                    code="lie_algebra.generated_subalgebra_height_bound",
                    message="iterated bracket and RREF coefficient-growth bound exceeds the admitted exact bound",
                )
            _check_generated_matrix_output_bound(
                entries=dimension**2,
                entry_digits=combined_height,
                location=("result", "generators"),
            )
            entry_height = combined_height


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


def lie_killing_form_radical(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieKillingRadicalResult:
    """Compute the radical of the Killing bilinear form over QQ.

    The result is the exact nullspace of the Killing matrix in the source
    algebra basis. It makes no claim that this subspace is the solvable
    radical.
    """

    from jacobian.math.matrices.operations import nullspace_result, rref_result

    killing = lie_killing_form(algebra)
    nullspace = nullspace_result(killing.killing_form)
    generators = (
        RationalMatrix(
            row_count=0,
            column_count=len(killing.algebra.basis),
            entries=(),
        )
        if nullspace.nullity == 0
        else rref_result(nullspace.basis_matrix).reduced_matrix
    )
    radical = LieSubspace(basis=killing.algebra.basis, generators=generators)
    return LieKillingRadicalResult._from_kernel(killing, radical)


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


def lie_subalgebra_centralizer(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    elements: tuple[LieAlgebraElement, ...] | list[LieAlgebraElement],
) -> LieCentralizerResult:
    """Compute the common centralizer of a finite family in a Lie algebra.

    The unknown vector x is constrained by [x, s]=0 for each supplied s.
    The stacked adjoint rows give the exact kernel, canonicalized as RREF
    coordinates on the source algebra axis.
    """

    from jacobian.math.matrices.operations import nullspace_result, rref_result

    algebra_value = _as_algebra(algebra)
    elements_value = tuple(_as_element(element) for element in elements)
    if len(elements_value) > len(algebra_value.basis):
        raise OperationDomainValidationError(
            location=("elements",),
            code="lie_algebra.centralizer_family_bound",
            message="the centralizer family cannot exceed the ambient dimension",
        )
    for element_index, element in enumerate(elements_value):
        if element.basis != algebra_value.basis:
            raise OperationDomainValidationError(
                location=("elements", element_index),
                code="lie_algebra.centralizer_element_basis",
                message="centralizer elements must use the algebra's ordered basis",
            )
    table = _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    for element_index, element in enumerate(elements_value):
        for coordinate_index, coordinate_value in enumerate(element.coordinates):
            _run_admission(
                lambda coordinate=coordinate_value: require_bounded_rational(
                    coordinate,
                    max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                    label="centralizer element coordinate",
                ),
                location=("elements", element_index, "coordinates", coordinate_index),
            )
    # Each supplied vector contributes n equations with n coefficients. The
    # first term bounds sparse structure-table scans, the second covers both
    # the entry-height admission pass and exact row construction, and the
    # final term bounds reduction of the n^2-by-n system.
    work = (
        len(elements_value) * dimension * MAX_STRUCTURE_NONZEROS
        + 2 * (len(elements_value) * dimension) * dimension**2
        + dimension**4
    )
    if work > MAX_CENTRALIZER_WORK:
        raise OperationResourceAdmissionError(
            location=("elements",),
            code="lie_algebra.centralizer_work_bound",
            message="centralizer linear-system work exceeds its admitted bound",
        )

    # Bound every stacked entry before building the exact row tuples. A term
    # is the product of one element coordinate and one structure constant;
    # summing t rational terms uses at most the sum of their denominator
    # digit widths, and the common-denominator numerator is bounded by the
    # largest scaled numerator plus ceil(log10(t)).
    largest_entry_height_bound = 1
    for element in elements_value:
        for target in range(dimension):
            for coordinate in range(dimension):
                term_heights: list[tuple[int, int]] = []
                for other in range(dimension):
                    vector_coordinate = element.coordinates[other].as_fraction()
                    structure_coefficient = table.get((coordinate, other), {}).get(
                        target
                    )
                    if not vector_coordinate or not structure_coefficient:
                        continue
                    term_heights.append(
                        (
                            decimal_digit_width(abs(vector_coordinate.numerator))
                            + decimal_digit_width(abs(structure_coefficient.numerator)),
                            decimal_digit_width(vector_coordinate.denominator)
                            + decimal_digit_width(structure_coefficient.denominator),
                        )
                    )
                if term_heights:
                    denominator_digits = sum(
                        denominator for _, denominator in term_heights
                    )
                    numerator_digits = max(
                        numerator + denominator_digits - denominator
                        for numerator, denominator in term_heights
                    ) + (
                        len(str(len(term_heights) - 1)) if len(term_heights) > 1 else 0
                    )
                    largest_entry_height_bound = max(
                        largest_entry_height_bound,
                        numerator_digits,
                        denominator_digits,
                    )
    # Clearing at most n row denominators makes each integer matrix entry
    # have at most (n+1)D digits. Every nullspace coordinate is a ratio of
    # minors of order at most n, bounded by n! products of n such entries.
    estimated_result_digit_bound = (
        dimension * (dimension + 1) * largest_entry_height_bound + dimension**2
    )
    if estimated_result_digit_bound > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("elements",),
            code="lie_algebra.centralizer_result_height_bound",
            message=(
                "the preflight exact nullspace coefficient-height bound "
                f"{estimated_result_digit_bound} exceeds "
                f"{MAX_CANONICAL_RATIONAL_DIGITS} digits"
            ),
        )
    rows = tuple(
        tuple(
            sum(
                (
                    element.coordinates[other].as_fraction()
                    * table.get((coordinate, other), {}).get(target, Fraction(0))
                    for other in range(dimension)
                ),
                start=Fraction(0),
            )
            for coordinate in range(dimension)
        )
        for element in elements_value
        for target in range(dimension)
    )
    matrix = rational_matrix_from_fractions(rows, column_count=dimension)
    nullspace = nullspace_result(matrix)
    generators = (
        RationalMatrix(row_count=0, column_count=dimension, entries=())
        if nullspace.nullity == 0
        else rref_result(nullspace.basis_matrix).reduced_matrix
    )
    centralizer = LieSubalgebra.model_construct(
        algebra=algebra_value,
        basis=algebra_value.basis,
        generators=generators,
    )
    return LieCentralizerResult._from_kernel(algebra_value, centralizer)


def lie_generated_subalgebra(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    generators: tuple[LieAlgebraElement, ...] | list[LieAlgebraElement],
) -> LieSubalgebra:
    """Return the smallest Lie subalgebra containing the supplied vectors.

    The finite closure is computed by adjoining brackets of the current RREF
    basis until its span stops growing. A conservative word-height estimate,
    generator count, and the dimension-derived closure work bound are admitted
    before bracket expansion.
    """
    from jacobian.math.matrices.operations import rref_result

    algebra_value = _as_algebra(algebra)
    generators_value = tuple(_as_element(generator) for generator in generators)
    _admit_generator_family(
        algebra_value,
        generators_value,
        coordinate_label="subalgebra generator coordinate",
        family_bound_code="lie_algebra.generated_subalgebra_generator_family_bound",
        family_bound_message=(
            "the generated-subalgebra generator family exceeds the admitted "
            f"{MAX_LIE_DIMENSION}-vector bound"
        ),
        basis_code="lie_algebra.generated_subalgebra_generator_basis",
        basis_message="generators must use the algebra's ordered basis",
    )
    dimension = len(algebra_value.basis)
    # Bound every pairwise bracket, each RREF in a closure round, and
    # every possible strict dimension increase plus the final closure check.
    closure_work = (dimension + 1) * (
        dimension**2 * len(algebra_value.structure_constants)
        + dimension**4
        + dimension**3
    )
    jacobi_work = 3 * dimension**5
    generator_rref_work = len(generators_value) * dimension**2 + dimension**3
    if (
        closure_work + generator_rref_work + jacobi_work
        > MAX_SUBALGEBRA_CHECK_WORK * 20
    ):
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="lie_algebra.generated_subalgebra_work_bound",
            message="generated-subalgebra closure exceeds its admitted work bound",
        )
    table = _admit_lie_algebra(algebra_value)
    source_digits = max(
        (
            max(decimal_digit_width(v.numerator), decimal_digit_width(v.denominator))
            for c in algebra_value.structure_constants
            for v in (c.coefficient.as_fraction(),)
        ),
        default=1,
    )
    source_digits = max(
        source_digits,
        max(
            (
                decimal_digit_width(v.numerator)
                for g in generators_value
                for c in g.coordinates
                for v in (c.as_fraction(),)
            ),
            default=1,
        ),
        max(
            (
                decimal_digit_width(v.denominator)
                for g in generators_value
                for c in g.coordinates
                for v in (c.as_fraction(),)
            ),
            default=1,
        ),
    )
    max_digits = MAX_BRACKET_RESULT_COEFFICIENT_DIGITS
    structure_digits = max(
        (
            max(
                decimal_digit_width(abs(value.numerator)),
                decimal_digit_width(value.denominator),
            )
            for constant in algebra_value.structure_constants
            for value in (constant.coefficient.as_fraction(),)
        ),
        default=1,
    )
    term_count = min(dimension**2, len(algebra_value.structure_constants))
    initial_rref_height = _rref_height_bound(dimension, source_digits)
    if initial_rref_height > max_digits:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="lie_algebra.generated_subalgebra_height_bound",
            message="initial generator RREF coefficient-growth bound exceeds the admitted exact bound",
        )
    _check_generated_matrix_output_bound(
        entries=dimension**2,
        entry_digits=initial_rref_height,
        location=("generators", "rref"),
    )
    rows = tuple(
        tuple(value.as_fraction() for value in item.coordinates)
        for item in generators_value
    )
    if rows:
        reduced = rref_result(
            rational_matrix_from_fractions(rows, column_count=dimension)
        )
        rows = tuple(
            tuple(v.as_fraction() for v in row)
            for row in reduced.reduced_matrix.entries[: reduced.rank]
        )
    # The initial RREF above is admitted by the generator-work, height, and
    # output checks, so its canonical rank is known before closure. Each
    # closure round either strictly increases the rank or terminates, so at
    # most dimension - rank + 1 rounds of worst-case growth can occur.
    _admit_generated_closure_height(
        dimension=dimension,
        rank_lower_bound=len(rows),
        initial_height=initial_rref_height,
        term_count=term_count,
        structure_digits=structure_digits,
    )

    while len(rows) > 1:
        brackets = _subspace_bracket_vectors(rows, rows, table, dimension)
        if not brackets:
            break
        for row in brackets:
            for value in row:
                _require_fraction_bound(
                    value,
                    label="generated bracket coordinate",
                    location=("generators",),
                )
        combined = rows + brackets
        reduced = rref_result(
            rational_matrix_from_fractions(combined, column_count=dimension)
        )
        next_rows = tuple(
            tuple(v.as_fraction() for v in row)
            for row in reduced.reduced_matrix.entries[: reduced.rank]
        )
        for row in next_rows:
            for value in row:
                _require_fraction_bound(
                    value,
                    label="generated subalgebra coordinate",
                    location=("result", "generators"),
                )
        if next_rows == rows:
            break
        rows = next_rows
    return LieSubalgebra.model_construct(
        algebra=algebra_value,
        basis=algebra_value.basis,
        generators=_subspace_value(algebra_value, rows).generators,
    )


def lie_generated_ideal(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    generators: tuple[LieAlgebraElement, ...] | list[LieAlgebraElement],
) -> LieIdeal:
    """Return the smallest ambient ideal containing the supplied vectors.

    The kernel repeatedly adjoins brackets of every ambient basis vector with
    the current RREF rows. Dimension and total work are admitted upfront, and
    each bracket or RREF stage admits its own coefficient growth and canonical
    matrix size from the actual incoming heights before that stage expands.
    """
    from jacobian.math.matrices.operations import rref_result

    algebra_value = _as_algebra(algebra)
    generators_value = tuple(_as_element(generator) for generator in generators)
    _admit_generator_family(
        algebra_value,
        generators_value,
        coordinate_label="ideal generator coordinate",
        family_bound_code="lie_algebra.generated_ideal_generator_family_bound",
        family_bound_message=(
            "the generated-ideal generator family exceeds the admitted "
            f"{MAX_LIE_DIMENSION}-vector bound"
        ),
        basis_code="lie_algebra.generated_ideal_generator_basis",
        basis_message="ideal generators must use the algebra's ordered basis",
    )
    dimension = len(algebra_value.basis)

    max_digits = MAX_CANONICAL_RATIONAL_DIGITS
    work = (
        len(generators_value) * dimension**2
        + dimension**3
        + 3 * dimension**5
        + (dimension + 1)
        * (
            dimension**2 * len(algebra_value.structure_constants)
            + dimension**4
            + dimension**3
        )
    )
    if work > MAX_SUBALGEBRA_CHECK_WORK * 20:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="lie_algebra.generated_ideal_work_bound",
            message="generated-ideal closure exceeds its admitted exact-work bound",
        )
    # No static envelope is admitted here: every matrix this closure can
    # construct (the initial generator RREF and each bracket/RREF round) is
    # admitted below from the input's actual coefficient heights before it
    # is built, and an empty generator family never expands at all.

    structure = tuple(
        item.coefficient.as_fraction() for item in algebra_value.structure_constants
    )
    structure_digits = max(
        (
            max(
                decimal_digit_width(abs(value.numerator)),
                decimal_digit_width(value.denominator),
            )
            for value in structure
        ),
        default=1,
    )

    table = _admit_lie_algebra(algebra_value)
    rows = tuple(
        tuple(value.as_fraction() for value in item.coordinates)
        for item in generators_value
    )
    if rows:
        source_digits = max(
            (
                max(
                    decimal_digit_width(abs(value.numerator)),
                    decimal_digit_width(value.denominator),
                )
                for row in rows
                for value in row
            ),
            default=1,
        )
        initial_height = _rref_height_bound(dimension, source_digits)
        initial_digits = dimension**2 * initial_height
        if (
            initial_height > max_digits
            or initial_digits > MAX_GENERATED_MATRIX_DECIMAL_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("generators",),
                code="lie_algebra.generated_ideal_height_bound",
                message="the initial ideal generator span exceeds its admitted RREF bound",
            )
        reduced = rref_result(
            rational_matrix_from_fractions(rows, column_count=dimension)
        )
        rows = tuple(
            tuple(value.as_fraction() for value in row)
            for row in reduced.reduced_matrix.entries[: reduced.rank]
        )

    ambient_basis = _identity_rows(dimension)
    while rows and len(rows) < dimension:
        current_height = max(
            (
                max(
                    decimal_digit_width(abs(value.numerator)),
                    decimal_digit_width(value.denominator),
                )
                for row in rows
                for value in row
            ),
            default=1,
        )
        bracket_height = _bracket_height_bound(
            min(dimension, len(structure)), structure_digits, current_height
        )
        bracket_digits = dimension**3 * bracket_height
        if (
            bracket_height > max_digits
            or bracket_digits > MAX_GENERATED_MATRIX_DECIMAL_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("generators",),
                code="lie_algebra.generated_ideal_height_bound",
                message="the next ambient bracket expansion exceeds its admitted exact bound",
            )
        brackets = _subspace_bracket_vectors(ambient_basis, rows, table, dimension)
        if not brackets:
            break
        rref_height = _rref_height_bound(dimension, max(current_height, bracket_height))
        rref_digits = (dimension**3 + dimension**2) * rref_height
        if (
            rref_height > max_digits
            or rref_digits > MAX_GENERATED_MATRIX_DECIMAL_DIGITS
        ):
            raise OperationResourceAdmissionError(
                location=("generators",),
                code="lie_algebra.generated_ideal_height_bound",
                message="the next ideal RREF expansion exceeds its admitted exact bound",
            )
        reduced = rref_result(
            rational_matrix_from_fractions(rows + brackets, column_count=dimension)
        )
        next_rows = tuple(
            tuple(value.as_fraction() for value in row)
            for row in reduced.reduced_matrix.entries[: reduced.rank]
        )
        if next_rows == rows:
            break
        rows = next_rows

    subspace = _subspace_value(algebra_value, rows)
    return LieIdeal.model_construct(
        algebra=algebra_value,
        basis=algebra_value.basis,
        generators=subspace.generators,
    )


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

    vectors = _subspace_bracket_vectors(left, right, table, dimension)
    if not vectors:
        return ()
    from jacobian.math.matrices.operations import rref_result

    reduced = rref_result(
        rational_matrix_from_fractions(vectors, column_count=dimension)
    )
    return tuple(
        tuple(value.as_fraction() for value in row)
        for row in reduced.reduced_matrix.entries[: reduced.rank]
    )


def _subspace_bracket_vectors(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    table: dict[tuple[int, int], dict[int, Fraction]],
    dimension: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Expand spanning-row pairs into exact bracket coordinate vectors."""
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
    return tuple(vectors)


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


def lie_derived_subalgebra(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieIdeal:
    """Return the exact derived ideal ``[g, g]`` in the ambient basis."""

    algebra_value = _as_algebra(algebra)
    table = _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    rows = _subspace_bracket_rows(
        _identity_rows(dimension),
        _identity_rows(dimension),
        table,
        dimension,
    )
    subspace = _subspace_value(algebra_value, rows)
    return LieIdeal.model_construct(
        algebra=algebra_value,
        basis=algebra_value.basis,
        generators=subspace.generators,
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


def _upper_central_successor_rows(
    rows: tuple[tuple[Fraction, ...], ...],
    table: dict[tuple[int, int], dict[int, Fraction]],
    dimension: int,
) -> tuple[tuple[Fraction, ...], ...]:
    """Compute the preimage of the center in the quotient by ``span(rows)``."""
    from jacobian.math.matrices.operations import nullspace_result, rref_result

    pivots = set(_rref_pivots(rows))
    free = tuple(column for column in range(dimension) if column not in pivots)
    reduced_brackets = tuple(
        tuple(
            _reduce_by_subspace(
                rows,
                tuple(
                    table.get((coordinate, basis_index), {}).get(target, Fraction(0))
                    for target in range(dimension)
                ),
            )
            for coordinate in range(dimension)
        )
        for basis_index in range(dimension)
    )
    constraints = []
    # For every basis vector b_j, require [x,b_j] to reduce to zero modulo Z.
    # The free coordinates after RREF reduction are canonical quotient coordinates.
    for basis_index in range(dimension):
        for quotient_coordinate in free:
            constraints.append(
                tuple(
                    reduced_brackets[basis_index][coordinate][quotient_coordinate]
                    for coordinate in range(dimension)
                )
            )
    if not constraints:
        return _identity_rows(dimension)
    matrix = rational_matrix_from_fractions(tuple(constraints), column_count=dimension)
    nullspace = nullspace_result(matrix)
    if nullspace.nullity == 0:
        return rows
    combined = rows + tuple(
        tuple(value.as_fraction() for value in row)
        for row in nullspace.basis_matrix.entries
    )
    reduced = rref_result(
        rational_matrix_from_fractions(combined, column_count=dimension)
    )
    return tuple(
        tuple(value.as_fraction() for value in row)
        for row in reduced.reduced_matrix.entries[: reduced.rank]
    )


def lie_upper_central_series(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieUpperCentralSeriesResult:
    """Compute Z_0=0 and Z_(i+1)/Z_i = center(L/Z_i) exactly over QQ.

    The chain stops at L for nilpotent algebras, or at the first stable proper
    term otherwise. Every term is an RREF subspace on the original ordered basis.
    """
    algebra_value = _as_algebra(algebra)
    table = _admit_lie_algebra(algebra_value)
    dimension = len(algebra_value.basis)
    # At most d strict dimension increases occur. Each stage performs at most
    # d^2 quotient-coordinate rows by d columns over the admitted dimension.
    max_work = (dimension + 1) * dimension**4
    max_result_coefficients = (dimension + 1) * dimension**2
    if max_work > MAX_UPPER_CENTRAL_WORK:
        raise OperationResourceAdmissionError(
            location=("algebra", "basis"),
            code="lie_algebra.upper_central_work_bound",
            message="upper central series exceeds its admitted exact-work bound",
        )
    if max_result_coefficients > MAX_UPPER_CENTRAL_RESULT_COEFFICIENTS:
        raise OperationResourceAdmissionError(
            location=("algebra", "basis"),
            code="lie_algebra.upper_central_result_bound",
            message="upper central series exceeds its admitted result-size bound",
        )
    rows: tuple[tuple[Fraction, ...], ...] = ()
    terms = [_subspace_value(algebra_value, rows)]
    while len(rows) < dimension:
        successor = _upper_central_successor_rows(rows, table, dimension)
        if successor == rows:
            break
        for row in successor:
            for value in row:
                _require_fraction_bound(
                    value,
                    label="upper central series coefficient",
                    location=("terms", len(terms)),
                )
        rows = successor
        terms.append(_subspace_value(algebra_value, rows))
    return LieUpperCentralSeriesResult._from_kernel(
        algebra_value, tuple(terms), nilpotent=len(rows) == dimension
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


def _as_subspace_candidate(
    candidate: LieSubspace | Mapping[str, Any],
) -> tuple[LieSubspace, FiniteDimensionalLieAlgebra | None]:
    if isinstance(candidate, (LieSubalgebra, LieIdeal)):
        return (
            LieSubspace.model_construct(
                basis=candidate.basis, generators=candidate.generators
            ),
            candidate.algebra,
        )
    if isinstance(candidate, Mapping) and "algebra" in candidate:
        # LieIdeal and LieSubalgebra have the same structural fields. Parsing
        # either as LieIdeal establishes its ambient-axis binding; consumers
        # below still establish whichever closure property they need.
        source_bound = LieIdeal.model_validate(candidate)
        return (
            LieSubspace.model_construct(
                basis=source_bound.basis, generators=source_bound.generators
            ),
            source_bound.algebra,
        )
    if isinstance(candidate, LieSubspace):
        return LieSubspace.model_validate(candidate.model_dump()), None
    return LieSubspace.model_validate(candidate), None


def _require_candidate_source(
    source: FiniteDimensionalLieAlgebra | None,
    algebra: FiniteDimensionalLieAlgebra,
    *,
    location: tuple[str | int, ...],
    code: str,
    label: str,
) -> None:
    if source is not None and source != algebra:
        raise OperationDomainValidationError(
            location=location,
            code=code,
            message=f"a source-bound {label} must retain the exact ambient algebra",
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
    candidate_value, candidate_source = _as_subspace_candidate(candidate)
    _require_candidate_source(
        candidate_source,
        algebra_value,
        location=("candidate",),
        code="lie_algebra.candidate_source",
        label="candidate",
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


def check_subalgebra(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    candidate: LieSubspace | Mapping[str, Any],
) -> LieSubalgebraCheckResult:
    """Decide whether every bracket of candidate basis rows stays inside it.

    Bilinearity makes the generator-pair check sufficient. The operation
    returns the first escaping bracket in lexicographic row order as a
    replayable witness.
    """

    algebra_value = _as_algebra(algebra)
    try:
        candidate_value, candidate_source = _as_subspace_candidate(candidate)
    except Exception as error:
        raise OperationDomainValidationError(
            location=("candidate",),
            code="lie_algebra.subspace_shape",
            message="candidate must be a canonical rational subspace",
        ) from error
    _require_candidate_source(
        candidate_source,
        algebra_value,
        location=("candidate",),
        code="lie_algebra.candidate_source",
        label="candidate",
    )
    table = _admit_lie_algebra(algebra_value)
    if candidate_value.basis != algebra_value.basis:
        raise OperationDomainValidationError(
            location=("candidate",),
            code="lie_algebra.candidate_basis",
            message="the candidate subspace must use the algebra's ordered basis",
        )
    rows = _subspace_rows(candidate_value)
    dimension = len(algebra_value.basis)
    for row_index, row in enumerate(rows):
        for coordinate_index, coordinate in enumerate(row):
            try:
                require_bounded_rational(
                    CanonicalRational.from_fraction(coordinate),
                    max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                    label="subspace generator coordinate",
                )
            except ValueError as error:
                raise OperationResourceAdmissionError(
                    location=("candidate", "generators", row_index, coordinate_index),
                    code="lie_algebra.subalgebra_input_bound",
                    message=str(error),
                ) from error
    pair_count = len(rows) * (len(rows) - 1) // 2
    work = pair_count * (dimension * (dimension - 1) // 2 + len(table))
    if work > MAX_SUBALGEBRA_CHECK_WORK:
        raise OperationResourceAdmissionError(
            location=("candidate",),
            code="lie_algebra.subalgebra_work_bound",
            message="subalgebra closure work exceeds its admitted exact envelope",
        )
    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            coordinates = [Fraction(0)] * dimension
            for first in range(dimension):
                for second in range(first + 1, dimension):
                    factor = left[first] * right[second] - left[second] * right[first]
                    if not factor:
                        continue
                    for target, structure_constant in table.get(
                        (first, second), {}
                    ).items():
                        coordinates[target] += factor * structure_constant
            for index, value in enumerate(coordinates):
                _require_fraction_bound(
                    value,
                    label="subalgebra bracket coordinate",
                    location=("candidate", "bracket", index),
                )
            if any(_reduce_by_subspace(rows, tuple(coordinates))):
                bracket = LieAlgebraElement.model_construct(
                    basis=algebra_value.basis,
                    coordinates=tuple(
                        CanonicalRational.from_fraction(value) for value in coordinates
                    ),
                )
                witness = LieSubalgebraViolationWitness.model_construct(
                    left_row=left_index,
                    right_row=right_index,
                    bracket=bracket,
                )
                return LieSubalgebraCheckResult._from_kernel(
                    algebra_value, candidate_value, False, witness
                )
    return LieSubalgebraCheckResult._from_kernel(
        algebra_value, candidate_value, True, None
    )


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
    ideal_value, ideal_source = _as_subspace_candidate(ideal)
    _require_candidate_source(
        ideal_source,
        algebra_value,
        location=("ideal",),
        code="lie_algebra.quotient_binding",
        label="ideal",
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


def lie_adjoint_representation(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
) -> LieAdjointRepresentationResult:
    """Return exact adjoint matrices together with their ordered source axis."""

    algebra_value = _as_algebra(algebra)
    matrices = lie_adjoint_matrices(algebra_value)
    return LieAdjointRepresentationResult._from_kernel(algebra_value, matrices)


def lie_adjoint(
    algebra: FiniteDimensionalLieAlgebra | Mapping[str, Any],
    element: LieAlgebraElement | Mapping[str, Any],
) -> LieAdjointResult:
    """Return the exact matrix of the element's adjoint action."""
    algebra_value = _as_algebra(algebra)
    element_value = _as_element(element)
    table = _admit_lie_algebra(algebra_value)
    if element_value.basis != algebra_value.basis:
        raise OperationDomainValidationError(
            location=("element", "basis"),
            code="lie_algebra.adjoint_element_basis",
            message="the adjoint element must use the algebra's ordered basis",
        )

    coordinate_digits = 1
    for index, coordinate in enumerate(element_value.coordinates):
        _run_admission(
            lambda coordinate=coordinate: require_bounded_rational(
                coordinate,
                max_digits=MAX_ELEMENT_COEFFICIENT_DIGITS,
                label="adjoint-element coordinate",
            ),
            location=("element", "coordinates", index),
        )
        coordinate_digits = max(
            coordinate_digits,
            decimal_digit_width(coordinate.as_fraction().numerator),
            decimal_digit_width(coordinate.as_fraction().denominator),
        )

    dimension = len(algebra_value.basis)
    structure_digits = max(
        (
            max(
                decimal_digit_width(constant.coefficient.as_fraction().numerator),
                decimal_digit_width(constant.coefficient.as_fraction().denominator),
            )
            for constant in algebra_value.structure_constants
        ),
        default=1,
    )
    # A matrix entry is a sum of at most n products x_i*c_ij^k. This
    # denominator-product bound also covers unreduced intermediate sums.
    result_digits = (
        dimension * (coordinate_digits + structure_digits) + len(str(dimension)) + 2
    )
    if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("result", "matrix"),
            code="lie_algebra.adjoint_height_bound",
            message="the adjoint matrix coefficient-growth bound exceeds the exact rational limit",
        )
    estimated_digits = dimension * dimension * result_digits
    if estimated_digits > MAX_GENERATED_MATRIX_DECIMAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("result", "matrix"),
            code="lie_algebra.adjoint_output_bound",
            message="the adjoint matrix exceeds the admitted canonical entry-count times digit-width budget",
        )
    work = dimension**3 + dimension * dimension * len(algebra_value.structure_constants)
    work_bound = MAX_LIE_DIMENSION**3 + MAX_LIE_DIMENSION**2 * MAX_STRUCTURE_NONZEROS
    if work > work_bound:
        raise OperationResourceAdmissionError(
            location=("result", "matrix"),
            code="lie_algebra.adjoint_work_bound",
            message="adjoint matrix construction exceeds the admitted arithmetic-work bound",
        )

    coordinates = tuple(value.as_fraction() for value in element_value.coordinates)
    matrix = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    for column in range(dimension):
        for basis_index, coefficient in enumerate(coordinates):
            for target, structure_value in table.get((basis_index, column), {}).items():
                matrix[target][column] += coefficient * structure_value
    exact_matrix = rational_matrix_from_fractions(tuple(tuple(row) for row in matrix))
    return LieAdjointResult._from_kernel(algebra_value, element_value, exact_matrix)


__all__ = [
    "check_ideal",
    "check_subalgebra",
    "lie_adjoint",
    "lie_adjoint_matrices",
    "lie_adjoint_representation",
    "lie_bracket",
    "lie_center",
    "lie_derived_series",
    "lie_direct_sum",
    "lie_killing_form",
    "lie_lower_central_series",
    "lie_quotient",
    "lie_upper_central_series",
]
