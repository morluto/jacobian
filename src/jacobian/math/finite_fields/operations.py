"""Exact operations on presentation-, parent-, and axis-bound finite-field values."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from itertools import combinations
from typing import Any, Literal

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._admission import (
    require_field,
    require_independent_basis,
)
from jacobian.math.finite_fields._fixed_subspace_process import (
    run_fixed_subspace_computation,
)
from jacobian.math.finite_fields._matrix_rank import compute_matrix_rank
from jacobian.math.finite_fields._matrix_rank_models import MatrixRankResult
from jacobian.math.finite_fields._models import (
    _MAX_DIRECTION_RANK_WORK,
    _MAX_PROJECTIVE_POINTS,
)
from jacobian.math.finite_fields.values import (
    MAX_ORBIT_DISTRIBUTION_COUNT_DIGITS,
    MAX_ORBIT_DISTRIBUTION_TOTAL_DIGITS,
    Axis,
    AxisBoundMatrix,
    CollisionResult,
    DirectionRankLedger,
    FiberPartition,
    FiniteDimensionalSubspace,
    FiniteFieldElement,
    FiniteFieldPresentation,
    FiniteLinearMap,
    FiniteMapTable,
    FinitePolynomial,
    FinitePolynomialMap,
    HomogeneousFixedSubspace,
    OrbitDistribution,
    PaleyTournamentResult,
    PermutationResult,
    PrimeFieldLinearAction,
    ProjectiveLine,
    ProjectivePoint,
    RankResult,
    _direction_rank_work,
    _fibers_for_table,
    _homogeneous_monomial_count,
    _orbit_counts,
)
from jacobian.math.graphs.directed._models import (
    MAX_DIRECTED_GRAPH_PARSE_EDGES,
    DirectedGraph,
)
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_ELIMINATION_WORK,
    MAX_PRIME_FIELD_MATRIX_AXIS,
    MAX_PRIME_FIELD_MATRIX_CELLS,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    _admit_prime,
)

_MAX_FINITE_MAP_WORK = 1_000_000
_MAX_PALEY_TOURNAMENT_WORK = 4_000_000
_PALEY_ORIENTATION: Literal["ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"] = (
    "ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"
)


def matrix_rank(matrix: AxisBoundMatrix) -> MatrixRankResult:
    """Return the exact rank certificate for an axis-bound finite-field matrix."""
    return compute_matrix_rank(matrix)


# Separate calibrated bound for the Python-level substitution loop in
# _induced_action_matrix. Each monomial substitution performs sum(exponents)
# calls to _multiply_by_linear_form, each iterating over at most
# monomial_count terms. The bound is monomial_count * degree * monomial_count
# per generator. FLINT elimination work and Python substitution work are not
# interchangeable: one billion FLINT matrix-rank cells finish in seconds, but
# one billion Python loop iterations take minutes.
_MAX_PYTHON_SUBSTITUTION_WORK = 10_000_000
_FIXED_SUBSPACE_WALL_SECONDS = 600.0


def _prime_exceeds_worker_json_limit(prime: int) -> bool:
    """Check the interpreter's decimal-integer JSON conversion boundary."""

    digit_limit = sys.get_int_max_str_digits()
    return digit_limit > 0 and prime >= 10**digit_limit


def _fixed_subspace_checkpoint(deadline: float | None, stage: str) -> None:
    request_checkpoint(stage)
    if deadline is not None and time.monotonic() >= deadline:
        raise OperationExecutionTimeoutError(
            f"finite-field fixed-subspace deadline expired {stage}"
        )


def _homogeneous_fixed_subspace_envelope(
    action: PrimeFieldLinearAction,
    degree: int,
    *,
    checkpoint: Callable[[str], None],
) -> int:
    checkpoint("before admission")
    if type(degree) is not int or degree < 0:
        raise OperationDomainValidationError(
            location=("degree",),
            code="finite_field.fixed_subspace_degree_bound",
            message="degree must be a nonnegative integer",
        )
    if _prime_exceeds_worker_json_limit(action.prime):
        raise OperationDomainValidationError(
            location=("action", "generator_matrices"),
            code="finite_field.linear_action_prime_serialization_bound",
            message="linear-action prime exceeds the worker JSON integer serialization bound",
        )
    variable_count = len(action.variable_axis.labels)
    generator_count = len(action.generator_matrices)
    monomial_count = _homogeneous_monomial_count(variable_count, degree)
    checkpoint("after shape admission")
    equation_rows = generator_count * monomial_count
    equation_entries = generator_count * monomial_count**2
    output_entries = monomial_count**2
    exponentiation_steps = max(1, degree.bit_length())
    expansion_work = (
        generator_count * exponentiation_steps
        if variable_count == 1
        else generator_count * max(1, degree) * variable_count * monomial_count**2
    )
    # One elimination solves the stacked fixed equations; the second puts the
    # returned nullspace rows into backend-independent canonical RREF form.
    elimination_work = (generator_count + 1) * monomial_count**3
    action_rank_work = generator_count * variable_count**3
    # The Python substitution loop does sum(exponents) calls to
    # _multiply_by_linear_form per monomial, each iterating over at most
    # monomial_count terms. Bound it separately from FLINT work.
    substitution_work = (
        generator_count * exponentiation_steps
        if variable_count == 1
        else generator_count * max(1, degree) * monomial_count**2
    )
    if equation_rows > MAX_PRIME_FIELD_MATRIX_AXIS:
        raise OperationDomainValidationError(
            location=("action", "generator_matrices"),
            code="finite_field.fixed_subspace_equation_axis_bound",
            message="stacked fixed-subspace equations exceed the matrix axis bound",
        )
    if (
        equation_entries > MAX_PRIME_FIELD_MATRIX_CELLS
        or output_entries > MAX_PRIME_FIELD_MATRIX_CELLS
    ):
        raise OperationDomainValidationError(
            location=("action",),
            code="finite_field.fixed_subspace_matrix_bound",
            message="fixed-subspace equation or result matrix exceeds its cell bound",
        )
    if substitution_work > _MAX_PYTHON_SUBSTITUTION_WORK:
        raise OperationDomainValidationError(
            location=("degree",),
            code="finite_field.fixed_subspace_substitution_bound",
            message="Python substitution loop exceeds the separately calibrated expansion bound",
        )
    work = expansion_work + elimination_work + action_rank_work
    if work > MAX_PRIME_FIELD_ELIMINATION_WORK:
        raise OperationDomainValidationError(
            location=("action",),
            code="finite_field.fixed_subspace_work_bound",
            message="homogeneous fixed-subspace computation exceeds its work bound",
        )
    checkpoint("after result admission")
    return monomial_count


def finite_field(
    characteristic: int,
    modulus_coefficients: tuple[int, ...],
    *,
    generator: str = "a",
) -> FiniteFieldPresentation:
    """Construct and validate an exact finite-extension presentation."""

    presentation = FiniteFieldPresentation(
        characteristic=characteristic,
        modulus_coefficients=modulus_coefficients,
        generator=generator,
    )
    require_field(presentation)
    return presentation


def element(
    presentation: FiniteFieldPresentation,
    coordinates: tuple[int, ...],
) -> FiniteFieldElement:
    """Construct one parent-bound element from canonical power-basis coordinates."""

    require_field(presentation)
    return FiniteFieldElement(presentation=presentation, coordinates=coordinates)


def projective_point(
    presentation: FiniteFieldPresentation,
    axis: Axis,
    coordinates: tuple[FiniteFieldElement, ...],
) -> ProjectivePoint:
    """Normalize nonzero homogeneous coordinates by their first nonzero entry."""

    if len(coordinates) != len(axis.labels):
        raise ValueError("projective coordinates must match their axis")
    if any(value.presentation != presentation for value in coordinates):
        raise ValueError("projective coordinates must share their presentation")
    from jacobian.math.finite_fields import _sympy

    normalized = _sympy.normalize_projective_coordinates(presentation, coordinates)
    return ProjectivePoint(
        presentation=presentation,
        axis=axis,
        coordinates=tuple(
            FiniteFieldElement(
                presentation=presentation,
                coordinates=value,
            )
            for value in normalized
        ),
    )


def projective_line(
    presentation: FiniteFieldPresentation,
    axis: Axis,
) -> ProjectiveLine:
    """Enumerate a projective line in deterministic power-basis encoding order."""

    if len(axis.labels) != 2:
        raise OperationDomainValidationError(
            location=("axis",),
            code="finite_field.projective_line_two_coordinate_axis",
            message="projective-line enumeration requires a two-coordinate axis",
        )
    if presentation.order + 1 > _MAX_PROJECTIVE_POINTS:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.projective_line_exceeds_output_size_budget",
            message="projective line exceeds the output-size budget",
        )
    require_field(presentation)
    zero = FiniteFieldElement(
        presentation=presentation, coordinates=(0,) * presentation.degree
    )
    one = FiniteFieldElement(
        presentation=presentation, coordinates=(1,) + (0,) * (presentation.degree - 1)
    )
    affine_elements = _field_elements(presentation)
    return ProjectiveLine(
        presentation=presentation,
        axis=axis,
        points=(
            ProjectivePoint(
                presentation=presentation, axis=axis, coordinates=(zero, one)
            ),
            *(
                ProjectivePoint(
                    presentation=presentation, axis=axis, coordinates=(one, value)
                )
                for value in affine_elements
            ),
        ),
    )


def _field_elements(
    presentation: FiniteFieldPresentation,
) -> tuple[FiniteFieldElement, ...]:
    return tuple(
        FiniteFieldElement(
            presentation=presentation,
            coordinates=tuple(
                (encoded // presentation.characteristic**power)
                % presentation.characteristic
                for power in range(presentation.degree)
            ),
        )
        for encoded in range(presentation.order)
    )


def paley_tournament(
    presentation: FiniteFieldPresentation,
) -> PaleyTournamentResult:
    """Construct the directed Paley tournament of an exact finite field."""

    order = presentation.order
    if order % 4 != 3:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.paley_tournament_order_congruent_to_three_mod_four",
            message="Paley tournament construction requires field order congruent to 3 modulo 4",
        )

    edge_count = order * (order - 1) // 2
    if edge_count > MAX_DIRECTED_GRAPH_PARSE_EDGES:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.paley_tournament_exceeds_graph_edge_envelope",
            message="Paley tournament exceeds the directed graph edge envelope",
        )
    work = order * presentation.degree + order + 2 * edge_count
    if work > _MAX_PALEY_TOURNAMENT_WORK:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.paley_tournament_exceeds_work_budget",
            message="Paley tournament construction exceeds the finite-field work budget",
        )

    from jacobian.math.finite_fields import _flint

    active_context = _flint.context(presentation)
    elements = _field_elements(presentation)
    backend_elements = tuple(
        _flint.to_backend(value, active_context=active_context) for value in elements
    )
    square_encodings = {
        _flint.coordinates(value.square(), degree=presentation.degree)
        for value in backend_elements[1:]
    }
    edges = tuple(
        sorted(
            (left, right)
            if _flint.coordinates(
                backend_elements[right] - backend_elements[left],
                degree=presentation.degree,
            )
            in square_encodings
            else (right, left)
            for left in range(order)
            for right in range(left + 1, order)
        )
    )
    return PaleyTournamentResult(
        presentation=presentation,
        vertex_axis=elements,
        graph=DirectedGraph(vertex_count=order, edges=edges),
        orientation=_PALEY_ORIENTATION,
    )


def verify_paley_tournament(claim: PaleyTournamentResult) -> bool:
    """Verify field-axis binding and every quadratic-residue arc."""

    try:
        expected = paley_tournament(claim.presentation)
        return (
            claim.vertex_axis == expected.vertex_axis
            and claim.graph == expected.graph
            and claim.orientation == expected.orientation
        )
    except Exception:
        return False


def _admit_restriction_shape(subspace: FiniteDimensionalSubspace) -> None:
    rows = len(subspace.column_axis.labels) * subspace.presentation.degree
    columns = len(subspace.basis_axis.labels)
    if (
        rows > MAX_PRIME_FIELD_MATRIX_AXIS
        or rows * columns > MAX_PRIME_FIELD_MATRIX_CELLS
    ):
        raise OperationDomainValidationError(
            location=("subspace",),
            code="finite_field.restriction_output_shape",
            message="restriction output exceeds the supported matrix axis or cell bound",
        )


def restrict_scalars(
    subspace: FiniteDimensionalSubspace,
    direction: ProjectivePoint,
) -> FiniteLinearMap:
    """Construct ``B -> B^T b`` over the exact prime-field coordinate basis."""

    if direction.presentation != subspace.presentation:
        raise OperationDomainValidationError(
            location=("direction", "presentation"),
            code="finite_field.direction_presentation_mismatch",
            message="direction and subspace must share their field presentation",
        )
    if direction.axis != subspace.row_axis:
        raise OperationDomainValidationError(
            location=("direction", "axis"),
            code="finite_field.direction_axis_mismatch",
            message="direction axis must match the subspace matrix row axis",
        )
    if _direction_rank_work(subspace, 1) > _MAX_DIRECTION_RANK_WORK:
        raise OperationDomainValidationError(
            location=("subspace",),
            code="finite_field.restriction_exceeds_operation_work_budget",
            message="restriction exceeds the operation work budget",
        )
    from jacobian.math.finite_fields import _flint

    _admit_restriction_shape(subspace)
    active_context = _flint.context(subspace.presentation)
    require_independent_basis(subspace)
    return _restrict_scalars_admitted(subspace, direction, active_context)


def _restrict_scalars_admitted(
    subspace: FiniteDimensionalSubspace,
    direction: ProjectivePoint,
    active_context: Any,
) -> FiniteLinearMap:
    from jacobian.math.finite_fields import _flint

    backend_direction = tuple(
        _flint.to_backend(value, active_context=active_context)
        for value in direction.coordinates
    )
    columns: list[tuple[int, ...]] = []
    for matrix in subspace.basis:
        backend_matrix = tuple(
            tuple(
                _flint.to_backend(value, active_context=active_context) for value in row
            )
            for row in matrix.entries
        )
        image = tuple(
            sum(
                (
                    backend_matrix[row][column] * backend_direction[row]
                    for row in range(len(matrix.row_axis.labels))
                ),
                active_context(0),
            )
            for column in range(len(matrix.column_axis.labels))
        )
        columns.append(
            tuple(
                coordinate
                for value in image
                for coordinate in _flint.coordinates(
                    value,
                    degree=subspace.presentation.degree,
                )
            )
        )
    target_axis = Axis(
        name=f"Res({subspace.column_axis.name})",
        labels=tuple(
            f"{label}:{basis}"
            for label in subspace.column_axis.labels
            for basis in subspace.presentation.ordered_basis
        ),
    )
    return FiniteLinearMap(
        source_axis=subspace.basis_axis,
        target_axis=target_axis,
        matrix=PrimeFieldMatrix(
            prime=subspace.presentation.characteristic,
            entries=tuple(zip(*columns, strict=True)),
            columns=len(subspace.basis),
        ),
    )


def linear_map_rank(
    subspace: FiniteDimensionalSubspace,
    direction: ProjectivePoint,
) -> RankResult:
    """Derive and rank the direction-bound prime-field map."""

    if _direction_rank_work(subspace, 1) > _MAX_DIRECTION_RANK_WORK:
        raise OperationDomainValidationError(
            location=("subspace",),
            code="finite_field.rank_derivation_exceeds_operation_work_budget",
            message="rank derivation exceeds the operation work budget",
        )
    from jacobian.math.finite_fields import _flint

    linear_map = restrict_scalars(subspace, direction)
    return RankResult._from_kernel(
        subspace=subspace,
        direction=direction,
        linear_map=linear_map,
        rank=_flint.matrix_rank(linear_map.matrix),
    )


def _linear_map_rank_admitted(
    subspace: FiniteDimensionalSubspace, direction: ProjectivePoint, active_context: Any
) -> RankResult:
    from jacobian.math.finite_fields import _flint

    linear_map = _restrict_scalars_admitted(subspace, direction, active_context)
    return RankResult._from_kernel(
        subspace=subspace,
        direction=direction,
        linear_map=linear_map,
        rank=_flint.matrix_rank(linear_map.matrix),
    )


def _homogeneous_monomial_basis(
    variable_count: int, degree: int
) -> tuple[tuple[int, ...], ...]:
    """Return degree-d exponent vectors in descending lexicographic order."""

    if variable_count == 1:
        return ((degree,),)

    # A weak composition is determined by the positions of its variable_count
    # minus one separators among degree + variable_count - 1 slots. Reversing
    # the lexicographically ordered separator choices gives the established
    # descending exponent order without recursive calls or a deep Python stack.
    slot_count = degree + variable_count - 1
    separator_count = variable_count - 1
    compositions: list[tuple[int, ...]] = []
    for separators in reversed(tuple(combinations(range(slot_count), separator_count))):
        previous = -1
        exponents: list[int] = []
        for separator in separators:
            exponents.append(separator - previous - 1)
            previous = separator
        exponents.append(slot_count - previous - 1)
        compositions.append(tuple(exponents))
    return tuple(compositions)


def _multiply_by_linear_form(
    polynomial: dict[tuple[int, ...], int],
    coefficients: tuple[int, ...],
    *,
    prime: int,
) -> dict[tuple[int, ...], int]:
    """Multiply one sparse homogeneous polynomial by a GF(p) linear form."""

    product: dict[tuple[int, ...], int] = {}
    for exponents, coefficient in polynomial.items():
        for index, linear_coefficient in enumerate(coefficients):
            if linear_coefficient == 0:
                continue
            target = list(exponents)
            target[index] += 1
            target_tuple = tuple(target)
            product[target_tuple] = (
                product.get(target_tuple, 0) + coefficient * linear_coefficient
            ) % prime
    return {
        exponents: coefficient
        for exponents, coefficient in product.items()
        if coefficient
    }


def _induced_action_matrix(
    action: PrimeFieldLinearAction,
    monomial_basis: tuple[tuple[int, ...], ...],
    generator: PrimeFieldMatrix,
    *,
    checkpoint: Callable[[str], None] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Substitute one generator into every ordered homogeneous monomial."""

    if len(action.variable_axis.labels) == 1:
        degree = monomial_basis[0][0]
        return ((pow(generator.entries[0][0], degree, action.prime),),)

    monomial_index = {
        exponents: index for index, exponents in enumerate(monomial_basis)
    }
    matrix = [[0] * len(monomial_basis) for _ in monomial_basis]
    variable_count = len(action.variable_axis.labels)
    zero_exponents = (0,) * variable_count
    substitution_steps = 0
    for column, source_exponents in enumerate(monomial_basis):
        if checkpoint is not None:
            checkpoint("during substitution")
        polynomial: dict[tuple[int, ...], int] = {zero_exponents: 1}
        for source_variable, exponent in enumerate(source_exponents):
            coefficients = tuple(
                generator.entries[target_variable][source_variable]
                for target_variable in range(variable_count)
            )
            for _ in range(exponent):
                polynomial = _multiply_by_linear_form(
                    polynomial, coefficients, prime=action.prime
                )
                substitution_steps += 1
                if checkpoint is not None and substitution_steps % 1_024 == 0:
                    checkpoint("during substitution")
        for target_exponents, coefficient in polynomial.items():
            matrix[monomial_index[target_exponents]][column] = coefficient
    return tuple(tuple(row) for row in matrix)


def homogeneous_fixed_subspace(
    action: PrimeFieldLinearAction,
    degree: int,
) -> HomogeneousFixedSubspace:
    """Compute one exact homogeneous simultaneous fixed subspace over GF(p)."""

    _admit_prime(action.prime)
    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else time.monotonic()
    deadline = min(
        started_at + _FIXED_SUBSPACE_WALL_SECONDS,
        execution.deadline
        if execution is not None and execution.deadline is not None
        else float("inf"),
    )
    bind_request_deadline(deadline)

    def checkpoint(stage: str) -> None:
        _fixed_subspace_checkpoint(deadline, stage)

    monomial_count = _homogeneous_fixed_subspace_envelope(
        action,
        degree,
        checkpoint=checkpoint,
    )
    variable_count = len(action.variable_axis.labels)
    if variable_count == 1:
        scalars = tuple(
            generator.entries[0][0] for generator in action.generator_matrices
        )
        if any(scalar == 0 for scalar in scalars):
            raise OperationDomainValidationError(
                location=("action", "generator_matrices"),
                code="finite_field.linear_action_generator_invertible",
                message="every linear-action generator matrix must be invertible",
            )
        scalar_basis_rows = (
            ((1,),)
            if all(pow(scalar, degree, action.prime) == 1 for scalar in scalars)
            else ()
        )
        result = HomogeneousFixedSubspace._from_kernel(
            action=action,
            degree=degree,
            monomial_basis=((degree,),),
            basis_matrix=PrimeFieldMatrix(
                prime=action.prime,
                entries=scalar_basis_rows,
                columns=monomial_count,
            ),
        )
        checkpoint("after result construction")
        return result
    monomial_basis = _homogeneous_monomial_basis(variable_count, degree)
    assert len(monomial_basis) == monomial_count
    equations: list[tuple[int, ...]] = []
    for generator in action.generator_matrices:
        induced = _induced_action_matrix(
            action, monomial_basis, generator, checkpoint=checkpoint
        )
        for row_index, row in enumerate(induced):
            checkpoint("during equation assembly")
            equation = list(row)
            equation[row_index] = (equation[row_index] - 1) % action.prime
            equations.append(tuple(equation))
    equation_matrix = PrimeFieldMatrix(
        prime=action.prime,
        entries=tuple(equations),
        columns=len(monomial_basis),
    )
    checkpoint("before nullspace")
    generators_invertible, basis_rows = run_fixed_subspace_computation(
        action.generator_matrices,
        equation_matrix,
        deadline=deadline,
    )
    if not generators_invertible:
        raise OperationDomainValidationError(
            location=("action", "generator_matrices"),
            code="finite_field.linear_action_generator_invertible",
            message="every linear-action generator matrix must be invertible",
        )
    checkpoint("after basis reduction")
    basis_matrix = PrimeFieldMatrix(
        prime=action.prime,
        entries=basis_rows,
        columns=len(monomial_basis),
    )
    result = HomogeneousFixedSubspace._from_kernel(
        action=action,
        degree=degree,
        monomial_basis=monomial_basis,
        basis_matrix=basis_matrix,
    )
    checkpoint("after result construction")
    return result


def direction_rank_ledger(
    subspace: FiniteDimensionalSubspace,
    directions: ProjectiveLine,
) -> DirectionRankLedger:
    """Restrict scalars and rank every supplied direction without losing order."""

    if directions.presentation != subspace.presentation:
        raise OperationDomainValidationError(
            location=("directions", "presentation"),
            code="finite_field.direction_presentation_mismatch",
            message="directions and subspace must share their field presentation",
        )
    if directions.axis != subspace.row_axis:
        raise OperationDomainValidationError(
            location=("directions", "axis"),
            code="finite_field.direction_axis_mismatch",
            message="direction axis must match the subspace matrix row axis",
        )
    if (
        _direction_rank_work(subspace, len(directions.points))
        > _MAX_DIRECTION_RANK_WORK
    ):
        raise OperationDomainValidationError(
            location=("directions",),
            code="finite_field.direction_rank_ledger_exceeds_operation_work_budget",
            message="direction-rank ledger exceeds the operation work budget",
        )
    from jacobian.math.finite_fields import _flint

    _admit_restriction_shape(subspace)
    active_context = _flint.context(subspace.presentation)
    require_independent_basis(subspace)
    return DirectionRankLedger._from_kernel(
        subspace=subspace,
        entries=tuple(
            _linear_map_rank_admitted(subspace, direction, active_context)
            for direction in directions.points
        ),
    )


def _orbit_count_digit_bound(base: int, exponent: int) -> int:
    """Conservatively bound decimal digits of ``base**exponent``.

    ``log10(2) < 30103 / 100000`` gives an integer-only upper bound from the
    result's bit length, avoiding construction of an over-budget power during
    admission.
    """

    if exponent == 0:
        return 1
    bits = base.bit_length() * exponent
    return (bits * 30103 + 99_999) // 100_000


def _admit_orbit_distribution(ledger: DirectionRankLedger) -> None:
    """Admit ledger authentication, histogram growth, and exact output size."""

    if (
        _direction_rank_work(ledger.subspace, len(ledger.entries))
        > _MAX_DIRECTION_RANK_WORK
    ):
        raise OperationDomainValidationError(
            location=("ledger",),
            code="finite_field.orbit_ledger_exceeds_operation_work_budget",
            message="orbit ledger authentication exceeds the operation work budget",
        )
    first = ledger.entries[0]
    target_dimension = len(first.linear_map.target_axis.labels)
    prime = first.direction.presentation.characteristic
    maximum_power_digits = _orbit_count_digit_bound(prime, target_dimension)
    maximum_count_digits = maximum_power_digits + len(str(len(ledger.entries))) + 1
    if maximum_count_digits > MAX_ORBIT_DISTRIBUTION_COUNT_DIGITS:
        raise OperationDomainValidationError(
            location=("ledger",),
            code="finite_field.orbit_distribution_count_digit_bound",
            message="orbit distribution counts exceed their exact output digit bound",
        )
    possible_rows = min(len(ledger.entries) + 1, target_dimension + 1)
    total_digits = possible_rows * 2 * maximum_count_digits
    if total_digits > MAX_ORBIT_DISTRIBUTION_TOTAL_DIGITS:
        raise OperationDomainValidationError(
            location=("ledger",),
            code="finite_field.orbit_distribution_output_bound",
            message="orbit distribution count rows exceed their exact output bound",
        )


def _authenticate_orbit_ledger(ledger: DirectionRankLedger) -> None:
    """Check every supplied ledger entry against its retained source."""

    from jacobian.math.finite_fields import _flint

    _admit_restriction_shape(ledger.subspace)
    active_context = _flint.context(ledger.subspace.presentation)
    require_independent_basis(ledger.subspace)
    for index, entry in enumerate(ledger.entries):
        request_checkpoint("before orbit ledger entry authentication")
        expected = _linear_map_rank_admitted(
            ledger.subspace, entry.direction, active_context
        )
        if entry.linear_map != expected.linear_map or entry.rank != expected.rank:
            raise OperationDomainValidationError(
                location=("ledger", "entries", index),
                code="finite_field.ledger_entry_matches_source_restriction",
                message="ledger entry must match the restricted map and rank of its bound source",
            )


def orbit_distribution(ledger: DirectionRankLedger) -> OrbitDistribution:
    """Aggregate projective orbit counts from a complete direction-rank ledger."""

    _admit_orbit_distribution(ledger)
    _authenticate_orbit_ledger(ledger)
    return OrbitDistribution._from_kernel(ledger)


def verify_orbit_distribution(claim: OrbitDistribution) -> bool:
    """Verify a serialized orbit histogram against its complete source ledger."""

    try:
        _admit_orbit_distribution(claim.ledger)
        _authenticate_orbit_ledger(claim.ledger)
        return claim.counts == _orbit_counts(claim.ledger)
    except Exception:
        return False


def finite_polynomial(
    presentation: FiniteFieldPresentation,
    coefficients: tuple[FiniteFieldElement, ...],
    *,
    variable: str = "x",
) -> FinitePolynomial:
    """Construct a canonical univariate polynomial over one exact field."""

    require_field(presentation)
    if not coefficients:
        raise ValueError("finite polynomial requires coefficients")
    last = next(
        (
            index
            for index in range(len(coefficients) - 1, -1, -1)
            if not coefficients[index].is_zero
        ),
        0,
    )
    return FinitePolynomial(
        presentation=presentation,
        variable=variable,
        coefficients=coefficients[: last + 1],
    )


def finite_polynomial_map(polynomial: FinitePolynomial) -> FinitePolynomialMap:
    """Bind a polynomial as a self-map of its exact field presentation."""

    require_field(polynomial.presentation)
    return FinitePolynomialMap(
        domain=polynomial.presentation,
        codomain=polynomial.presentation,
        polynomial=polynomial,
    )


def evaluate_finite_polynomial(
    polynomial: FinitePolynomial,
    value: FiniteFieldElement,
) -> FiniteFieldElement:
    """Evaluate with Python-FLINT while preserving the exact parent."""

    if value.presentation != polynomial.presentation:
        raise ValueError("polynomial and value must share their exact presentation")
    from jacobian.math.finite_fields import _flint

    return FiniteFieldElement(
        presentation=polynomial.presentation,
        coordinates=_flint.evaluate_polynomial(polynomial.coefficients, value),
    )


def _admit_map_evaluation(
    polynomial_map: FinitePolynomialMap, *, location: tuple[str, ...]
) -> None:
    work = (
        polynomial_map.domain.order
        * len(polynomial_map.polynomial.coefficients)
        * polynomial_map.domain.degree
    )
    if work > _MAX_FINITE_MAP_WORK:
        raise OperationDomainValidationError(
            location=location,
            code="finite_field.finite_map_exceeds_operation_work_budget",
            message="finite map exceeds the operation work budget",
        )


def finite_map_table(polynomial_map: FinitePolynomialMap) -> FiniteMapTable:
    """Enumerate a complete finite polynomial-map table in canonical order."""

    _admit_map_evaluation(polynomial_map, location=("polynomial_map",))
    from jacobian.math.finite_fields import _flint

    sources = _field_elements(polynomial_map.domain)
    targets = _flint.evaluate_polynomial_values(
        polynomial_map.polynomial.coefficients,
        sources,
    )
    return FiniteMapTable._from_kernel(
        polynomial_map,
        tuple(
            (
                source,
                FiniteFieldElement(
                    presentation=polynomial_map.codomain, coordinates=coordinates
                ),
            )
            for source, coordinates in zip(sources, targets, strict=True)
        ),
    )


def _authenticate_map_table(table: FiniteMapTable) -> None:
    """Establish the caller-supplied evaluation relation within consumer work.

    Shape validation and serialization cannot establish polynomial evaluation.
    Each public consumer admits and computes that relation once; trusted result
    construction and deserialization never invoke this recognition step.
    """
    _admit_map_evaluation(table.map, location=("table", "map"))
    from jacobian.math.finite_fields import _flint

    expected = _flint.evaluate_polynomial_values(
        table.map.polynomial.coefficients,
        tuple(source for source, _ in table.entries),
    )
    if any(
        target.coordinates != coordinates
        for (_, target), coordinates in zip(table.entries, expected, strict=True)
    ):
        raise OperationDomainValidationError(
            location=("table", "entries"),
            code="finite_field.finite_map_table_targets_match_bound_polynomial",
            message="finite map table targets must match the bound polynomial",
        )


def fiber_partition(table: FiniteMapTable) -> FiberPartition:
    """Partition the complete domain by exact map image."""

    _authenticate_map_table(table)
    return FiberPartition.from_table(table)


def verify_fiber_partition(claim: FiberPartition) -> bool:
    """Verify every fiber in a serialized finite-map partition."""

    try:
        _authenticate_map_table(claim.table)
        return claim.fibers == _fibers_for_table(claim.table)
    except Exception:
        return False


def analyze_collisions(table: FiniteMapTable) -> CollisionResult:
    """Return either the first canonical collision or an injectivity result."""

    _authenticate_map_table(table)
    seen: dict[str, tuple[FiniteFieldElement, FiniteFieldElement]] = {}
    for source, target in table.entries:
        previous = seen.get(target.digest)
        if previous is not None:
            return CollisionResult._from_kernel(
                table=table,
                status="COLLISION",
                left=previous[0],
                right=source,
                image=target,
            )
        seen[target.digest] = (source, target)
    return CollisionResult._from_kernel(table=table, status="INJECTIVE")


def verify_collisions(claim: CollisionResult) -> bool:
    """Verify a serialized collision or injectivity conclusion."""

    try:
        expected = analyze_collisions(claim.table)
        return (
            claim.status == expected.status
            and claim.left == expected.left
            and claim.right == expected.right
            and claim.image == expected.image
        )
    except Exception:
        return False


def analyze_permutation(table: FiniteMapTable) -> PermutationResult:
    """Return either an inverse table or a non-permutation result."""

    _authenticate_map_table(table)
    inverse_entries = tuple(
        sorted(
            ((target, source) for source, target in table.entries),
            key=lambda entry: sum(
                coordinate * table.map.codomain.characteristic**power
                for power, coordinate in enumerate(entry[0].coordinates)
            ),
        )
    )
    if len({target.digest for _, target in table.entries}) != len(table.entries):
        return PermutationResult._from_kernel(table=table, status="NOT_PERMUTATION")
    return PermutationResult._from_kernel(
        table=table, status="PERMUTATION", inverse_entries=inverse_entries
    )


def verify_permutation(claim: PermutationResult) -> bool:
    """Verify a serialized permutation or non-permutation conclusion."""

    try:
        expected = analyze_permutation(claim.table)
        return (
            claim.status == expected.status
            and claim.inverse_entries == expected.inverse_entries
        )
    except Exception:
        return False
