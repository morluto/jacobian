"""Exact operations on presentation-, parent-, and axis-bound finite-field values."""

from __future__ import annotations

from math import comb
from typing import Literal

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._models import (
    _MAX_DIRECTION_RANK_WORK,
    _MAX_PROJECTIVE_POINTS,
)
from jacobian.math.finite_fields.values import (
    _MAX_HOMOGENEOUS_DEGREE,
    _MAX_HOMOGENEOUS_MONOMIALS,
    Axis,
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
)
from jacobian.math.graphs.directed._models import (
    MAX_DIRECTED_GRAPH_PARSE_EDGES,
    DirectedGraph,
)
from jacobian.math.matrices.finite_fields._bounds import (
    MAX_PRIME_FIELD_ELIMINATION_WORK,
    MAX_PRIME_FIELD_MATRIX_CELLS,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)

_MAX_FINITE_MAP_WORK = 1_000_000
_MAX_PALEY_TOURNAMENT_WORK = 4_000_000
_PALEY_ORIENTATION: Literal["ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"] = (
    "ARC_X_TO_Y_IFF_Y_MINUS_X_IS_NONZERO_SQUARE"
)


def _homogeneous_fixed_subspace_envelope(
    action: PrimeFieldLinearAction, degree: int
) -> int:
    if type(degree) is not int or not 0 <= degree <= _MAX_HOMOGENEOUS_DEGREE:
        raise OperationDomainValidationError(
            location=("degree",),
            code="finite_field.fixed_subspace_degree_bound",
            message=(
                f"degree must be an integer from 0 through {_MAX_HOMOGENEOUS_DEGREE}"
            ),
        )
    variable_count = len(action.variable_axis.labels)
    generator_count = len(action.generator_matrices)
    monomial_count = comb(variable_count + degree - 1, degree)
    equation_entries = generator_count * monomial_count**2
    output_entries = monomial_count**2
    expansion_work = (
        generator_count * max(1, degree) * variable_count * monomial_count**2
    )
    # One elimination solves the stacked fixed equations; the second puts the
    # returned nullspace rows into backend-independent canonical RREF form.
    elimination_work = (generator_count + 1) * monomial_count**3
    action_rank_work = generator_count * variable_count**3
    if monomial_count > _MAX_HOMOGENEOUS_MONOMIALS:
        raise OperationDomainValidationError(
            location=("degree",),
            code="finite_field.fixed_subspace_monomial_bound",
            message="homogeneous monomial basis exceeds the operation bound",
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
    work = expansion_work + elimination_work + action_rank_work
    if work > MAX_PRIME_FIELD_ELIMINATION_WORK:
        raise OperationDomainValidationError(
            location=("action",),
            code="finite_field.fixed_subspace_work_bound",
            message="homogeneous fixed-subspace computation exceeds its work bound",
        )
    return monomial_count


def finite_field(
    characteristic: int,
    modulus_coefficients: tuple[int, ...],
    *,
    generator: str = "a",
) -> FiniteFieldPresentation:
    """Construct and validate an exact finite-extension presentation."""

    return FiniteFieldPresentation(
        characteristic=characteristic,
        modulus_coefficients=modulus_coefficients,
        generator=generator,
    )


def element(
    presentation: FiniteFieldPresentation,
    coordinates: tuple[int, ...],
) -> FiniteFieldElement:
    """Construct one parent-bound element from canonical power-basis coordinates."""

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
    zero = element(presentation, (0,) * presentation.degree)
    one = element(presentation, (1,) + (0,) * (presentation.degree - 1))
    affine_elements = _field_elements(presentation)
    return ProjectiveLine(
        presentation=presentation,
        axis=axis,
        points=(
            projective_point(presentation, axis, (zero, one)),
            *(
                projective_point(presentation, axis, (one, value))
                for value in affine_elements
            ),
        ),
    )


def _field_elements(
    presentation: FiniteFieldPresentation,
) -> tuple[FiniteFieldElement, ...]:
    return tuple(
        element(
            presentation,
            tuple(
                (encoded // presentation.characteristic**power)
                % presentation.characteristic
                for power in range(presentation.degree)
            ),
        )
        for encoded in range(presentation.order)
    )


def _paley_result_wire_bytes(
    presentation: FiniteFieldPresentation,
    order: int,
) -> int:
    """Return the exact canonical result size before allocating its arc tuple."""

    try:
        empty_size = len(
            encode_strict_json(
                {
                    "presentation": presentation.model_dump(mode="json"),
                    "graph": {"vertex_count": order, "edges": []},
                    "orientation": _PALEY_ORIENTATION,
                },
                # The request envelope is at most one canonical input document;
                # the fixed graph and orientation fields add only a small amount
                # before the result-size check below.  A relaxed measurement
                # limit keeps an oversized result on the typed domain path.
                limits=CanonicalLimits(
                    max_output_bytes=2 * CanonicalLimits().max_output_bytes
                ),
            )
        )
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.paley_tournament_exceeds_output_budget",
            message="complete Paley tournament exceeds the canonical output budget",
        ) from exc
    edge_count = order * (order - 1) // 2
    digit_lengths = tuple(len(str(vertex)) for vertex in range(order))
    edge_bytes = 3 * edge_count + sum(
        digits * ((order - 1 - vertex) + vertex)
        for vertex, digits in enumerate(digit_lengths)
    )
    return empty_size + edge_bytes + max(0, edge_count - 1)


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
    output_bytes = _paley_result_wire_bytes(presentation, order)
    if output_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("presentation",),
            code="finite_field.paley_tournament_exceeds_output_budget",
            message="complete Paley tournament exceeds the canonical output budget",
        )
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

    elements = _field_elements(presentation)
    active_context = _flint.context(presentation)
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
        graph=DirectedGraph(vertex_count=order, edges=edges),
        orientation=_PALEY_ORIENTATION,
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

    active_context = _flint.context(subspace.presentation)
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


def _homogeneous_monomial_basis(
    variable_count: int, degree: int
) -> tuple[tuple[int, ...], ...]:
    """Return degree-d exponent vectors in descending lexicographic order."""

    def compositions(remaining: int, positions: int) -> tuple[tuple[int, ...], ...]:
        if positions == 1:
            return ((remaining,),)
        return tuple(
            (first, *tail)
            for first in range(remaining, -1, -1)
            for tail in compositions(remaining - first, positions - 1)
        )

    return compositions(degree, variable_count)


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
) -> tuple[tuple[int, ...], ...]:
    """Substitute one generator into every ordered homogeneous monomial."""

    monomial_index = {
        exponents: index for index, exponents in enumerate(monomial_basis)
    }
    matrix = [[0] * len(monomial_basis) for _ in monomial_basis]
    variable_count = len(action.variable_axis.labels)
    zero_exponents = (0,) * variable_count
    for column, source_exponents in enumerate(monomial_basis):
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
        for target_exponents, coefficient in polynomial.items():
            matrix[monomial_index[target_exponents]][column] = coefficient
    return tuple(tuple(row) for row in matrix)


def homogeneous_fixed_subspace(
    action: PrimeFieldLinearAction, degree: int
) -> HomogeneousFixedSubspace:
    """Compute one exact homogeneous simultaneous fixed subspace over GF(p)."""

    monomial_count = _homogeneous_fixed_subspace_envelope(action, degree)
    variable_count = len(action.variable_axis.labels)
    if any(rank(matrix) != variable_count for matrix in action.generator_matrices):
        raise OperationDomainValidationError(
            location=("action", "generator_matrices"),
            code="finite_field.linear_action_generator_invertible",
            message="every linear-action generator matrix must be invertible",
        )
    monomial_basis = _homogeneous_monomial_basis(variable_count, degree)
    assert len(monomial_basis) == monomial_count
    equations: list[tuple[int, ...]] = []
    for generator in action.generator_matrices:
        induced = _induced_action_matrix(action, monomial_basis, generator)
        for row_index, row in enumerate(induced):
            equation = list(row)
            equation[row_index] = (equation[row_index] - 1) % action.prime
            equations.append(tuple(equation))
    equation_matrix = PrimeFieldMatrix(
        prime=action.prime,
        entries=tuple(equations),
        columns=len(monomial_basis),
    )
    nullspace_rows = nullspace(equation_matrix)
    if nullspace_rows:
        reduced, pivots = rref(
            PrimeFieldMatrix(
                prime=action.prime,
                entries=nullspace_rows,
                columns=len(monomial_basis),
            )
        )
        basis_rows = reduced[: len(pivots)]
    else:
        basis_rows = ()
    basis_matrix = PrimeFieldMatrix(
        prime=action.prime,
        entries=basis_rows,
        columns=len(monomial_basis),
    )
    return HomogeneousFixedSubspace._from_kernel(
        action=action,
        degree=degree,
        monomial_basis=monomial_basis,
        basis_matrix=basis_matrix,
    )


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
    return DirectionRankLedger._from_kernel(
        subspace=subspace,
        entries=tuple(
            linear_map_rank(subspace, direction) for direction in directions.points
        ),
    )


def orbit_distribution(ledger: DirectionRankLedger) -> OrbitDistribution:
    """Aggregate projective orbit counts from a complete direction-rank ledger."""

    return OrbitDistribution.from_ledger(ledger)


def finite_polynomial(
    presentation: FiniteFieldPresentation,
    coefficients: tuple[FiniteFieldElement, ...],
    *,
    variable: str = "x",
) -> FinitePolynomial:
    """Construct a canonical univariate polynomial over one exact field."""

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

    return element(
        polynomial.presentation,
        _flint.evaluate_polynomial(polynomial.coefficients, value),
    )


def finite_map_table(polynomial_map: FinitePolynomialMap) -> FiniteMapTable:
    """Enumerate a complete finite polynomial-map table in canonical order."""

    work = (
        polynomial_map.domain.order
        * len(polynomial_map.polynomial.coefficients)
        * polynomial_map.domain.degree
    )
    if work > _MAX_FINITE_MAP_WORK:
        raise OperationDomainValidationError(
            location=("polynomial_map",),
            code="finite_field.finite_map_exceeds_operation_work_budget",
            message="finite map exceeds the operation work budget",
        )
    from jacobian.math.finite_fields import _flint

    sources = _field_elements(polynomial_map.domain)
    targets = _flint.evaluate_polynomial_values(
        polynomial_map.polynomial.coefficients,
        sources,
    )
    return FiniteMapTable._from_kernel(
        polynomial_map,
        tuple(
            (source, element(polynomial_map.codomain, coordinates))
            for source, coordinates in zip(sources, targets, strict=True)
        ),
    )


def fiber_partition(table: FiniteMapTable) -> FiberPartition:
    """Partition the complete domain by exact map image."""

    return FiberPartition.from_table(table)


def analyze_collisions(table: FiniteMapTable) -> CollisionResult:
    """Return either the first canonical collision or an injectivity result."""

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


def analyze_permutation(table: FiniteMapTable) -> PermutationResult:
    """Return either an inverse table or a non-permutation result."""

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
