"""Exact operations on presentation-, parent-, and axis-bound finite-field values."""

from __future__ import annotations

from jacobian.math.finite_fields.values import (
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
    OrbitDistribution,
    PermutationResult,
    ProjectiveLine,
    ProjectivePoint,
    RankResult,
)
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix


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
        raise ValueError("projective-line enumeration requires a two-coordinate axis")
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


def restrict_scalars(
    subspace: FiniteDimensionalSubspace,
    direction: ProjectivePoint,
) -> FiniteLinearMap:
    """Construct ``B -> B^T b`` over the exact prime-field coordinate basis."""

    if direction.presentation != subspace.presentation:
        raise ValueError("direction and subspace must share their field presentation")
    if direction.axis != subspace.row_axis:
        raise ValueError("direction axis must match the subspace matrix row axis")
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

    from jacobian.math.finite_fields import _flint

    linear_map = restrict_scalars(subspace, direction)
    return RankResult(
        subspace=subspace,
        direction=direction,
        linear_map=linear_map,
        rank=_flint.matrix_rank(linear_map.matrix),
    )


def direction_rank_ledger(
    subspace: FiniteDimensionalSubspace,
    directions: ProjectiveLine,
) -> DirectionRankLedger:
    """Restrict scalars and rank every supplied direction without losing order."""

    return DirectionRankLedger(
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

    from jacobian.math.finite_fields import _flint

    sources = _field_elements(polynomial_map.domain)
    targets = _flint.evaluate_polynomial_values(
        polynomial_map.polynomial.coefficients,
        sources,
    )
    return FiniteMapTable(
        map=polynomial_map,
        entries=tuple(
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
            return CollisionResult(
                table=table,
                status="COLLISION",
                left=previous[0],
                right=source,
                image=target,
            )
        seen[target.digest] = (source, target)
    return CollisionResult(table=table, status="INJECTIVE")


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
        return PermutationResult(table=table, status="NOT_PERMUTATION")
    return PermutationResult(
        table=table, status="PERMUTATION", inverse_entries=inverse_entries
    )
