"""Exact native geometry operations."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._line_arithmetic import (
    canonical_line_coefficients,
    squared_point_line_distance,
)
from jacobian.math.geometry.exact._models import (
    DistanceMultiplicityEntry,
    DistanceProfileResult,
    EuclideanOrbitProfileResult,
    LabelledRationalPoint,
    PinnedLineConfiguration,
    PinnedLineDistanceResult,
    PointConfiguration,
    _require_bounded_point_configuration,
    _validation_error,
)
from jacobian.math.geometry.exact._orbit_bounds import admit_orbit_profile
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.values import RationalMatrix


def _to_fraction_point(point: LabelledRationalPoint) -> tuple[Fraction, ...]:
    return tuple(coordinate.as_fraction() for coordinate in point.coordinates)


def _squared_distance(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> Fraction:
    result = Fraction(0)
    for left_coordinate, right_coordinate in zip(left, right, strict=True):
        result += (left_coordinate - right_coordinate) ** 2
    return result


def distance_profile(configuration: PointConfiguration) -> DistanceProfileResult:
    """Compute exact pairwise squared distances for every unordered pair."""
    points = [_to_fraction_point(point) for point in configuration.points]
    distances: Counter[Fraction] = Counter(
        _squared_distance(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )
    entries = tuple(
        DistanceMultiplicityEntry(
            squared_distance=CanonicalRational.from_fraction(distance),
            pair_count=count,
        )
        for distance, count in sorted(distances.items())
    )
    return DistanceProfileResult(
        dimension=len(configuration.points[0].coordinates),
        point_count=len(configuration.points),
        entries=entries,
    )


def distance_graph(
    configuration: PointConfiguration,
    target_squared_distance: CanonicalRational,
) -> IndexedSimpleUndirectedGraph:
    """Build the graph whose edges connect pairs at the target distance."""
    points = [_to_fraction_point(point) for point in configuration.points]
    target = target_squared_distance.as_fraction()
    edges = tuple(
        (left, right)
        for left in range(len(points))
        for right in range(left + 1, len(points))
        if _squared_distance(points[left], points[right]) == target
    )
    return IndexedSimpleUndirectedGraph(
        vertex_count=len(points),
        edges=edges,
    )


def euclidean_orbit_profile(
    configuration: PointConfiguration,
) -> EuclideanOrbitProfileResult:
    """Return canonical unlabeled isometry and similarity distance forms."""
    admit_orbit_profile(configuration)
    points = [_to_fraction_point(point) for point in configuration.points]
    size = len(points)
    distances = [
        [_squared_distance(points[left], points[right]) for right in range(size)]
        for left in range(size)
    ]

    def canonical_form(
        scale: Fraction,
    ) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[int, ...]]:
        best_form: tuple[tuple[Fraction, ...], ...] | None = None
        best_relabeling: tuple[int, ...] | None = None
        for permutation in permutations(range(size)):
            form = tuple(
                tuple(
                    distances[permutation[left]][permutation[right]] / scale
                    for right in range(size)
                )
                for left in range(size)
            )
            if best_form is None or form < best_form:
                best_form = form
                best_relabeling = permutation
        assert best_form is not None and best_relabeling is not None
        # ``permutation`` indexes the source row for each canonical position;
        # the public relabeling maps each source index to its canonical position.
        inverse = [0] * size
        for canonical_index, source_index in enumerate(best_relabeling):
            inverse[source_index] = canonical_index
        return best_form, tuple(inverse)

    positive_distances = [
        distances[left][right]
        for left in range(size)
        for right in range(size)
        if right > left and distances[left][right] > 0
    ]
    if not positive_distances:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="point_configuration.orbit_degenerate",
            message="orbit profiles require at least one positive squared distance",
        )
    minimum_positive_distance = min(positive_distances)
    isometry_form, isometry_relabeling = canonical_form(Fraction(1))
    similarity_form, similarity_relabeling = canonical_form(minimum_positive_distance)

    def canonical_rational_matrix(
        form: tuple[tuple[Fraction, ...], ...],
    ) -> RationalMatrix:
        return RationalMatrix(
            entries=tuple(
                tuple(CanonicalRational.from_fraction(value) for value in row)
                for row in form
            )
        )

    return EuclideanOrbitProfileResult(
        configuration=configuration,
        isometry_form=canonical_rational_matrix(isometry_form),
        isometry_relabeling=isometry_relabeling,
        similarity_form=canonical_rational_matrix(similarity_form),
        similarity_relabeling=similarity_relabeling,
        normalizing_squared_distance=CanonicalRational.from_fraction(
            minimum_positive_distance
        ),
    )


def pinned_line_distance_profile(
    configuration: PinnedLineConfiguration,
    anchor_value: tuple[CanonicalRational, ...],
) -> PinnedLineDistanceResult:
    """Compute the pinned line-distance profile of a point configuration."""
    from jacobian.math.geometry.exact._models import PinnedLineEntry

    try:
        _require_bounded_point_configuration(configuration, anchor_value)
        if len(configuration.points[0].coordinates) != 2:
            raise _validation_error(
                "pinned_line_distance_profile_requires_a",
                "pinned line-distance profile requires a planar configuration",
            )
        coordinates = {
            tuple(component.as_fraction() for component in point.coordinates)
            for point in configuration.points
        }
        if len(coordinates) != len(configuration.points):
            raise _validation_error(
                "pinned_line_distance_profile_requires_distinct",
                "pinned line-distance profile requires distinct point coordinates",
            )
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("configuration",), code=exc.type, message=exc.message()
        ) from exc

    points = [_to_fraction_point(point) for point in configuration.points]
    anchor = tuple(coordinate.as_fraction() for coordinate in anchor_value)
    lines: dict[tuple[Fraction, Fraction, Fraction], list[tuple[int, int]]] = {}
    distances: dict[tuple[Fraction, Fraction, Fraction], Fraction] = {}
    for left, right in combinations(range(len(points)), 2):
        coefficients = canonical_line_coefficients(points[left], points[right])
        lines.setdefault(coefficients, []).append((left, right))
        if coefficients not in distances:
            distances[coefficients] = squared_point_line_distance(
                anchor, points[left], points[right]
            )

    ordered = sorted(
        lines, key=lambda coefficients: (distances[coefficients], coefficients)
    )
    entries = tuple(
        PinnedLineEntry(
            line_coefficients=tuple(
                CanonicalRational.from_fraction(value) for value in coefficients
            ),
            squared_distance=CanonicalRational.from_fraction(distances[coefficients]),
            pairs=tuple(lines[coefficients]),
        )
        for coefficients in ordered
    )
    multiplicities: dict[Fraction, int] = {}
    for entry in entries:
        distance = entry.squared_distance.as_fraction()
        multiplicities[distance] = multiplicities.get(distance, 0) + 1
    return PinnedLineDistanceResult._from_kernel(
        configuration,
        anchor_value,
        lines=entries,
        distance_multiplicities=tuple(
            (CanonicalRational.from_fraction(distance), count)
            for distance, count in sorted(multiplicities.items())
        ),
    )


__all__ = [
    "distance_graph",
    "distance_profile",
    "euclidean_orbit_profile",
    "pinned_line_distance_profile",
]
