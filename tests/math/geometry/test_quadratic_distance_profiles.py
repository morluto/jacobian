"""Exact quadratic pair classes and composition with distance graphs."""

import json
import time
from fractions import Fraction

import pytest
from pydantic import ValidationError
from sympy import Rational, expand, sqrt

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.exact import (
    LabelledQuadraticPoint,
    QuadraticPointConfiguration,
    distance_graph,
    distance_profile,
)
from jacobian.math.geometry.exact._models import DistanceProfileResult
from jacobian.math.geometry.exact._tools import EQUILATERAL_QUADRATIC, TOOLS
from jacobian.math.number_theory.algebraic_numbers.quadratic import RealQuadraticValue


def value(a: int | Fraction, b: int | Fraction = 0, d: int = 3) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(Fraction(a)),
        radical_coefficient=CanonicalRational.from_fraction(Fraction(b)),
        radicand=d,
    )


def configuration(
    points: tuple[tuple[RealQuadraticValue, RealQuadraticValue], ...], d: int = 3
) -> QuadraticPointConfiguration:
    return QuadraticPointConfiguration(
        radicand=d,
        points=tuple(
            LabelledQuadraticPoint(label=str(i), coordinates=p)
            for i, p in enumerate(points)
        ),
    )


def test_equilateral_profile_and_selected_graph() -> None:
    source = QuadraticPointConfiguration.model_validate_json(
        json.dumps(EQUILATERAL_QUADRATIC["configuration"])
    )
    result = distance_profile(source)
    assert len(result.entries) == 1
    assert result.entries[0].squared_distance == value(1)
    assert result.entries[0].pairs == ((0, 1), (0, 2), (1, 2))
    assert result.entries[0].pair_count == 3
    decoded = DistanceProfileResult[
        QuadraticPointConfiguration, RealQuadraticValue
    ].model_validate_json(result.model_dump_json())
    selected = distance_graph(
        decoded.configuration, decoded.entries[0].squared_distance
    )
    assert selected.graph.edges == ((0, 1), (0, 2), (1, 2))


def test_every_class_replays_independent_symbolic_pair_identity() -> None:
    source = configuration(
        tuple(
            (
                value(Fraction(i, 3), Fraction(i % 3, 2)),
                value(Fraction(i * i, 5), Fraction((-1) ** i, 3)),
            )
            for i in range(7)
        )
    )
    result = distance_profile(source)
    seen = set()
    for entry in result.entries:
        actual = entry.squared_distance
        exact = Rational(actual.rational_part.num, actual.rational_part.den) + Rational(
            actual.radical_coefficient.num, actual.radical_coefficient.den
        ) * sqrt(3)
        for i, j in entry.pairs:
            expected = 0
            for left, right in zip(
                source.points[i].coordinates, source.points[j].coordinates, strict=True
            ):
                a = left.rational_part.as_fraction() - right.rational_part.as_fraction()
                b = (
                    left.radical_coefficient.as_fraction()
                    - right.radical_coefficient.as_fraction()
                )
                expected += (
                    Rational(a.numerator, a.denominator)
                    + Rational(b.numerator, b.denominator) * sqrt(3)
                ) ** 2
            assert expand(expected - exact) == 0
            seen.add((i, j))
    assert len(seen) == 21
    assert sum(entry.pair_count for entry in result.entries) == 21


def test_anisotropic_lattice_retains_each_pinned_pair_partition() -> None:
    positions = ((0, 0), (1, 0), (0, 1), (1, 1))
    source = configuration(
        tuple((value(x, d=2), value(0, y, d=2)) for x, y in positions), 2
    )
    result = distance_profile(source)
    assert [
        (e.squared_distance.rational_part.num, e.pair_count) for e in result.entries
    ] == [(1, 2), (2, 2), (3, 2)]
    for center in range(4):
        assert (
            sum(center in pair for entry in result.entries for pair in entry.pairs) == 3
        )


def test_source_permutation_preserves_labelled_classes() -> None:
    source = configuration(
        ((value(0), value(0)), (value(1, 1), value(0)), (value(1, -1), value(0)))
    )
    permuted = QuadraticPointConfiguration(
        radicand=3, points=tuple(reversed(source.points))
    )

    def labelled(s: QuadraticPointConfiguration) -> dict[str, set[tuple[str, ...]]]:
        return {
            e.squared_distance.model_dump_json(): {
                tuple(sorted((s.points[i].label, s.points[j].label)))
                for i, j in e.pairs
            }
            for e in distance_profile(s).entries
        }

    assert labelled(source) == labelled(permuted)


@pytest.mark.parametrize("size", [0, 1, 2])
def test_empty_singleton_and_coincident_configurations(size: int) -> None:
    source = configuration(tuple((value(0), value(0)) for _ in range(size)))
    result = distance_profile(source)
    assert sum(e.pair_count for e in result.entries) == size * (size - 1) // 2
    assert all(e.squared_distance == value(0) for e in result.entries)
    assert distance_graph(source, value(0)).graph.vertex_count == size


def test_shared_large_translation_is_accepted_for_64_points() -> None:
    translation = 10**250
    source = configuration(tuple((value(translation + i), value(0)) for i in range(64)))
    result = distance_profile(source)
    assert sum(e.pair_count for e in result.entries) == 2016
    assert len(result.entries) == 63
    decoded = DistanceProfileResult[
        QuadraticPointConfiguration, RealQuadraticValue
    ].model_validate_json(result.model_dump_json())
    assert distance_graph(
        decoded.configuration, decoded.entries[0].squared_distance
    ).graph.edges == tuple((i, i + 1) for i in range(63))


def test_graph_selection_does_not_require_serializing_unselected_large_distances() -> (
    None
):
    source = configuration(
        ((value(0), value(0)), (value(1), value(0)), (value(10**250), value(0)))
    )
    with pytest.raises(OperationResourceAdmissionError):
        distance_profile(source)
    assert distance_graph(source, value(1)).graph.edges == ((0, 1),)


def test_mixed_and_non_square_free_fields_reject() -> None:
    with pytest.raises(ValidationError, match="radicand"):
        configuration(((value(0), value(0, d=2)),))
    invalid = configuration(((value(0, d=4), value(0, d=4)),), 4)
    with pytest.raises(OperationDomainValidationError, match="square-free"):
        distance_profile(invalid)
    source = configuration(((value(0), value(0)),))
    with pytest.raises(OperationDomainValidationError, match="same quadratic field"):
        distance_graph(source, value(1, d=2))
    with pytest.raises(OperationDomainValidationError, match="nonnegative"):
        distance_graph(source, value(-1))


def test_existing_operation_ids_accept_quadratic_sources_and_retain_types() -> None:
    tool = next(
        t for t in TOOLS if t.operation_id == "geometry.points.distance_profile.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(EQUILATERAL_QUADRATIC))
    result = tool.run(request)
    assert tool.result_type.model_validate_json(result.model_dump_json()) == result


def test_expired_deadline_is_preserved() -> None:
    source = configuration(((value(0), value(0)), (value(1), value(0))))
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            distance_profile(source)


def test_graph_selection_does_not_inherit_profile_output_bits() -> None:
    huge = Fraction(1, 10**250 + 7)
    points = tuple(
        (
            value(huge, Fraction(1, 10**249 + i + 3)),
            value(Fraction(1, 10**248 + i + 5), Fraction(1, 10**247 + i + 11)),
        )
        for i in range(64)
    )
    selected = distance_graph(configuration(points), value(0))
    assert selected.graph.edges == ()
