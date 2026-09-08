"""Independent reconstruction and accepted exact geometry boundaries."""

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs import FiniteHypergraph
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    HyperedgeColorAssignment,
    IndexedHyperedgeColoring,
)
from jacobian.math.geometry.exact._models import (
    LabelledRationalPoint,
    PointConfiguration,
)
from jacobian.math.geometry.exact.distance_edge_coloring import (
    DistanceEdgeColoringResult,
    compute_distance_edge_coloring,
)


def configuration(
    rows: Sequence[Sequence[int | Fraction]], labels: Sequence[str] | None = None
) -> PointConfiguration:
    return PointConfiguration(
        points=tuple(
            LabelledRationalPoint(
                label=labels[i] if labels is not None else f"p{i}",
                coordinates=tuple(
                    CanonicalRational.from_fraction(Fraction(value)) for value in row
                ),
            )
            for i, row in enumerate(rows)
        )
    )


def assert_reconstructs(source: PointConfiguration) -> DistanceEdgeColoringResult:
    result = compute_distance_edge_coloring(source)
    assert result.configuration == source
    palette = tuple(value.as_fraction() for value in result.squared_distances)
    assert palette == tuple(sorted(set(palette)))
    expected = {}
    for i, j in combinations(range(len(source.points)), 2):
        left, right = source.points[i], source.points[j]
        distance = sum(
            (x.as_fraction() - y.as_fraction()) ** 2
            for x, y in zip(left.coordinates, right.coordinates, strict=True)
        )
        expected[f"{i}:{j}"] = (tuple(sorted((left.label, right.label))), distance)
    graph = result.coloring.hypergraph
    assert dict(graph.edges) == {key: value[0] for key, value in expected.items()}
    assert {
        item.edge_id: palette[item.color_index] for item in result.coloring.assignments
    } == {key: value[1] for key, value in expected.items()}
    assert (
        DistanceEdgeColoringResult.model_validate_json(result.model_dump_json())
        == result
    )
    return result


def test_square() -> None:
    result = assert_reconstructs(configuration([(0, 0), (1, 0), (0, 1), (1, 1)]))
    assert [value.num for value in result.squared_distances] == [1, 2]
    assert Counter(item.color_index for item in result.coloring.assignments) == {
        0: 4,
        1: 2,
    }


def test_deserialized_palette_must_be_strictly_increasing() -> None:
    result = assert_reconstructs(configuration([(0, 0), (1, 0), (0, 1), (1, 1)]))
    reversed_palette = result.model_dump()
    reversed_palette["squared_distances"] = list(
        reversed(reversed_palette["squared_distances"])
    )
    reversed_palette["coloring"]["assignments"] = [
        {**item, "color_index": 1 - item["color_index"]}
        for item in reversed_palette["coloring"]["assignments"]
    ]
    with pytest.raises(ValidationError, match="strictly increasing"):
        DistanceEdgeColoringResult.model_validate(reversed_palette)
    duplicate = result.model_dump()
    duplicate["squared_distances"] = [
        duplicate["squared_distances"][0],
        duplicate["squared_distances"][0],
    ]
    with pytest.raises(ValidationError, match="strictly increasing"):
        DistanceEdgeColoringResult.model_validate(duplicate)


def test_pythagorean_pair_admits_reduced_unit_distance() -> None:
    m = 10**9000
    point = (
        Fraction(m * m - 1, m * m + 1),
        Fraction(2 * m, m * m + 1),
    )
    result = assert_reconstructs(configuration([(0, 0), point]))
    assert result.squared_distances[0].as_fraction() == 1


@pytest.mark.parametrize(
    "rows",
    [
        [(0,), (1,)],
        [(Fraction(2, 3),), (Fraction(2, 3),)],
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(0,), (1,), (3,), (7,)],
        [(Fraction(i, 7), Fraction(i * i, 11)) for i in range(8)],
        [(i % 4, i // 4, i % 3) for i in range(20)],
    ],
)
def test_complete_pairwise_reconstruction(
    rows: Sequence[Sequence[int | Fraction]],
) -> None:
    assert_reconstructs(configuration(rows))


def test_source_labels_and_edge_ids_survive_reordering() -> None:
    source = configuration([(0,), (1,), (3,)], ["z", "e\u0301", "é"])
    result = assert_reconstructs(source)
    assert result.coloring.hypergraph.vertices == ("z", "e\u0301", "é")
    coloring = result.coloring.model_copy()
    payload = coloring.model_dump(mode="json")
    payload["assignments"].reverse()
    assert IndexedHyperedgeColoring.model_validate(payload) == coloring
    reversed_result = assert_reconstructs(
        source.model_copy(update={"points": source.points[::-1]})
    )
    assert reversed_result.squared_distances == result.squared_distances


def test_large_translation_and_coincident_coordinates() -> None:
    shift = 10**30_000
    translated = configuration([(shift + i,) for i in range(64)])
    result = assert_reconstructs(translated)
    assert len(result.coloring.assignments) == 2016
    assert len(result.squared_distances) == 63
    coincident = assert_reconstructs(configuration([(shift, shift), (shift, shift)]))
    assert coincident.squared_distances == (CanonicalRational(num=0, den=1),)


def test_large_representable_squared_distance() -> None:
    result = assert_reconstructs(configuration([(0,), (10**16_000,)]))
    assert result.squared_distances[0].num == 10**32_000


def test_required_distance_height_rejects() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="growth"):
        compute_distance_edge_coloring(configuration([(0,), (10**20_000,)]))


def test_distinct_rationals_do_not_collapse() -> None:
    result = assert_reconstructs(
        configuration([(0,), (Fraction(1, 10**50),), (Fraction(1, 10**50 + 1),)])
    )
    assert len(result.squared_distances) == 3


def test_empty_and_duplicate_hyperedge_partition() -> None:
    empty = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(vertices=(), edges=()),
        color_count=0,
        assignments=(),
    )
    assert (
        IndexedHyperedgeColoring.model_validate_json(empty.model_dump_json()) == empty
    )
    duplicate = IndexedHyperedgeColoring(
        hypergraph=FiniteHypergraph(
            vertices=("v",), edges=(("a", ("v",)), ("b", ("v",)))
        ),
        color_count=1,
        assignments=(
            HyperedgeColorAssignment(edge_id="b", color_index=0),
            HyperedgeColorAssignment(edge_id="a", color_index=0),
        ),
    )
    assert tuple(item.edge_id for item in duplicate.assignments) == ("a", "b")


@pytest.mark.parametrize(
    "assignments,count",
    [([], 0), ([("a", 0), ("a", 0)], 1), ([("a", 1)], 2), ([("x", 0)], 1)],
)
def test_malformed_partition_rejects(
    assignments: list[tuple[str, int]], count: int
) -> None:
    with pytest.raises(ValidationError):
        IndexedHyperedgeColoring(
            hypergraph=FiniteHypergraph(vertices=("v",), edges=(("a", ("v",)),)),
            color_count=count,
            assignments=tuple(
                HyperedgeColorAssignment(edge_id=key, color_index=color)
                for key, color in assignments
            ),
        )


def test_source_coefficient_storage_rejects_before_pair_work() -> None:
    value = 10**2100
    with pytest.raises(OperationResourceAdmissionError, match="source coefficients"):
        compute_distance_edge_coloring(configuration([(value,) * 20] * 64))


def test_incommensurate_denominator_growth_rejects() -> None:
    values = tuple(Fraction(1, 10**1000 + 2 * i + 1) for i in range(20))
    with pytest.raises(OperationResourceAdmissionError, match="coefficient"):
        compute_distance_edge_coloring(configuration([(0,) * 20, values]))


def test_work_budget_rejects_large_rational_palette() -> None:
    values = [(Fraction(1, 10**2000 + 2 * i + 1),) for i in range(64)]
    with pytest.raises(OperationResourceAdmissionError, match="work bound"):
        compute_distance_edge_coloring(configuration(values))


@pytest.mark.parametrize("bind_expired", [False, True])
def test_expired_request_context_is_an_operational_timeout(bind_expired: bool) -> None:
    from time import monotonic

    from jacobian._execution import (
        OperationExecutionTimeoutError,
        bind_request_deadline,
        request_execution,
    )

    source = configuration([(0,), (1,)])
    with request_execution(monotonic() if bind_expired else monotonic() - 100):
        if bind_expired:
            bind_request_deadline(monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            compute_distance_edge_coloring(source)


def test_colliding_numeric_hashes_retain_distance_classes() -> None:
    import sys

    modulus = sys.hash_info.modulus
    result = assert_reconstructs(configuration([(i * modulus,) for i in range(64)]))
    assert len(result.squared_distances) == 63
    assert all(hash(value.as_fraction()) == 0 for value in result.squared_distances)


def test_surrogate_point_label_is_rejected_before_distances() -> None:
    source = configuration([(0,), (1,)])
    forged = source.model_copy(
        update={
            "points": (
                source.points[0],
                source.points[1].model_copy(update={"label": "\ud800"}),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError, match="UTF-8"):
        compute_distance_edge_coloring(forged)


def test_deserialized_palette_must_be_strictly_increasing() -> None:
    result = assert_reconstructs(configuration([(0,), (1,), (3,)]))
    duplicate = result.model_dump()
    palette = list(duplicate["squared_distances"])
    palette[1] = palette[0]
    duplicate["squared_distances"] = palette
    with pytest.raises(ValidationError, match="strictly increasing"):
        DistanceEdgeColoringResult.model_validate(duplicate)
    reordered = result.model_dump()
    reordered["squared_distances"] = list(reversed(reordered["squared_distances"]))
    with pytest.raises(ValidationError, match="strictly increasing"):
        DistanceEdgeColoringResult.model_validate(reordered)
