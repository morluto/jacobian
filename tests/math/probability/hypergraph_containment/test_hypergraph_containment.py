from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.canonical import parse_canonical_integer
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.probability.hypergraph_containment.operations import (
    compute_hypergraph_vertex_containment,
    verify_hypergraph_vertex_containment,
)


def _hg(vertices, edges):
    return FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((eid, tuple(m)) for eid, m in edges),
    )


def test_single_edge_p1() -> None:
    """With p=1, every vertex is retained, probability 1."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )
    assert parse_canonical_integer(result.success_count) == 1
    assert parse_canonical_integer(result.total_state_count) == 4
    assert result.probability.as_fraction() == Fraction(1)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert decoded.cardinality_axis == (0, 1, 2)
    assert verify_hypergraph_vertex_containment(decoded)
    assert not verify_hypergraph_vertex_containment(
        decoded.model_copy(update={"success_count": "0"})
    )


def test_single_edge_p0() -> None:
    """With p=0, no vertices retained, probability 0."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(0))
    )
    assert result.probability.as_fraction() == Fraction(0)


def test_single_edge_p_half() -> None:
    """With p=1/2: P(contains edge) = P(both retained) = (1/2)^2 = 1/4."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )
    assert result.probability.as_fraction() == Fraction(1, 4)
    assert parse_canonical_integer(result.success_count) == 1


def test_empty_hypergraph() -> None:
    """Empty hypergraph: no edge can be contained, probability 0."""
    hg = _hg(["a", "b"], [])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )
    assert parse_canonical_integer(result.success_count) == 0
    assert result.probability.as_fraction() == Fraction(0)


def test_two_edges() -> None:
    """Two edges: success = contains edge 0 OR contains edge 1."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b")), ("e1", ("b", "c"))])
    p = CanonicalRational.from_fraction(Fraction(1))
    result = compute_hypergraph_vertex_containment(hg, p)
    assert parse_canonical_integer(result.success_count) == 3


def test_subset_counts_sum() -> None:
    """Sum of counts equals success_count."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 3))
    )
    assert sum(
        parse_canonical_integer(value) for value in result.containing_subset_counts
    ) == parse_canonical_integer(result.success_count)


def test_total_state_count() -> None:
    """Total state count = 2^n."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b"))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )
    assert parse_canonical_integer(result.total_state_count) == 8


def test_result_preserves_source() -> None:
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    p = CanonicalRational.from_fraction(Fraction(1, 2))
    result = compute_hypergraph_vertex_containment(hg, p)
    assert result.hypergraph == hg
    assert result.retention_probability == p


def test_edgeless_large_hypergraph_uses_closed_form() -> None:
    hg = _hg([f"v{i}" for i in range(23)], [])

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.total_state_count) == 1 << 23
    assert parse_canonical_integer(result.success_count) == 0
    assert result.probability.as_fraction() == 0


def test_empty_edge_large_hypergraph_uses_closed_form() -> None:
    hg = _hg([f"v{i}" for i in range(23)], [("empty", ())])

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.success_count) == 1 << 23
    assert parse_canonical_integer(result.containing_subset_counts[11]) == comb(23, 11)


def test_single_large_edge_uses_closed_form_before_state_cap() -> None:
    vertices = [f"v{i}" for i in range(23)]
    hg = _hg(vertices, [("all", tuple(vertices))])

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.success_count) == 1
    assert parse_canonical_integer(result.containing_subset_counts[22]) == 0
    assert parse_canonical_integer(result.containing_subset_counts[23]) == 1
    assert result.probability.as_fraction() == Fraction(1, 2**23)


def test_two_large_minimal_edges_use_inclusion_exclusion() -> None:
    vertices = [f"v{i}" for i in range(23)]
    hg = _hg(
        vertices,
        [
            ("left", tuple(vertices[:22])),
            ("right", tuple(vertices[1:])),
        ],
    )

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.success_count) == 3
    assert parse_canonical_integer(result.containing_subset_counts[22]) == 2
    assert parse_canonical_integer(result.containing_subset_counts[23]) == 1
    assert result.probability.as_fraction() == Fraction(3, 2**23)


def test_three_large_minimal_edges_use_inclusion_exclusion() -> None:
    vertices = [f"v{i}" for i in range(23)]
    hg = _hg(
        vertices,
        [
            ("e0", tuple(vertices[1:])),
            ("e1", (vertices[0], *vertices[2:])),
            ("e2", (vertices[0], vertices[1], *vertices[3:])),
        ],
    )

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.success_count) == 4
    assert parse_canonical_integer(result.containing_subset_counts[22]) == 3
    assert parse_canonical_integer(result.containing_subset_counts[23]) == 1
    assert result.probability.as_fraction() == Fraction(4, 2**23)


def test_duplicate_edge_members_are_scanned_once() -> None:
    hg = _hg(
        ["a", "b", "c"],
        [("e0", ("a", "b")), ("e1", ("a", "b"))],
    )

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )

    assert parse_canonical_integer(result.success_count) == 2


def test_probability_growth_uses_event_support() -> None:
    hg = _hg([f"v{i}" for i in range(22)], [("e0", ("v0",))])
    p = CanonicalRational.from_fraction(Fraction(1, 10**2000))

    result = compute_hypergraph_vertex_containment(hg, p)

    assert result.probability == p


def test_enumerates_active_support_and_lifts_isolates() -> None:
    hg = _hg([f"v{i}" for i in range(23)], [("e0", ("v0",))])

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )

    assert parse_canonical_integer(result.success_count) == 1 << 22
    assert parse_canonical_integer(result.containing_subset_counts[1]) == 1
    assert parse_canonical_integer(result.containing_subset_counts[23]) == 1


def test_dominated_edges_are_removed_before_work_admission() -> None:
    vertices = [f"v{i}" for i in range(13)]
    edges = []
    for subset in range(1 << 12):
        members = tuple(
            ["v0"]
            + [vertices[index + 1] for index in range(12) if subset & (1 << index)]
        )
        edges.append((f"e{subset}", members))
    hg = _hg(vertices, edges)

    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1))
    )

    assert parse_canonical_integer(result.success_count) == 1 << 12


def test_result_rejects_incomplete_cardinality_axis() -> None:
    hg = _hg(["a", "b"], [("e", ("a",))])
    result = compute_hypergraph_vertex_containment(
        hg, CanonicalRational.from_fraction(Fraction(1, 2))
    )
    forged = result.model_dump(mode="json")
    forged["cardinality_axis"] = []
    with pytest.raises(ValidationError, match="every subset cardinality"):
        type(result).model_validate(forged)
