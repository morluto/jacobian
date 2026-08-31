from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternProfileRequest,
)
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile.operations import (
    compute_edge_pattern_profile,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def _hg(vertices, edges):
    return FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((eid, tuple(m)) for eid, m in edges),
    )


def test_mixed_coloring() -> None:
    """Edge with colors (red, red, blue) has equality partition (0,0,1) and 2 blocks."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    colors = {"a": "red", "b": "red", "c": "blue"}
    result = compute_edge_pattern_profile(hg, colors)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.equality_partition == (0, 0, 1)
    assert entry.num_color_blocks == 2
    assert entry.color_labels == ("red", "blue")
    assert "e0" not in result.monochromatic_edge_ids
    assert "e0" not in result.rainbow_edge_ids


def test_monochromatic_edge() -> None:
    """All-same colour edge is monochromatic."""
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    colors = {"a": "red", "b": "red"}
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.monochromatic_edge_ids
    assert "e0" not in result.rainbow_edge_ids


def test_rainbow_edge() -> None:
    """All-distinct colour edge is rainbow."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    colors = {"a": "red", "b": "green", "c": "blue"}
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.rainbow_edge_ids
    assert "e0" not in result.monochromatic_edge_ids


def test_multiple_edges() -> None:
    """Multiple edges with different patterns."""
    hg = _hg(
        ["a", "b", "c"], [("e0", ("a", "b")), ("e1", ("b", "c")), ("e2", ("a", "c"))]
    )
    colors = {"a": "red", "b": "red", "c": "blue"}
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.monochromatic_edge_ids
    assert "e1" in result.rainbow_edge_ids
    assert "e2" in result.rainbow_edge_ids


def test_empty_hypergraph() -> None:
    """Empty hypergraph has no entries."""
    hg = _hg(["a"], [])
    colors = {"a": "red"}
    result = compute_edge_pattern_profile(hg, colors)
    assert len(result.entries) == 0
    assert len(result.monochromatic_edge_ids) == 0
    assert len(result.rainbow_edge_ids) == 0


def test_empty_edge_is_monochromatic() -> None:
    hg = _hg(["a"], [("empty", ())])
    result = compute_edge_pattern_profile(hg, {"a": "red"})

    assert result.entries[0].num_color_blocks == 0
    assert result.monochromatic_edge_ids == ("empty",)
    assert result.rainbow_edge_ids == ("empty",)


def test_numeric_color_keys_use_strict_delivery_sizing() -> None:
    hg = _hg(["num", "den"], [("e0", ("num", "den"))])
    result = compute_edge_pattern_profile(hg, {"num": "red", "den": "blue"})

    assert {(p.vertex, p.color) for p in result.vertex_colors} == {("den", "blue"), ("num", "red")}


def test_result_preserves_source() -> None:
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    colors = {"a": "red", "b": "blue"}
    result = compute_edge_pattern_profile(hg, colors)
    assert result.hypergraph == hg
    assert {(p.vertex, p.color) for p in result.vertex_colors} == {("a", "red"), ("b", "blue")}


def test_rejects_incomplete_colors() -> None:
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b"))])
    with pytest.raises(ValidationError):
        EdgePatternProfileRequest(
            hypergraph=hg, vertex_colors={"a": "red", "b": "blue"}
        )


def test_native_rejects_oversized_color_before_normalization() -> None:
    # Thread 3: The fixed per-label cap was replaced by an aggregate UTF-8 bound.
    # A 65-character label on a one-vertex hypergraph is now admitted.
    hg = _hg(["a"], [])
    result = compute_edge_pattern_profile(hg, {"a": "x" * 65})
    assert len(result.vertex_colors) == 1
    assert result.vertex_colors[0].vertex == "a"
    assert result.vertex_colors[0].color == "x" * 65


def test_native_rejects_unencodable_color() -> None:
    hg = _hg(["a"], [])
    with pytest.raises(OperationDomainValidationError, match="valid UTF-8"):
        compute_edge_pattern_profile(hg, {"a": "\ud800"})
