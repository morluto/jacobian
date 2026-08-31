from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError, OperationResult
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternProfileRequest,
    VertexColorPair,
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


def _colors(mapping: dict[str, str]) -> tuple[VertexColorPair, ...]:
    return tuple(VertexColorPair(vertex=k, color=v) for k, v in mapping.items())


def test_mixed_coloring() -> None:
    """Edge with colors (red, red, blue) has equality partition (0,0,1) and 2 blocks."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    colors = _colors({"a": "red", "b": "red", "c": "blue"})
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
    colors = _colors({"a": "red", "b": "red"})
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.monochromatic_edge_ids
    assert "e0" not in result.rainbow_edge_ids


def test_rainbow_edge() -> None:
    """All-distinct colour edge is rainbow."""
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b", "c"))])
    colors = _colors({"a": "red", "b": "green", "c": "blue"})
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.rainbow_edge_ids
    assert "e0" not in result.monochromatic_edge_ids


def test_multiple_edges() -> None:
    """Multiple edges with different patterns."""
    hg = _hg(
        ["a", "b", "c"], [("e0", ("a", "b")), ("e1", ("b", "c")), ("e2", ("a", "c"))]
    )
    colors = _colors({"a": "red", "b": "red", "c": "blue"})
    result = compute_edge_pattern_profile(hg, colors)
    assert "e0" in result.monochromatic_edge_ids
    assert "e1" in result.rainbow_edge_ids
    assert "e2" in result.rainbow_edge_ids


def test_empty_hypergraph() -> None:
    """Empty hypergraph has no entries."""
    hg = _hg(["a"], [])
    colors = _colors({"a": "red"})
    result = compute_edge_pattern_profile(hg, colors)
    assert len(result.entries) == 0
    assert len(result.monochromatic_edge_ids) == 0
    assert len(result.rainbow_edge_ids) == 0


def test_empty_edge_is_monochromatic() -> None:
    hg = _hg(["a"], [("empty", ())])
    result = compute_edge_pattern_profile(hg, _colors({"a": "red"}))

    assert result.entries[0].num_color_blocks == 0
    assert result.monochromatic_edge_ids == ("empty",)
    assert result.rainbow_edge_ids == ("empty",)


def test_numeric_color_keys_survive_canonical_transport() -> None:
    # Thread 2 resolution: vertices named num/den with non-integer colors must
    # not be reinterpreted as a rational object by transport canonicalization.
    hg = _hg(["num", "den"], [("e0", ("num", "den"))])
    result = compute_edge_pattern_profile(hg, _colors({"num": "red", "den": "blue"}))
    output = result.model_dump(mode="json")
    assert {(p["vertex"], p["color"]) for p in output["vertex_colors"]} == {
        ("num", "red"),
        ("den", "blue"),
    }

    envelope = OperationResult(
        operation_id="hypergraph.vertex_coloring.edge_pattern_profile.compute",
        runtime_ms=0,
        output=output,
    )
    assert len(envelope.output["vertex_colors"]) == 2


def test_result_preserves_source() -> None:
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    colors = _colors({"a": "red", "b": "blue"})
    result = compute_edge_pattern_profile(hg, colors)
    assert result.hypergraph == hg
    assert {(p.vertex, p.color) for p in result.vertex_colors} == {
        ("a", "red"),
        ("b", "blue"),
    }


def test_produced_coloring_is_accepted_unchanged_by_the_operation() -> None:
    # Thread 6dmPem resolution: the operation returns the same domain-owned
    # VertexColorPair carrier it accepts, so a produced coloring can be fed
    # straight back in, on the native path and through the request model.
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b")), ("e1", ("b", "c"))])
    colors = _colors({"a": "red", "b": "red", "c": "blue"})
    first = compute_edge_pattern_profile(hg, colors)

    second = compute_edge_pattern_profile(hg, first.vertex_colors)
    assert second.entries == first.entries
    assert second.monochromatic_edge_ids == first.monochromatic_edge_ids
    assert second.rainbow_edge_ids == first.rainbow_edge_ids

    parsed = EdgePatternProfileRequest(hypergraph=hg, vertex_colors=first.vertex_colors)
    assert {p.vertex for p in parsed.vertex_colors} == {"a", "b", "c"}


def test_nfc_colliding_vertex_labels_are_all_admitted() -> None:
    # Thread 6dmPen resolution: two distinct vertex labels that normalize to
    # the same NFC string (precomposed and decomposed e-acute) are both
    # represented, since the pair carrier is not keyed by normalized labels.
    precomposed = "\u00e9"  # é
    decomposed = "e\u0301"  # e + combining acute
    hg = _hg([precomposed, decomposed], [("e0", (precomposed, decomposed))])
    result = compute_edge_pattern_profile(
        hg,
        _colors({precomposed: "red", decomposed: "blue"}),
    )
    assert {(p.vertex, p.color) for p in result.vertex_colors} == {
        (precomposed, "red"),
        (decomposed, "blue"),
    }
    assert result.entries[0].equality_partition == (0, 1)


def test_rejects_incomplete_colors() -> None:
    hg = _hg(["a", "b", "c"], [("e0", ("a", "b"))])
    with pytest.raises(ValidationError):
        EdgePatternProfileRequest(
            hypergraph=hg, vertex_colors=_colors({"a": "red", "b": "blue"})
        )


def test_rejects_repeated_vertex() -> None:
    hg = _hg(["a", "b"], [("e0", ("a", "b"))])
    pairs = (
        VertexColorPair(vertex="a", color="red"),
        VertexColorPair(vertex="a", color="blue"),
        VertexColorPair(vertex="b", color="green"),
    )
    with pytest.raises(ValidationError):
        EdgePatternProfileRequest(hypergraph=hg, vertex_colors=pairs)


def test_color_labels_use_the_output_envelope_without_a_fixed_length_cap() -> None:
    # Thread 3 resolution: no borrowed 64-character ceiling; modest labels are
    # admitted solely by the aggregate UTF-8 check and result-sensitive output
    # admission.
    hg = _hg(["a"], [])
    for size in (65, 200, 5_000):
        long_color = "x" * size
        result = compute_edge_pattern_profile(hg, _colors({"a": long_color}))
        assert result.vertex_colors[0].vertex == "a"
        assert result.vertex_colors[0].color == long_color


def test_native_rejects_unencodable_color() -> None:
    hg = _hg(["a"], [])
    with pytest.raises(OperationDomainValidationError, match="valid UTF-8"):
        compute_edge_pattern_profile(hg, _colors({"a": "\ud800"}))


def test_single_label_over_the_aggregate_utf8_bound_is_rejected() -> None:
    # Thread 3 resolution: the cheap aggregate raw UTF-8 bound rejects a single
    # color label whose raw size already exceeds the output envelope.
    hg = _hg(["a"], [])
    with pytest.raises(OperationDomainValidationError, match="output envelope"):
        compute_edge_pattern_profile(hg, _colors({"a": "x" * (11 * 1024 * 1024)}))


def test_admitted_equality_partitions_are_reused_in_the_kernel() -> None:
    # Thread 1 resolution: admission computes the equality partition, block
    # count, and distinct-label sequence once and the kernel constructs the
    # result from that plan without recomputing the defining mathematical work.
    hg = _hg(
        ["a", "b", "c"],
        [("e0", ("a", "b", "c")), ("e1", ("a", "b")), ("e2", ("b", "c"))],
    )
    result = compute_edge_pattern_profile(
        hg, _colors({"a": "red", "b": "red", "c": "blue"})
    )
    assert result.entries[0].equality_partition == (0, 0, 1)
    assert result.entries[0].num_color_blocks == 2
    assert result.entries[0].color_labels == ("red", "blue")
    assert "e0" not in result.monochromatic_edge_ids
    assert "e1" in result.monochromatic_edge_ids
    assert "e1" not in result.rainbow_edge_ids
    assert "e2" in result.rainbow_edge_ids


def test_decomposed_labels_are_sized_for_non_normalizing_delivery() -> None:
    # Sizing matches direct delivery (encode_strict_json on model_dump), which
    # does not NFC-normalize. A decomposed e-acute edge/member label is kept
    # in its raw form, so admission never undercounts the shipped bytes.
    decomposed = "e\u0301"  # e + combining acute
    hg = _hg([decomposed], [("edge\u0301id", (decomposed,))])
    result = compute_edge_pattern_profile(hg, _colors({decomposed: "red"}))
    assert result.entries[0].edge_id == "edge\u0301id"
    assert result.entries[0].members == (decomposed,)
    assert result.entries[0].color_labels == ("red",)


def test_oversized_color_is_rejected_by_character_count_before_encoding() -> None:
    # The aggregate character-count precheck rejects a color label whose total
    # character count already exceeds the byte envelope, before the full UTF-8
    # string is ever encoded (no host exhaustion).
    hg = _hg(["a"], [])
    with pytest.raises(OperationDomainValidationError, match="output envelope"):
        compute_edge_pattern_profile(hg, _colors({"a": "x" * (12 * 1024 * 1024)}))
