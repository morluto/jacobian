from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring.operations import (
    decide_nonmonochromatic_coloring,
    verify_coloring_witness,
    verify_non_colorable,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def _hg(vertices, edges):
    return FiniteHypergraph(
        vertices=tuple(vertices),
        edges=tuple((eid, tuple(m)) for eid, m in edges),
    )


def test_colorable_3edge() -> None:
    """One 3-edge hypergraph on {0,1,2} with q=2 is colourable."""
    h = _hg(["0", "1", "2"], [("e0", ("0", "1", "2"))])
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "COLORABLE"
    assert result.witness is not None
    color_map = dict(result.witness.assignments)
    assert color_map["0"] != color_map["2"] or color_map["1"] != color_map["2"]


def test_serialized_colorable_witness_is_verifiable_and_forgery_resistant() -> None:
    h = _hg(["0", "1", "2"], [("e0", ("0", "1", "2"))])
    result = decide_nonmonochromatic_coloring(h, 2)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_coloring_witness(decoded)

    forged = result.model_dump(mode="json")
    forged["witness"]["assignments"] = [["0", 0], ["1", 0], ["2", 0]]
    assert not verify_coloring_witness(type(result).model_validate(forged))

    forged["witness"]["assignments"] = [["0", 0], ["1", 0]]
    assert not verify_coloring_witness(type(result).model_validate(forged))


def test_not_colorable_k3() -> None:
    """K3 as 2-uniform hypergraph with q=2: not colourable."""
    h = _hg(
        ["0", "1", "2"],
        [("e0", ("0", "1")), ("e1", ("1", "2")), ("e2", ("0", "2"))],
    )
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "NOT_COLORABLE"
    assert result.witness is None


def test_serialized_not_colorable_claim_is_explicitly_verifiable() -> None:
    h = _hg(
        ["0", "1", "2"],
        [("e0", ("0", "1")), ("e1", ("1", "2")), ("e2", ("0", "2"))],
    )
    result = decide_nonmonochromatic_coloring(h, 2)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_non_colorable(decoded)

    forged = result.model_dump(mode="json")
    forged["outcome"] = "COLORABLE"
    assert not verify_non_colorable(type(result).model_validate(forged))


def test_empty_edges_are_vacuously_colorable() -> None:
    """An edgeless hypergraph has a vacuous non-monochromatic colouring."""
    h = _hg(["0", "1"], [])
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "COLORABLE"


def test_q1_singleton_edge_not_colorable() -> None:
    """q=1 makes every edge monochromatic."""
    h = _hg(["0", "1"], [("e0", ("0", "1"))])
    result = decide_nonmonochromatic_coloring(h, 1)
    assert result.outcome == "NOT_COLORABLE"


def test_q1_empty_edges_colorable() -> None:
    """A positive palette still colors an edgeless hypergraph vacuously."""
    h = _hg(["0"], [])
    result = decide_nonmonochromatic_coloring(h, 1)
    assert result.outcome == "COLORABLE"


def test_witness_replay() -> None:
    """Replay: every edge is non-monochromatic under the witness colouring."""
    h = _hg(
        ["0", "1", "2", "3"],
        [("e0", ("0", "1", "2")), ("e1", ("1", "2", "3"))],
    )
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "COLORABLE"
    color_map = dict(result.witness.assignments)
    for _, members in h.edges:
        colors = {color_map[m] for m in members}
        assert len(colors) >= 2


def test_q2_k3_colorable_with_q3() -> None:
    """K3 as 2-uniform with q=3 is colourable."""
    h = _hg(
        ["0", "1", "2"],
        [("e0", ("0", "1")), ("e1", ("1", "2")), ("e2", ("0", "2"))],
    )
    result = decide_nonmonochromatic_coloring(h, 3)
    assert result.outcome == "COLORABLE"


def test_native_admission_allows_palette_above_legacy_cap_for_injective_case() -> None:
    h = _hg(["0", "1"], [("e0", ("0", "1"))])
    result = decide_nonmonochromatic_coloring(h, 17)
    assert result.outcome == "COLORABLE"


def test_large_injective_palette_does_not_overflow_deadline_budget() -> None:
    h = _hg(
        [str(index) for index in range(20)],
        [("e0", tuple(str(index) for index in range(20)))],
    )
    result = decide_nonmonochromatic_coloring(h, (1 << 53) - 1)
    assert result.outcome == "COLORABLE"


def test_colorable_verifier_rejects_palette_outside_operation_domain() -> None:
    h = _hg(["a", "b"], [("e0", ("a", "b"))])
    result = decide_nonmonochromatic_coloring(h, 2)
    forged = result.model_dump(mode="json")
    forged["palette_size"] = 1 << 53
    claim = type(result).model_validate(forged)
    assert not verify_coloring_witness(claim)


def test_empty_hyperedge_is_not_colorable() -> None:
    h = _hg(["0"], [("e0", ())])
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "NOT_COLORABLE"


def test_exhaustive_oracle() -> None:
    """Compare against independent exhaustive colouring check."""
    from itertools import product

    h = _hg(
        ["a", "b", "c"],
        [("e0", ("a", "b")), ("e1", ("b", "c")), ("e2", ("a", "c"))],
    )
    q = 2
    result = decide_nonmonochromatic_coloring(h, q)
    vertices = list(h.vertices)
    found = False
    for coloring in product(range(q), repeat=len(vertices)):
        color_map = {vertices[i]: coloring[i] for i in range(len(vertices))}
        valid = True
        for _, members in h.edges:
            colors = {color_map[m] for m in members}
            if len(colors) == 1:
                valid = False
                break
        if valid:
            found = True
            break
    assert (result.outcome == "COLORABLE") == found


def test_singleton_edge_not_colorable() -> None:
    """A singleton edge makes the hypergraph non-colourable for any q."""
    h = _hg(["0"], [("e0", ("0",))])
    result = decide_nonmonochromatic_coloring(h, 2)
    assert result.outcome == "NOT_COLORABLE"


def test_native_admission_matches_request_bounds() -> None:
    h = _hg([str(i) for i in range(256)], [("e0", ("0", "1"))])
    with pytest.raises(OperationDomainValidationError, match="edge checks"):
        decide_nonmonochromatic_coloring(h, 16)


def test_large_carrier_with_cheap_search_is_admitted() -> None:
    h = _hg([str(i) for i in range(17)], [("e0", ("0", "1"))])
    result = decide_nonmonochromatic_coloring(h, 1)
    assert result.outcome == "NOT_COLORABLE"


def test_rejects_search_work_before_enumeration() -> None:
    h = _hg([str(i) for i in range(16)], [("e0", ("0", "1"))])
    result = decide_nonmonochromatic_coloring(h, 16)
    assert result.outcome == "COLORABLE"
    assert result.witness is not None
