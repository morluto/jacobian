from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.topology import (
    ChainCoefficientRing,
    ChainComplexRequest,
    FiniteSimplicialComplex,
    IntegralSimplicialHomologyRequest,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)
from jacobian.domains.topology.operations import (
    _canonical_complex,
    _canonicalize,
    _chain_result,
    _homology,
)


def test_facet_request_rejects_duplicates_nonmaximal_faces_and_hidden_isolates() -> (
    None
):
    with pytest.raises(ValidationError, match="distinct"):
        SimplicialComplexRequest(
            vertices=("a", "b"),
            facets=(("a", "b"), ("b", "a")),
        )
    with pytest.raises(ValidationError, match="maximal"):
        SimplicialComplexRequest(
            vertices=("a", "b"),
            facets=(("a",), ("a", "b")),
        )
    with pytest.raises(ValidationError, match="singleton"):
        SimplicialComplexRequest(
            vertices=("a", "b", "isolated"),
            facets=(("a", "b"),),
        )


def test_chain_and_homology_requests_validate_prime_semantics() -> None:
    complex_ = _canonical_complex(("a", "b"), (("a", "b"),))

    with pytest.raises(ValidationError, match="must not declare a prime"):
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.INTEGER,
            prime=2,
        )
    with pytest.raises(ValidationError, match="bounded prime"):
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=9,
        )
    with pytest.raises(ValidationError, match="bounded prime"):
        SimplicialHomologyRequest(complex=complex_, prime=15)


def test_canonical_complex_composes_as_the_authoritative_object() -> None:
    canonical = _canonicalize(
        SimplicialComplexRequest(
            vertices=("c", "a", "b"),
            facets=(("b", "a"), ("c", "b"), ("c", "a")),
        )
    ).value

    complex_ = canonical.complex
    chain = _chain_result(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
        )
    )
    homology = _homology(SimplicialHomologyRequest(complex=complex_, prime=2)).value

    assert chain.complex_digest == complex_.complex_digest
    assert homology.complex_digest == complex_.complex_digest
    assert tuple(group.betti_number for group in homology.groups) == (1, 1)


def test_chain_bounds_are_checked_after_materialization_but_before_computation() -> (
    None
):
    vertices = tuple(f"v{index}" for index in range(64))
    facets = tuple(
        tuple(f"v{start + offset}" for offset in range(8)) for start in range(0, 64, 8)
    )
    complex_ = _canonical_complex(vertices, facets)

    assert complex_.closure_size == 8 * 255
    with pytest.raises(ValidationError, match="chain group"):
        SimplicialHomologyRequest(complex=complex_, prime=2)


def test_inline_homology_rejects_basis_that_exceeds_its_inline_budget() -> None:
    vertices = tuple(f"v{index}" for index in range(64))
    edges = (
        *((f"v{index}", f"v{(index + 1) % 64}") for index in range(64)),
        ("v0", "v2"),
    )
    complex_ = _canonical_complex(vertices, edges)

    with pytest.raises(ValidationError, match="inline homology bases"):
        SimplicialHomologyRequest(complex=complex_, prime=2)


def test_integral_homology_has_tighter_certificate_size_bounds() -> None:
    too_many_vertices = tuple(f"v{index}" for index in range(17))
    vertex_complex = _canonical_complex(
        too_many_vertices,
        tuple((vertex,) for vertex in too_many_vertices),
    )
    with pytest.raises(ValidationError, match="at most 16 simplices"):
        IntegralSimplicialHomologyRequest(complex=vertex_complex)

    projective_plane_facets = (
        ("0", "1", "2"),
        ("0", "1", "3"),
        ("0", "2", "4"),
        ("0", "3", "5"),
        ("0", "4", "5"),
        ("1", "2", "5"),
        ("1", "3", "4"),
        ("1", "4", "5"),
        ("2", "3", "4"),
        ("2", "3", "5"),
        ("6",),
        ("7",),
    )
    total_rank_too_large = _canonical_complex(
        tuple(str(index) for index in range(8)),
        projective_plane_facets,
    )
    assert max(total_rank_too_large.f_vector) <= 16
    assert sum(total_rank_too_large.f_vector) == 33
    with pytest.raises(ValidationError, match="total chain rank at most 32"):
        IntegralSimplicialHomologyRequest(complex=total_rank_too_large)


def test_stale_complex_digest_reports_field_level_loc() -> None:
    """A stale ``complex_digest`` must produce a Pydantic error whose ``loc``
    targets the ``complex_digest`` field (not a model-level ``()``), so the
    enrichment helper can surface ``complex/complex_digest`` to the agent.
    """

    good = _canonical_complex(("a", "b", "c"), (("a", "b"), ("b", "c")))
    bad_payload = good.model_dump(mode="python")
    bad_payload["complex_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError) as exc_info:
        FiniteSimplicialComplex.model_validate(bad_payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("complex_digest",)
    assert "complex_digest" in errors[0]["msg"]
