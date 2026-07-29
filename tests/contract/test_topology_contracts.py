from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.topology import (
    ChainCoefficientRing,
    ChainComplexRequest,
    SimplicialComplexRequest,
    SimplicialHomologyRequest,
)
from jacobian.domains.topology.operations import _materialized_complex


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
    complex_ = _materialized_complex(("a", "b"), (("a", "b"),))

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


def test_chain_bounds_are_checked_after_materialization_but_before_computation() -> (
    None
):
    vertices = tuple(f"v{index}" for index in range(64))
    facets = tuple(
        tuple(f"v{start + offset}" for offset in range(8)) for start in range(0, 64, 8)
    )
    complex_ = _materialized_complex(vertices, facets)

    assert complex_.closure_size == 8 * 255
    with pytest.raises(ValidationError, match="chain group"):
        SimplicialHomologyRequest(complex=complex_, prime=2)
