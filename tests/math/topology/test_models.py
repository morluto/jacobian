from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.topology._homology import (
    IntegralSimplicialHomologyRequest,
    SimplicialHomologyRequest,
)
from jacobian.math.topology._models import (
    ChainCoefficientRing,
    ChainComplexRequest,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
)
from jacobian.math.topology._tools import TOOLS


def _operation(operation_id: str) -> MathTool:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def _canonical_complex(vertices, facets):
    """Build a canonical FiniteSimplicialComplex via its owner declaration."""
    request = SimplicialComplexRequest(vertices=vertices, facets=facets)
    operation = _operation("topology.simplicial_complex.canonicalize")
    return operation.run(request).complex


def test_facet_request_rejects_duplicates_nonmaximal_faces_and_hidden_isolates() -> (
    None
):
    with pytest.raises(ValidationError):
        SimplicialComplexRequest(
            vertices=("a", "b"),
            facets=(("a", "b"), ("b", "a")),
        )
    with pytest.raises(ValidationError):
        SimplicialComplexRequest(
            vertices=("a", "b"),
            facets=(("a",), ("a", "b")),
        )
    with pytest.raises(ValidationError):
        SimplicialComplexRequest(
            vertices=("a", "b", "isolated"),
            facets=(("a", "b"),),
        )


def test_chain_and_homology_requests_validate_prime_semantics() -> None:
    complex_ = _canonical_complex(("a", "b"), (("a", "b"),))

    with pytest.raises(ValidationError):
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.INTEGER,
            prime=2,
        )
    with pytest.raises(ValidationError):
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=9,
        )
    with pytest.raises(ValidationError):
        SimplicialHomologyRequest(complex=complex_, prime=15)


def test_canonical_complex_composes_as_the_authoritative_object() -> None:
    request = SimplicialComplexRequest(
        vertices=("c", "a", "b"),
        facets=(("b", "a"), ("c", "b"), ("c", "a")),
    )
    canonical = _operation("topology.simplicial_complex.canonicalize").run(request)
    complex_ = canonical.complex

    chain_operation = _operation("topology.simplicial_complex.chain_complex.compute")
    chain = chain_operation.run(
        ChainComplexRequest(
            complex=complex_,
            coefficient_ring=ChainCoefficientRing.PRIME_FIELD,
            prime=2,
        )
    )

    homology_operation = _operation("topology.simplicial_homology.compute")
    homology = homology_operation.run(
        SimplicialHomologyRequest(complex=complex_, prime=2)
    )

    assert chain.complex_digest == complex_.complex_digest
    assert homology.complex_digest == complex_.complex_digest
    assert tuple(group.betti_number for group in homology.groups) == (1, 1)


def test_integral_homology_runs_through_the_public_operation() -> None:
    complex_ = _canonical_complex(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    operation = _operation("topology.simplicial_homology.integral.compute")

    result = operation.run(IntegralSimplicialHomologyRequest(complex=complex_))

    assert result.complex_digest == complex_.complex_digest
    assert result.coefficient_ring == "ZZ"
    assert tuple(group.betti_number for group in result.groups) == (1, 1)


def test_chain_bounds_are_checked_after_materialization_but_before_computation() -> (
    None
):
    vertices = tuple(f"v{index}" for index in range(64))
    facets = tuple(
        tuple(f"v{start + offset}" for offset in range(8)) for start in range(0, 64, 8)
    )
    complex_ = _canonical_complex(vertices, facets)

    assert complex_.closure_size == 8 * (
        2**8 - 1
    )  # 8 simplices, each closing to 2^8-1 faces
    with pytest.raises(ValidationError):
        SimplicialHomologyRequest(complex=complex_, prime=2)


def test_inline_homology_rejects_basis_that_exceeds_its_inline_budget() -> None:
    vertices = tuple(f"v{index}" for index in range(64))
    edges = (
        *((f"v{index}", f"v{(index + 1) % 64}") for index in range(64)),
        ("v0", "v2"),
    )
    complex_ = _canonical_complex(vertices, edges)

    with pytest.raises(ValidationError):
        SimplicialHomologyRequest(complex=complex_, prime=2)


def test_integral_homology_chain_groups_derive_from_certificate_dimension() -> None:
    """Every integral-homology certificate matrix is a ``CertifiedIntegerMatrix``
    bounded at ``MAX_CERTIFIED_SNF_DIMENSION`` = 32, so a chain group of 33
    simplices must be rejected at admission rather than fail construction."""
    too_many_vertices = tuple(f"v{index}" for index in range(33))
    vertex_complex = _canonical_complex(
        too_many_vertices,
        tuple((vertex,) for vertex in too_many_vertices),
    )
    assert max(vertex_complex.f_vector) == 33
    with pytest.raises(ValidationError):
        IntegralSimplicialHomologyRequest(complex=vertex_complex)


def test_integral_homology_certificate_boundary_runs_the_public_operation() -> None:
    """32 isolated vertices sit exactly on the certificate-dimension boundary:
    the public operation returns a typed result whose H_0 carries one free
    generator per component instead of failing result construction."""
    vertices = tuple(f"v{index}" for index in range(32))
    complex_ = _canonical_complex(vertices, tuple((vertex,) for vertex in vertices))
    operation = _operation("topology.simplicial_homology.integral.compute")

    result = operation.run(IntegralSimplicialHomologyRequest(complex=complex_))

    assert result.groups[0].betti_number == 32
    assert len(result.groups[0].free_generators) == 32


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
    assert errors[0]["type"] == "topology.require_digest_binds_canonical_complex_1"
    assert "complex_digest" in errors[0]["msg"]
