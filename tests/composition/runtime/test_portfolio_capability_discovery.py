"""Whole-portfolio discovery and catalog projection contracts."""

from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityDiscoveryRequest
from jacobian.runtime.model import JacobianRuntime

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "DISCOVERY"


def test_catalog_keeps_chromatic_number_without_unused_encoding_workflow(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    capability_ids = {
        descriptor.capability_id
        for descriptor in (
            attached_complete_runtime_read_only.core.capabilities.catalog().capabilities
        )
    }

    assert "graph.invariant.chromatic_number.compute" in capability_ids
    assert {
        "graph.coloring.encode_k_cnf",
        "graph.coloring.encoding.verify",
    }.isdisjoint(capability_ids)


def test_discovery_filters_hidden_and_nonmatching_domains(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    discovered = attached_complete_runtime_read_only.core.capabilities.discover(
        CapabilityDiscoveryRequest(query="artifact", domain="artifact")
    )

    assert discovered.domain == "artifact"
    assert discovered.matches == ()


def test_small_bounded_operation_is_published_inline(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in (
            attached_complete_runtime_read_only.core.capabilities.catalog().capabilities
        )
    }
    induced_tree = descriptors["graph.induced_tree.maximum.compute"]
    assert induced_tree.produced_artifact_types == ()


def test_materialize_to_width_produced_types_are_symmetric_and_discoverable(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in (
            attached_complete_runtime_read_only.core.capabilities.catalog().capabilities
        )
    }
    assert descriptors["poset.finite.compute"].produced_artifact_types == ()
    assert descriptors["poset.width.compute"].accepted_artifact_types == ()


def test_discovery_ranks_lexical_matches_and_returns_no_synthetic_fit_labels(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    capabilities = attached_complete_runtime_read_only.core.capabilities

    strong = capabilities.discover(
        CapabilityDiscoveryRequest(
            query="graded Jacobian syzygy minimum degree",
            limit=3,
        )
    )
    assert strong.matches[0].capability_id == (
        "polynomial.jacobian_syzygy.minimum_degree.compute"
    )
    assert strong.matches[0].relevance_score > 0
    assert strong.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    gaussian = capabilities.discover(
        CapabilityDiscoveryRequest(
            query="exact fixed order Gaussian polynomial moment Wick contraction",
            limit=3,
        )
    )
    assert gaussian.matches[0].capability_id == (
        "probability.gaussian_polynomial.moment.compute"
    )
    assert gaussian.matches[0].relevance_score > 0
    assert "does not establish an identity for every order" in (
        gaussian.matches[0].description
    )

    reliability = capabilities.discover(
        CapabilityDiscoveryRequest(
            query="exact small graph reliability terminal connection probability",
            limit=3,
        )
    )
    assert reliability.matches[0].capability_id == (
        "probability.graph_reliability.connection_probability.compute"
    )

    symmetry = capabilities.discover(
        CapabilityDiscoveryRequest(
            query=(
                "declared graph automorphism generators vertex edge orbit "
                "symmetry compression"
            ),
            limit=3,
        )
    )
    assert symmetry.matches[0].capability_id == (
        "graph.symmetry.generator_orbits.compute"
    )

    absent = capabilities.discover(
        CapabilityDiscoveryRequest(query="quuxonium frobnicator", limit=3)
    )
    assert absent.matches == ()


def test_discovery_finds_resultant_producer_from_plain_language(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    capabilities = attached_complete_runtime_read_only.core.capabilities

    for query in (
        "compute the exact resultant of two univariate rational polynomials",
        "compute and independently verify the exact resultant of two univariate rational polynomials",
    ):
        discovered = capabilities.discover(
            CapabilityDiscoveryRequest(query=query, limit=8)
        )

        ids = [match.capability_id for match in discovered.matches]
        assert "polynomial.compute.resultant" in ids
        assert ids.index("polynomial.compute.resultant") <= 1
        resultant = next(
            match
            for match in discovered.matches
            if match.capability_id == "polynomial.compute.resultant"
        )
        assert resultant.relevance_score > 0


def test_domain_intents_discover_poset_and_topology_operations(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    capabilities = attached_complete_runtime_read_only.core.capabilities
    cases = (
        (
            "maximum antichain and minimum chain decomposition of a finite poset",
            "poset.width.compute",
        ),
        (
            "compute the width of a finite partially ordered set",
            "poset.width.compute",
        ),
        (
            "homology of a finite simplicial complex over F_2",
            "topology.simplicial_homology.compute",
        ),
    )
    for query, capability_id in cases:
        discovered = capabilities.discover(
            CapabilityDiscoveryRequest(query=query, limit=5)
        )

        assert discovered.matches[0].capability_id == capability_id, query
        assert discovered.matches[0].relevance_score > 0, query
