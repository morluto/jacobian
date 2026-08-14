"""Compiled-catalog discovery and projection contracts."""

from __future__ import annotations

from jacobian.contracts.operations import OperationDiscoveryRequest
from jacobian.runtime.model import JacobianRuntime


def test_catalog_keeps_chromatic_number_without_unused_encoding_workflow(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    operation_ids = {
        descriptor.operation_id
        for descriptor in (
            attached_complete_runtime_read_only.core.operations.snapshot().operations
        )
    }

    assert "graph.invariant.chromatic_number.compute" in operation_ids
    assert {
        "graph.coloring.encode_k_cnf",
        "graph.coloring.encoding.verify",
    }.isdisjoint(operation_ids)


def test_discovery_filters_hidden_and_nonmatching_domains(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    discovered = attached_complete_runtime_read_only.core.operations.search(
        OperationDiscoveryRequest(query="artifact", domain="artifact")
    )

    assert discovered.domain == "artifact"
    assert discovered.matches == ()


def test_small_bounded_operation_is_published_inline(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.operation_id: descriptor
        for descriptor in (
            attached_complete_runtime_read_only.core.operations.snapshot().operations
        )
    }
    induced_tree = descriptors["graph.induced_tree.maximum.compute"]
    assert induced_tree.produced_artifact_types == ()


def test_materialize_to_width_produced_types_are_symmetric_and_discoverable(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    descriptors = {
        descriptor.operation_id: descriptor
        for descriptor in (
            attached_complete_runtime_read_only.core.operations.snapshot().operations
        )
    }
    assert descriptors["poset.finite.compute"].produced_artifact_types == ()
    assert descriptors["poset.width.compute"].accepted_artifact_types == ()


def test_discovery_ranks_lexical_matches_and_returns_no_synthetic_fit_labels(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    operations = attached_complete_runtime_read_only.core.operations

    strong = operations.search(
        OperationDiscoveryRequest(
            query="graded Jacobian syzygy minimum degree",
            limit=3,
        )
    )
    assert strong.matches[0].operation_id == (
        "polynomial.jacobian_syzygy.minimum_degree.compute"
    )
    assert strong.matches[0].relevance_score > 0
    assert strong.matches[0].applicability == "NEEDS_MORE_TYPED_REQUIREMENTS"

    gaussian = operations.search(
        OperationDiscoveryRequest(
            query="exact fixed order Gaussian polynomial moment Wick contraction",
            limit=3,
        )
    )
    assert gaussian.matches[0].operation_id == (
        "probability.gaussian_polynomial.moment.compute"
    )
    assert gaussian.matches[0].relevance_score > 0
    assert "does not establish an identity for every order" in (
        gaussian.matches[0].description
    )

    reliability = operations.search(
        OperationDiscoveryRequest(
            query="exact small graph reliability terminal connection probability",
            limit=3,
        )
    )
    assert reliability.matches[0].operation_id == (
        "probability.graph_reliability.connection_probability.compute"
    )

    symmetry = operations.search(
        OperationDiscoveryRequest(
            query=(
                "declared graph automorphism generators vertex edge orbit "
                "symmetry compression"
            ),
            limit=3,
        )
    )
    assert symmetry.matches[0].operation_id == (
        "graph.symmetry.generator_orbits.compute"
    )

    absent = operations.search(
        OperationDiscoveryRequest(query="quuxonium frobnicator", limit=3)
    )
    assert absent.matches == ()


def test_discovery_finds_resultant_producer_from_plain_language(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    operations = attached_complete_runtime_read_only.core.operations

    for query in (
        "compute the exact resultant of two univariate rational polynomials",
        "compute and independently verify the exact resultant of two univariate rational polynomials",
    ):
        discovered = operations.search(OperationDiscoveryRequest(query=query, limit=8))

        ids = [match.operation_id for match in discovered.matches]
        assert "polynomial.compute.resultant" in ids
        assert ids.index("polynomial.compute.resultant") <= 1
        resultant = next(
            match
            for match in discovered.matches
            if match.operation_id == "polynomial.compute.resultant"
        )
        assert resultant.relevance_score > 0


def test_discovery_maps_exceeds_bound_language_to_interval_positivity(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    discovered = attached_complete_runtime_read_only.core.operations.search(
        OperationDiscoveryRequest(
            query="verify a rational derivative exceeds a proposed bound",
            domain="polynomial",
            limit=10,
        )
    )

    matches = {match.operation_id: match for match in discovered.matches}
    assert "polynomial.interval.positivity.decide" in matches
    assert matches["polynomial.interval.positivity.decide"].relevance_score > 0


def test_sum_of_squares_intent_discovers_the_identity_verifier(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    discovered = attached_complete_runtime_read_only.core.operations.search(
        OperationDiscoveryRequest(
            query="independently verify a polynomial sum-of-squares identity",
            limit=8,
        )
    )

    matches = {match.operation_id: match for match in discovered.matches}
    assert "polynomial.identity.verify" in matches
    assert matches["polynomial.identity.verify"].relevance_score > 0


def test_domain_intents_discover_poset_and_topology_operations(
    attached_complete_runtime_read_only: JacobianRuntime,
) -> None:
    operations = attached_complete_runtime_read_only.core.operations
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
    for query, operation_id in cases:
        discovered = operations.search(OperationDiscoveryRequest(query=query, limit=5))

        assert discovered.matches[0].operation_id == operation_id, query
        assert discovered.matches[0].relevance_score > 0, query
