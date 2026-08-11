from __future__ import annotations

from jacobian.contracts.capabilities import CapabilityCatalogRelationshipKind

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


def test_atomic_capability_catalog_includes_required_and_excludes_composite_operations(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    catalog = runtime.core.capabilities.catalog().capabilities
    ids = {item.capability_id for item in catalog}
    descriptors = {item.capability_id: item for item in catalog}

    assert {
        "artifact.put",
        "claim.validate",
        "evaluate.batch",
        "witness.find",
        "witness.verify",
        "certificate.verify",
        "shrink.run",
        "structure.canonicalize",
        "search.enumerate",
        "experiment.inspect",
        "experiment.wait",
        "experiment.cancel",
        "transform.apply",
        "transform.verify",
        "polytope.separate",
        "parameter.region.promote",
    }.issubset(ids)
    assert {
        "reference.solve",
        "verification.run",
        "search.run",
        "conjecture.generate",
        "conjecture.repair",
        "parameter.generalize",
    }.isdisjoint(ids)
    assert "witness_uri" in descriptors["witness.find"].output_schema["properties"]
    assert [
        (item.capability_id, item.kind)
        for item in descriptors["witness.find"].related_capabilities
    ] == [("witness.verify", CapabilityCatalogRelationshipKind.INDEPENDENT_VERIFIER)]
    assert [
        (item.capability_id, item.kind)
        for item in descriptors["witness.verify"].related_capabilities
    ] == [
        (
            "witness.find",
            CapabilityCatalogRelationshipKind.VERIFIABLE_RESULT_PRODUCER,
        )
    ]
    assert (
        "experiment_uri" in descriptors["search.enumerate"].output_schema["properties"]
    )
