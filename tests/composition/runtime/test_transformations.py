from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.support.services import (
    ReferenceTestServices,
    atomic_installation,
    open_reference_services,
)

from jacobian.atomic_capabilities import install_atomic_capabilities

# Composition-lane admission category for architecture ratchets.
COMPOSITION_ADMISSION = "AUTHORITY"


@pytest.fixture
def verified_matrix_reference_services(
    tmp_path,
) -> Iterator[ReferenceTestServices]:
    with open_reference_services(
        tmp_path / "state", "matrices", authorize_checkers=True
    ) as services:
        with atomic_installation(services.core):
            for adapter in install_atomic_capabilities(
                services.installation,
                services.application,
            ):
                services.installation.register_capability(adapter)
        yield services


def test_matrix_representation_change_is_independently_verified(
    verified_matrix_reference_services: ReferenceTestServices,
) -> None:
    services = verified_matrix_reference_services
    reference = services.references["matrices"]
    source = services.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={
            "rows": 2,
            "cols": 2,
            "entries": [["1", "2"], ["3", "4"]],
        },
    )

    applied = services.application.transformations.apply(
        source_uri=source.artifact_uri,
        plugin_id=reference.plugin_id,
        target_schema_uri=reference.representation_schema_uris["row_major"],
        target_semantics_uri=reference.representation_semantics_uris["row_major"],
        requested_relation="EQUIVALENT",
        wall_seconds=30,
    )

    assert applied.transformation_uri is not None
    assert applied.result.assurance.verification.value == "UNVERIFIED"
    verified = services.application.verification.verify_transformation(
        transformation_uri=applied.transformation_uri,
    )
    assert verified.conclusion.value == "TRUE"
    assert verified.assurance.verification.value == "VERIFIED"
    assert (
        verified.assurance.checker_id
        == reference.transformation_checker_ids["matrix.row_major"]
    )


def test_transformation_tampering_fails_closed(
    verified_matrix_reference_services: ReferenceTestServices,
) -> None:
    services = verified_matrix_reference_services
    reference = services.references["matrices"]
    source = services.core.artifacts.put(
        schema_uri=reference.candidate_schema_uri,
        semantics_uri=reference.semantics_uri,
        payload={"rows": 1, "cols": 1, "entries": [["1"]]},
    )
    applied = services.application.transformations.apply(
        source_uri=source.artifact_uri,
        plugin_id=reference.plugin_id,
        target_schema_uri=reference.representation_schema_uris["row_major"],
        target_semantics_uri=reference.representation_semantics_uris["row_major"],
        requested_relation="EQUIVALENT",
        wall_seconds=30,
    )
    assert applied.transformation_uri is not None
    transformation = services.core.store.get(applied.transformation_uri)
    replacement = services.core.artifacts.put(
        schema_uri=reference.representation_schema_uris["row_major"],
        semantics_uri=reference.representation_semantics_uris["row_major"],
        payload={"rows": 1, "cols": 1, "values": ["2"]},
    )
    cases = (
        (
            "target rebinding",
            {"target_uri": replacement.artifact_uri},
            (
                transformation.payload["claim_uri"],
                transformation.payload["source_uri"],
                replacement.artifact_uri,
            ),
        ),
        (
            "relation rebinding",
            {"relation": "HEURISTIC"},
            transformation.manifest.parents,
        ),
        (
            "obligation tampering",
            {"obligation": {"rows": 999, "cols": 999}},
            transformation.manifest.parents,
        ),
    )
    for label, mutation, parents in cases:
        tampered_payload = dict(transformation.payload)
        tampered_payload.update(mutation)
        tampered = services.core.store.put(
            schema_uri=transformation.manifest.schema_uri,
            semantics_uri=transformation.manifest.semantics_uri,
            payload=tampered_payload,
            parents=parents,
            summary=f"adversarial {label}",
        )

        result = services.application.verification.verify_transformation(
            transformation_uri=tampered.artifact_uri
        )

        assert result.input.status.value == "REJECTED", label
        assert result.conclusion.value == "UNKNOWN", label
        assert result.assurance.verification.value == "UNVERIFIED", label
        assert result.verification_record_uri is None, label
