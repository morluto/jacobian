"""Behavioral coverage for shared checker-evidence envelope publication."""

from __future__ import annotations

from tests.support.exact_domain import (
    VerifiedDomainTestServices,
    open_exact_domain_services,
)

from jacobian.checker_artifacts import put_witness_envelope
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.matrices import RationalMatrix
from jacobian.domains.matrix_lattice import build_matrix_bundle
from jacobian.storage.models import StoredArtifact


def _claim_and_candidate(
    services: VerifiedDomainTestServices,
) -> tuple[StoredArtifact, StoredArtifact]:
    matrix = {
        "matrix_schema_version": "1",
        "domain": "QQ",
        "entries": [
            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
            [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
        ],
    }
    installed = services.bundles["matrix"]
    matrix_schema_uri = services.core.schemas.register_model(
        name="jacobian.exact-rational-matrix",
        version="1",
        model=RationalMatrix,
    )
    claim_put = services.core.artifacts.put(
        schema_uri=matrix_schema_uri,
        semantics_uri=installed.semantics_uri,
        payload=matrix,
        summary="exact rational matrix verification claim",
    )
    claim = services.core.store.get(claim_put.artifact_uri)
    candidate_put = services.core.artifacts.put(
        schema_uri=installed.result_schema_uris["matrix.determinant.compute"],
        semantics_uri=installed.semantics_uri,
        payload={
            "determinant": {"num": "-2", "den": "1"},
            "method": "FRACTION_FREE_BAREISS",
        },
        parents=(claim.artifact_uri,),
        summary="exact rational matrix determinant verification candidate",
    )
    return claim, services.core.store.get(candidate_put.artifact_uri)


def test_put_witness_envelope_binds_digests_and_parents(tmp_path) -> None:
    with open_exact_domain_services(
        tmp_path / "state", build_matrix_bundle()
    ) as services:
        claim, candidate = _claim_and_candidate(services)
        installed = services.bundles["matrix"]
        semantics = services.core.store.get(installed.semantics_uri)
        witness_schema_uri = services.core.schemas.register_model(
            name="jacobian.witness-envelope",
            version="1",
            model=WitnessEnvelope,
        )

        result = put_witness_envelope(
            services.core.artifacts,
            witness_schema_uri=witness_schema_uri,
            witness_format="matrix.rational_determinant",
            claim_artifact=claim,
            semantics_artifact=semantics,
            candidate_artifact=candidate,
            payload={
                "matrix_uri": claim.artifact_uri,
                "determinant_uri": candidate.artifact_uri,
            },
            summary="exact rational determinant verification witness",
        )

        stored = services.core.store.get(result.artifact_uri)
        assert stored.manifest.schema_uri == witness_schema_uri
        assert stored.manifest.semantics_uri == semantics.artifact_uri
        assert stored.manifest.parents == tuple(
            sorted((claim.artifact_uri, candidate.artifact_uri))
        )
        witness = WitnessEnvelope.model_validate(stored.payload)
        assert witness.witness_format == "matrix.rational_determinant"
        assert witness.format_version == "1"
        assert witness.role is WitnessRole.SUPPORTS_CLAIM
        assert witness.bindings == EvidenceBindings(
            claim_digest=claim.manifest.object_digest,
            semantics_digest=semantics.manifest.object_digest,
            candidate_digest=candidate.manifest.object_digest,
        )
        assert witness.payload == {
            "matrix_uri": claim.artifact_uri,
            "determinant_uri": candidate.artifact_uri,
        }
