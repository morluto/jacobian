"""Adapter implementations for sparse rational polynomial-map capabilities."""

from __future__ import annotations

import hashlib

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityInvocationExample,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
)
from jacobian.contracts.polynomials import (
    PolynomialIdentityClaim,
    PolynomialIdentityOutput,
    PolynomialIdentityReplayPayload,
    PolynomialIdentityRequest,
    RationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
)
from jacobian.polynomials._support import (
    _polynomial_error,
    _validate_request,
)
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema


class PolynomialIdentityAdapter:
    """Verify equality in one exact sparse polynomial ring."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_ids = (
            (resources.installation.identity_checker_id,)
            if resources.installation.identity_checker_id is not None
            else ()
        )
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.identity.verify",
            version="1",
            title="Verify an exact polynomial identity",
            description=(
                "Independently compare every exact coefficient of two sparse "
                "polynomials in one declared QQ polynomial ring."
            ),
            provider="jacobian.sparse-polynomial-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.sparse-polynomial-checker",
                features=("polynomial-identity", "exact-rational"),
                checker_ids=checker_ids,
            ),
            input_schema=model_schema(PolynomialIdentityRequest),
            output_schema=model_schema(PolynomialIdentityOutput),
            tags=("polynomial", "identity", "verification", "exact-rational"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="zero_identity",
                    description=(
                        "Independently verify that two zero polynomials are equal "
                        "in QQ[x]."
                    ),
                    input=PolynomialIdentityRequest.model_validate(
                        {
                            "variables": ["x"],
                            "left": {"terms": []},
                            "right": {"terms": []},
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialIdentityRequest,
            request.input,
            code="INVALID_POLYNOMIAL_IDENTITY_REQUEST",
            operation="identity verification",
        )
        checker_id = self.resources.installation.identity_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_IDENTITY_CHECKER_UNAVAILABLE",
                "identity_verification",
                "No authorized polynomial identity checker is installed.",
            )
        left = self.resources.artifacts.put(
            schema_uri=self.resources.installation.left_polynomial_schema_uri,
            semantics_uri=self.resources.installation.identity_semantics_uri,
            payload=RationalPolynomial(
                variables=validated.variables,
                polynomial=validated.left,
            ).model_dump(mode="json"),
            summary="left exact rational polynomial",
        )
        right = self.resources.artifacts.put(
            schema_uri=self.resources.installation.right_polynomial_schema_uri,
            semantics_uri=self.resources.installation.identity_semantics_uri,
            payload=RationalPolynomial(
                variables=validated.variables,
                polynomial=validated.right,
            ).model_dump(mode="json"),
            summary="right exact rational polynomial",
        )
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.installation.identity_claim_schema_uri,
            semantics_uri=self.resources.installation.identity_semantics_uri,
            payload=PolynomialIdentityClaim(
                variables=validated.variables,
                left_uri=left.artifact_uri,
                right_uri=right.artifact_uri,
            ).model_dump(mode="json"),
            parents=(left.artifact_uri, right.artifact_uri),
            summary="exact polynomial identity claim",
        )
        semantics = self.resources.store.get(
            self.resources.installation.identity_semantics_uri
        )
        certificate_payload = PolynomialIdentityReplayPayload(
            variables=validated.variables,
            left_uri=left.artifact_uri,
            right_uri=right.artifact_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.identity_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=right.object_digest,
                scope_digest=left.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(certificate_payload)).hexdigest()
            ),
            payload=certificate_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.certificate_schema_uri,
            semantics_uri=self.resources.installation.identity_semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(claim.artifact_uri, right.artifact_uri, left.artifact_uri),
            summary="exact sparse polynomial identity replay certificate",
        )
        checked = self.resources.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
        )
        record_uri = checked.verification_record_uri
        verified = record_uri is not None
        conclusion = (
            checked.conclusion
            if verified and checked.conclusion in {Conclusion.TRUE, Conclusion.FALSE}
            else Conclusion.UNKNOWN
        )
        identical = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }[conclusion]
        output = PolynomialIdentityOutput(
            identical=identical,
            conclusion=conclusion,
            left_uri=left.artifact_uri,
            right_uri=right.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="global coefficient equality in one declared QQ ring",
                parameters={"variables": list(validated.variables)},
                artifact_uri=left.artifact_uri,
            ),
            completeness=(
                CapabilityCompleteness(
                    status=CapabilityCompletenessStatus.COMPLETE,
                    basis=(
                        "every canonical sparse coefficient was replayed independently"
                    ),
                    assurance_level=CapabilityAssuranceLevel.VERIFIED,
                    verification_record_uri=record_uri,
                )
                if verified
                else CapabilityCompleteness(
                    status=CapabilityCompletenessStatus.UNKNOWN,
                    basis="the independent checker did not accept the replay",
                )
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.identity",
                    source_artifact_uris=(left.artifact_uri,),
                    target_artifact_uris=(right.artifact_uri,),
                    status=CapabilityRelationshipStatus.VERIFIED,
                    verification_record_uri=record_uri,
                ),
            )
            if conclusion is Conclusion.TRUE and verified
            else (),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted by the authorized independent sparse-polynomial checker"
                    if verified
                    else "the independent checker did not accept the identity request"
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=(
                left.artifact_uri,
                right.artifact_uri,
                claim.artifact_uri,
                certificate_artifact.artifact_uri,
                *((record_uri,) if record_uri is not None else ()),
            ),
        )
