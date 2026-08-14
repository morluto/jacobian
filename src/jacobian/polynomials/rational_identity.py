"""Exact identity verification in a bounded rational-function field."""

from __future__ import annotations

import hashlib

from jacobian.canonical import canonicalize_json
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationExample,
    OperationRequest,
)
from jacobian.contracts.polynomials import (
    RationalFunctionArtifact,
    RationalFunctionIdentityClaim,
    RationalFunctionIdentityOutput,
    RationalFunctionIdentityReplayPayload,
    RationalFunctionIdentityRequest,
)
from jacobian.contracts.results import Conclusion
from jacobian.operation_projection import OperationProjection
from jacobian.polynomials._support import (
    PolynomialOperationResult,
    _polynomial_error,
    _validate_request,
)
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema


class RationalFunctionIdentityAdapter:
    """Verify equality in one declared QQ rational-function field."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.contracts.rational_function_identity_checker_id
        self._descriptor = OperationDescriptor(
            operation_id="polynomial.rational_function.identity.verify",
            version="1",
            title="Verify an exact rational-function identity",
            description=(
                "Independently verify equality of two bounded sparse rational "
                "functions in a declared QQ fraction field by exact cross "
                "multiplication. This does not prove pointwise definedness."
            ),
            provider="jacobian.rational-function-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.rational-function-checker",
                features=("rational-function-identity", "exact-rational"),
                checker_ids=((checker_id,) if checker_id is not None else ()),
            ),
            input_schema=model_schema(RationalFunctionIdentityRequest),
            output_schema=model_schema(RationalFunctionIdentityOutput),
            tags=(
                "polynomial",
                "rational-function",
                "identity",
                "verification",
            ),
            examples=(
                OperationExample(
                    name="cancel_common_factor",
                    description="Verify that (x²-1)/(x-1) equals x+1 in QQ(x).",
                    input=RationalFunctionIdentityRequest.model_validate(
                        {
                            "variables": ["x"],
                            "left": {
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        },
                                        {
                                            "coefficient": {"num": "-1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                            "right": {
                                "numerator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        },
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                                "denominator": {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [0],
                                        },
                                    ]
                                },
                            },
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> RationalFunctionIdentityRequest:
        return _validate_request(
            RationalFunctionIdentityRequest,
            request.input,
            code="INVALID_RATIONAL_FUNCTION_IDENTITY_REQUEST",
            operation="rational-function identity verification",
        )

    def invoke(self, validated: RationalFunctionIdentityRequest) -> OperationProjection:
        checker_id = self.resources.contracts.rational_function_identity_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "RATIONAL_FUNCTION_IDENTITY_CHECKER_UNAVAILABLE",
                "rational_function_identity_verification",
                "No authorized rational-function identity checker is installed.",
            )
        semantics_uri = (
            self.resources.contracts.rational_function_identity_semantics_uri
        )
        left_payload = RationalFunctionArtifact(
            variables=validated.variables,
            numerator=validated.left.numerator,
            denominator=validated.left.denominator,
        )
        right_payload = RationalFunctionArtifact(
            variables=validated.variables,
            numerator=validated.right.numerator,
            denominator=validated.right.denominator,
        )
        left = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.rational_function_left_schema_uri,
            semantics_uri=semantics_uri,
            payload=left_payload.model_dump(mode="json"),
            summary="left exact rational function",
        )
        right = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.rational_function_right_schema_uri,
            semantics_uri=semantics_uri,
            payload=right_payload.model_dump(mode="json"),
            summary="right exact rational function",
        )
        claim = self.resources.artifacts.put(
            schema_uri=(
                self.resources.contracts.rational_function_identity_claim_schema_uri
            ),
            semantics_uri=semantics_uri,
            payload=RationalFunctionIdentityClaim(
                variables=validated.variables,
                left_uri=left.artifact_uri,
                right_uri=right.artifact_uri,
            ).model_dump(mode="json"),
            parents=(left.artifact_uri, right.artifact_uri),
            summary="exact rational-function identity claim",
        )
        semantics = self.resources.store.get(semantics_uri)
        replay_payload = RationalFunctionIdentityReplayPayload(
            variables=validated.variables,
            left_uri=left.artifact_uri,
            right_uri=right.artifact_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.rational_function.identity_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=right.object_digest,
                scope_digest=left.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(replay_payload)).hexdigest()
            ),
            payload=replay_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.certificate_schema_uri,
            semantics_uri=semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(claim.artifact_uri, right.artifact_uri, left.artifact_uri),
            summary="rational-function cross-multiplication replay certificate",
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
        output = RationalFunctionIdentityOutput(
            identical=identical,
            conclusion=conclusion,
            left_uri=left.artifact_uri,
            right_uri=right.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
        )
        return PolynomialOperationResult(
            execution=checked.execution,
            value=output,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=(
                left.artifact_uri,
                right.artifact_uri,
                claim.artifact_uri,
                certificate_artifact.artifact_uri,
                *((record_uri,) if record_uri is not None else ()),
            ),
        ).project(self.descriptor)
