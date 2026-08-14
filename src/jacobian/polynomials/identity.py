"""Adapter implementations for sparse rational polynomial-map operations."""

from __future__ import annotations

import hashlib
from fractions import Fraction

from jacobian.canonical import canonicalize_json
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.operations import (
    OperationDescriptor,
    OperationExample,
    OperationRequest,
)
from jacobian.contracts.polynomials import (
    PolynomialCoefficientMismatch,
    PolynomialIdentityClaim,
    PolynomialIdentityOutput,
    PolynomialIdentityReplayPayload,
    PolynomialIdentityRequest,
    RationalPolynomial,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
)
from jacobian.operation_projection import OperationProjection
from jacobian.polynomials._support import (
    PolynomialOperationResult,
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
            (resources.contracts.identity_checker_id,)
            if resources.contracts.identity_checker_id is not None
            else ()
        )
        self._descriptor = OperationDescriptor(
            operation_id="polynomial.identity.verify",
            version="2",
            title="Compare exact polynomials coefficient by coefficient",
            description=(
                "Independently compare every exact coefficient of two sparse "
                "polynomials in one declared QQ polynomial ring. A false identity "
                "returns the first canonical monomial coefficient mismatch and its "
                "exact rational difference."
            ),
            provider="jacobian.sparse-polynomial-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.sparse-polynomial-checker",
                features=("polynomial-identity", "exact-rational"),
                checker_ids=checker_ids,
            ),
            input_schema=model_schema(PolynomialIdentityRequest),
            output_schema=model_schema(PolynomialIdentityOutput),
            tags=(
                "polynomial",
                "identity",
                "equality",
                "verification",
                "exact-rational",
                "coefficient-equality",
                "sum-of-squares",
                "coefficient-mismatch",
                "counter-witness",
            ),
            examples=(
                OperationExample(
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
    def descriptor(self) -> OperationDescriptor:
        return self._descriptor

    def prepare(self, request: OperationRequest) -> PolynomialIdentityRequest:
        return _validate_request(
            PolynomialIdentityRequest,
            request.input,
            code="INVALID_POLYNOMIAL_IDENTITY_REQUEST",
            operation="identity verification",
        )

    def invoke(self, validated: PolynomialIdentityRequest) -> OperationProjection:
        return self.verify(validated).project(self.descriptor)

    def verify(
        self,
        validated: PolynomialIdentityRequest,
    ) -> PolynomialOperationResult[PolynomialIdentityOutput]:
        """Verify one validated identity without re-entering the adapter."""

        checker_id = self.resources.contracts.identity_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_IDENTITY_CHECKER_UNAVAILABLE",
                "identity_verification",
                "No authorized polynomial identity checker is installed.",
            )
        left = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.left_polynomial_schema_uri,
            semantics_uri=self.resources.contracts.identity_semantics_uri,
            payload=RationalPolynomial(
                variables=validated.variables,
                polynomial=validated.left,
            ).model_dump(mode="json"),
            summary="left exact rational polynomial",
        )
        right = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.right_polynomial_schema_uri,
            semantics_uri=self.resources.contracts.identity_semantics_uri,
            payload=RationalPolynomial(
                variables=validated.variables,
                polynomial=validated.right,
            ).model_dump(mode="json"),
            summary="right exact rational polynomial",
        )
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.contracts.identity_claim_schema_uri,
            semantics_uri=self.resources.contracts.identity_semantics_uri,
            payload=PolynomialIdentityClaim(
                variables=validated.variables,
                left_uri=left.artifact_uri,
                right_uri=right.artifact_uri,
            ).model_dump(mode="json"),
            parents=(left.artifact_uri, right.artifact_uri),
            summary="exact polynomial identity claim",
        )
        semantics = self.resources.store.get(
            self.resources.contracts.identity_semantics_uri
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
            schema_uri=self.resources.contracts.certificate_schema_uri,
            semantics_uri=self.resources.contracts.identity_semantics_uri,
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
        mismatch = (
            _first_coefficient_mismatch(validated.left, validated.right)
            if conclusion is Conclusion.FALSE
            else None
        )
        output = PolynomialIdentityOutput(
            identical=identical,
            conclusion=conclusion,
            left_uri=left.artifact_uri,
            right_uri=right.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
            first_coefficient_mismatch=mismatch,
        )
        return PolynomialOperationResult(
            value=output,
            execution=checked.execution,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=(
                left.artifact_uri,
                right.artifact_uri,
                claim.artifact_uri,
                certificate_artifact.artifact_uri,
                *((record_uri,) if record_uri is not None else ()),
            ),
        )


def _first_coefficient_mismatch(
    left: SparseRationalPolynomial,
    right: SparseRationalPolynomial,
) -> PolynomialCoefficientMismatch:
    left_terms = {term.exponents: term.coefficient for term in left.terms}
    right_terms = {term.exponents: term.coefficient for term in right.terms}
    zero = CanonicalRational.from_fraction(Fraction(0))
    for exponents in sorted(set(left_terms) | set(right_terms), reverse=True):
        left_coefficient = left_terms.get(exponents, zero)
        right_coefficient = right_terms.get(exponents, zero)
        if left_coefficient != right_coefficient:
            return PolynomialCoefficientMismatch(
                exponents=exponents,
                left_coefficient=left_coefficient,
                right_coefficient=right_coefficient,
                left_minus_right=CanonicalRational.from_fraction(
                    left_coefficient.as_fraction() - right_coefficient.as_fraction()
                ),
            )
    raise RuntimeError("false polynomial identity has no coefficient mismatch")
