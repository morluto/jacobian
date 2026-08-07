"""Exact rational polynomial-system solution verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, cast

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.capability_service import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomial_systems import (
    PolynomialSystemSolutionClaim,
    PolynomialSystemSolutionOutput,
    PolynomialSystemSolutionReplay,
    PolynomialSystemSolutionRequest,
    RationalPolynomialAssignment,
    RationalPolynomialSystem,
)
from jacobian.contracts.polynomials import SparseRationalPolynomial
from jacobian.contracts.results import Conclusion, ExecutionStatus, Verification
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class PolynomialSystemInstallation:
    semantics_uri: str
    system_schema_uri: str
    assignment_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    checker_id: str | None


@dataclass(frozen=True, slots=True)
class PolynomialSystemResources:
    store: ArtifactRepository
    artifacts: ArtifactService
    verification: VerificationService
    installation: PolynomialSystemInstallation


def install_polynomial_system_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[PolynomialSystemSolutionAdapter | None, PolynomialSystemInstallation]:
    """Register exact polynomial-system schemas and optional verifier."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.rational-polynomial-system",
        version="1",
        definition={
            "description": (
                "finite equations and inequations over QQ with an explicit "
                "variable order and canonical sparse coefficients"
            ),
            "maximum_variables": 4,
            "maximum_equations": 64,
            "maximum_inequations": 64,
        },
    )
    system_schema_uri = schemas.register(
        name="jacobian.rational-polynomial-system",
        version="1",
        schema=model_schema(RationalPolynomialSystem),
    )
    assignment_schema_uri = schemas.register(
        name="jacobian.rational-polynomial-assignment",
        version="1",
        schema=model_schema(RationalPolynomialAssignment),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.polynomial-system-solution-claim",
        version="1",
        schema=model_schema(PolynomialSystemSolutionClaim),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact rational polynomial-system solution checker",
                entrypoint="jacobian_checkers.polynomial_systems:check_solution",
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="polynomial.system_solution_replay",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(assignment_schema_uri,),
                reason="bundled independent exact polynomial-system evaluator",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = PolynomialSystemInstallation(
        semantics_uri=semantics_uri,
        system_schema_uri=system_schema_uri,
        assignment_schema_uri=assignment_schema_uri,
        claim_schema_uri=claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        checker_id=checker_id,
    )
    if checker_id is None:
        return None, installation
    return (
        PolynomialSystemSolutionAdapter(
            PolynomialSystemResources(
                store=store,
                artifacts=artifacts,
                verification=verification,
                installation=installation,
            )
        ),
        installation,
    )


class PolynomialSystemSolutionAdapter:
    """Verify one exact assignment against every declared constraint."""

    def __init__(self, resources: PolynomialSystemResources) -> None:
        self.resources = resources
        checker_id = resources.installation.checker_id
        assert checker_id is not None
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.system.solution.verify",
            version="1",
            title="Verify an exact polynomial-system solution",
            description=(
                "Independently evaluate every equation and inequation at one exact "
                "rational assignment."
            ),
            provider="jacobian.exact-polynomial-system-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.exact-polynomial-system-checker",
                features=("polynomial-system", "solution", "exact-rational"),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(PolynomialSystemSolutionRequest),
            output_schema=model_schema(PolynomialSystemSolutionOutput),
            tags=("polynomial", "system", "solution", "verification"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            validated = PolynomialSystemSolutionRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_POLYNOMIAL_SYSTEM_SOLUTION_REQUEST",
                    stage="request_validation",
                    message=(
                        "The complete polynomial-system solution request is invalid."
                    ),
                    path="system/assignment",
                    expected=(
                        "one canonical rational assignment value for every declared "
                        "variable, with every monomial using that variable order"
                    ),
                    hint=(
                        "Match the assignment and every exponent vector to the "
                        "system.variables order, then retry."
                    ),
                )
            ) from exc
        installation = self.resources.installation
        checker_id = installation.checker_id
        assert checker_id is not None
        equation_residuals, inequation_values = _evaluate_request(validated)
        system = self.resources.artifacts.put(
            schema_uri=installation.system_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=validated.system.model_dump(mode="json"),
            summary="exact rational polynomial system",
        )
        assignment = self.resources.artifacts.put(
            schema_uri=installation.assignment_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=RationalPolynomialAssignment(
                values=validated.assignment
            ).model_dump(mode="json"),
            summary="exact rational polynomial-system assignment",
        )
        claim = self.resources.artifacts.put(
            schema_uri=installation.claim_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=PolynomialSystemSolutionClaim(
                system_uri=system.artifact_uri,
                assignment_uri=assignment.artifact_uri,
            ).model_dump(mode="json"),
            parents=(system.artifact_uri, assignment.artifact_uri),
            summary="polynomial-system assignment satisfaction claim",
        )
        semantics = self.resources.store.get(installation.semantics_uri)
        replay = PolynomialSystemSolutionReplay(
            system_uri=system.artifact_uri,
            assignment_uri=assignment.artifact_uri,
            equation_residuals=equation_residuals,
            inequation_values=inequation_values,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.system_solution_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=assignment.object_digest,
                scope_digest=system.object_digest,
            ),
            payload_digest=(
                "sha256:" + hashlib.sha256(canonicalize_json(replay)).hexdigest()
            ),
            payload=replay,
        )
        evidence = self.resources.artifacts.put(
            schema_uri=installation.certificate_schema_uri,
            semantics_uri=installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                claim.artifact_uri,
                assignment.artifact_uri,
                system.artifact_uri,
            ),
            summary="exact polynomial-system solution replay certificate",
        )
        checked = self.resources.verification.verify_certificate(
            certificate_uri=evidence.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.execution.status is ExecutionStatus.COMPLETED
            and checked.conclusion in {Conclusion.TRUE, Conclusion.FALSE}
            and checked.assurance.verification is Verification.VERIFIED
            and checked.verification_record_uri is not None
        )
        conclusion = cast(
            Literal["TRUE", "FALSE", "UNKNOWN"],
            checked.conclusion.value,
        )
        satisfies = {
            "TRUE": True,
            "FALSE": False,
            "UNKNOWN": None,
        }[conclusion]
        record_uri = checked.verification_record_uri if verified else None
        output = PolynomialSystemSolutionOutput(
            satisfies=satisfies,
            conclusion=conclusion,
            equation_residuals=equation_residuals,
            inequation_values=inequation_values,
            residuals_assurance="VERIFIED" if verified else "COMPUTED",
            system_uri=system.artifact_uri,
            assignment_uri=assignment.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=evidence.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
        )
        artifact_uris = [
            system.artifact_uri,
            assignment.artifact_uri,
            claim.artifact_uri,
            evidence.artifact_uri,
        ]
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="all equations and inequations in one finite system",
                parameters={
                    "equation_count": len(validated.system.equations),
                    "inequation_count": len(validated.system.inequations),
                    "variables": list(validated.system.variables),
                },
                artifact_uri=system.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if checked.execution.status is ExecutionStatus.COMPLETED
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "the independent checker evaluated every declared constraint"
                    if verified
                    else (
                        "the adapter computed every declared residual, but the "
                        "checker did not accept the bound replay"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not establish complete coverage"
                    )
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                verification_record_uri=record_uri,
            ),
            relationships=(
                (
                    CapabilityRelationship(
                        relation_id="polynomial.relation.satisfies-system",
                        source_artifact_uris=(assignment.artifact_uri,),
                        target_artifact_uris=(system.artifact_uri,),
                        status=CapabilityRelationshipStatus.VERIFIED,
                        verification_record_uri=record_uri,
                    ),
                )
                if conclusion == "TRUE" and verified
                else ()
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else (
                        CapabilityAssuranceLevel.COMPUTED
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else CapabilityAssuranceLevel.HEURISTIC
                    )
                ),
                basis=(
                    "accepted by the authorized independent exact evaluator"
                    if verified
                    else (
                        "exact residuals were computed, but the independent checker "
                        "did not accept the bound replay"
                        if checked.execution.status is ExecutionStatus.COMPLETED
                        else "checker execution did not complete; no mathematical "
                        "conclusion follows"
                    )
                ),
                verification_record_uri=record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )


def _evaluate_request(
    request: PolynomialSystemSolutionRequest,
) -> tuple[tuple[CanonicalRational, ...], tuple[CanonicalRational, ...]]:
    """Compute projected values; the isolated checker independently replays them."""

    assignment = tuple(value.as_fraction() for value in request.assignment)

    def evaluate(polynomial: SparseRationalPolynomial) -> CanonicalRational:
        total = Fraction(0)
        for term in polynomial.terms:
            value = term.coefficient.as_fraction()
            for coordinate, exponent in zip(
                assignment,
                term.exponents,
                strict=True,
            ):
                value *= coordinate**exponent
            total += value
        return CanonicalRational(
            num=format_canonical_integer(total.numerator),
            den=format_canonical_integer(total.denominator),
        )

    return (
        tuple(evaluate(polynomial) for polynomial in request.system.equations),
        tuple(evaluate(polynomial) for polynomial in request.system.inequations),
    )
