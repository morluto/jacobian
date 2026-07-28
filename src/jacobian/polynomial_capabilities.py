"""SymPy-backed exact operations for sparse rational polynomial maps."""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod as multiply
from queue import Empty
from typing import Any, cast

import sympy
from pydantic import ValidationError
from sympy import QQ, Matrix, Poly, expand, solve, symbols, sympify
from sympy.polys.polyerrors import PolynomialError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capabilities import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRelationshipStatus,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.polynomials import (
    PolynomialCollisionOutput,
    PolynomialCollisionPayload,
    PolynomialCollisionRequest,
    PolynomialCollisionSearchOutput,
    PolynomialCollisionSearchRequest,
    PolynomialCollisionSearchStopReason,
    PolynomialCollisionVerifyOutput,
    PolynomialCollisionVerifyRequest,
    PolynomialEvaluationOutput,
    PolynomialEvaluationRequest,
    PolynomialFactorizationArtifact,
    PolynomialFactorOutput,
    PolynomialFactorRecord,
    PolynomialFactorRequest,
    PolynomialIdentityClaim,
    PolynomialIdentityOutput,
    PolynomialIdentityReplayPayload,
    PolynomialIdentityRequest,
    PolynomialInjectivityClaim,
    PolynomialInverseAnsatzSpecification,
    PolynomialInverseCoefficientEquation,
    PolynomialInverseSolverProvenance,
    PolynomialInverseSupportMode,
    PolynomialInverseSynthesisStatus,
    PolynomialJacobian,
    PolynomialJacobianClaim,
    PolynomialJacobianOutput,
    PolynomialJacobianReplayPayload,
    PolynomialJacobianRequest,
    PolynomialMapCompositionResiduals,
    PolynomialMapEvaluation,
    PolynomialMapInverseClaim,
    PolynomialMapInverseReplayPayload,
    PolynomialMapInverseSynthesisArtifact,
    PolynomialMapInverseSynthesisOutput,
    PolynomialMapInverseSynthesisRequest,
    PolynomialMapInverseVerifyOutput,
    PolynomialMapInverseVerifyRequest,
    RationalPolynomial,
    RationalPolynomialMap,
    RationalPolynomialPoint,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import (
    Conclusion,
    ContractModel,
    Execution,
    ExecutionStatus,
    Verification,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.store import ArtifactStore, StoredArtifact, StoreError
from jacobian.verification import VerificationService


@dataclass(frozen=True, slots=True)
class PolynomialInstallation:
    semantics_uri: str
    polynomial_semantics_uri: str
    factorization_semantics_uri: str
    identity_semantics_uri: str
    inverse_semantics_uri: str
    map_schema_uri: str
    evaluation_schema_uri: str
    jacobian_schema_uri: str
    claim_schema_uri: str
    jacobian_claim_schema_uri: str
    right_polynomial_schema_uri: str
    left_polynomial_schema_uri: str
    identity_claim_schema_uri: str
    inverse_claim_schema_uri: str
    inverse_residual_schema_uri: str
    inverse_synthesis_schema_uri: str
    witness_schema_uri: str
    certificate_schema_uri: str
    polynomial_schema_uri: str
    factorization_schema_uri: str
    collision_checker_id: str | None
    jacobian_checker_id: str | None
    identity_checker_id: str | None
    inverse_checker_id: str | None


@dataclass(frozen=True, slots=True)
class PolynomialResources:
    store: ArtifactStore
    artifacts: ArtifactService
    verification: VerificationService
    installation: PolynomialInstallation


def install_polynomial_capabilities(
    store: ArtifactStore,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    tuple[
        PolynomialMapEvaluationAdapter,
        PolynomialJacobianAdapter,
        PolynomialCollisionAdapter,
        PolynomialIdentityAdapter,
        PolynomialCollisionSearchAdapter,
        PolynomialCollisionVerifyAdapter,
        PolynomialFactorAdapter,
        PolynomialMapInverseSynthesizeAdapter,
        PolynomialMapInverseVerifyAdapter,
    ],
    PolynomialInstallation,
]:
    """Register exact polynomial-map schemas, adapters, and optional checker."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.rational-polynomial-map",
        version="1",
        definition={
            "description": (
                "square sparse polynomial maps over QQ with an explicit variable "
                "order and canonical reduced rational coefficients"
            ),
            "domain": "QQ",
            "map_shape": "square",
            "maximum_dimension": 4,
            "maximum_terms_per_coordinate": 1024,
            "maximum_exponent": 32,
            "maximum_derived_exponent": 127,
            "maximum_jacobian_product_term_estimate": 1024,
        },
    )
    identity_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.sparse-rational-polynomial-ring",
        version="1",
        definition={
            "description": (
                "canonical sparse polynomials over QQ in an explicit ordered "
                "tuple of variables"
            ),
            "coefficient_field": "QQ",
            "maximum_dimension": 4,
            "maximum_terms": 1024,
            "maximum_exponent": 127,
            "monomial_order": "descending lexicographic",
            "zero_terms": "omitted",
        },
    )
    inverse_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.rational-polynomial-map-two-sided-inverse",
        version="1",
        definition={
            "description": (
                "two square sparse polynomial maps over QQ are inverse only when "
                "both ordered exact compositions are identity"
            ),
            "coefficient_field": "QQ",
            "directions": ["inverse_after_forward", "forward_after_inverse"],
            "one_sided_identity": "insufficient",
            "synthesis_scope": "bounded polynomial coefficient ansatz only",
            "bounded_no_candidate": "does not prove noninvertibility",
            "rational_map_inverses": "unsupported",
        },
    )
    polynomial_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial",
        version="1",
        definition={
            "description": (
                "univariate sparse polynomials over QQ with canonical reduced "
                "rational coefficients"
            ),
            "domain": "QQ",
            "maximum_terms": 1024,
        },
    )
    factorization_semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.univariate-rational-polynomial-factorization",
        version="1",
        definition={
            "description": (
                "a rational coefficient and multiplicity-bearing irreducible "
                "factors over QQ whose product reconstructs one source polynomial"
            ),
            "domain": "QQ",
            "zero_representation": {
                "coefficient": {"num": "0", "den": "1"},
                "factors": [],
            },
            "irreducibility_assurance": "unverified",
        },
    )
    map_schema_uri = schemas.register(
        name="jacobian.rational-polynomial-map",
        version="1",
        schema=model_schema(RationalPolynomialMap),
    )
    evaluation_schema_uri = schemas.register(
        name="jacobian.polynomial-map-evaluation",
        version="1",
        schema=model_schema(PolynomialMapEvaluation),
    )
    jacobian_schema_uri = schemas.register(
        name="jacobian.polynomial-jacobian",
        version="1",
        schema=model_schema(PolynomialJacobian),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-injectivity-claim",
        version="1",
        schema=model_schema(PolynomialInjectivityClaim),
    )
    jacobian_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-jacobian-claim",
        version="1",
        schema=model_schema(PolynomialJacobianClaim),
    )
    right_polynomial_schema_uri = schemas.register(
        name="jacobian.sparse-rational-polynomial-right",
        version="1",
        schema=model_schema(RationalPolynomial),
    )
    left_polynomial_schema_uri = schemas.register(
        name="jacobian.sparse-rational-polynomial-left",
        version="1",
        schema=model_schema(RationalPolynomial),
    )
    identity_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-identity-claim",
        version="1",
        schema=model_schema(PolynomialIdentityClaim),
    )
    inverse_claim_schema_uri = schemas.register(
        name="jacobian.polynomial-map-inverse-claim",
        version="1",
        schema=model_schema(PolynomialMapInverseClaim),
    )
    inverse_residual_schema_uri = schemas.register(
        name="jacobian.polynomial-map-composition-residuals",
        version="1",
        schema=model_schema(PolynomialMapCompositionResiduals),
    )
    inverse_synthesis_schema_uri = schemas.register(
        name="jacobian.polynomial-map-inverse-synthesis",
        version="1",
        schema=model_schema(PolynomialMapInverseSynthesisArtifact),
    )
    witness_schema_uri = schemas.register(
        name="jacobian.witness-envelope",
        version="1",
        schema=model_schema(WitnessEnvelope),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    polynomial_schema_uri = schemas.register(
        name="jacobian.sparse-rational-polynomial",
        version="1",
        schema=model_schema(SparseRationalPolynomial),
    )
    factorization_schema_uri = schemas.register(
        name="jacobian.polynomial-factorization",
        version="1",
        schema=model_schema(PolynomialFactorizationArtifact),
    )
    collision_checker_id = None
    jacobian_checker_id = None
    identity_checker_id = None
    inverse_checker_id = None
    if authorize_checker:
        collision_checker_id = (
            CheckerInstaller(checkers)
            .install(
                CheckerOperation(
                    name="exact rational polynomial-map collision checker",
                    entrypoint="jacobian_checkers.polynomial_maps:check_collision",
                    evidence_kind=EvidenceKind.WITNESS,
                    format_id="polynomial.map_collision",
                    format_version="1",
                    claim_schema_uris=(claim_schema_uri,),
                    semantics_uris=(semantics_uri,),
                    candidate_schema_uris=(map_schema_uri,),
                    reason="bundled polynomial-map reference checker",
                ),
                authorize=True,
            )
            .checker_id
        )
        jacobian_checker_id = (
            CheckerInstaller(checkers)
            .install(
                CheckerOperation(
                    name="exact sparse polynomial Jacobian replay checker",
                    entrypoint="jacobian_checkers.polynomial_maps:check_jacobian",
                    evidence_kind=EvidenceKind.CERTIFICATE,
                    format_id="polynomial.jacobian_replay",
                    format_version="1",
                    claim_schema_uris=(jacobian_claim_schema_uri,),
                    semantics_uris=(semantics_uri,),
                    candidate_schema_uris=(jacobian_schema_uri,),
                    reason="bundled independent sparse-polynomial Jacobian checker",
                ),
                authorize=True,
            )
            .checker_id
        )
        identity_checker_id = (
            CheckerInstaller(checkers)
            .install(
                CheckerOperation(
                    name="exact sparse rational polynomial identity checker",
                    entrypoint="jacobian_checkers.polynomial_maps:check_identity",
                    evidence_kind=EvidenceKind.CERTIFICATE,
                    format_id="polynomial.identity_replay",
                    format_version="1",
                    claim_schema_uris=(identity_claim_schema_uri,),
                    semantics_uris=(identity_semantics_uri,),
                    candidate_schema_uris=(right_polynomial_schema_uri,),
                    reason="bundled independent sparse-polynomial identity checker",
                ),
                authorize=True,
            )
            .checker_id
        )
        inverse_checker_id = (
            CheckerInstaller(checkers)
            .install(
                CheckerOperation(
                    name="exact two-sided polynomial-map inverse checker",
                    entrypoint="jacobian_checkers.polynomial_maps:check_map_inverse",
                    evidence_kind=EvidenceKind.CERTIFICATE,
                    format_id="polynomial.map.inverse.two_sided_replay",
                    format_version="1",
                    claim_schema_uris=(inverse_claim_schema_uri,),
                    semantics_uris=(inverse_semantics_uri,),
                    candidate_schema_uris=(inverse_residual_schema_uri,),
                    reason=(
                        "bundled independent two-sided sparse-polynomial map checker"
                    ),
                ),
                authorize=True,
            )
            .checker_id
        )
    installation = PolynomialInstallation(
        semantics_uri=semantics_uri,
        polynomial_semantics_uri=polynomial_semantics_uri,
        factorization_semantics_uri=factorization_semantics_uri,
        identity_semantics_uri=identity_semantics_uri,
        inverse_semantics_uri=inverse_semantics_uri,
        map_schema_uri=map_schema_uri,
        evaluation_schema_uri=evaluation_schema_uri,
        jacobian_schema_uri=jacobian_schema_uri,
        claim_schema_uri=claim_schema_uri,
        jacobian_claim_schema_uri=jacobian_claim_schema_uri,
        right_polynomial_schema_uri=right_polynomial_schema_uri,
        left_polynomial_schema_uri=left_polynomial_schema_uri,
        identity_claim_schema_uri=identity_claim_schema_uri,
        inverse_claim_schema_uri=inverse_claim_schema_uri,
        inverse_residual_schema_uri=inverse_residual_schema_uri,
        inverse_synthesis_schema_uri=inverse_synthesis_schema_uri,
        witness_schema_uri=witness_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        polynomial_schema_uri=polynomial_schema_uri,
        factorization_schema_uri=factorization_schema_uri,
        collision_checker_id=collision_checker_id,
        jacobian_checker_id=jacobian_checker_id,
        identity_checker_id=identity_checker_id,
        inverse_checker_id=inverse_checker_id,
    )
    resources = PolynomialResources(
        store=store,
        artifacts=artifacts,
        verification=verification,
        installation=installation,
    )
    return (
        (
            PolynomialMapEvaluationAdapter(resources),
            PolynomialJacobianAdapter(resources),
            PolynomialCollisionAdapter(resources),
            PolynomialIdentityAdapter(resources),
            PolynomialCollisionSearchAdapter(resources),
            *(
                (PolynomialCollisionVerifyAdapter(resources),)
                if collision_checker_id is not None
                else ()
            ),
            PolynomialFactorAdapter(resources),
            PolynomialMapInverseSynthesizeAdapter(resources),
            *(
                (PolynomialMapInverseVerifyAdapter(resources),)
                if inverse_checker_id is not None and identity_checker_id is not None
                else ()
            ),
        ),
        installation,
    )


class PolynomialMapEvaluationAdapter:
    """Evaluate one exact rational polynomial map at one exact point."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.evaluate",
            version="1",
            title="Evaluate a rational polynomial map",
            description=(
                "Compute the exact rational image of one point under one sparse "
                "square polynomial map over QQ."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("rational-polynomial-evaluation",),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialEvaluationRequest),
            output_schema=model_schema(PolynomialEvaluationOutput),
            tags=("polynomial", "map", "evaluation", "exact-computation"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialEvaluationRequest,
            request.input,
            code="INVALID_POLYNOMIAL_EVALUATION_REQUEST",
            operation="evaluation",
        )
        started = time.monotonic()
        polynomial_map = validated.map
        polynomial_map, map_uri = _materialize_map(self.resources, polynomial_map)
        point = RationalPolynomialPoint(values=validated.point)
        image = _evaluate(polynomial_map, point)
        evaluation, evaluation_uri = _materialize_evaluation(
            self.resources,
            map_uri=map_uri,
            point=point,
            image=image,
        )
        output = PolynomialEvaluationOutput(
            map_uri=map_uri,
            evaluation_uri=evaluation_uri,
            point=point.values,
            image=evaluation.image,
            backend_version=sympy.__version__,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one exact point evaluation for one polynomial map",
                parameters={
                    "map_uri": map_uri,
                    "point": point.model_dump(mode="json")["values"],
                },
                artifact_uri=map_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.evaluation-of",
                    source_artifact_uris=(map_uri,),
                    target_artifact_uris=(evaluation_uri,),
                ),
            ),
            artifact_uris=(map_uri, evaluation_uri),
            completeness_basis="every coordinate was evaluated exactly at the point",
        )


class PolynomialJacobianAdapter:
    """Compute the exact Jacobian matrix and determinant of one square map."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.compute_jacobian",
            version="1",
            title="Compute a polynomial-map Jacobian",
            description=(
                "Compute the exact Jacobian matrix and determinant of one sparse "
                "square polynomial map over QQ."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("symbolic-jacobian", "rational-polynomials"),
                checker_ids=(
                    (resources.installation.jacobian_checker_id,)
                    if resources.installation.jacobian_checker_id is not None
                    else ()
                ),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialJacobianRequest),
            output_schema=model_schema(PolynomialJacobianOutput),
            tags=("polynomial", "jacobian", "determinant", "exact-computation"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialJacobianRequest,
            request.input,
            code="INVALID_POLYNOMIAL_JACOBIAN_REQUEST",
            operation="Jacobian computation",
        )
        started = time.monotonic()
        polynomial_map = validated.map
        polynomial_map, map_uri = _materialize_map(self.resources, polynomial_map)
        try:
            generators, coordinates = _sympy_map(polynomial_map)
            matrix_polys = tuple(
                tuple(coordinate.diff(generator) for generator in generators)
                for coordinate in coordinates
            )
            determinant = Poly(
                expand(
                    Matrix(
                        [[entry.as_expr() for entry in row] for row in matrix_polys]
                    ).det()
                ),
                *generators,
                domain=QQ,
            )
        except (PolynomialError, TypeError, ValueError) as exc:
            raise _polynomial_error(
                "POLYNOMIAL_JACOBIAN_FAILED",
                "jacobian_computation",
                "The exact polynomial Jacobian computation failed.",
            ) from exc
        matrix = tuple(
            tuple(_wire_polynomial(entry) for entry in row) for row in matrix_polys
        )
        jacobian = PolynomialJacobian(
            map_uri=map_uri,
            variable_order=polynomial_map.variables,
            matrix=matrix,
            determinant=_wire_polynomial(determinant),
            backend_version=sympy.__version__,
        )
        jacobian_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.jacobian_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=jacobian.model_dump(mode="json"),
            parents=(map_uri,),
            summary="exact rational polynomial-map Jacobian",
        )
        claim = PolynomialJacobianClaim(source_map_uri=map_uri)
        claim_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.jacobian_claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(map_uri, jacobian_artifact.artifact_uri),
            summary="exact polynomial Jacobian replay claim",
        )
        semantics = self.resources.store.get(self.resources.installation.semantics_uri)
        source_map = self.resources.store.get(map_uri)
        certificate_payload = PolynomialJacobianReplayPayload(
            source_map_uri=map_uri,
            jacobian_uri=jacobian_artifact.artifact_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.jacobian_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=jacobian_artifact.object_digest,
                scope_digest=source_map.manifest.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(certificate_payload)).hexdigest()
            ),
            payload=certificate_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.certificate_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                claim_artifact.artifact_uri,
                jacobian_artifact.artifact_uri,
                map_uri,
            ),
            summary="unverified exact polynomial Jacobian replay certificate",
        )
        output = PolynomialJacobianOutput(
            map_uri=map_uri,
            jacobian_uri=jacobian_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=self.resources.installation.jacobian_checker_id,
            matrix=jacobian.matrix,
            determinant=jacobian.determinant,
            backend_version=sympy.__version__,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="the full Jacobian matrix for one square polynomial map",
                parameters={
                    "map_uri": map_uri,
                    "variable_order": list(polynomial_map.variables),
                },
                artifact_uri=map_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.jacobian-of",
                    source_artifact_uris=(map_uri,),
                    target_artifact_uris=(jacobian_artifact.artifact_uri,),
                ),
            ),
            artifact_uris=(
                map_uri,
                jacobian_artifact.artifact_uri,
                claim_artifact.artifact_uri,
                certificate_artifact.artifact_uri,
            ),
            completeness_basis=(
                "every partial derivative and the exact determinant were computed"
            ),
        )


class PolynomialCollisionAdapter:
    """Compare exact evaluation artifacts and materialize collision evidence."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.collision_witness",
            version="1",
            title="Construct a polynomial-map collision witness",
            description=(
                "Compare the declared canonical rational values in two structurally "
                "compatible point-evaluation artifacts for the same polynomial map "
                "and materialize an unverified candidate collision witness."
            ),
            provider="jacobian.artifact-comparison",
            provider_runtime=known_provider_runtime(
                "jacobian.artifact-comparison",
                features=("polynomial-collision-witness",),
                checker_ids=(
                    (resources.installation.collision_checker_id,)
                    if resources.installation.collision_checker_id is not None
                    else ()
                ),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialCollisionRequest),
            output_schema=model_schema(PolynomialCollisionOutput),
            tags=("polynomial", "map", "collision", "witness", "artifact-composition"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialCollisionRequest,
            request.input,
            code="INVALID_POLYNOMIAL_COLLISION_REQUEST",
            operation="collision construction",
        )
        started = time.monotonic()
        first_evaluation, first_evaluation_artifact = _load_evaluation(
            self.resources,
            validated.first_evaluation_uri,
            path="first_evaluation_uri",
        )
        second_evaluation, second_evaluation_artifact = _load_evaluation(
            self.resources,
            validated.second_evaluation_uri,
            path="second_evaluation_uri",
        )
        if first_evaluation.map_uri != second_evaluation.map_uri:
            raise _polynomial_error(
                "POLYNOMIAL_EVALUATION_MAP_MISMATCH",
                "collision_validation",
                "Collision evaluation artifacts must reference the same polynomial map.",
            )
        candidate_uri = first_evaluation.map_uri
        polynomial_map, candidate = _load_polynomial_map(
            self.resources,
            candidate_uri,
        )
        dimension = len(polynomial_map.variables)
        if any(
            len(evaluation.point.values) != dimension
            for evaluation in (first_evaluation, second_evaluation)
        ):
            raise _polynomial_error(
                "POLYNOMIAL_EVALUATION_DIMENSION_MISMATCH",
                "collision_validation",
                "Collision evaluation dimensions must match the polynomial map.",
            )
        first_point = first_evaluation.point
        second_point = second_evaluation.point
        first_image = first_evaluation.image
        second_image = second_evaluation.image
        claim = PolynomialInjectivityClaim(map_uri=candidate_uri)
        claim_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(candidate_uri,),
            summary="rational polynomial-map injectivity claim",
        )
        candidate_collision = (
            first_point.values != second_point.values and first_image == second_image
        )
        witness_uri = None
        if candidate_collision:
            # Evaluation payloads are candidate evidence. The independent checker,
            # not this comparison adapter, replays the map at both points.
            semantics = self.resources.store.get(
                self.resources.installation.semantics_uri
            )
            witness = WitnessEnvelope(
                witness_format="polynomial.map_collision",
                format_version="1",
                role=WitnessRole.REFUTES_CLAIM,
                bindings=EvidenceBindings(
                    claim_digest=claim_artifact.object_digest,
                    semantics_digest=semantics.manifest.object_digest,
                    candidate_digest=candidate.manifest.object_digest,
                ),
                payload=PolynomialCollisionPayload(
                    first_point=first_point.values,
                    second_point=second_point.values,
                    image=first_image,
                ).model_dump(mode="json"),
            )
            witness_artifact = self.resources.store.put(
                schema_uri=self.resources.installation.witness_schema_uri,
                semantics_uri=self.resources.installation.semantics_uri,
                payload=witness.model_dump(mode="json"),
                parents=(
                    claim_artifact.artifact_uri,
                    candidate_uri,
                    first_evaluation_artifact.artifact_uri,
                    second_evaluation_artifact.artifact_uri,
                ),
                summary="unverified rational polynomial-map collision witness",
            )
            witness_uri = witness_artifact.artifact_uri
        checker_id = self.resources.installation.collision_checker_id
        output = PolynomialCollisionOutput(
            claim_uri=claim_artifact.artifact_uri,
            candidate_uri=candidate_uri,
            first_evaluation_uri=first_evaluation_artifact.artifact_uri,
            second_evaluation_uri=second_evaluation_artifact.artifact_uri,
            first_point=first_point.values,
            second_point=second_point.values,
            first_image=first_image,
            second_image=second_image,
            candidate_collision=candidate_collision,
            witness_uri=witness_uri,
            checker_id=checker_id,
            certificate_available=witness_uri is not None and checker_id is not None,
        )
        artifact_uris = [
            candidate_uri,
            claim_artifact.artifact_uri,
            first_evaluation_artifact.artifact_uri,
            second_evaluation_artifact.artifact_uri,
        ]
        if witness_uri is not None:
            artifact_uris.append(witness_uri)
        relationships = [
            CapabilityRelationship(
                relation_id="polynomial.relation.evaluation-of",
                source_artifact_uris=(candidate_uri,),
                target_artifact_uris=(
                    first_evaluation_artifact.artifact_uri,
                    second_evaluation_artifact.artifact_uri,
                ),
            )
        ]
        if witness_uri is not None:
            relationships.append(
                CapabilityRelationship(
                    relation_id="polynomial.relation.collision-derived-from",
                    source_artifact_uris=(
                        first_evaluation_artifact.artifact_uri,
                        second_evaluation_artifact.artifact_uri,
                    ),
                    target_artifact_uris=(witness_uri,),
                )
            )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description=(
                    "exact comparison of two point-evaluation artifacts for one "
                    "polynomial map"
                ),
                parameters={
                    "candidate_uri": candidate_uri,
                    "first_evaluation_uri": first_evaluation_artifact.artifact_uri,
                    "second_evaluation_uri": second_evaluation_artifact.artifact_uri,
                },
                artifact_uri=candidate_uri,
            ),
            relationships=tuple(relationships),
            artifact_uris=tuple(artifact_uris),
            completeness_basis=(
                "both supplied evaluation artifact payloads were structurally "
                "validated and their declared values were compared exactly"
            ),
            assurance_basis=(
                "deterministic structural comparison of canonical rational payloads; "
                "the source evaluations were not replayed and any candidate witness "
                "remains unverified"
            ),
        )


class PolynomialCollisionSearchAdapter:
    """Search one fully declared finite rational grid for a collision."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.collision.search",
            version="1",
            title="Search a bounded rational grid for a collision",
            description=(
                "Enumerate one deterministic finite rational grid and return its "
                "first exact polynomial-map collision with reconciled accounting."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("bounded-rational-grid-search",),
                checker_ids=(
                    (resources.installation.collision_checker_id,)
                    if resources.installation.collision_checker_id is not None
                    else ()
                ),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialCollisionSearchRequest),
            output_schema=model_schema(PolynomialCollisionSearchOutput),
            tags=("polynomial", "map", "collision", "bounded-search"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialCollisionSearchRequest,
            request.input,
            code="INVALID_POLYNOMIAL_COLLISION_SEARCH_REQUEST",
            operation="collision search",
        )
        started = time.monotonic()
        polynomial_map, map_uri = _materialize_map(self.resources, validated.map)
        scalar_values = tuple(
            CanonicalRational(num=str(value.numerator), den=str(value.denominator))
            for value in sorted(
                {
                    Fraction(numerator, denominator)
                    for denominator in range(1, validated.max_denominator + 1)
                    for numerator in range(
                        -validated.max_abs_numerator,
                        validated.max_abs_numerator + 1,
                    )
                }
            )
        )
        grid_point_count = len(scalar_values) ** len(polynomial_map.variables)
        seen: dict[
            tuple[tuple[str, str], ...],
            tuple[tuple[CanonicalRational, ...], str],
        ] = {}
        found: (
            tuple[
                tuple[CanonicalRational, ...],
                tuple[CanonicalRational, ...],
                tuple[CanonicalRational, ...],
                str,
                str,
            ]
            | None
        ) = None
        evaluation_uris: list[str] = []
        examined = 0
        for point_values in product(
            scalar_values,
            repeat=len(polynomial_map.variables),
        ):
            examined += 1
            point = RationalPolynomialPoint(values=point_values)
            image = _evaluate(polynomial_map, point)
            _, evaluation_uri = _materialize_evaluation(
                self.resources,
                map_uri=map_uri,
                point=point,
                image=image,
            )
            evaluation_uris.append(evaluation_uri)
            key = tuple((value.num, value.den) for value in image)
            previous = seen.get(key)
            if previous is not None and previous[0] != point_values:
                found = (
                    previous[0],
                    point_values,
                    image,
                    previous[1],
                    evaluation_uri,
                )
                break
            seen[key] = (point_values, evaluation_uri)
        claim_uri: str | None = None
        witness_uri: str | None = None
        first_point_result: tuple[CanonicalRational, ...] | None = None
        second_point_result: tuple[CanonicalRational, ...] | None = None
        image_result: tuple[CanonicalRational, ...] | None = None
        first_evaluation_result: str | None = None
        second_evaluation_result: str | None = None
        if found is not None:
            (
                first_point_result,
                second_point_result,
                image_result,
                first_evaluation_result,
                second_evaluation_result,
            ) = found
            assert first_evaluation_result is not None
            assert second_evaluation_result is not None
            candidate = self.resources.store.get(map_uri)
            claim = self.resources.artifacts.put(
                schema_uri=self.resources.installation.claim_schema_uri,
                semantics_uri=self.resources.installation.semantics_uri,
                payload=PolynomialInjectivityClaim(map_uri=map_uri).model_dump(
                    mode="json"
                ),
                parents=(map_uri,),
                summary="rational polynomial-map injectivity claim",
            )
            semantics = self.resources.store.get(
                self.resources.installation.semantics_uri
            )
            witness = WitnessEnvelope(
                witness_format="polynomial.map_collision",
                format_version="1",
                role=WitnessRole.REFUTES_CLAIM,
                bindings=EvidenceBindings(
                    claim_digest=claim.object_digest,
                    semantics_digest=semantics.manifest.object_digest,
                    candidate_digest=candidate.manifest.object_digest,
                ),
                payload=PolynomialCollisionPayload(
                    first_point=first_point_result,
                    second_point=second_point_result,
                    image=image_result,
                ).model_dump(mode="json"),
            )
            witness_artifact = self.resources.artifacts.put(
                schema_uri=self.resources.installation.witness_schema_uri,
                semantics_uri=self.resources.installation.semantics_uri,
                payload=witness.model_dump(mode="json"),
                parents=(
                    claim.artifact_uri,
                    map_uri,
                    first_evaluation_result,
                    second_evaluation_result,
                ),
                summary="unverified bounded-search collision witness",
            )
            claim_uri = claim.artifact_uri
            witness_uri = witness_artifact.artifact_uri
        output = PolynomialCollisionSearchOutput(
            found=found is not None,
            map_uri=map_uri,
            examined_point_count=examined,
            grid_point_count=grid_point_count,
            first_point=first_point_result,
            second_point=second_point_result,
            common_image=image_result,
            first_evaluation_uri=first_evaluation_result,
            second_evaluation_uri=second_evaluation_result,
            claim_uri=claim_uri,
            witness_uri=witness_uri,
            checker_id=self.resources.installation.collision_checker_id,
            stop_reason=(
                PolynomialCollisionSearchStopReason.FIRST_COLLISION
                if found is not None
                else PolynomialCollisionSearchStopReason.GRID_EXHAUSTED
            ),
        )
        artifacts = [map_uri, *evaluation_uris]
        relationships = [
            CapabilityRelationship(
                relation_id="polynomial.relation.evaluation-of",
                source_artifact_uris=(map_uri,),
                target_artifact_uris=tuple(evaluation_uris),
            )
        ]
        if found is not None:
            assert first_evaluation_result is not None
            assert second_evaluation_result is not None
            assert claim_uri is not None
            assert witness_uri is not None
            artifacts.extend(
                [
                    second_evaluation_result,
                    claim_uri,
                    witness_uri,
                ]
            )
            relationships.extend(
                (
                    CapabilityRelationship(
                        relation_id="polynomial.relation.injectivity-claim-of",
                        source_artifact_uris=(map_uri,),
                        target_artifact_uris=(claim_uri,),
                    ),
                    CapabilityRelationship(
                        relation_id="polynomial.relation.collision-derived-from",
                        source_artifact_uris=(
                            first_evaluation_result,
                            second_evaluation_result,
                        ),
                        target_artifact_uris=(witness_uri,),
                    ),
                    CapabilityRelationship(
                        relation_id=(
                            "polynomial.relation.collision-refutes-injectivity"
                        ),
                        source_artifact_uris=(witness_uri,),
                        target_artifact_uris=(claim_uri,),
                    ),
                )
            )
        exhausted_grid = examined == grid_point_count
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="declared finite rational grid",
                parameters={
                    "max_abs_numerator": validated.max_abs_numerator,
                    "max_denominator": validated.max_denominator,
                    "grid_point_count": grid_point_count,
                },
                artifact_uri=map_uri,
            ),
            relationships=tuple(relationships),
            artifact_uris=tuple(
                dict.fromkeys(uri for uri in artifacts if uri is not None)
            ),
            completeness_basis=(
                "the deterministic grid was fully enumerated"
                if exhausted_grid
                else "the canonical prefix through the first collision was enumerated"
            ),
            completeness_status=(
                CapabilityCompletenessStatus.COMPLETE
                if exhausted_grid
                else CapabilityCompletenessStatus.PARTIAL
            ),
            assurance_basis=(
                "deterministic exact SymPy search; any returned witness remains "
                "unverified until independent replay"
            ),
        )


class PolynomialCollisionVerifyAdapter:
    """Independently verify one explicit exact rational collision."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.collision_checker_id
        assert checker_id is not None
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.collision.verify",
            version="1",
            title="Verify a polynomial-map collision",
            description=(
                "Independently reevaluate one exact map at two supplied distinct "
                "rational points and verify their claimed common image."
            ),
            provider="jacobian.polynomial-collision-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.polynomial-collision-checker",
                features=("exact-rational-collision-replay",),
                checker_ids=(checker_id,),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(PolynomialCollisionVerifyRequest),
            output_schema=model_schema(PolynomialCollisionVerifyOutput),
            tags=("polynomial", "map", "collision", "verification"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialCollisionVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_COLLISION_VERIFY_REQUEST",
            operation="collision verification",
        )
        _, map_uri = _materialize_map(self.resources, validated.map)
        candidate = self.resources.store.get(map_uri)
        claim_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=PolynomialInjectivityClaim(map_uri=map_uri).model_dump(mode="json"),
            parents=(map_uri,),
            summary="rational polynomial-map injectivity claim",
        )
        semantics = self.resources.store.get(self.resources.installation.semantics_uri)
        witness = WitnessEnvelope(
            witness_format="polynomial.map_collision",
            format_version="1",
            role=WitnessRole.REFUTES_CLAIM,
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=candidate.manifest.object_digest,
            ),
            payload=PolynomialCollisionPayload(
                first_point=validated.first_point,
                second_point=validated.second_point,
                image=validated.claimed_image,
            ).model_dump(mode="json"),
        )
        witness_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.witness_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(claim_artifact.artifact_uri, map_uri),
            summary="exact rational polynomial-map collision witness",
        )
        checker_id = self.resources.installation.collision_checker_id
        assert checker_id is not None
        checked = self.resources.verification.verify_witness(
            claim_uri=claim_artifact.artifact_uri,
            candidate_uri=map_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.assurance.verification is Verification.VERIFIED
            and checked.conclusion is Conclusion.FALSE
        )
        output = PolynomialCollisionVerifyOutput(
            collision_verified=verified,
            conclusion="FALSE" if verified else "UNKNOWN",
            verification_input=checked.input,
            map_uri=map_uri,
            claim_uri=claim_artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            verification_record_uri=checked.verification_record_uri,
            checker_id=checker_id,
            first_point=validated.first_point,
            second_point=validated.second_point,
            claimed_image=validated.claimed_image,
        )
        artifact_uris = [
            map_uri,
            claim_artifact.artifact_uri,
            witness_artifact.artifact_uri,
        ]
        if checked.verification_record_uri is not None:
            artifact_uris.append(checked.verification_record_uri)
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one direct collision witness over QQ",
                parameters={"map_uri": map_uri},
                artifact_uri=map_uri,
            ),
            completeness=CapabilityCompleteness(
                status=CapabilityCompletenessStatus.NOT_APPLICABLE,
                basis="direct witness verification makes no search coverage claim",
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.collision-refutes-injectivity",
                    source_artifact_uris=(witness_artifact.artifact_uri,),
                    target_artifact_uris=(claim_artifact.artifact_uri,),
                    status=(
                        CapabilityRelationshipStatus.VERIFIED
                        if verified
                        else CapabilityRelationshipStatus.PROPOSED
                    ),
                    verification_record_uri=(
                        checked.verification_record_uri if verified else None
                    ),
                ),
            ),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted by the authorized independent Fraction-based checker"
                    if verified
                    else "the checker did not accept the claimed collision"
                ),
                verification_record_uri=checked.verification_record_uri,
            ),
            artifact_uris=tuple(artifact_uris),
        )


class PolynomialFactorAdapter:
    """Factor one univariate sparse polynomial over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.factor.compute",
            version="1",
            title="Factor a univariate rational polynomial",
            description=(
                "Compute a coefficient and multiplicity-bearing factor list over QQ, "
                "together with an exact reconstructed product."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("univariate-polynomial-factorization",),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialFactorRequest),
            output_schema=model_schema(PolynomialFactorOutput),
            tags=("polynomial", "factorization", "exact-computation"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate_request(
            PolynomialFactorRequest,
            request.input,
            code="INVALID_POLYNOMIAL_FACTOR_REQUEST",
            operation="factorization",
        )
        started = time.monotonic()
        source = self.resources.artifacts.put(
            schema_uri=self.resources.installation.polynomial_schema_uri,
            semantics_uri=self.resources.installation.polynomial_semantics_uri,
            payload=validated.polynomial.model_dump(mode="json"),
            summary="univariate sparse rational polynomial",
        )
        generator = cast(tuple[Any, ...], symbols(validated.variable, seq=True))
        polynomial = _sympy_polynomial(validated.polynomial, generator)
        coefficient_value, raw_factors = polynomial.factor_list()
        factors = tuple(
            PolynomialFactorRecord(
                factor=_wire_polynomial(factor),
                multiplicity=multiplicity,
            )
            for factor, multiplicity in raw_factors
        )
        reconstructed_expression = sympy.Rational(coefficient_value)
        for factor, multiplicity in raw_factors:
            reconstructed_expression *= factor.as_expr() ** multiplicity
        reconstructed = _wire_polynomial(
            Poly(
                sympy.expand(reconstructed_expression),
                *generator,
                domain=QQ,
            )
        )
        artifact_payload = PolynomialFactorizationArtifact(
            variable=validated.variable,
            source_polynomial_uri=source.artifact_uri,
            coefficient=_wire_rational(coefficient_value),
            factors=factors,
            reconstructed=reconstructed,
            backend_version=sympy.__version__,
        )
        factorization = self.resources.artifacts.put(
            schema_uri=self.resources.installation.factorization_schema_uri,
            semantics_uri=self.resources.installation.factorization_semantics_uri,
            payload=artifact_payload.model_dump(mode="json"),
            parents=(source.artifact_uri,),
            summary="computed univariate rational polynomial factorization",
        )
        output = PolynomialFactorOutput(
            source_polynomial_uri=source.artifact_uri,
            factorization_uri=factorization.artifact_uri,
            variable=validated.variable,
            coefficient=artifact_payload.coefficient,
            factors=factors,
            reconstructed=reconstructed,
            backend_version=sympy.__version__,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one univariate polynomial over QQ",
                parameters={"variable": validated.variable},
                artifact_uri=source.artifact_uri,
            ),
            relationships=(
                CapabilityRelationship(
                    relation_id="polynomial.relation.factorization-of",
                    source_artifact_uris=(source.artifact_uri,),
                    target_artifact_uris=(factorization.artifact_uri,),
                ),
            ),
            artifact_uris=(source.artifact_uri, factorization.artifact_uri),
            completeness_basis=(
                "SymPy returned a factor list and its product was reconstructed exactly"
            ),
            assurance_basis=(
                "exact SymPy factorization and product reconstruction over QQ; "
                "factor irreducibility was not independently verified"
            ),
        )


def _materialize_map(
    resources: PolynomialResources,
    polynomial_map: RationalPolynomialMap,
) -> tuple[RationalPolynomialMap, str]:
    artifact = resources.artifacts.put(
        schema_uri=resources.installation.map_schema_uri,
        semantics_uri=resources.installation.semantics_uri,
        payload=polynomial_map.model_dump(mode="json"),
        summary="exact sparse rational polynomial map",
    )
    return polynomial_map, artifact.artifact_uri


def _load_evaluation(
    resources: PolynomialResources,
    evaluation_uri: str,
    *,
    path: str,
) -> tuple[PolynomialMapEvaluation, StoredArtifact]:
    try:
        artifact = resources.store.get(evaluation_uri)
    except StoreError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="POLYNOMIAL_EVALUATION_ARTIFACT_NOT_FOUND",
                stage="evaluation_resolution",
                message="The requested polynomial evaluation artifact is unavailable.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Use an evaluation URI returned by polynomial.map.evaluate.",
            )
        ) from exc
    if (
        artifact.manifest.schema_uri != resources.installation.evaluation_schema_uri
        or artifact.manifest.semantics_uri != resources.installation.semantics_uri
        or not isinstance(artifact.payload, dict)
    ):
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The artifact is not a compatible polynomial-map evaluation.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Use an evaluation URI returned by polynomial.map.evaluate.",
            )
        )
    try:
        evaluation = PolynomialMapEvaluation.model_validate(artifact.payload)
    except ValidationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The polynomial-map evaluation artifact payload is malformed.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Recreate the artifact through polynomial.map.evaluate.",
            )
        ) from exc
    if evaluation.map_uri not in artifact.manifest.parents:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="MISBOUND_POLYNOMIAL_EVALUATION_ARTIFACT",
                stage="evaluation_validation",
                message="The evaluation artifact is not bound to its declared map.",
                path=path,
                schema_uri=resources.installation.evaluation_schema_uri,
                hint="Recreate the artifact through polynomial.map.evaluate.",
            )
        )
    return evaluation, artifact


def _load_polynomial_map(
    resources: PolynomialResources,
    map_uri: str,
) -> tuple[RationalPolynomialMap, StoredArtifact]:
    try:
        artifact = resources.store.get(map_uri)
    except StoreError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="POLYNOMIAL_MAP_ARTIFACT_NOT_FOUND",
                stage="map_resolution",
                message="The polynomial map referenced by an evaluation is unavailable.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        ) from exc
    if (
        artifact.manifest.schema_uri != resources.installation.map_schema_uri
        or artifact.manifest.semantics_uri != resources.installation.semantics_uri
        or not isinstance(artifact.payload, dict)
    ):
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_MAP_ARTIFACT",
                stage="map_validation",
                message="An evaluation references an incompatible polynomial map.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        )
    try:
        polynomial_map = RationalPolynomialMap.model_validate(artifact.payload)
    except ValidationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INCOMPATIBLE_POLYNOMIAL_MAP_ARTIFACT",
                stage="map_validation",
                message="The referenced polynomial map artifact payload is malformed.",
                path="evaluation.map_uri",
                schema_uri=resources.installation.map_schema_uri,
                hint="Recreate the evaluations through polynomial.map.evaluate.",
            )
        ) from exc
    return polynomial_map, artifact


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
            modes=(CapabilityMode.VERIFY,),
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
                    mode=CapabilityMode.VERIFY,
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
        verified = checked.verification_record_uri is not None
        conclusion = checked.conclusion
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
            verification_record_uri=checked.verification_record_uri,
            checker_id=checker_id,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
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
                    verification_record_uri=checked.verification_record_uri,
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
                    verification_record_uri=checked.verification_record_uri,
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
                verification_record_uri=checked.verification_record_uri,
            ),
            artifact_uris=(
                left.artifact_uri,
                right.artifact_uri,
                claim.artifact_uri,
                certificate_artifact.artifact_uri,
                *(
                    (checked.verification_record_uri,)
                    if checked.verification_record_uri is not None
                    else ()
                ),
            ),
        )


def _inverse_solver_worker(
    equation_texts: tuple[str, ...],
    unknown_names: tuple[str, ...],
    result_queue: Any,
) -> None:
    """Solve one exact coefficient system in an isolated process."""

    try:
        unknowns = symbols(" ".join(unknown_names), seq=True)
        locals_by_name = dict(zip(unknown_names, unknowns, strict=True))
        equations = tuple(
            sympify(expression, locals=locals_by_name) for expression in equation_texts
        )
        solutions = solve(equations, unknowns, dict=True, simplify=False)
        serialized = tuple(
            tuple(
                (name, str(solution.get(symbol, symbol)))
                for name, symbol in zip(
                    unknown_names,
                    unknowns,
                    strict=True,
                )
            )
            for solution in solutions
        )
        result_queue.put(("OK", tuple(sorted(serialized))))
    except Exception as exc:  # pragma: no cover - defensive child boundary
        result_queue.put(("ERROR", type(exc).__name__))


class PolynomialMapInverseSynthesizeAdapter:
    """Solve one bounded exact polynomial inverse ansatz over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.inverse.candidate_synthesize",
            version="1",
            title="Synthesize a bounded polynomial-map inverse candidate",
            description=(
                "Solve an explicit finite polynomial inverse ansatz over QQ, "
                "then submit every found candidate to the independent two-sided "
                "inverse verifier."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("polynomial-map-inverse-ansatz", "exact-equation-solving"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(PolynomialMapInverseSynthesisRequest),
            output_schema=model_schema(PolynomialMapInverseSynthesisOutput),
            tags=("polynomial", "map", "inverse", "synthesis", "exact-rational"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="triangular_inverse",
                    description=(
                        "Synthesize and independently check the degree-two "
                        "inverse of (x + y^2, y)."
                    ),
                    mode=CapabilityMode.EXPLORE,
                    input=PolynomialMapInverseSynthesisRequest.model_validate(
                        {
                            "forward_map": {
                                "variables": ["x", "y"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1, 0],
                                            },
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [0, 2],
                                            },
                                        ]
                                    },
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [0, 1],
                                            }
                                        ]
                                    },
                                ],
                            },
                            "source_variables": ["x", "y"],
                            "target_variables": ["u", "v"],
                            "inverse_degree_bound": 2,
                            "support_mode": "FULL_TOTAL_DEGREE",
                            "solver": "sympy.solve",
                            "limits": {
                                "timeout_ms": 10000,
                                "max_inverse_degree": 4,
                                "max_composition_degree": 32,
                                "max_unknown_coefficients": 64,
                                "max_coefficient_equations": 512,
                                "max_residual_terms": 1024,
                            },
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        started = time.monotonic()
        validated = _validate_request(
            PolynomialMapInverseSynthesisRequest,
            request.input,
            code="INVALID_POLYNOMIAL_MAP_INVERSE_SYNTHESIS_REQUEST",
            operation="map inverse candidate synthesis",
        )
        forward_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=self.resources.installation.inverse_semantics_uri,
            payload=validated.forward_map.model_dump(mode="json"),
            summary="forward map for bounded inverse synthesis",
        )
        supports = _inverse_supports(validated)
        coefficient_names = tuple(
            tuple(f"c_{coordinate}_{index}" for index in range(len(support)))
            for coordinate, support in enumerate(supports)
        )
        ansatz = PolynomialInverseAnsatzSpecification(
            support_mode=validated.support_mode,
            inverse_degree_bound=validated.inverse_degree_bound,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            coordinate_supports=supports,
            coefficient_symbols=coefficient_names,
        )
        unknown_names = tuple(name for row in coefficient_names for name in row)

        status: PolynomialInverseSynthesisStatus
        equations: tuple[PolynomialInverseCoefficientEquation, ...] = ()
        candidate: RationalPolynomialMap | None = None
        left_residuals: tuple[SparseRationalPolynomial, ...] = ()
        right_residuals: tuple[SparseRationalPolynomial, ...] = ()
        verification_output: dict[str, Any] | None = None
        verification_artifact_uri: str | None = None
        verification_failure: str | None = None
        residual_term_count = 0

        if validated.solver != "sympy.solve":
            status = PolynomialInverseSynthesisStatus.UNSUPPORTED
            verification_failure = (
                f"solver {validated.solver!r} is unsupported; use 'sympy.solve'"
            )
        elif validated.limits.timeout_ms == 0:
            status = PolynomialInverseSynthesisStatus.TIMEOUT
        elif len(unknown_names) > validated.limits.max_unknown_coefficients:
            status = PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED
            verification_failure = "ansatz unknown count exceeds the declared limit"
        else:
            forward_degree = max(
                (
                    sum(term.exponents)
                    for coordinate in validated.forward_map.coordinates
                    for term in coordinate.terms
                ),
                default=0,
            )
            residual_term_bound = _inverse_residual_term_bound(validated, supports)
            precheck_exhausted = (
                forward_degree * validated.inverse_degree_bound
                > validated.limits.max_composition_degree
                or residual_term_bound > validated.limits.max_residual_terms
            )
            if precheck_exhausted:
                status = PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED
                verification_failure = (
                    "conservative composition degree or residual-term bound "
                    "exceeds a declared limit"
                )
                ansatz_expressions = ()
                unknown_symbols = ()
            else:
                (
                    ansatz_expressions,
                    unknown_symbols,
                    equations,
                    residual_term_count,
                ) = _inverse_coefficient_system(
                    validated,
                    supports,
                    coefficient_names,
                )
            if precheck_exhausted:
                pass
            elif (
                residual_term_count > validated.limits.max_residual_terms
                or len(equations) > validated.limits.max_coefficient_equations
            ):
                status = PolynomialInverseSynthesisStatus.BUDGET_EXHAUSTED
                verification_failure = (
                    "derived coefficient system exceeds a declared residual "
                    "or equation limit"
                )
            elif not equations:
                status = PolynomialInverseSynthesisStatus.UNDERDETERMINED
                verification_failure = "the ansatz produced no coefficient equations"
            else:
                remaining_ms = validated.limits.timeout_ms - int(
                    (time.monotonic() - started) * 1000
                )
                if remaining_ms <= 0:
                    solve_status, solution = "TIMEOUT", None
                else:
                    solve_status, solution = _solve_inverse_system(
                        equations,
                        unknown_names,
                        timeout_ms=remaining_ms,
                    )
                if solve_status == "TIMEOUT":
                    status = PolynomialInverseSynthesisStatus.TIMEOUT
                elif solve_status == "ERROR":
                    status = PolynomialInverseSynthesisStatus.UNSUPPORTED
                    verification_failure = "the configured exact solver failed"
                elif solution is None:
                    status = PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ
                elif any(value.free_symbols for value in solution.values()) or set(
                    solution
                ) != set(unknown_symbols):
                    status = PolynomialInverseSynthesisStatus.UNDERDETERMINED
                    verification_failure = (
                        "the coefficient equations have free parameters"
                    )
                elif not all(value.is_Rational for value in solution.values()):
                    status = PolynomialInverseSynthesisStatus.UNSUPPORTED
                    verification_failure = (
                        "the selected solution is not rational over QQ"
                    )
                else:
                    candidate = _inverse_candidate_map(
                        validated,
                        ansatz_expressions,
                        solution,
                    )
                    verify_request = PolynomialMapInverseVerifyRequest(
                        forward_map=validated.forward_map,
                        inverse_map=candidate,
                        source_variables=validated.source_variables,
                        target_variables=validated.target_variables,
                    )
                    left_residuals, right_residuals = _map_inverse_residuals(
                        verify_request
                    )
                    status = PolynomialInverseSynthesisStatus.FOUND
                    try:
                        verified = PolynomialMapInverseVerifyAdapter(
                            self.resources
                        ).invoke(
                            CapabilityRequest(
                                capability_id="polynomial.map.inverse.verify",
                                mode=CapabilityMode.VERIFY,
                                input=verify_request.model_dump(mode="json"),
                            )
                        )
                        verification_output = verified.output
                        artifact = verified.output.get(
                            "verification_record_uri"
                        ) or verified.output.get("certificate_uri")
                        verification_artifact_uri = (
                            artifact if isinstance(artifact, str) else None
                        )
                        if verified.output.get("inverse_verified") is not True:
                            verification_failure = (
                                "the independent two-sided verifier rejected "
                                "the synthesized candidate"
                            )
                    except CapabilityInvocationError as exc:
                        verification_failure = exc.diagnostic.message

        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        provenance = PolynomialInverseSolverProvenance(
            solver=validated.solver,
            backend_version=sympy.__version__,
            timeout_ms=validated.limits.timeout_ms,
            unknown_count=len(unknown_names),
            equation_count=len(equations),
            residual_term_count=residual_term_count,
            elapsed_ms=elapsed_ms,
        )
        payload = PolynomialMapInverseSynthesisArtifact(
            status=status,
            forward_map=validated.forward_map,
            ansatz=ansatz,
            coefficient_equations=equations,
            solver_provenance=provenance,
            candidate_inverse_map=candidate,
            inverse_after_forward=left_residuals,
            forward_after_inverse=right_residuals,
            verification_output=verification_output,
            verification_artifact_uri=verification_artifact_uri,
            verification_failure=verification_failure,
        )
        parents = tuple(
            dict.fromkeys(
                (
                    forward_artifact.artifact_uri,
                    *(
                        (verification_artifact_uri,)
                        if verification_artifact_uri
                        else ()
                    ),
                )
            )
        )
        synthesis = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_synthesis_schema_uri,
            semantics_uri=self.resources.installation.inverse_semantics_uri,
            payload=payload.model_dump(mode="json"),
            parents=parents,
            summary=f"bounded polynomial inverse synthesis: {status.value}",
        )
        output = PolynomialMapInverseSynthesisOutput(
            **payload.model_dump(mode="python"),
            synthesis_uri=synthesis.artifact_uri,
            forward_map_uri=forward_artifact.artifact_uri,
        )
        artifact_uris = tuple(
            dict.fromkeys(
                (
                    forward_artifact.artifact_uri,
                    synthesis.artifact_uri,
                    *(
                        (verification_artifact_uri,)
                        if verification_artifact_uri
                        else ()
                    ),
                )
            )
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=Execution(
                status=(
                    ExecutionStatus.TIMEOUT
                    if status is PolynomialInverseSynthesisStatus.TIMEOUT
                    else ExecutionStatus.COMPLETED
                ),
                runtime_ms=elapsed_ms,
            ),
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="one finite polynomial inverse coefficient ansatz over QQ",
                parameters={
                    "inverse_degree_bound": validated.inverse_degree_bound,
                    "support_mode": validated.support_mode.value,
                    "unknown_count": len(unknown_names),
                },
                artifact_uri=synthesis.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if status
                    in {
                        PolynomialInverseSynthesisStatus.FOUND,
                        PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ,
                        PolynomialInverseSynthesisStatus.UNDERDETERMINED,
                    }
                    else CapabilityCompletenessStatus.PARTIAL
                ),
                basis=(
                    "the declared finite ansatz and exact coefficient system were solved"
                    if status
                    in {
                        PolynomialInverseSynthesisStatus.FOUND,
                        PolynomialInverseSynthesisStatus.NO_CANDIDATE_WITHIN_ANSATZ,
                        PolynomialInverseSynthesisStatus.UNDERDETERMINED,
                    }
                    else "the declared synthesis operation did not complete"
                ),
                assurance_level=CapabilityAssuranceLevel.COMPUTED,
            ),
            assurance=CapabilityAssurance(
                level=CapabilityAssuranceLevel.COMPUTED,
                basis=(
                    "bounded exact symbolic synthesis; no failure status proves "
                    "noninvertibility and only the separate verifier may certify "
                    "a found candidate"
                ),
            ),
            artifact_uris=artifact_uris,
        )


def _inverse_supports(
    request: PolynomialMapInverseSynthesisRequest,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if request.support_mode is PolynomialInverseSupportMode.EXPLICIT:
        assert request.explicit_support is not None
        return request.explicit_support
    dimension = len(request.target_variables)
    support = tuple(
        _reverse_lex_degree_bounded_exponents(
            dimension=dimension,
            degree_bound=request.inverse_degree_bound,
        )
    )
    return tuple(support for _ in range(dimension))


def _reverse_lex_degree_bounded_exponents(
    *,
    dimension: int,
    degree_bound: int,
) -> Iterator[tuple[int, ...]]:
    if dimension == 1:
        for exponent in range(degree_bound, -1, -1):
            yield (exponent,)
        return
    for exponent in range(degree_bound, -1, -1):
        for suffix in _reverse_lex_degree_bounded_exponents(
            dimension=dimension - 1,
            degree_bound=degree_bound - exponent,
        ):
            yield (exponent, *suffix)


def _inverse_residual_term_bound(
    request: PolynomialMapInverseSynthesisRequest,
    supports: tuple[tuple[tuple[int, ...], ...], ...],
) -> int:
    forward_counts = tuple(
        len(coordinate.terms) for coordinate in request.forward_map.coordinates
    )
    support_counts = tuple(len(support) for support in supports)
    left = sum(
        multiply(
            count**exponent
            for count, exponent in zip(
                forward_counts,
                monomial,
                strict=True,
            )
        )
        for support in supports
        for monomial in support
    )
    right = sum(
        multiply(
            count**exponent
            for count, exponent in zip(
                support_counts,
                term.exponents,
                strict=True,
            )
        )
        for coordinate in request.forward_map.coordinates
        for term in coordinate.terms
    )
    return int(left + right + 2 * len(request.source_variables))


def _inverse_coefficient_system(
    request: PolynomialMapInverseSynthesisRequest,
    supports: tuple[tuple[tuple[int, ...], ...], ...],
    coefficient_names: tuple[tuple[str, ...], ...],
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[PolynomialInverseCoefficientEquation, ...],
    int,
]:
    source_generators, forward = _sympy_map(request.forward_map)
    target_generators = tuple(symbols(request.target_variables))
    flat_names = tuple(name for row in coefficient_names for name in row)
    unknowns = tuple(symbols(" ".join(flat_names), seq=True))
    unknown_by_name = dict(zip(flat_names, unknowns, strict=True))
    ansatz = tuple(
        expand(
            sum(
                unknown_by_name[name]
                * multiply(
                    generator**exponent
                    for generator, exponent in zip(
                        target_generators,
                        exponents,
                        strict=True,
                    )
                )
                for name, exponents in zip(names, support, strict=True)
            )
        )
        for names, support in zip(coefficient_names, supports, strict=True)
    )
    left_substitutions = {
        generator: polynomial.as_expr()
        for generator, polynomial in zip(
            target_generators,
            forward,
            strict=True,
        )
    }
    left = tuple(
        expand(
            expression.subs(left_substitutions, simultaneous=True)
            - source_generators[index]
        )
        for index, expression in enumerate(ansatz)
    )
    right_substitutions = dict(zip(source_generators, ansatz, strict=True))
    right = tuple(
        expand(
            polynomial.as_expr().subs(right_substitutions, simultaneous=True)
            - target_generators[index]
        )
        for index, polynomial in enumerate(forward)
    )
    records: list[PolynomialInverseCoefficientEquation] = []
    residual_term_count = 0
    for direction, generators, residuals in (
        ("INVERSE_AFTER_FORWARD", source_generators, left),
        ("FORWARD_AFTER_INVERSE", target_generators, right),
    ):
        for coordinate, residual in enumerate(residuals):
            polynomial = Poly(residual, *generators)
            terms = polynomial.terms()
            residual_term_count += len(terms)
            records.extend(
                PolynomialInverseCoefficientEquation(
                    direction=cast(Any, direction),
                    coordinate=coordinate,
                    monomial_exponents=monomial,
                    expression=str(coefficient),
                )
                for monomial, coefficient in terms
                if coefficient != 0
            )
    return ansatz, unknowns, tuple(records), residual_term_count


def _solve_inverse_system(
    equations: tuple[PolynomialInverseCoefficientEquation, ...],
    unknown_names: tuple[str, ...],
    *,
    timeout_ms: int,
) -> tuple[str, dict[Any, Any] | None]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_inverse_solver_worker,
        args=(
            tuple(item.expression for item in equations),
            unknown_names,
            result_queue,
        ),
    )
    process.start()
    process.join(timeout_ms / 1000)
    if process.is_alive():
        process.terminate()
        process.join()
        return "TIMEOUT", None
    try:
        status, raw = result_queue.get_nowait()
    except Empty:
        return "ERROR", None
    if status != "OK":
        return "ERROR", None
    if not raw:
        return "OK", None
    symbols_by_name = dict(
        zip(unknown_names, symbols(" ".join(unknown_names), seq=True), strict=True)
    )
    first = raw[0]
    solution = {
        symbols_by_name[name]: sympify(value, locals=symbols_by_name)
        for name, value in first
    }
    return "OK", solution


def _inverse_candidate_map(
    request: PolynomialMapInverseSynthesisRequest,
    ansatz: tuple[Any, ...],
    solution: dict[Any, Any],
) -> RationalPolynomialMap:
    target_generators = tuple(symbols(request.target_variables))
    return RationalPolynomialMap(
        variables=request.target_variables,
        coordinates=tuple(
            _wire_polynomial(
                Poly(
                    expand(expression.subs(solution, simultaneous=True)),
                    *target_generators,
                    domain=QQ,
                )
            )
            for expression in ansatz
        ),
    )


class PolynomialMapInverseVerifyAdapter:
    """Verify both exact compositions of two square polynomial maps."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.inverse_checker_id
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.inverse.verify",
            version="1",
            title="Verify a two-sided polynomial-map inverse",
            description=(
                "Independently replay both ordered polynomial-map compositions "
                "over QQ and accept an inverse only when both are identity."
            ),
            provider="jacobian.sparse-polynomial-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.sparse-polynomial-checker",
                features=("polynomial-map-composition", "two-sided-inverse"),
                checker_ids=((checker_id,) if checker_id is not None else ()),
            ),
            modes=(CapabilityMode.VERIFY,),
            input_schema=model_schema(PolynomialMapInverseVerifyRequest),
            output_schema=model_schema(PolynomialMapInverseVerifyOutput),
            tags=("polynomial", "map", "inverse", "verification", "exact-rational"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="identity_map_inverse",
                    description=(
                        "Independently verify the identity map as its own "
                        "two-sided inverse over QQ."
                    ),
                    mode=CapabilityMode.VERIFY,
                    input=PolynomialMapInverseVerifyRequest.model_validate(
                        {
                            "forward_map": {
                                "variables": ["x"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1],
                                            }
                                        ]
                                    }
                                ],
                            },
                            "inverse_map": {
                                "variables": ["u"],
                                "coordinates": [
                                    {
                                        "terms": [
                                            {
                                                "coefficient": {
                                                    "num": "1",
                                                    "den": "1",
                                                },
                                                "exponents": [1],
                                            }
                                        ]
                                    }
                                ],
                            },
                            "source_variables": ["x"],
                            "target_variables": ["u"],
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
            PolynomialMapInverseVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_MAP_INVERSE_REQUEST",
            operation="map inverse verification",
        )
        checker_id = self.resources.installation.inverse_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_MAP_INVERSE_CHECKER_UNAVAILABLE",
                "inverse_verification",
                "No authorized polynomial-map inverse checker is installed.",
            )
        semantics_uri = self.resources.installation.inverse_semantics_uri
        forward = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=semantics_uri,
            payload=validated.forward_map.model_dump(mode="json"),
            summary="forward sparse rational polynomial map",
        )
        inverse = self.resources.artifacts.put(
            schema_uri=self.resources.installation.map_schema_uri,
            semantics_uri=semantics_uri,
            payload=validated.inverse_map.model_dump(mode="json"),
            summary="candidate inverse sparse rational polynomial map",
        )
        left_residuals, right_residuals = _map_inverse_residuals(validated)
        identity_adapter = PolynomialIdentityAdapter(self.resources)
        zero = SparseRationalPolynomial()
        left_records: list[str] = []
        right_records: list[str] = []
        identity_artifacts: list[str] = []
        for variables, residuals, records in (
            (validated.source_variables, left_residuals, left_records),
            (validated.target_variables, right_residuals, right_records),
        ):
            for residual in residuals:
                checked = identity_adapter.invoke(
                    CapabilityRequest(
                        capability_id="polynomial.identity.verify",
                        mode=CapabilityMode.VERIFY,
                        input=PolynomialIdentityRequest(
                            variables=variables,
                            left=residual,
                            right=zero,
                        ).model_dump(mode="json"),
                    )
                )
                record_uri = checked.output.get("verification_record_uri")
                if not isinstance(record_uri, str):
                    raise _polynomial_error(
                        "POLYNOMIAL_IDENTITY_REPLAY_UNAVAILABLE",
                        "composition_identity_replay",
                        "A composition residual could not obtain a checker record.",
                    )
                records.append(record_uri)
                identity_artifacts.extend(checked.artifact_uris)
        residual_payload = PolynomialMapCompositionResiduals(
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            inverse_after_forward=left_residuals,
            forward_after_inverse=right_residuals,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
        )
        residual_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_residual_schema_uri,
            semantics_uri=semantics_uri,
            payload=residual_payload.model_dump(mode="json"),
            parents=tuple(
                dict.fromkeys(
                    (
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        *left_records,
                        *right_records,
                    )
                )
            ),
            summary="both exact polynomial-map composition residual families",
        )
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_claim_schema_uri,
            semantics_uri=semantics_uri,
            payload=PolynomialMapInverseClaim(
                forward_map_uri=forward.artifact_uri,
                inverse_map_uri=inverse.artifact_uri,
                source_variables=validated.source_variables,
                target_variables=validated.target_variables,
            ).model_dump(mode="json"),
            parents=(forward.artifact_uri, inverse.artifact_uri),
            summary="two-sided polynomial-map inverse claim",
        )
        semantics = self.resources.store.get(semantics_uri)
        replay = PolynomialMapInverseReplayPayload(
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            residuals_uri=residual_artifact.artifact_uri,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
        )
        replay_payload = replay.model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.map.inverse.two_sided_replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=residual_artifact.object_digest,
                scope_digest=forward.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(replay_payload)).hexdigest()
            ),
            payload=replay_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.certificate_schema_uri,
            semantics_uri=semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=tuple(
                dict.fromkeys(
                    (
                        claim.artifact_uri,
                        residual_artifact.artifact_uri,
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        *left_records,
                        *right_records,
                    )
                )
            ),
            summary="two-sided exact polynomial-map inverse replay certificate",
        )
        supporting = (inverse.artifact_uri, *left_records, *right_records)
        aggregate_checked = self.resources.verification.verify_certificate(
            certificate_uri=certificate_artifact.artifact_uri,
            checker_id=checker_id,
            supporting_artifact_uris=supporting,
        )
        verified = aggregate_checked.verification_record_uri is not None
        conclusion = aggregate_checked.conclusion
        output = PolynomialMapInverseVerifyOutput(
            inverse_verified={
                Conclusion.TRUE: True,
                Conclusion.FALSE: False,
                Conclusion.UNKNOWN: None,
            }[conclusion],
            conclusion=conclusion,
            forward_map_uri=forward.artifact_uri,
            inverse_map_uri=inverse.artifact_uri,
            residuals_uri=residual_artifact.artifact_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            inverse_after_forward_checker_records=tuple(left_records),
            forward_after_inverse_checker_records=tuple(right_records),
            verification_record_uri=aggregate_checked.verification_record_uri,
            checker_id=checker_id,
            source_variables=validated.source_variables,
            target_variables=validated.target_variables,
        )
        return CapabilityResult(
            capability_id=self.descriptor.capability_id,
            capability_version=self.descriptor.version,
            mode=request.mode,
            execution=aggregate_checked.execution,
            output=output.model_dump(mode="json"),
            scope=CapabilityScope(
                description="both ordered polynomial compositions over QQ",
                parameters={
                    "source_variables": list(validated.source_variables),
                    "target_variables": list(validated.target_variables),
                },
                artifact_uri=forward.artifact_uri,
            ),
            completeness=CapabilityCompleteness(
                status=(
                    CapabilityCompletenessStatus.COMPLETE
                    if verified
                    else CapabilityCompletenessStatus.UNKNOWN
                ),
                basis=(
                    "both complete composition residual families were independently replayed"
                    if verified
                    else "the aggregate independent checker did not accept the replay"
                ),
                assurance_level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.COMPUTED
                ),
                verification_record_uri=aggregate_checked.verification_record_uri,
            ),
            relationships=(),
            assurance=CapabilityAssurance(
                level=(
                    CapabilityAssuranceLevel.VERIFIED
                    if verified
                    else CapabilityAssuranceLevel.HEURISTIC
                ),
                basis=(
                    "accepted by the authorized independent two-sided map checker"
                    if verified
                    else "the independent checker did not accept the inverse request"
                ),
                verification_record_uri=aggregate_checked.verification_record_uri,
            ),
            artifact_uris=tuple(
                dict.fromkeys(
                    (
                        forward.artifact_uri,
                        inverse.artifact_uri,
                        residual_artifact.artifact_uri,
                        claim.artifact_uri,
                        certificate_artifact.artifact_uri,
                        *identity_artifacts,
                        *(
                            (aggregate_checked.verification_record_uri,)
                            if aggregate_checked.verification_record_uri is not None
                            else ()
                        ),
                    )
                )
            ),
        )


def _map_inverse_residuals(
    request: PolynomialMapInverseVerifyRequest,
) -> tuple[
    tuple[SparseRationalPolynomial, ...],
    tuple[SparseRationalPolynomial, ...],
]:
    source_generators, forward = _sympy_map(request.forward_map)
    target_generators, inverse = _sympy_map(request.inverse_map)

    def compose_residuals(
        outer: tuple[Poly, ...],
        outer_generators: tuple[Any, ...],
        inner: tuple[Poly, ...],
        result_generators: tuple[Any, ...],
    ) -> tuple[SparseRationalPolynomial, ...]:
        substitutions = {
            generator: polynomial.as_expr()
            for generator, polynomial in zip(outer_generators, inner, strict=True)
        }
        return tuple(
            _wire_polynomial(
                Poly(
                    expand(
                        polynomial.as_expr().subs(
                            substitutions,
                            simultaneous=True,
                        )
                    )
                    - result_generators[index],
                    *result_generators,
                    domain=QQ,
                )
            )
            for index, polynomial in enumerate(outer)
        )

    return (
        compose_residuals(inverse, target_generators, forward, source_generators),
        compose_residuals(forward, source_generators, inverse, target_generators),
    )


def _validate_request[RequestModel: ContractModel](
    model: type[RequestModel],
    payload: object,
    *,
    code: str,
    operation: str,
) -> RequestModel:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _polynomial_error(
            code,
            "request_validation",
            f"The complete polynomial {operation} request is invalid.",
        ) from exc


def _sympy_map(
    polynomial_map: RationalPolynomialMap,
) -> tuple[tuple[Any, ...], tuple[Poly, ...]]:
    generators = cast(
        tuple[Any, ...],
        symbols(" ".join(polynomial_map.variables), seq=True),
    )
    coordinates = tuple(
        _sympy_polynomial(polynomial, generators)
        for polynomial in polynomial_map.coordinates
    )
    return generators, coordinates


def _sympy_polynomial(
    polynomial: SparseRationalPolynomial,
    generators: tuple[Any, ...],
) -> Poly:
    terms = {
        term.exponents: QQ(
            int(term.coefficient.num),
            int(term.coefficient.den),
        )
        for term in polynomial.terms
    }
    return Poly.from_dict(terms, generators, domain=QQ)


def _wire_polynomial(polynomial: Poly) -> SparseRationalPolynomial:
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=_wire_rational(coefficient),
                exponents=exponents,
            )
            for exponents, coefficient in polynomial.terms()
            if coefficient != 0
        )
    )


def _wire_rational(value: object) -> CanonicalRational:
    rational = sympy.Rational(value)
    return CanonicalRational(num=str(rational.p), den=str(rational.q))


def _evaluate(
    polynomial_map: RationalPolynomialMap,
    point: RationalPolynomialPoint,
) -> tuple[CanonicalRational, ...]:
    try:
        generators, coordinates = _sympy_map(polynomial_map)
        substitutions = {
            generator: QQ(
                int(value.num),
                int(value.den),
            )
            for generator, value in zip(
                generators,
                point.values,
                strict=True,
            )
        }
        return tuple(_wire_rational(poly.eval(substitutions)) for poly in coordinates)
    except (PolynomialError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise _polynomial_error(
            "POLYNOMIAL_EVALUATION_FAILED",
            "evaluation",
            "The exact polynomial-map evaluation failed.",
        ) from exc


def _materialize_evaluation(
    resources: PolynomialResources,
    *,
    map_uri: str,
    point: RationalPolynomialPoint,
    image: tuple[CanonicalRational, ...],
) -> tuple[PolynomialMapEvaluation, str]:
    evaluation = PolynomialMapEvaluation(
        map_uri=map_uri,
        point=point,
        image=image,
        backend_version=sympy.__version__,
    )
    artifact = resources.artifacts.put(
        schema_uri=resources.installation.evaluation_schema_uri,
        semantics_uri=resources.installation.semantics_uri,
        payload=evaluation.model_dump(mode="json"),
        parents=(map_uri,),
        summary="exact rational polynomial-map point evaluation",
    )
    return evaluation, artifact.artifact_uri


def _computed_result(
    *,
    descriptor: CapabilityDescriptor,
    request: CapabilityRequest,
    started: float,
    output: dict[str, Any],
    scope: CapabilityScope,
    relationships: tuple[CapabilityRelationship, ...],
    artifact_uris: tuple[str, ...],
    completeness_basis: str,
    completeness_status: CapabilityCompletenessStatus = (
        CapabilityCompletenessStatus.COMPLETE
    ),
    assurance_basis: str = (
        "deterministic exact SymPy arithmetic over QQ; the computation did not "
        "authorize or invoke an independent checker"
    ),
) -> CapabilityResult:
    return CapabilityResult(
        capability_id=descriptor.capability_id,
        capability_version=descriptor.version,
        mode=request.mode,
        execution=Execution(
            status=ExecutionStatus.COMPLETED,
            runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
        ),
        output=output,
        scope=scope,
        completeness=CapabilityCompleteness(
            status=completeness_status,
            basis=(
                f"{completeness_basis}; no mathematical conclusion or independent "
                "verification is claimed"
            ),
            assurance_level=CapabilityAssuranceLevel.COMPUTED,
        ),
        relationships=relationships,
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis=assurance_basis,
        ),
        artifact_uris=artifact_uris,
    )


def _polynomial_error(
    code: str,
    stage: str,
    message: str,
) -> CapabilityInvocationError:
    return CapabilityInvocationError(
        CapabilityDiagnostic(
            code=code,
            stage=stage,
            message=message,
            hint=(
                "Use the advertised sparse QQ schema with reduced rationals, "
                "matching dimensions, unique exponent vectors in descending "
                "monomial order, and no zero-coefficient terms. Combine duplicate "
                "exponent vectors before invoking."
            ),
        )
    )
