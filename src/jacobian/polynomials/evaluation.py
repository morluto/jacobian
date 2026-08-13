"""Adapter implementations for sparse rational polynomial-map capabilities."""

from __future__ import annotations

import hashlib
import time
from typing import cast

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityInvocationExample,
    CapabilityRequest,
)
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
)
from jacobian.contracts.polynomials import (
    PolynomialEvaluationOutput,
    PolynomialEvaluationRequest,
    PolynomialJacobian,
    PolynomialJacobianClaim,
    PolynomialJacobianOutput,
    PolynomialJacobianReplayPayload,
    PolynomialJacobianRequest,
    PolynomialKellerConditionClaim,
    PolynomialKellerConditionReplayPayload,
    PolynomialKellerConditionVerifyOutput,
    PolynomialKellerConditionVerifyRequest,
    RationalPolynomialPoint,
)
from jacobian.contracts.results import Conclusion
from jacobian.domains._examples import example
from jacobian.operation_projection import OperationProjection
from jacobian.operations import Completed
from jacobian.polynomials._support import (
    PolynomialOperationResult,
    _completed_projection,
    _computed_result,
    _evaluate,
    _materialize_evaluation,
    _materialize_map,
    _polynomial_error,
    _sympy_map,
    _validate_request,
    _wire_polynomial,
)
from jacobian.polynomials._sympy import _sympy
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.schema_registry import model_schema


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
            input_schema=model_schema(PolynomialEvaluationRequest),
            output_schema=model_schema(PolynomialEvaluationOutput),
            tags=("polynomial", "map", "evaluation", "exact-computation"),
            invocation_examples=(
                example(
                    "identity_at_zero",
                    "Evaluate x at zero.",
                    {
                        "map": {
                            "variables": ["x"],
                            "coordinates": [
                                {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                }
                            ],
                        },
                        "point": [{"num": "0", "den": "1"}],
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> PolynomialEvaluationRequest:
        return _validate_request(
            PolynomialEvaluationRequest,
            request.input,
            code="INVALID_POLYNOMIAL_EVALUATION_REQUEST",
            operation="evaluation",
        )

    def invoke(self, validated: PolynomialEvaluationRequest) -> OperationProjection:
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
            backend_version=SYMPY_VERSION,
        )
        return _computed_result(
            descriptor=self.descriptor,
            started=started,
            output=output,
            artifact_uris=(map_uri, evaluation_uri),
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
                "square polynomial map over QQ. Each map coordinate is limited "
                "to 1,024 terms and exponent 32 per variable; the determinant "
                "expansion estimate is limited to 1,024 products."
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
            input_schema=model_schema(PolynomialJacobianRequest),
            output_schema=model_schema(PolynomialJacobianOutput),
            tags=("polynomial", "jacobian", "determinant", "exact-computation"),
            invocation_examples=(
                example(
                    "identity_jacobian",
                    "Compute the Jacobian of x.",
                    {
                        "map": {
                            "variables": ["x"],
                            "coordinates": [
                                {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [1],
                                        }
                                    ]
                                }
                            ],
                        }
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(self, request: CapabilityRequest) -> PolynomialJacobianRequest:
        return _validate_request(
            PolynomialJacobianRequest,
            request.input,
            code="INVALID_POLYNOMIAL_JACOBIAN_REQUEST",
            operation="Jacobian computation",
        )

    def invoke(self, validated: PolynomialJacobianRequest) -> OperationProjection:
        return self.compute(validated)

    def compute(
        self,
        validated: PolynomialJacobianRequest,
    ) -> OperationProjection:
        """Compute from one validated request without re-entering the adapter."""

        started = time.monotonic()
        polynomial_map = validated.map
        polynomial_map, map_uri = _materialize_map(self.resources, polynomial_map)
        sp = _sympy.get()
        try:
            generators, coordinates = _sympy_map(polynomial_map)
            matrix_polys = tuple(
                tuple(coordinate.diff(generator) for generator in generators)
                for coordinate in coordinates
            )
            determinant = sp.Poly(
                sp.expand(
                    sp.Matrix(
                        [[entry.as_expr() for entry in row] for row in matrix_polys]
                    ).det()
                ),
                *generators,
                domain=sp.QQ,
            )
        except (
            cast(type[BaseException], sp.PolynomialError),
            TypeError,
            ValueError,
        ) as exc:
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
            backend_version=SYMPY_VERSION,
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
            matrix=jacobian.matrix,
            determinant=jacobian.determinant,
            backend_version=SYMPY_VERSION,
        )
        return _completed_projection(
            descriptor=self.descriptor,
            output=output,
            runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            artifact_uris=(
                map_uri,
                jacobian_artifact.artifact_uri,
                claim_artifact.artifact_uri,
                certificate_artifact.artifact_uri,
            ),
        )


class PolynomialKellerConditionVerifyAdapter:
    """Verify the exact nonzero-constant Jacobian condition over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.keller_checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.keller_condition.verify",
            version="1",
            title="Verify a polynomial-map Keller condition",
            description=(
                "Independently replay the exact Jacobian of a sparse square map "
                "over QQ and decide whether its determinant is a nonzero constant."
            ),
            provider="jacobian.polynomial-keller-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.polynomial-keller-checker",
                features=("exact-rational-keller-condition",),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(PolynomialKellerConditionVerifyRequest),
            output_schema=model_schema(PolynomialKellerConditionVerifyOutput),
            tags=("polynomial", "map", "jacobian", "Keller", "verification"),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="identity_keller_condition",
                    description=(
                        "Verify the identity map's constant nonzero Jacobian "
                        "determinant over QQ."
                    ),
                    input=PolynomialKellerConditionVerifyRequest.model_validate(
                        {
                            "map": {
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
                            }
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def prepare(
        self, request: CapabilityRequest
    ) -> PolynomialKellerConditionVerifyRequest:
        return _validate_request(
            PolynomialKellerConditionVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_KELLER_CONDITION_REQUEST",
            operation="Keller-condition verification",
        )

    def invoke(
        self, validated: PolynomialKellerConditionVerifyRequest
    ) -> OperationProjection:
        checker_id = self.resources.installation.keller_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_KELLER_CHECKER_UNAVAILABLE",
                "keller_condition_verification",
                "No authorized polynomial Keller-condition checker is installed.",
            )
        jacobian_projection = PolynomialJacobianAdapter(self.resources).compute(
            PolynomialJacobianRequest(map=validated.map)
        )
        assert isinstance(jacobian_projection.terminal, Completed)
        assert jacobian_projection.publication is not None
        jacobian = cast(
            PolynomialJacobianOutput,
            jacobian_projection.terminal.value,
        )
        jacobian_artifact_uris = jacobian_projection.publication.artifact_uris
        map_uri = jacobian.map_uri
        jacobian_uri = jacobian.jacobian_uri
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.installation.keller_claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=PolynomialKellerConditionClaim(
                map_uri=map_uri,
                jacobian_uri=jacobian_uri,
            ).model_dump(mode="json"),
            parents=(map_uri, jacobian_uri),
            summary="polynomial-map Keller-condition claim",
        )
        semantics = self.resources.store.get(self.resources.installation.semantics_uri)
        map_artifact = self.resources.store.get(map_uri)
        jacobian_artifact = self.resources.store.get(jacobian_uri)
        replay_payload = PolynomialKellerConditionReplayPayload(
            map_uri=map_uri,
            jacobian_uri=jacobian_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="polynomial.map.keller_condition.replay",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=jacobian_artifact.manifest.object_digest,
                scope_digest=map_artifact.manifest.object_digest,
            ),
            payload_digest=(
                "sha256:"
                + hashlib.sha256(canonicalize_json(replay_payload)).hexdigest()
            ),
            payload=replay_payload,
        )
        certificate_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.certificate_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(claim.artifact_uri, jacobian_uri, map_uri),
            summary="exact polynomial-map Keller-condition replay certificate",
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
        condition = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }[conclusion]
        output = PolynomialKellerConditionVerifyOutput(
            keller_condition_verified=condition if verified else None,
            conclusion=conclusion if verified else Conclusion.UNKNOWN,
            map_uri=map_uri,
            jacobian_uri=jacobian_uri,
            claim_uri=claim.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            determinant=jacobian.determinant,
            verification_record_uri=record_uri,
            checker_id=checker_id,
        )
        artifact_uris = list(
            dict.fromkeys(
                (
                    *jacobian_artifact_uris,
                    claim.artifact_uri,
                    certificate_artifact.artifact_uri,
                )
            )
        )
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return PolynomialOperationResult(
            execution=checked.execution,
            value=output,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=tuple(artifact_uris),
        ).project(self.descriptor)
