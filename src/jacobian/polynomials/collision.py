"""Adapter implementations for sparse rational polynomial-map capabilities."""

from __future__ import annotations

import time
from itertools import product

from jacobian.bounded_process import bounded_process_cancelled
from jacobian.canonical import format_canonical_integer
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityRequest,
)
from jacobian.contracts.evidence import (
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.exact import CanonicalRational, bounded_rational_scalars
from jacobian.contracts.polynomials import (
    PolynomialCollisionOutput,
    PolynomialCollisionPayload,
    PolynomialCollisionRequest,
    PolynomialCollisionSearchOutput,
    PolynomialCollisionSearchRequest,
    PolynomialCollisionSearchStopReason,
    PolynomialCollisionVerifyOutput,
    PolynomialCollisionVerifyRequest,
    PolynomialInjectivityClaim,
    PolynomialMapInverseCollisionVerifyOutput,
    PolynomialMapInverseCollisionVerifyRequest,
    PolynomialNoTwoSidedInverseClaim,
    RationalPolynomialPoint,
)
from jacobian.contracts.results import (
    Conclusion,
    Execution,
    ExecutionStatus,
)
from jacobian.domains._examples import example
from jacobian.operation_projection import OperationProjection
from jacobian.polynomials._support import (
    PolynomialOperationResult,
    _computed_result,
    _evaluate,
    _load_evaluation,
    _load_polynomial_map,
    _materialize_evaluation,
    _materialize_map,
    _polynomial_error,
    _validate_request,
)
from jacobian.polynomials.resources import PolynomialResources
from jacobian.provider_runtime import known_provider_runtime
from jacobian.schema_registry import model_schema


def _require_found_evaluation(
    first: str | None,
    second: str | None,
) -> tuple[str, str]:
    """Return both evaluation URIs or fail closed for an impossible result."""

    if first is None:
        raise RuntimeError("first evaluation result is unexpectedly None")
    if second is None:
        raise RuntimeError("second evaluation result is unexpectedly None")
    return first, second


def _require_found_result(
    first_evaluation_result: str | None,
    second_evaluation_result: str | None,
    claim_uri: str | None,
    witness_uri: str | None,
) -> tuple[str, str, str, str]:
    """Return every collision result URI or fail closed for an impossible result."""

    first_evaluation_result, second_evaluation_result = _require_found_evaluation(
        first_evaluation_result,
        second_evaluation_result,
    )
    if claim_uri is None:
        raise RuntimeError("claim URI is unexpectedly None")
    if witness_uri is None:
        raise RuntimeError("witness URI is unexpectedly None")
    return first_evaluation_result, second_evaluation_result, claim_uri, witness_uri


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
            input_schema=model_schema(PolynomialCollisionRequest),
            output_schema=model_schema(PolynomialCollisionOutput),
            tags=("polynomial", "map", "collision", "witness", "artifact-composition"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
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
        )
        artifact_uris = [
            candidate_uri,
            claim_artifact.artifact_uri,
            first_evaluation_artifact.artifact_uri,
            second_evaluation_artifact.artifact_uri,
        ]
        if witness_uri is not None:
            artifact_uris.append(witness_uri)
        return _computed_result(
            descriptor=self.descriptor,
            started=started,
            output=output,
            artifact_uris=tuple(artifact_uris),
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
            input_schema=model_schema(PolynomialCollisionSearchRequest),
            output_schema=model_schema(PolynomialCollisionSearchOutput),
            tags=("polynomial", "map", "collision", "bounded-search"),
            invocation_examples=(
                example(
                    "constant_map_collision",
                    "Find a collision for the constant zero map on a tiny grid.",
                    {
                        "map": {"variables": ["x"], "coordinates": [{"terms": []}]},
                        "max_abs_numerator": 1,
                        "max_denominator": 1,
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(
        self,
        request: CapabilityRequest,
    ) -> OperationProjection:
        validated = _validate_request(
            PolynomialCollisionSearchRequest,
            request.input,
            code="INVALID_POLYNOMIAL_COLLISION_SEARCH_REQUEST",
            operation="collision search",
        )
        started = time.monotonic()
        polynomial_map, map_uri = _materialize_map(self.resources, validated.map)
        scalar_values = tuple(
            CanonicalRational(
                num=format_canonical_integer(value.numerator),
                den=format_canonical_integer(value.denominator),
            )
            for value in bounded_rational_scalars(
                validated.max_abs_numerator, validated.max_denominator
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
        cancelled = False
        for point_values in product(
            scalar_values,
            repeat=len(polynomial_map.variables),
        ):
            if bounded_process_cancelled():
                cancelled = True
                break
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
            first_evaluation_result, second_evaluation_result = (
                _require_found_evaluation(
                    first_evaluation_result, second_evaluation_result
                )
            )
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
            stop_reason=(
                PolynomialCollisionSearchStopReason.FIRST_COLLISION
                if found is not None
                else (
                    PolynomialCollisionSearchStopReason.CANCELLED
                    if cancelled
                    else PolynomialCollisionSearchStopReason.GRID_EXHAUSTED
                )
            ),
        )
        artifacts = [map_uri, *evaluation_uris]
        if found is not None:
            (
                first_evaluation_result,
                second_evaluation_result,
                claim_uri,
                witness_uri,
            ) = _require_found_result(
                first_evaluation_result,
                second_evaluation_result,
                claim_uri,
                witness_uri,
            )
            artifacts.extend(
                [
                    second_evaluation_result,
                    claim_uri,
                    witness_uri,
                ]
            )
        artifact_uris = tuple(
            dict.fromkeys(uri for uri in artifacts if uri is not None)
        )
        if cancelled:
            return PolynomialOperationResult(
                value=output,
                execution=Execution(
                    status=ExecutionStatus.CANCELLED,
                    runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
                    detail="The client cancelled the collision-grid search.",
                ),
                artifact_uris=artifact_uris,
            ).project(self.descriptor)
        return _computed_result(
            descriptor=self.descriptor,
            started=started,
            output=output,
            artifact_uris=artifact_uris,
        )


class PolynomialCollisionVerifyAdapter:
    """Independently verify one explicit exact rational collision."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.collision_checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.collision.verify",
            version="1",
            title="Verify a polynomial-map collision",
            description=(
                "Independently reevaluate one exact map at two supplied distinct "
                "rational points and verify their claimed common image. Each map "
                "coordinate is limited to 1,024 terms and exponent 32 per variable."
            ),
            provider="jacobian.polynomial-collision-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.polynomial-collision-checker",
                features=("exact-rational-collision-replay",),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(PolynomialCollisionVerifyRequest),
            output_schema=model_schema(PolynomialCollisionVerifyOutput),
            tags=("polynomial", "map", "collision", "verification"),
            invocation_examples=(
                example(
                    "square_map_collision",
                    "Verify that x squared maps -1 and 1 to the common image 1.",
                    {
                        "map": {
                            "variables": ["x"],
                            "coordinates": [
                                {
                                    "terms": [
                                        {
                                            "coefficient": {"num": "1", "den": "1"},
                                            "exponents": [2],
                                        }
                                    ]
                                }
                            ],
                        },
                        "first_point": [{"num": "-1", "den": "1"}],
                        "second_point": [{"num": "1", "den": "1"}],
                        "claimed_image": [{"num": "1", "den": "1"}],
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
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
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        checked = self.resources.verification.verify_witness(
            claim_uri=claim_artifact.artifact_uri,
            candidate_uri=map_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.conclusion is Conclusion.FALSE
            and checked.verification_record_uri is not None
        )
        record_uri = checked.verification_record_uri if verified else None
        output = PolynomialCollisionVerifyOutput(
            collision_verified=verified,
            conclusion="FALSE" if verified else "UNKNOWN",
            verification_input=checked.input,
            map_uri=map_uri,
            claim_uri=claim_artifact.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            verification_record_uri=record_uri,
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
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return PolynomialOperationResult(
            value=output,
            execution=checked.execution,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=tuple(artifact_uris),
        ).project(self.descriptor)


class PolynomialMapInverseCollisionVerifyAdapter:
    """Verify that an exact collision rules out a two-sided inverse over QQ."""

    def __init__(self, resources: PolynomialResources) -> None:
        self.resources = resources
        checker_id = resources.installation.inverse_collision_checker_id
        if checker_id is None:
            raise RuntimeError("checker is not installed")
        self._descriptor = CapabilityDescriptor(
            capability_id="polynomial.map.inverse.refute_by_collision",
            version="1",
            title="Refute a polynomial-map inverse by collision",
            description=(
                "Independently replay two distinct rational preimages with the "
                "same image and bind that collision to the absence of a two-sided "
                "polynomial inverse over QQ."
            ),
            provider="jacobian.polynomial-inverse-obstruction-checker",
            provider_runtime=known_provider_runtime(
                "jacobian.polynomial-inverse-obstruction-checker",
                features=("exact-rational-collision", "inverse-obstruction"),
                checker_ids=(checker_id,),
            ),
            input_schema=model_schema(PolynomialMapInverseCollisionVerifyRequest),
            output_schema=model_schema(PolynomialMapInverseCollisionVerifyOutput),
            tags=("polynomial", "map", "inverse", "collision", "verification"),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        validated = _validate_request(
            PolynomialMapInverseCollisionVerifyRequest,
            request.input,
            code="INVALID_POLYNOMIAL_MAP_INVERSE_COLLISION_REQUEST",
            operation="polynomial-map inverse obstruction",
        )
        checker_id = self.resources.installation.inverse_collision_checker_id
        if checker_id is None:
            raise _polynomial_error(
                "POLYNOMIAL_INVERSE_COLLISION_CHECKER_UNAVAILABLE",
                "inverse_obstruction_verification",
                "No authorized collision inverse-obstruction checker is installed.",
            )
        _, map_uri = _materialize_map(self.resources, validated.map)
        map_artifact = self.resources.store.get(map_uri)
        candidate = self.resources.store.get(map_uri)
        claim = self.resources.artifacts.put(
            schema_uri=self.resources.installation.inverse_collision_claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=PolynomialNoTwoSidedInverseClaim(
                map_uri=map_uri,
            ).model_dump(mode="json"),
            parents=(map_uri,),
            summary="polynomial-map no-two-sided-inverse claim",
        )
        semantics = self.resources.store.get(self.resources.installation.semantics_uri)
        witness = WitnessEnvelope(
            witness_format="polynomial.map_collision_refutes_inverse",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=EvidenceBindings(
                claim_digest=claim.object_digest,
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
            parents=(claim.artifact_uri, map_uri),
            summary="exact collision obstructing a two-sided polynomial inverse",
        )
        checked = self.resources.verification.verify_witness(
            claim_uri=claim.artifact_uri,
            candidate_uri=map_uri,
            witness_uri=witness_artifact.artifact_uri,
            checker_id=checker_id,
        )
        verified = (
            checked.verification_record_uri is not None
            and checked.conclusion is Conclusion.TRUE
        )
        record_uri = checked.verification_record_uri if verified else None
        output = PolynomialMapInverseCollisionVerifyOutput(
            noninvertibility_verified=verified if verified else None,
            conclusion=Conclusion.TRUE if verified else Conclusion.UNKNOWN,
            verification_input=checked.input,
            map_uri=map_uri,
            claim_uri=claim.artifact_uri,
            witness_uri=witness_artifact.artifact_uri,
            verification_record_uri=record_uri,
            checker_id=checker_id,
            first_point=validated.first_point,
            second_point=validated.second_point,
            claimed_image=validated.claimed_image,
        )
        artifact_uris = [map_artifact.artifact_uri, claim.artifact_uri]
        artifact_uris.append(witness_artifact.artifact_uri)
        if record_uri is not None:
            artifact_uris.append(record_uri)
        return PolynomialOperationResult(
            value=output,
            execution=checked.execution,
            verification_record_uri=(record_uri if verified else None),
            artifact_uris=tuple(artifact_uris),
        ).project(self.descriptor)
