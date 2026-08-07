"""Exact finite-polytope membership and separator generation."""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import time
from fractions import Fraction
from functools import reduce
from typing import Any

from pydantic import ValidationError

from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.contracts.evidence import (
    CertificateEnvelope,
    EvidenceBindings,
    WitnessEnvelope,
    WitnessRole,
)
from jacobian.contracts.polytope import (
    FiniteGeneratorSet,
    PolytopeClaim,
    PolytopePredicate,
    PolytopeSeparateRequest,
    PolytopeSeparateResult,
    PolytopeStatus,
    RationalPoint,
    RationalVector,
)
from jacobian.contracts.results import (
    Arithmetic,
    Assurance,
    Conclusion,
    Coverage,
    Execution,
    ExecutionStatus,
    InputStatus,
    InputValidation,
    Method,
    ResultEnvelope,
    Verification,
)
from jacobian.schema_registry import SchemaRegistry, SchemaRegistryError, model_schema
from jacobian.storage.errors import StorageError
from jacobian.storage.models import StoredArtifact
from jacobian.storage.repository import ArtifactRepository

_LOGGER = logging.getLogger(__name__)


def _z3() -> Any:
    """Load the optional discovery backend only for a polytope solve."""

    return importlib.import_module("z3")


def _wire_rational(value: Fraction) -> dict[str, str]:
    return {
        "num": format_canonical_integer(value.numerator),
        "den": format_canonical_integer(value.denominator),
    }


def _require_schema_version(
    schema: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    result = dict(schema)
    required = list(result.get("required", []))
    if field not in required:
        required.append(field)
    result["required"] = required
    return result


def _z3_rational(value: Fraction) -> Any:
    z3 = _z3()
    return z3.Q(value.numerator, value.denominator)


def _model_fraction(
    model: Any,
    expression: Any,
) -> Fraction:
    z3 = _z3()
    value = model.eval(expression, model_completion=True)
    if not isinstance(value, z3.RatNumRef):
        raise ValueError("exact rational model value required")
    return Fraction(value.as_fraction())


def _polytope_input_failure_detail(exc: Exception) -> str:
    _LOGGER.warning("polytope input validation failed", exc_info=exc)
    if isinstance(exc, (StorageError, SchemaRegistryError, ValidationError)):
        return (
            "The point or generator artifact is unavailable or invalid. Check the "
            "artifact URIs and the polytope reference contract, then retry."
        )
    return (
        "The point or generator input is invalid. Check that values are exact "
        "rationals with matching dimensions and compatible semantics, then retry."
    )


class PolytopeService:
    """Generate exact evidence over explicitly finite rational generator sets.

    Z3 is an untrusted discovery backend here. Generated convex-combination
    witnesses and separating inequalities become verified only after a
    separately implemented checker replays them.
    """

    def __init__(self, store: ArtifactRepository, schemas: SchemaRegistry) -> None:
        self.store = store
        self.schemas = schemas
        self.semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.finite-rational-polytope",
            version="1",
            definition={
                "description": (
                    "finite generator sets and points over exact rational coordinates"
                )
            },
        )
        self.point_schema_uri = schemas.register(
            name="jacobian.rational-point",
            version="1",
            schema=_require_schema_version(
                model_schema(RationalPoint),
                "point_schema_version",
            ),
        )
        self.generator_set_schema_uri = schemas.register(
            name="jacobian.finite-generator-set",
            version="1",
            schema=_require_schema_version(
                model_schema(FiniteGeneratorSet),
                "generator_set_schema_version",
            ),
        )
        self.claim_schema_uri = schemas.register(
            name="jacobian.polytope-claim",
            version="1",
            schema=model_schema(PolytopeClaim),
        )
        self.witness_schema_uri = schemas.register(
            name="jacobian.witness-envelope",
            version="1",
            schema=model_schema(WitnessEnvelope),
        )
        self.certificate_schema_uri = schemas.register(
            name="jacobian.certificate-envelope",
            version="1",
            schema=model_schema(CertificateEnvelope),
        )

    def separate(
        self,
        request: PolytopeSeparateRequest,
    ) -> PolytopeSeparateResult:
        """Find exact convex weights or an exact rational separator."""

        started = time.monotonic()
        try:
            point_artifact = self.store.get(request.point_uri)
            generators_artifact = self.store.get(request.generator_set_uri)
            point = self._load_point(point_artifact)
            generators = self._load_generators(generators_artifact)
            if len(point.coordinates) != generators.dimension:
                raise ValueError("point and generator dimensions differ")
            projection = request.projection
            if projection is not None and max(projection) >= generators.dimension:
                raise ValueError("projection index exceeds the ambient dimension")
            (
                effective_point_artifact,
                effective_generators_artifact,
                effective_point,
                effective_generators,
            ) = self._project(
                point_artifact,
                generators_artifact,
                point,
                generators,
                projection,
            )
            point_values = tuple(
                value.as_fraction() for value in effective_point.coordinates
            )
            generator_values = tuple(
                tuple(value.as_fraction() for value in generator.values)
                for generator in effective_generators.generators
            )
            deadline = started + request.wall_seconds
            membership = self._convex_weights(
                point_values,
                generator_values,
                deadline=deadline,
            )
            if membership is None:
                if time.monotonic() >= deadline:
                    return self._unknown(
                        request,
                        started,
                        ExecutionStatus.TIMEOUT,
                        "convex-hull membership solve timed out",
                    )
                separator = self._separator(
                    point_values,
                    generator_values,
                    deadline=deadline,
                )
                if separator is None:
                    status = (
                        ExecutionStatus.TIMEOUT
                        if time.monotonic() >= deadline
                        else ExecutionStatus.ERROR
                    )
                    return self._unknown(
                        request,
                        started,
                        status,
                        "exact separator backend did not produce a model",
                    )
                return self._separated_result(
                    request=request,
                    started=started,
                    point=effective_point_artifact,
                    generators=effective_generators_artifact,
                    point_values=point_values,
                    generator_values=generator_values,
                    separator=separator,
                )
            return self._member_result(
                request=request,
                started=started,
                point=effective_point_artifact,
                generators=effective_generators_artifact,
                point_values=point_values,
                generator_values=generator_values,
                weights=membership,
            )
        except (
            SchemaRegistryError,
            StorageError,
            ValidationError,
            ValueError,
            ZeroDivisionError,
        ) as exc:
            return self._rejected(
                request,
                started,
                _polytope_input_failure_detail(exc),
            )
        except ImportError:
            _LOGGER.exception("exact polytope backend is unavailable")
            return self._unknown(
                request,
                started,
                ExecutionStatus.ERROR,
                "The optional exact polytope backend is unavailable.",
            )
        except _z3().Z3Exception:
            _LOGGER.exception("exact polytope backend failed")
            return self._unknown(
                request,
                started,
                ExecutionStatus.ERROR,
                "The exact polytope check failed. Retry with a smaller input; "
                "if it fails again, inspect the local Jacobian log.",
            )

    def _load_point(self, artifact: StoredArtifact) -> RationalPoint:
        if artifact.manifest.schema_uri != self.point_schema_uri:
            raise ValueError("point artifact uses an unsupported schema")
        if artifact.manifest.semantics_uri != self.semantics_uri:
            raise ValueError("point artifact uses incompatible semantics")
        normalized = self.schemas.validate(self.point_schema_uri, artifact.payload)
        return RationalPoint.model_validate(normalized)

    def _load_generators(self, artifact: StoredArtifact) -> FiniteGeneratorSet:
        if artifact.manifest.schema_uri != self.generator_set_schema_uri:
            raise ValueError("generator-set artifact uses an unsupported schema")
        if artifact.manifest.semantics_uri != self.semantics_uri:
            raise ValueError("generator-set artifact uses incompatible semantics")
        normalized = self.schemas.validate(
            self.generator_set_schema_uri,
            artifact.payload,
        )
        return FiniteGeneratorSet.model_validate(normalized)

    def _project(
        self,
        point_artifact: StoredArtifact,
        generators_artifact: StoredArtifact,
        point: RationalPoint,
        generators: FiniteGeneratorSet,
        projection: tuple[int, ...] | None,
    ) -> tuple[
        StoredArtifact,
        StoredArtifact,
        RationalPoint,
        FiniteGeneratorSet,
    ]:
        if projection is None:
            return point_artifact, generators_artifact, point, generators
        projected_point = RationalPoint(
            coordinates=tuple(point.coordinates[index] for index in projection)
        )
        projected_generators = FiniteGeneratorSet(
            dimension=len(projection),
            generators=tuple(
                RationalVector(
                    values=tuple(generator.values[index] for index in projection)
                )
                for generator in generators.generators
            ),
        )
        point_result = self.store.put(
            schema_uri=self.point_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=projected_point.model_dump(mode="json"),
            parents=(point_artifact.artifact_uri,),
            summary="exact projected rational point",
        )
        generators_result = self.store.put(
            schema_uri=self.generator_set_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=projected_generators.model_dump(mode="json"),
            parents=(generators_artifact.artifact_uri,),
            summary="exact projected finite generator set",
        )
        return (
            self.store.get(point_result.artifact_uri),
            self.store.get(generators_result.artifact_uri),
            projected_point,
            projected_generators,
        )

    @staticmethod
    def _solver(deadline: float) -> Any:
        z3 = _z3()
        solver = z3.Solver()
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        solver.set(timeout=remaining_ms)
        return solver

    def _convex_weights(
        self,
        point: tuple[Fraction, ...],
        generators: tuple[tuple[Fraction, ...], ...],
        *,
        deadline: float,
    ) -> tuple[Fraction, ...] | None:
        z3 = _z3()
        solver = self._solver(deadline)
        weights = tuple(z3.Real(f"lambda_{index}") for index in range(len(generators)))
        solver.add(*(weight >= 0 for weight in weights))
        solver.add(z3.Sum(*weights) == 1)
        for coordinate, expected in enumerate(point):
            solver.add(
                z3.Sum(
                    *(
                        weight * _z3_rational(generator[coordinate])
                        for weight, generator in zip(
                            weights,
                            generators,
                            strict=True,
                        )
                    )
                )
                == _z3_rational(expected)
            )
        status = solver.check()
        if status == z3.sat:
            model = solver.model()
            return tuple(_model_fraction(model, weight) for weight in weights)
        if status == z3.unsat or status == z3.unknown:
            return None
        raise ValueError("unexpected convex-hull backend status")

    def _separator(
        self,
        point: tuple[Fraction, ...],
        generators: tuple[tuple[Fraction, ...], ...],
        *,
        deadline: float,
    ) -> tuple[tuple[int, ...], int] | None:
        z3 = _z3()
        solver = self._solver(deadline)
        coefficients = tuple(
            z3.Real(f"separator_{index}") for index in range(len(point))
        )
        rhs = z3.Real("separator_rhs")
        for generator in generators:
            solver.add(
                z3.Sum(
                    *(
                        coefficient * _z3_rational(value)
                        for coefficient, value in zip(
                            coefficients,
                            generator,
                            strict=True,
                        )
                    )
                )
                <= rhs
            )
        solver.add(
            z3.Sum(
                *(
                    coefficient * _z3_rational(value)
                    for coefficient, value in zip(
                        coefficients,
                        point,
                        strict=True,
                    )
                )
            )
            >= rhs + 1
        )
        status = solver.check()
        if status != z3.sat:
            return None
        model = solver.model()
        rational_values = tuple(
            _model_fraction(model, coefficient) for coefficient in coefficients
        )
        rational_rhs = _model_fraction(model, rhs)
        common_denominator = math.lcm(
            *(value.denominator for value in (*rational_values, rational_rhs))
        )
        integer_coefficients = tuple(
            value.numerator * (common_denominator // value.denominator)
            for value in rational_values
        )
        integer_rhs = rational_rhs.numerator * (
            common_denominator // rational_rhs.denominator
        )
        divisor = reduce(
            math.gcd,
            (abs(value) for value in (*integer_coefficients, integer_rhs)),
        )
        if divisor > 1:
            integer_coefficients = tuple(
                value // divisor for value in integer_coefficients
            )
            integer_rhs //= divisor
        return integer_coefficients, integer_rhs

    def _member_result(
        self,
        *,
        request: PolytopeSeparateRequest,
        started: float,
        point: StoredArtifact,
        generators: StoredArtifact,
        point_values: tuple[Fraction, ...],
        generator_values: tuple[tuple[Fraction, ...], ...],
        weights: tuple[Fraction, ...],
    ) -> PolytopeSeparateResult:
        claim = self._put_claim(
            PolytopePredicate.INSIDE_CONVEX_HULL,
            point,
            generators,
            len(point_values),
        )
        bindings = self._bindings(claim, point, generators)
        reconstructed = tuple(
            sum(
                (
                    weight * generator[index]
                    for weight, generator in zip(
                        weights,
                        generator_values,
                        strict=True,
                    )
                ),
                Fraction(0),
            )
            for index in range(len(point_values))
        )
        witness = WitnessEnvelope(
            witness_format="polytope.convex_combination",
            format_version="1",
            role=WitnessRole.SUPPORTS_CLAIM,
            bindings=bindings,
            payload={
                "weights": [_wire_rational(weight) for weight in weights],
                "reconstructed_point": [
                    _wire_rational(value) for value in reconstructed
                ],
            },
        )
        stored = self.store.put(
            schema_uri=self.witness_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=witness.model_dump(mode="json"),
            parents=(
                claim.artifact_uri,
                point.artifact_uri,
                generators.artifact_uri,
            ),
            summary="unverified exact convex-combination witness",
        )
        return self._decisive_result(
            request=request,
            started=started,
            point=point,
            generators=generators,
            claim=claim,
            status=PolytopeStatus.MEMBER,
            evidence_uri=stored.artifact_uri,
            witness_uri=stored.artifact_uri,
        )

    def _separated_result(
        self,
        *,
        request: PolytopeSeparateRequest,
        started: float,
        point: StoredArtifact,
        generators: StoredArtifact,
        point_values: tuple[Fraction, ...],
        generator_values: tuple[tuple[Fraction, ...], ...],
        separator: tuple[tuple[int, ...], int],
    ) -> PolytopeSeparateResult:
        coefficients, rhs = separator
        claim = self._put_claim(
            PolytopePredicate.OUTSIDE_CONVEX_HULL,
            point,
            generators,
            len(point_values),
        )
        bindings = self._bindings(claim, point, generators)
        point_value = sum(
            (
                Fraction(coefficient) * value
                for coefficient, value in zip(
                    coefficients,
                    point_values,
                    strict=True,
                )
            ),
            Fraction(0),
        )
        generator_totals = tuple(
            sum(
                (
                    Fraction(coefficient) * value
                    for coefficient, value in zip(
                        coefficients,
                        generator,
                        strict=True,
                    )
                ),
                Fraction(0),
            )
            for generator in generator_values
        )
        max_generator_value = max(generator_totals)
        payload = {
            "coefficients": [_wire_rational(Fraction(value)) for value in coefficients],
            "sense": "<=",
            "rhs": _wire_rational(Fraction(rhs)),
            "point_value": _wire_rational(point_value),
            "max_generator_value": _wire_rational(max_generator_value),
            "margin": _wire_rational(point_value - max_generator_value),
        }
        payload_digest = (
            "sha256:" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
        )
        certificate = CertificateEnvelope(
            certificate_type="polytope.linear_separator",
            format_version="1",
            bindings=bindings,
            payload_digest=payload_digest,
            payload=payload,
        )
        stored = self.store.put(
            schema_uri=self.certificate_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=certificate.model_dump(mode="json"),
            parents=(
                claim.artifact_uri,
                point.artifact_uri,
                generators.artifact_uri,
            ),
            summary="unverified exact rational separation certificate",
        )
        return self._decisive_result(
            request=request,
            started=started,
            point=point,
            generators=generators,
            claim=claim,
            status=PolytopeStatus.SEPARATED,
            evidence_uri=stored.artifact_uri,
            certificate_uri=stored.artifact_uri,
        )

    def _put_claim(
        self,
        predicate: PolytopePredicate,
        point: StoredArtifact,
        generators: StoredArtifact,
        dimension: int,
    ) -> StoredArtifact:
        result = self.store.put(
            schema_uri=self.claim_schema_uri,
            semantics_uri=self.semantics_uri,
            payload=PolytopeClaim(
                predicate=predicate,
                dimension=dimension,
                point_uri=point.artifact_uri,
                generator_set_uri=generators.artifact_uri,
            ).model_dump(mode="json"),
            parents=(point.artifact_uri, generators.artifact_uri),
            summary="finite rational convex-hull claim",
        )
        return self.store.get(result.artifact_uri)

    def _bindings(
        self,
        claim: StoredArtifact,
        point: StoredArtifact,
        generators: StoredArtifact,
    ) -> EvidenceBindings:
        semantics_digest = self.store.get(self.semantics_uri).manifest.object_digest
        return EvidenceBindings(
            claim_digest=claim.manifest.object_digest,
            semantics_digest=semantics_digest,
            candidate_digest=point.manifest.object_digest,
            scope_digest=generators.manifest.object_digest,
        )

    def _decisive_result(
        self,
        *,
        request: PolytopeSeparateRequest,
        started: float,
        point: StoredArtifact,
        generators: StoredArtifact,
        claim: StoredArtifact,
        status: PolytopeStatus,
        evidence_uri: str,
        witness_uri: str | None = None,
        certificate_uri: str | None = None,
    ) -> PolytopeSeparateResult:
        return PolytopeSeparateResult(
            status=status,
            point_uri=request.point_uri,
            generator_set_uri=request.generator_set_uri,
            effective_point_uri=point.artifact_uri,
            effective_generator_set_uri=generators.artifact_uri,
            claim_uri=claim.artifact_uri,
            witness_uri=witness_uri,
            certificate_uri=certificate_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=Conclusion.TRUE,
                assurance=Assurance(
                    arithmetic=Arithmetic.EXACT_RATIONAL,
                    method=Method.BOUNDED_SEARCH,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
                claim_digest=claim.manifest.object_digest,
                semantics_digest=self.store.get(
                    self.semantics_uri
                ).manifest.object_digest,
                candidate_digest=point.manifest.object_digest,
                evidence_uris=(evidence_uri,),
            ),
        )

    @staticmethod
    def _unknown(
        request: PolytopeSeparateRequest,
        started: float,
        status: ExecutionStatus,
        detail: str,
    ) -> PolytopeSeparateResult:
        return PolytopeSeparateResult(
            status=PolytopeStatus.UNKNOWN,
            point_uri=request.point_uri,
            generator_set_uri=request.generator_set_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=status,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                    detail=detail,
                ),
                input=InputValidation(status=InputStatus.ACCEPTED),
                conclusion=Conclusion.UNKNOWN,
                assurance=Assurance(
                    arithmetic=Arithmetic.EXACT_RATIONAL,
                    method=Method.BOUNDED_SEARCH,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
            ),
        )

    @staticmethod
    def _rejected(
        request: PolytopeSeparateRequest,
        started: float,
        detail: str,
    ) -> PolytopeSeparateResult:
        return PolytopeSeparateResult(
            status=PolytopeStatus.UNKNOWN,
            point_uri=request.point_uri,
            generator_set_uri=request.generator_set_uri,
            result=ResultEnvelope(
                execution=Execution(
                    status=ExecutionStatus.COMPLETED,
                    runtime_ms=int((time.monotonic() - started) * 1000),
                ),
                input=InputValidation(
                    status=InputStatus.REJECTED,
                    errors=(detail,),
                ),
                conclusion=Conclusion.UNKNOWN,
                assurance=Assurance(
                    arithmetic=Arithmetic.EXACT_RATIONAL,
                    method=Method.BOUNDED_SEARCH,
                    coverage=Coverage.NOT_APPLICABLE,
                    verification=Verification.UNVERIFIED,
                ),
            ),
        )
