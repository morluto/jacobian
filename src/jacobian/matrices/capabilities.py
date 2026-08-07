"""Exact rational determinant and rank capability adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import format_canonical_integer
from jacobian.capability_service import CapabilityInvocationError
from jacobian.contracts.capabilities import (
    CapabilityAssurance,
    CapabilityAssuranceLevel,
    CapabilityCompleteness,
    CapabilityCompletenessStatus,
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityMode,
    CapabilityRelationship,
    CapabilityRequest,
    CapabilityResult,
    CapabilityScope,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.matrices import (
    ExactRationalMatrix,
    MatrixDeterminantArtifact,
    MatrixDeterminantOutput,
    MatrixDeterminantRequest,
    MatrixRankArtifact,
    MatrixRankOutput,
    MatrixRankRequest,
)
from jacobian.contracts.results import Execution, ExecutionStatus
from jacobian.domains._examples import example
from jacobian.provider_runtime import SYMPY_VERSION, known_provider_runtime
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository

if TYPE_CHECKING:
    from sympy import Matrix, Rational


@dataclass(frozen=True, slots=True)
class MatrixInstallation:
    semantics_uri: str
    matrix_schema_uri: str
    determinant_schema_uri: str
    rank_schema_uri: str


@dataclass(frozen=True, slots=True)
class MatrixResources:
    artifacts: ArtifactService
    installation: MatrixInstallation


def install_matrix_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
) -> tuple[
    tuple[MatrixDeterminantAdapter, MatrixRankAdapter],
    MatrixInstallation,
]:
    """Register exact QQ matrix contracts and computation adapters."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.exact-rational-matrix",
        version="1",
        definition={
            "description": (
                "rectangular matrices over QQ with canonical reduced rational entries"
            ),
            "domain": "QQ",
            "maximum_rows": 32,
            "maximum_columns": 32,
        },
    )
    installation = MatrixInstallation(
        semantics_uri=semantics_uri,
        matrix_schema_uri=schemas.register(
            name="jacobian.exact-rational-matrix",
            version="1",
            schema=model_schema(ExactRationalMatrix),
        ),
        determinant_schema_uri=schemas.register(
            name="jacobian.matrix-determinant",
            version="1",
            schema=model_schema(MatrixDeterminantArtifact),
        ),
        rank_schema_uri=schemas.register(
            name="jacobian.matrix-rank",
            version="1",
            schema=model_schema(MatrixRankArtifact),
        ),
    )
    resources = MatrixResources(artifacts=artifacts, installation=installation)
    return (
        (MatrixDeterminantAdapter(resources), MatrixRankAdapter(resources)),
        installation,
    )


class MatrixDeterminantAdapter:
    """Compute one exact determinant without claiming verification."""

    def __init__(self, resources: MatrixResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="matrix.determinant.compute",
            version="1",
            title="Compute an exact rational matrix determinant",
            description=(
                "Compute the determinant of one square matrix over QQ using "
                "SymPy's exact Bareiss algorithm."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("matrix", "determinant", "exact-rational"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(MatrixDeterminantRequest),
            output_schema=model_schema(MatrixDeterminantOutput),
            tags=("matrix", "determinant", "exact-computation"),
            invocation_examples=(
                example(
                    "determinant_minus_six",
                    "Compute the determinant of [[0,2],[3,4]].",
                    {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [
                                [{"num": "0", "den": "1"}, {"num": "2", "den": "1"}],
                                [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                            ],
                        }
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate(MatrixDeterminantRequest, request.input)
        started = time.monotonic()
        matrix_uri = _materialize_matrix(self.resources, validated.matrix)
        from sympy import Rational

        determinant = Rational(_sympy_matrix(validated.matrix).det(method="bareiss"))
        determinant_value = _wire(determinant)
        artifact = MatrixDeterminantArtifact(
            matrix_uri=matrix_uri,
            determinant=determinant_value,
            backend_version=SYMPY_VERSION,
        )
        result_uri = self.resources.artifacts.put(
            schema_uri=self.resources.installation.determinant_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=artifact.model_dump(mode="json"),
            parents=(matrix_uri,),
            summary="exact rational matrix determinant",
        ).artifact_uri
        output = MatrixDeterminantOutput(
            matrix_uri=matrix_uri,
            determinant_uri=result_uri,
            determinant=determinant_value,
            backend_version=SYMPY_VERSION,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope_description="one square exact rational matrix",
            matrix_uri=matrix_uri,
            result_uri=result_uri,
            relation_id="matrix.relation.determinant-of",
            basis="fraction-free Bareiss elimination completed for the full matrix",
        )


class MatrixRankAdapter:
    """Compute one exact rank and expose its pivot columns."""

    def __init__(self, resources: MatrixResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="matrix.rank.compute",
            version="1",
            title="Compute exact rational matrix rank",
            description=(
                "Compute the rank and pivot columns of one rectangular matrix over QQ."
            ),
            provider="jacobian.sympy",
            provider_runtime=known_provider_runtime(
                "jacobian.sympy",
                features=("matrix", "rank", "exact-rational"),
            ),
            modes=(CapabilityMode.EXPLORE,),
            input_schema=model_schema(MatrixRankRequest),
            output_schema=model_schema(MatrixRankOutput),
            tags=("matrix", "rank", "exact-computation"),
            invocation_examples=(
                example(
                    "rank_three_by_four",
                    "Compute rank and pivots of a rectangular rational matrix.",
                    {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [
                                [
                                    {"num": "1", "den": "1"},
                                    {"num": "2", "den": "1"},
                                    {"num": "3", "den": "1"},
                                    {"num": "4", "den": "1"},
                                ],
                                [
                                    {"num": "2", "den": "1"},
                                    {"num": "4", "den": "1"},
                                    {"num": "6", "den": "1"},
                                    {"num": "8", "den": "1"},
                                ],
                                [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            ],
                        }
                    },
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        validated = _validate(MatrixRankRequest, request.input)
        started = time.monotonic()
        matrix_uri = _materialize_matrix(self.resources, validated.matrix)
        _, pivot_columns = _sympy_matrix(validated.matrix).rref()
        artifact = MatrixRankArtifact(
            matrix_uri=matrix_uri,
            rank=len(pivot_columns),
            pivot_columns=pivot_columns,
            backend_version=SYMPY_VERSION,
        )
        result_uri = self.resources.artifacts.put(
            schema_uri=self.resources.installation.rank_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=artifact.model_dump(mode="json"),
            parents=(matrix_uri,),
            summary="exact rational matrix rank",
        ).artifact_uri
        output = MatrixRankOutput(
            matrix_uri=matrix_uri,
            rank_uri=result_uri,
            rank=len(pivot_columns),
            pivot_columns=pivot_columns,
            backend_version=SYMPY_VERSION,
        )
        return _computed_result(
            descriptor=self.descriptor,
            request=request,
            started=started,
            output=output.model_dump(mode="json"),
            scope_description="one rectangular exact rational matrix",
            matrix_uri=matrix_uri,
            result_uri=result_uri,
            relation_id="matrix.relation.rank-of",
            basis="exact rational row reduction completed for the full matrix",
        )


def _validate(model: Any, payload: dict[str, Any]) -> Any:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise CapabilityInvocationError(
            CapabilityDiagnostic(
                code="INVALID_EXACT_MATRIX_REQUEST",
                stage="matrix_input_validation",
                message="The matrix does not satisfy the advertised exact QQ contract.",
                hint=(
                    "Use a nonempty rectangular matrix of canonical reduced "
                    "rationals; determinant inputs must be square."
                ),
            )
        ) from exc


def _materialize_matrix(
    resources: MatrixResources,
    matrix: ExactRationalMatrix,
) -> str:
    return resources.artifacts.put(
        schema_uri=resources.installation.matrix_schema_uri,
        semantics_uri=resources.installation.semantics_uri,
        payload=matrix.model_dump(mode="json"),
        summary="exact rational matrix",
    ).artifact_uri


def _sympy_matrix(matrix: ExactRationalMatrix) -> Matrix:
    from sympy import Matrix, Rational

    return Matrix(
        [[Rational(entry.as_fraction()) for entry in row] for row in matrix.entries]
    )


def _wire(value: Rational) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def _computed_result(
    *,
    descriptor: CapabilityDescriptor,
    request: CapabilityRequest,
    started: float,
    output: dict[str, Any],
    scope_description: str,
    matrix_uri: str,
    result_uri: str,
    relation_id: str,
    basis: str,
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
        scope=CapabilityScope(
            description=scope_description,
            parameters={"matrix_uri": matrix_uri},
            artifact_uri=matrix_uri,
        ),
        completeness=CapabilityCompleteness(
            status=CapabilityCompletenessStatus.COMPLETE,
            basis=f"{basis}; this is not independent verification",
            assurance_level=CapabilityAssuranceLevel.COMPUTED,
        ),
        relationships=(
            CapabilityRelationship(
                relation_id=relation_id,
                source_artifact_uris=(matrix_uri,),
                target_artifact_uris=(result_uri,),
            ),
        ),
        assurance=CapabilityAssurance(
            level=CapabilityAssuranceLevel.COMPUTED,
            basis=(
                "deterministic exact rational arithmetic; no independent checker "
                "was invoked"
            ),
        ),
        artifact_uris=(matrix_uri, result_uri),
    )
