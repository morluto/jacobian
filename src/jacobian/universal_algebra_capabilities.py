"""Exact finite universal-algebra capabilities."""

from __future__ import annotations

import hashlib
import importlib
import time
from dataclasses import dataclass
from itertools import product
from typing import Any

from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.canonical import canonicalize_json
from jacobian.capability_adapters import CapabilityAdapter
from jacobian.capability_errors import CapabilityInvocationError
from jacobian.checker_installation import CheckerInstaller
from jacobian.checker_operations import CheckerOperation
from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityDiagnostic,
    CapabilityInvocationExample,
    CapabilityProviderAvailability,
    CapabilityProviderRuntime,
    CapabilityRequest,
)
from jacobian.contracts.checkers import EvidenceKind
from jacobian.contracts.evidence import CertificateEnvelope, EvidenceBindings
from jacobian.contracts.universal_algebra import (
    CountermodelSearchStatus,
    FiniteMagma,
    FiniteMagmaCountermodelArtifact,
    FiniteMagmaLawEvaluationArtifact,
    FiniteMagmaLawEvaluationClaim,
    FiniteMagmaLawProblem,
    FiniteMagmaLawReplayPayload,
    FiniteMagmaTableEnumerationArtifact,
    FiniteMagmaTableEnumerationOutput,
    FiniteMagmaTableEnumerationRequest,
    MagmaAssignmentValue,
    MagmaLaw,
    MagmaLawCounterexample,
    MagmaLawCoverage,
    MagmaLawEvaluationRecord,
    MagmaTerm,
    UniversalAlgebraCountermodelSearchOutput,
    UniversalAlgebraCountermodelSearchRequest,
    UniversalAlgebraEvaluationOutput,
    UniversalAlgebraEvaluationRequest,
)
from jacobian.domains._examples import example
from jacobian.operation_projection import OperationProjection
from jacobian.operation_publication import PublishedOperation
from jacobian.operations import Completed
from jacobian.provider_runtime import known_provider_runtime
from jacobian.registry import CheckerRegistry
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.storage.repository import ArtifactRepository
from jacobian.verification.service import VerificationService
from jacobian.verification_capabilities import certificate_verification_adapter

_COUNTERMODEL_TIMEOUT_MS = 10_000


@dataclass(frozen=True, slots=True)
class UniversalAlgebraInstallation:
    semantics_uri: str
    magma_schema_uri: str
    problem_schema_uri: str
    evaluation_schema_uri: str
    countermodel_schema_uri: str
    table_enumeration_schema_uri: str
    claim_schema_uri: str
    certificate_schema_uri: str
    evaluation_checker_id: str | None


@dataclass(frozen=True, slots=True)
class UniversalAlgebraResources:
    store: ArtifactRepository
    artifacts: ArtifactService
    installation: UniversalAlgebraInstallation


def install_universal_algebra_capabilities(
    store: ArtifactRepository,
    schemas: SchemaRegistry,
    artifacts: ArtifactService,
    verification: VerificationService,
    checkers: CheckerRegistry,
    *,
    authorize_checker: bool,
) -> tuple[
    tuple[CapabilityAdapter, ...],
    UniversalAlgebraInstallation,
]:
    """Install exact bounded finite-magma law evaluation."""

    semantics_uri = store.register_descriptor(
        kind="semantics",
        name="jacobian.finite-magma-laws",
        version="1",
        definition={
            "description": (
                "one total binary operation on the carrier 0 through n-1 and "
                "equational laws represented by ordered binary term trees"
            ),
            "maximum_order": 8,
            "maximum_variables_per_law": 4,
            "maximum_term_nodes": 31,
            "maximum_total_valuations": 1_000_000,
            "valuation_order": "lexicographic in sorted variable order",
            "countermodel_timeout_ms": _COUNTERMODEL_TIMEOUT_MS,
        },
    )
    magma_schema_uri = schemas.register(
        name="jacobian.finite-magma",
        version="1",
        schema=model_schema(FiniteMagma),
    )
    problem_schema_uri = schemas.register(
        name="jacobian.finite-magma-law-problem",
        version="1",
        schema=model_schema(FiniteMagmaLawProblem),
    )
    evaluation_schema_uri = schemas.register(
        name="jacobian.finite-magma-law-evaluation",
        version="1",
        schema=model_schema(FiniteMagmaLawEvaluationArtifact),
    )
    countermodel_schema_uri = schemas.register(
        name="jacobian.finite-magma-countermodel-search",
        version="1",
        schema=model_schema(FiniteMagmaCountermodelArtifact),
    )
    table_enumeration_schema_uri = schemas.register(
        name="jacobian.finite-magma-table-enumeration",
        version="1",
        schema=model_schema(FiniteMagmaTableEnumerationArtifact),
    )
    claim_schema_uri = schemas.register(
        name="jacobian.finite-magma-law-evaluation-claim",
        version="1",
        schema=model_schema(FiniteMagmaLawEvaluationClaim),
    )
    certificate_schema_uri = schemas.register(
        name="jacobian.certificate-envelope",
        version="1",
        schema=model_schema(CertificateEnvelope),
    )
    evaluation_checker_id = (
        CheckerInstaller(checkers)
        .install(
            CheckerOperation(
                name="exact finite magma law evaluation replay checker",
                entrypoint=("jacobian_checkers.universal_algebra:check_law_evaluation"),
                evidence_kind=EvidenceKind.CERTIFICATE,
                format_id="universal_algebra.law_evaluation",
                format_version="1",
                claim_schema_uris=(claim_schema_uri,),
                semantics_uris=(semantics_uri,),
                candidate_schema_uris=(evaluation_schema_uri,),
                reason="bundled independent finite table evaluator",
            ),
            authorize=authorize_checker,
        )
        .checker_id
    )
    installation = UniversalAlgebraInstallation(
        semantics_uri=semantics_uri,
        magma_schema_uri=magma_schema_uri,
        problem_schema_uri=problem_schema_uri,
        evaluation_schema_uri=evaluation_schema_uri,
        countermodel_schema_uri=countermodel_schema_uri,
        table_enumeration_schema_uri=table_enumeration_schema_uri,
        claim_schema_uri=claim_schema_uri,
        certificate_schema_uri=certificate_schema_uri,
        evaluation_checker_id=evaluation_checker_id,
    )
    resources = UniversalAlgebraResources(
        store=store,
        artifacts=artifacts,
        installation=installation,
    )
    evaluation = UniversalAlgebraEvaluateLawsAdapter(resources)
    search_runtime = known_provider_runtime(
        "jacobian.z3",
        features=("finite-magma-countermodel-search",),
    )
    adapters: tuple[CapabilityAdapter, ...] = (evaluation,)
    if search_runtime.availability is CapabilityProviderAvailability.AVAILABLE:
        adapters += (
            UniversalAlgebraSearchCountermodelAdapter(resources, search_runtime),
        )
    adapters += (FiniteMagmaTableEnumerateAdapter(resources),)
    verify = certificate_verification_adapter(
        capability_id="universal_algebra.law_evaluation.verify",
        title="Verify a finite-magma law evaluation",
        description=(
            "Independently replay one exhaustive finite-magma law evaluation "
            "certificate."
        ),
        checker_id=evaluation_checker_id,
        tags=("universal-algebra", "finite-magma", "law-evaluation"),
        verification=verification,
    )
    if verify is not None:
        adapters += (verify,)
    return adapters, installation


class UniversalAlgebraEvaluateLawsAdapter:
    """Evaluate finite magma laws in deterministic lexicographic order."""

    def __init__(self, resources: UniversalAlgebraResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="universal_algebra.evaluate_laws",
            version="1",
            title="Evaluate laws on a finite magma",
            description=(
                "Evaluate each equational law exactly on a finite binary-operation "
                "table, returning exhaustive truth evidence or the first failing "
                "valuation."
            ),
            provider="jacobian.finite-table",
            provider_runtime=known_provider_runtime(
                "jacobian.finite-table",
                features=("finite-magma-law-evaluation",),
                checker_ids=(
                    (resources.installation.evaluation_checker_id,)
                    if resources.installation.evaluation_checker_id is not None
                    else ()
                ),
            ),
            input_schema=model_schema(UniversalAlgebraEvaluationRequest),
            output_schema=model_schema(UniversalAlgebraEvaluationOutput),
            tags=(
                "universal-algebra",
                "finite-model",
                "law-evaluation",
                "counterexample",
            ),
            invocation_examples=(
                example(
                    "one_element_idempotence",
                    "Evaluate x*x=x on the one-element magma.",
                    {
                        "problem": {
                            "structure": {"order": 1, "table": [[0]]},
                            "laws": [
                                {
                                    "law_id": "idempotence",
                                    "variables": ["x"],
                                    "left": {
                                        "kind": "PRODUCT",
                                        "left": {"kind": "VARIABLE", "variable": "x"},
                                        "right": {"kind": "VARIABLE", "variable": "x"},
                                    },
                                    "right": {"kind": "VARIABLE", "variable": "x"},
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

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        try:
            validated = UniversalAlgebraEvaluationRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_FINITE_MAGMA_LAW_REQUEST",
                    stage="request_validation",
                    message="The complete finite-magma law request is invalid.",
                )
            ) from exc
        started = time.monotonic()
        problem = validated.problem
        problem_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.problem_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=problem.model_dump(mode="json"),
            summary="finite magma and equational laws",
        )
        records = tuple(_evaluate_law(problem.structure, law) for law in problem.laws)
        evaluation = FiniteMagmaLawEvaluationArtifact(
            problem_uri=problem_artifact.artifact_uri,
            records=records,
        )
        evaluation_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.evaluation_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=evaluation.model_dump(mode="json"),
            parents=(problem_artifact.artifact_uri,),
            summary="exact finite magma law evaluation",
        )
        claim = FiniteMagmaLawEvaluationClaim(problem_uri=problem_artifact.artifact_uri)
        claim_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.claim_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=claim.model_dump(mode="json"),
            parents=(
                problem_artifact.artifact_uri,
                evaluation_artifact.artifact_uri,
            ),
            summary="exact finite magma law evaluation claim",
        )
        semantics = self.resources.store.get(self.resources.installation.semantics_uri)
        certificate_payload = FiniteMagmaLawReplayPayload(
            problem_uri=problem_artifact.artifact_uri,
            evaluation_uri=evaluation_artifact.artifact_uri,
        ).model_dump(mode="json")
        certificate = CertificateEnvelope(
            certificate_type="universal_algebra.law_evaluation",
            format_version="1",
            bindings=EvidenceBindings(
                claim_digest=claim_artifact.object_digest,
                semantics_digest=semantics.manifest.object_digest,
                candidate_digest=evaluation_artifact.object_digest,
                scope_digest=problem_artifact.object_digest,
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
                evaluation_artifact.artifact_uri,
                problem_artifact.artifact_uri,
            ),
            summary="unverified finite magma law evaluation certificate",
        )
        output = UniversalAlgebraEvaluationOutput(
            problem_uri=problem_artifact.artifact_uri,
            evaluation_uri=evaluation_artifact.artifact_uri,
            claim_uri=claim_artifact.artifact_uri,
            certificate_uri=certificate_artifact.artifact_uri,
            records=records,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(
                    problem_artifact.artifact_uri,
                    evaluation_artifact.artifact_uri,
                    claim_artifact.artifact_uri,
                    certificate_artifact.artifact_uri,
                ),
            ),
        )


class UniversalAlgebraSearchCountermodelAdapter:
    """Search all fixed-order operation tables through a complete SMT encoding."""

    def __init__(
        self,
        resources: UniversalAlgebraResources,
        provider_runtime: CapabilityProviderRuntime | None = None,
    ) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="universal_algebra.search.countermodel",
            version="1",
            title="Search for a fixed-order finite magma countermodel",
            description=(
                "Search binary-operation tables of one exact carrier order for a "
                "magma satisfying every source law and falsifying one target law."
            ),
            provider="jacobian.z3",
            provider_runtime=provider_runtime
            or known_provider_runtime(
                "jacobian.z3",
                features=("finite-magma-countermodel-search",),
            ),
            input_schema=model_schema(UniversalAlgebraCountermodelSearchRequest),
            output_schema=model_schema(UniversalAlgebraCountermodelSearchOutput),
            tags=(
                "universal-algebra",
                "finite-model",
                "countermodel",
                "counterexample",
                "bounded-search",
            ),
            invocation_examples=(
                CapabilityInvocationExample(
                    name="commutative_nonassociative_magma",
                    description=(
                        "Search order-two commutative magmas for a counterexample "
                        "to associativity."
                    ),
                    input=UniversalAlgebraCountermodelSearchRequest.model_validate(
                        {
                            "order": 2,
                            "source_laws": [
                                {
                                    "law_id": "commutative",
                                    "variables": ["x", "y"],
                                    "left": {
                                        "kind": "PRODUCT",
                                        "left": {
                                            "kind": "VARIABLE",
                                            "variable": "x",
                                        },
                                        "right": {
                                            "kind": "VARIABLE",
                                            "variable": "y",
                                        },
                                    },
                                    "right": {
                                        "kind": "PRODUCT",
                                        "left": {
                                            "kind": "VARIABLE",
                                            "variable": "y",
                                        },
                                        "right": {
                                            "kind": "VARIABLE",
                                            "variable": "x",
                                        },
                                    },
                                }
                            ],
                            "target_law": {
                                "law_id": "associative",
                                "variables": ["x", "y", "z"],
                                "left": {
                                    "kind": "PRODUCT",
                                    "left": {
                                        "kind": "PRODUCT",
                                        "left": {
                                            "kind": "VARIABLE",
                                            "variable": "x",
                                        },
                                        "right": {
                                            "kind": "VARIABLE",
                                            "variable": "y",
                                        },
                                    },
                                    "right": {
                                        "kind": "VARIABLE",
                                        "variable": "z",
                                    },
                                },
                                "right": {
                                    "kind": "PRODUCT",
                                    "left": {
                                        "kind": "VARIABLE",
                                        "variable": "x",
                                    },
                                    "right": {
                                        "kind": "PRODUCT",
                                        "left": {
                                            "kind": "VARIABLE",
                                            "variable": "y",
                                        },
                                        "right": {
                                            "kind": "VARIABLE",
                                            "variable": "z",
                                        },
                                    },
                                },
                            },
                        }
                    ).model_dump(mode="json"),
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        try:
            validated = UniversalAlgebraCountermodelSearchRequest.model_validate(
                request.input
            )
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_FINITE_MAGMA_COUNTERMODEL_REQUEST",
                    stage="request_validation",
                    message="The complete finite-magma countermodel request is invalid.",
                )
            ) from exc
        started = time.monotonic()
        try:
            search = _search_countermodel(validated)
        except ImportError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="FINITE_MAGMA_COUNTERMODEL_PROVIDER_UNAVAILABLE",
                    stage="provider_runtime",
                    message="The optional Z3 countermodel provider is unavailable.",
                )
            ) from exc
        search_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.countermodel_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=search.model_dump(mode="json"),
            summary="fixed-order finite magma countermodel search",
        )
        output = UniversalAlgebraCountermodelSearchOutput(
            search_uri=search_artifact.artifact_uri,
            status=search.status,
            structure=search.structure,
            source_records=search.source_records,
            target_record=search.target_record,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(search_artifact.artifact_uri,),
            ),
        )


class FiniteMagmaTableEnumerateAdapter:
    """Enumerate every small finite-magma table in canonical order."""

    def __init__(self, resources: UniversalAlgebraResources) -> None:
        self.resources = resources
        self._descriptor = CapabilityDescriptor(
            capability_id="finite_magma.table.enumerate",
            version="1",
            title="Enumerate finite magma tables",
            description=(
                "Enumerate every total binary-operation table of order one or two "
                "in exact lexicographic row-major order."
            ),
            provider="jacobian.finite-table",
            provider_runtime=known_provider_runtime(
                "jacobian.finite-table",
                features=("finite-magma-table-enumeration",),
            ),
            input_schema=model_schema(FiniteMagmaTableEnumerationRequest),
            output_schema=model_schema(FiniteMagmaTableEnumerationOutput),
            tags=("universal-algebra", "finite-model", "enumeration"),
            invocation_examples=(
                example(
                    "order_one",
                    "Enumerate the unique binary table of order one.",
                    {"order": 1},
                ),
            ),
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def invoke(self, request: CapabilityRequest) -> OperationProjection:
        try:
            validated = FiniteMagmaTableEnumerationRequest.model_validate(request.input)
        except ValidationError as exc:
            raise CapabilityInvocationError(
                CapabilityDiagnostic(
                    code="INVALID_FINITE_MAGMA_TABLE_ENUMERATION_REQUEST",
                    stage="request_validation",
                    message=(
                        "Finite-magma table enumeration supports exact carrier "
                        "orders one and two."
                    ),
                )
            ) from exc
        started = time.monotonic()
        order = validated.order
        table_uris: list[str] = []
        for cells in product(range(order), repeat=order * order):
            structure = FiniteMagma(
                order=order,
                table=tuple(
                    tuple(cells[row * order : (row + 1) * order])
                    for row in range(order)
                ),
            )
            table = self.resources.artifacts.put(
                schema_uri=self.resources.installation.magma_schema_uri,
                semantics_uri=self.resources.installation.semantics_uri,
                payload=structure.model_dump(mode="json"),
                summary=f"finite magma table of order {order}",
            )
            table_uris.append(table.artifact_uri)
        total_count = order ** (order * order)
        enumeration = FiniteMagmaTableEnumerationArtifact(
            order=order,
            table_uris=tuple(table_uris),
            enumerated_count=len(table_uris),
            total_count=total_count,
        )
        enumeration_artifact = self.resources.artifacts.put(
            schema_uri=self.resources.installation.table_enumeration_schema_uri,
            semantics_uri=self.resources.installation.semantics_uri,
            payload=enumeration.model_dump(mode="json"),
            parents=tuple(table_uris),
            summary=f"complete finite magma table enumeration of order {order}",
        )
        output = FiniteMagmaTableEnumerationOutput(
            enumeration_uri=enumeration_artifact.artifact_uri,
            order=order,
            table_uris=tuple(table_uris),
            enumerated_count=len(table_uris),
            total_count=total_count,
        )
        return OperationProjection(
            operation_id=self.descriptor.capability_id,
            version=self.descriptor.version,
            terminal=Completed(
                value=output,
                runtime_ms=max(0, round((time.monotonic() - started) * 1000)),
            ),
            publication=PublishedOperation(
                output=output,
                artifact_uris=(enumeration_artifact.artifact_uri, *table_uris),
            ),
        )


def _search_countermodel(
    request: UniversalAlgebraCountermodelSearchRequest,
) -> FiniteMagmaCountermodelArtifact:
    z3: Any = importlib.import_module("z3")
    order = request.order
    solver = z3.Solver()
    solver.set(random_seed=0, timeout=_COUNTERMODEL_TIMEOUT_MS)
    cells = tuple(
        tuple(z3.Int(f"mul_{left}_{right}") for right in range(order))
        for left in range(order)
    )
    for row in cells:
        for cell in row:
            solver.add(cell >= 0, cell < order)
    for law in request.source_laws:
        for values in product(range(order), repeat=len(law.variables)):
            assignment = dict(zip(law.variables, values, strict=True))
            solver.add(
                _z3_evaluate_term(law.left, cells, assignment, order, z3)
                == _z3_evaluate_term(law.right, cells, assignment, order, z3)
            )
    target_assignment = {
        variable: z3.Int(f"target_{variable}")
        for variable in request.target_law.variables
    }
    for value in target_assignment.values():
        solver.add(value >= 0, value < order)
    solver.add(
        _z3_evaluate_term(
            request.target_law.left,
            cells,
            target_assignment,
            order,
            z3,
        )
        != _z3_evaluate_term(
            request.target_law.right,
            cells,
            target_assignment,
            order,
            z3,
        )
    )
    result = solver.check()
    if result == z3.unknown:
        return FiniteMagmaCountermodelArtifact(
            order=order,
            source_laws=request.source_laws,
            target_law=request.target_law,
            status=CountermodelSearchStatus.INDETERMINATE,
            backend_version=z3.get_version_string(),
        )
    if result == z3.unsat:
        return FiniteMagmaCountermodelArtifact(
            order=order,
            source_laws=request.source_laws,
            target_law=request.target_law,
            status=CountermodelSearchStatus.NO_WITNESS_FOUND,
            backend_version=z3.get_version_string(),
        )
    model = solver.model()
    structure = FiniteMagma(
        order=order,
        table=tuple(
            tuple(model.eval(cell, model_completion=True).as_long() for cell in row)
            for row in cells
        ),
    )
    source_records = tuple(_evaluate_law(structure, law) for law in request.source_laws)
    target_record = _evaluate_law(structure, request.target_law)
    return FiniteMagmaCountermodelArtifact(
        order=order,
        source_laws=request.source_laws,
        target_law=request.target_law,
        status=CountermodelSearchStatus.WITNESS_FOUND,
        structure=structure,
        source_records=source_records,
        target_record=target_record,
        backend_version=z3.get_version_string(),
    )


def _z3_evaluate_term(
    term: MagmaTerm,
    cells: tuple[tuple[Any, ...], ...],
    assignment: dict[str, Any],
    order: int,
    z3: Any,
) -> Any:
    if term.kind == "VARIABLE":
        if term.variable is None:
            raise ValueError("variable terms require only a variable name")
        return assignment[term.variable]
    if term.left is None or term.right is None:
        raise ValueError("product terms require exactly two child terms")
    left = _z3_evaluate_term(term.left, cells, assignment, order, z3)
    right = _z3_evaluate_term(term.right, cells, assignment, order, z3)
    selected: Any = cells[-1][-1]
    for left_index in reversed(range(order)):
        for right_index in reversed(range(order)):
            selected = z3.If(
                z3.And(left == left_index, right == right_index),
                cells[left_index][right_index],
                selected,
            )
    return selected


def _evaluate_law(
    structure: FiniteMagma,
    law: MagmaLaw,
) -> MagmaLawEvaluationRecord:
    checked = 0
    for values in product(range(structure.order), repeat=len(law.variables)):
        checked += 1
        assignment = dict(zip(law.variables, values, strict=True))
        left_value = _evaluate_term(law.left, structure.table, assignment)
        right_value = _evaluate_term(law.right, structure.table, assignment)
        if left_value != right_value:
            return MagmaLawEvaluationRecord(
                law_id=law.law_id,
                holds=False,
                coverage=MagmaLawCoverage.COUNTEREXAMPLE_FOUND,
                checked_valuations=checked,
                counterexample=MagmaLawCounterexample(
                    assignment=tuple(
                        MagmaAssignmentValue(variable=variable, value=value)
                        for variable, value in zip(
                            law.variables,
                            values,
                            strict=True,
                        )
                    ),
                    left_value=left_value,
                    right_value=right_value,
                ),
            )
    return MagmaLawEvaluationRecord(
        law_id=law.law_id,
        holds=True,
        coverage=MagmaLawCoverage.EXHAUSTIVE,
        checked_valuations=checked,
    )


def _evaluate_term(
    term: MagmaTerm,
    table: tuple[tuple[int, ...], ...],
    assignment: dict[str, int],
) -> int:
    if term.kind == "VARIABLE":
        if term.variable is None:
            raise ValueError("variable terms require only a variable name")
        return assignment[term.variable]
    if term.left is None or term.right is None:
        raise ValueError("product terms require exactly two child terms")
    left = _evaluate_term(term.left, table, assignment)
    right = _evaluate_term(term.right, table, assignment)
    return table[left][right]
