from dataclasses import replace
from typing import Any, Self, cast

import pytest
from pydantic import Field, model_validator

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiagnostic,
    CapabilityRequest,
)
from jacobian.contracts.results import ContractModel, ExecutionStatus
from jacobian.operation_installation import OperationInstaller
from jacobian.operations import (
    BoundedSearchInterrupted,
    BoundedSearchOperation,
    BoundedSearchWitness,
    ComputedNotApplicable,
    ComputedOperation,
    ComputedSuccess,
    DomainBundle,
    DomainDiagnostics,
    DomainSemantics,
    MaterializedOperation,
    MaterializedOperationFactory,
    OperationExecutionFailure,
    OperationFailure,
)
from jacobian.provider_runtime import known_provider_runtime
from jacobian.runtime.model import JacobianRuntime


class _SyntheticRequest(ContractModel):
    value: int = Field(ge=0, le=100)


class _CrossFieldRequest(ContractModel):
    """A request whose cross-field validator rejects a value that passes
    JSON Schema (both fields are valid integers in range) but fails
    Pydantic's ``model_validator`` — exercising the adapter-level
    ``ValidationError`` path, not the dispatch-level schema path.
    """

    value: int = Field(ge=0, le=100)
    limit: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def require_value_below_limit(self) -> Self:
        if self.value >= self.limit:
            raise ValueError("value must be strictly less than limit")
        return self


class _SyntheticResult(ContractModel):
    doubled: int


class _BoundedResult(ContractModel):
    complete: bool


class _SyntheticPreview(ContractModel):
    summary: str


def _synthetic_bundle() -> DomainBundle:
    not_applicable = CapabilityDiagnostic(
        code="SYNTHETIC_NOT_APPLICABLE",
        stage="synthetic_computation",
        message="Thirteen is excluded from this synthetic operation.",
    )

    def compute(
        request: _SyntheticRequest,
    ) -> ComputedSuccess[_SyntheticResult] | ComputedNotApplicable:
        if request.value == 13:
            return ComputedNotApplicable(not_applicable)
        return ComputedSuccess(_SyntheticResult(doubled=request.value * 2))

    return DomainBundle(
        domain_id="synthetic",
        schema_namespace="jacobian.synthetic",
        semantics=DomainSemantics(
            name="jacobian.synthetic",
            version="1",
            definition={"description": "synthetic capability test semantics"},
        ),
        provider_runtime=known_provider_runtime(
            "jacobian.synthetic",
            features=("deterministic",),
        ),
        backend_version="synthetic-1",
        capabilities=(
            ComputedOperation(
                capability_id="synthetic.compute.double",
                title="Double a bounded integer",
                description="Double one bounded nonnegative integer.",
                request_model=_SyntheticRequest,
                result_model=_SyntheticResult,
                implementation=compute,
                relation_id="synthetic.relation.double",
                tags=("synthetic",),
            ),
        ),
        diagnostics=DomainDiagnostics(
            invalid_request=CapabilityDiagnostic(
                code="INVALID_SYNTHETIC_REQUEST",
                stage="synthetic_input_validation",
                message="Input does not satisfy the synthetic contract.",
            )
        ),
        scope_description="the complete supplied synthetic input",
        completeness_basis="deterministic computation covered the supplied input",
        assurance_basis="deterministic synthetic computation; no checker invoked",
    )


def _install(runtime: JacobianRuntime, bundle: DomainBundle) -> None:
    installation = OperationInstaller(
        runtime.core.store,
        runtime.core.schemas,
        runtime.core.artifacts,
    ).install(bundle)
    for adapter in installation.adapters:
        runtime.core.capabilities.register(adapter)


def test_synthetic_bundle_returns_an_inline_typed_result(
    fresh_complete_runtime,
) -> None:
    _install(fresh_complete_runtime, _synthetic_bundle())

    descriptor = next(
        descriptor
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "synthetic.compute.double"
    )
    assert descriptor.provider == "jacobian.synthetic"
    assert descriptor.input_schema["additionalProperties"] is False
    result_schema = descriptor.output_schema["properties"]["result"]
    assert result_schema == {"$ref": "#/$defs/_SyntheticResult"}

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.compute.double",
            input={"value": 6},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"doubled": 12}
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()
    assert result.relationships == ()
    assert result.scope is not None
    assert result.scope.parameters == {"value": 6}
    assert result.scope.artifact_uri is None


def test_materialized_operation_retains_artifacts_lineage_and_typed_preview(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    materialized = MaterializedOperation(
        capability_id="synthetic.materialize.double",
        title="Materialize a doubled integer",
        description="Materialize one doubled integer with lineage and a typed preview.",
        request_model=_SyntheticRequest,
        result_model=_SyntheticResult,
        implementation=lambda request: ComputedSuccess(
            _SyntheticResult(doubled=request.value * 2)
        ),
        relation_id="synthetic.relation.materialize",
        tags=("synthetic",),
        resource_reason="the test exercises durable lineage and preview behavior",
        preview_model=_SyntheticPreview,
        preview=lambda result: _SyntheticPreview(summary=f"doubled={result.doubled}"),
        preview_complete=True,
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(materialized,)))

    descriptor = next(
        descriptor
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
        if descriptor.capability_id == "synthetic.materialize.double"
    )
    assert descriptor.read_only is False
    assert set(descriptor.output_schema["properties"]) == {
        "input_uri",
        "result_uri",
        "preview",
        "preview_complete",
        "backend_version",
    }

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.materialize.double",
            input={"value": 6},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert len(result.artifact_uris) == 2
    input_uri, result_uri = result.artifact_uris
    assert result.output["input_uri"] == input_uri
    assert result.output["result_uri"] == result_uri
    assert result.output["backend_version"] == "synthetic-1"
    assert result.output["preview"] == {"summary": "doubled=12"}
    assert result.output["preview_complete"] is True
    assert result.scope is not None
    assert result.scope.artifact_uri == input_uri
    assert result.scope.parameters == {"input_uri": input_uri}
    input_artifact = fresh_complete_runtime.core.store.get(input_uri)
    output_artifact = fresh_complete_runtime.core.store.get(result_uri)
    assert input_artifact.payload == {"value": 6}
    assert output_artifact.payload == {"doubled": 12}
    assert output_artifact.manifest.parents == (input_uri,)
    assert result.relationships[0].relation_id == "synthetic.relation.materialize"
    assert result.relationships[0].source_artifact_uris == (input_uri,)
    assert result.relationships[0].target_artifact_uris == (result_uri,)


def test_materialized_operation_omits_preview_without_projection(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    materialized = MaterializedOperation(
        capability_id="synthetic.materialize.no_preview",
        title="Materialize a doubled integer without preview",
        description="Materialize one doubled integer with lineage but no inline preview.",
        request_model=_SyntheticRequest,
        result_model=_SyntheticResult,
        implementation=lambda request: ComputedSuccess(
            _SyntheticResult(doubled=request.value * 2)
        ),
        relation_id="synthetic.relation.no_preview",
        tags=("synthetic",),
        resource_reason="the test exercises durable lineage without a preview",
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(materialized,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.materialize.no_preview",
            input={"value": 4},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert len(result.artifact_uris) == 2
    assert result.output["preview"] is None
    assert result.output["preview_complete"] is False
    assert result.output["backend_version"] == "synthetic-1"
    assert fresh_complete_runtime.core.store.get(result.artifact_uris[1]).payload == {
        "doubled": 8
    }


def test_materialized_factory_derives_terminal_materialize_relation_id() -> None:
    operation = MaterializedOperationFactory(
        OperationFailure(
            code="SYNTHETIC_NOT_APPLICABLE",
            stage="synthetic_computation",
            hint="Use a synthetic value.",
        )
    )(
        "synthetic.materialize",
        "Materialize a synthetic value",
        "Materialize one synthetic value with durable lineage.",
        _SyntheticRequest,
        _SyntheticResult,
        lambda request: _SyntheticResult(doubled=request.value * 2),
        resource_reason="the test exercises durable lineage and relation derivation",
    )

    assert operation.relation_id == "synthetic.relation"


def test_materialized_operation_fails_closed_before_artifact_writes(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    not_applicable = CapabilityDiagnostic(
        code="SYNTHETIC_NOT_APPLICABLE",
        stage="synthetic_computation",
        message="Thirteen is excluded from this synthetic operation.",
    )
    materialized = MaterializedOperation(
        capability_id="synthetic.materialize.excluded",
        title="Materialize a doubled integer that excludes thirteen",
        description="Exercise fail-closed materialization.",
        request_model=_SyntheticRequest,
        result_model=_SyntheticResult,
        implementation=lambda _request: ComputedNotApplicable(not_applicable),
        relation_id="synthetic.relation.excluded",
        tags=("synthetic",),
        resource_reason="the test exercises fail-closed durable computation",
        preview_model=_SyntheticPreview,
        preview=lambda result: _SyntheticPreview(summary=f"doubled={result.doubled}"),
        preview_complete=True,
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(materialized,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.materialize.excluded",
            input={"value": 13},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "SYNTHETIC_NOT_APPLICABLE"
    assert result.artifact_uris == ()


def test_synthetic_bundle_fails_closed_before_artifact_writes(
    fresh_complete_runtime,
) -> None:
    _install(fresh_complete_runtime, _synthetic_bundle())

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.compute.double",
            input={"value": 13},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "SYNTHETIC_NOT_APPLICABLE"
    assert result.artifact_uris == ()


@pytest.mark.parametrize(
    "status",
    (
        ExecutionStatus.ERROR,
        ExecutionStatus.TIMEOUT,
        ExecutionStatus.CANCELLED,
    ),
)
def test_computed_adapter_preserves_operational_failure_status(
    fresh_complete_runtime,
    status: ExecutionStatus,
) -> None:
    bundle = _synthetic_bundle()
    diagnostic = CapabilityDiagnostic(
        code="SYNTHETIC_OPERATION_FAILED",
        stage="synthetic_computation",
        message="The synthetic operation did not complete.",
    )
    failed = replace(
        bundle.capabilities[0],
        implementation=lambda _request: OperationExecutionFailure(status, diagnostic),
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(failed,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.compute.double",
            input={"value": 2},
        )
    )

    assert result.execution.status is status
    assert result.diagnostics == (diagnostic,)
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.artifact_uris == ()


def test_computed_failure_rejects_conclusive_status() -> None:
    diagnostic = CapabilityDiagnostic(
        code="SYNTHETIC_OPERATION_FAILED",
        stage="synthetic_computation",
        message="The synthetic operation did not complete.",
    )

    with pytest.raises(ValueError, match="operational failure status"):
        OperationExecutionFailure(ExecutionStatus.COMPLETED, diagnostic)


def test_bounded_adapter_preserves_timeout_without_partial_artifacts(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    diagnostic = CapabilityDiagnostic(
        code="SYNTHETIC_SEARCH_TIMEOUT",
        stage="synthetic_search",
        message="The synthetic search exceeded its wall budget.",
    )
    operation = BoundedSearchOperation(
        capability_id="synthetic.search.timeout",
        title="Timed out synthetic search",
        description="Exercise bounded operational failure mapping.",
        request_model=_SyntheticRequest,
        result_model=_BoundedResult,
        implementation=lambda _request: OperationExecutionFailure(
            ExecutionStatus.TIMEOUT,
            diagnostic,
        ),
        relation_id="synthetic.search.timeout.relation",
        scope_parameters=lambda _request, _result: {},
        is_complete=lambda result: result.complete,
        obligation_model=_BoundedResult,
        obligation=lambda _request, result: result,
        incomplete_basis="the synthetic search did not complete",
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(operation,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.search.timeout",
            input={"value": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics == (diagnostic,)
    assert result.artifact_uris == ()


def test_bounded_adapter_materializes_interrupted_partial_result(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    diagnostic = CapabilityDiagnostic(
        code="SYNTHETIC_SEARCH_TIMEOUT",
        stage="synthetic_search",
        message="The synthetic search retained a partial result.",
    )
    operation = BoundedSearchOperation(
        capability_id="synthetic.search.partial_timeout",
        title="Interrupted synthetic search",
        description="Exercise inspectable bounded interruption mapping.",
        request_model=_SyntheticRequest,
        result_model=_BoundedResult,
        implementation=lambda _request: BoundedSearchInterrupted(
            value=_BoundedResult(complete=False),
            status=ExecutionStatus.TIMEOUT,
            diagnostic=diagnostic,
        ),
        relation_id="synthetic.search.partial-timeout.relation",
        scope_parameters=lambda _request, _result: {},
        is_complete=lambda result: result.complete,
        obligation_model=_BoundedResult,
        obligation=lambda _request, result: result,
        incomplete_basis="the synthetic search did not complete",
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(operation,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.search.partial_timeout",
            input={"value": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.output == {"complete": False}
    assert result.diagnostics == (diagnostic,)
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert len(result.artifact_uris) == 3
    assert len(result.obligations) == 1


def test_computed_adapter_rejects_invalid_implementation_result(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    original = bundle.capabilities[0]
    invalid = ComputedOperation(
        capability_id="synthetic.compute.invalid",
        title=original.title,
        description=original.description,
        request_model=_SyntheticRequest,
        result_model=_SyntheticResult,
        implementation=lambda _request: ComputedSuccess(
            cast(Any, {"doubled": "not-an-integer"})
        ),
        relation_id="synthetic.relation.invalid",
    )
    _install(
        fresh_complete_runtime,
        DomainBundle(
            domain_id=bundle.domain_id,
            schema_namespace=bundle.schema_namespace,
            semantics=bundle.semantics,
            provider_runtime=bundle.provider_runtime,
            backend_version=bundle.backend_version,
            capabilities=(invalid,),
            diagnostics=bundle.diagnostics,
            scope_description=bundle.scope_description,
            completeness_basis=bundle.completeness_basis,
            assurance_basis=bundle.assurance_basis,
        ),
    )
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.compute.invalid",
            input={"value": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.artifact_uris == ()


def test_operation_specific_invalid_request_is_not_enriched(
    fresh_complete_runtime,
) -> None:
    """An operation that sets its own ``invalid_request`` diagnostic must
    receive it verbatim on ``ValidationError`` — the enrichment helper must
    not overwrite its ``path``, ``hint``, or any other field.

    Uses ``_CrossFieldRequest`` so the input passes JSON Schema validation
    (both fields are valid integers in range) but fails the Pydantic
    ``model_validator``, reaching the adapter-level ``except ValidationError``
    block where enrichment is gated on ``operation.invalid_request is None``.
    """

    bundle = _synthetic_bundle()
    operation_diagnostic = CapabilityDiagnostic(
        code="SYNTHETIC_OPERATION_INVALID_REQUEST",
        stage="synthetic_operation_validation",
        message="This operation-specific diagnostic must be preserved verbatim.",
        hint="Use a value between 0 and 100.",
        path="value",
    )
    operation = ComputedOperation(
        capability_id="synthetic.compute.operation_diagnostic",
        title="Cross-field validated synthetic operation",
        description="Exercise operation-specific invalid_request preservation.",
        request_model=_CrossFieldRequest,
        result_model=_SyntheticResult,
        implementation=lambda request: ComputedSuccess(
            _SyntheticResult(doubled=request.value * 2)
        ),
        relation_id="synthetic.relation.cross_field",
        tags=("synthetic",),
        invalid_request=operation_diagnostic,
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(operation,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="synthetic.compute.operation_diagnostic",
            input={"value": 50, "limit": 10},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0] == operation_diagnostic
    assert result.artifact_uris == ()


def test_installer_rejects_empty_and_duplicate_domain_bundles(
    fresh_complete_runtime,
) -> None:
    installer = OperationInstaller(
        fresh_complete_runtime.core.store,
        fresh_complete_runtime.core.schemas,
        fresh_complete_runtime.core.artifacts,
    )
    bundle = _synthetic_bundle()

    with pytest.raises(ValueError, match="must not be empty"):
        installer.install(replace(bundle, capabilities=()))
    with pytest.raises(ValueError, match="duplicate capability ID"):
        installer.install(replace(bundle, capabilities=(bundle.capabilities[0],) * 2))


def test_bounded_outcome_cannot_contradict_completion_semantics(
    fresh_complete_runtime,
) -> None:
    bundle = _synthetic_bundle()
    contradictory = BoundedSearchOperation(
        capability_id="synthetic.search.contradictory",
        title="Contradictory bounded search",
        description="Exercise fail-closed completion binding.",
        request_model=_SyntheticRequest,
        result_model=_BoundedResult,
        implementation=lambda _request: BoundedSearchWitness(
            _BoundedResult(complete=False)
        ),
        relation_id="synthetic.search.relation",
        scope_parameters=lambda _request, _result: {},
        is_complete=lambda result: result.complete,
        obligation_model=_BoundedResult,
        obligation=lambda _request, result: result,
        incomplete_basis="the synthetic search did not complete",
    )
    _install(fresh_complete_runtime, replace(bundle, capabilities=(contradictory,)))

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=contradictory.capability_id,
            input={"value": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "ADAPTER_EXECUTION_FAILED"
    assert result.artifact_uris == ()
