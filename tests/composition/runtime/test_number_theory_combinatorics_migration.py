from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jacobian.artifacts import ArtifactService
from jacobian.bounded_process import BoundedProcessResult, ProcessResourceLimits
from jacobian.canonical import loads_strict_json
from jacobian.capability_service import CapabilityService
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityCompletenessStatus,
    CapabilityRequest,
)
from jacobian.contracts.number_theory import (
    ChineseRemainderRequest,
    FactorialValuationRequest,
    ModularValueRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    PowerfulNumberResult,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import build_combinatorics_bundle
from jacobian.domains.number_theory import build_number_theory_bundle
from jacobian.memory import ResearchMemory
from jacobian.operation_installation import OperationInstaller
from jacobian.runtime import create_runtime
from jacobian.schema_registry import SchemaRegistry
from jacobian.storage.repository import ArtifactRepository


def _service(tmp_path: Path) -> CapabilityService:
    store = ArtifactRepository(tmp_path)
    schemas = SchemaRegistry(store)
    artifacts = ArtifactService(store, schemas)
    service = CapabilityService(store, ResearchMemory(store, schemas))
    installer = OperationInstaller(store, schemas, artifacts)
    for bundle in (build_number_theory_bundle(), build_combinatorics_bundle()):
        for adapter in installer.install(bundle).adapters:
            service.register(adapter)
    return service


def test_runtime_catalog_uses_only_domain_owned_operation_ids(tmp_path: Path) -> None:
    catalog_ids = {
        descriptor.capability_id
        for descriptor in create_runtime(tmp_path)
        .core.capabilities.catalog()
        .capabilities
    }

    assert {
        "number_theory.compute.jacobi_symbol",
        "modular.compute.discrete_logarithm",
        "combinatorics.enumerate.integer_partitions",
    } <= catalog_ids
    assert {
        "number_theory.jacobi_symbol.compute",
        "number_theory.discrete_log.bounded",
        "combinatorics.integer_partition.enumerate",
    }.isdisjoint(catalog_ids)


def test_jacobi_symbol_is_domain_owned_exact_computation(tmp_path: Path) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="number_theory.compute.jacobi_symbol",
            input={"a": "10", "n": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"a": "10", "n": 21, "jacobi": -1}
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_even_jacobi_denominator_fails_before_artifact_writes(
    tmp_path: Path,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="number_theory.compute.jacobi_symbol",
            input={"a": "10", "n": 20},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_NUMBER_THEORY_REQUEST"


def test_chinese_remainder_returns_canonical_exact_solution(tmp_path: Path) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="modular.solve.chinese_remainder",
            input={"residues": [2, 3, 2], "moduli": [3, 5, 7]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"residue": "23", "modulus": "105"}
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


def test_chinese_remainder_reports_inconsistent_system_without_artifacts(
    tmp_path: Path,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="modular.solve.chinese_remainder",
            input={"residues": [0, 1], "moduli": [2, 2]},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    assert result.diagnostics[0].code == "NUMBER_THEORY_OPERATION_NOT_APPLICABLE"


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    with pytest.raises(ValidationError, match="canonical"):
        ChineseRemainderRequest(residues=(residue,), moduli=(3,))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"residues": [1, 2], "moduli": [3]}, "equal length"),
        ({"residues": [0], "moduli": [1]}, "between 2 and 10,000"),
        ({"residues": [0], "moduli": [10_001]}, "between 2 and 10,000"),
    ],
)
def test_chinese_remainder_rejects_invalid_system_bounds(
    payload: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChineseRemainderRequest.model_validate(payload)


def test_discrete_logarithm_materializes_bound_result_and_obligation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    result = service.invoke(
        CapabilityRequest(
            capability_id="modular.compute.discrete_logarithm",
            input={
                "base": 7,
                "target": 15,
                "modulus": 41,
                "resource_budget": {"wall_seconds": 30},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output == {
        "status": "SOLVED",
        "base": 7,
        "target": 15,
        "modulus": 41,
        "discrete_log": 3,
    }
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert len(result.artifact_uris) == 3
    assert len(result.obligations) == 1
    obligation = service.store.get(result.obligations[0].obligation_uri)
    assert obligation.payload["required_checks"] == ["DISCRETE_LOG_WITNESS_REPLAY"]


def test_discrete_logarithm_reports_unsolvable_without_false_witness(
    tmp_path: Path,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="modular.compute.discrete_logarithm",
            input={
                "base": 2,
                "target": 3,
                "modulus": 8,
                "resource_budget": {"wall_seconds": 30},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["status"] == "UNSOLVABLE"
    assert result.output["discrete_log"] is None
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE


def test_discrete_logarithm_timeout_is_an_artifact_free_non_conclusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def timeout_worker(*args, **kwargs):
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(
        "jacobian.domains.number_theory.discrete_logarithm.run_bounded_process",
        timeout_worker,
    )
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="modular.compute.discrete_logarithm",
            input={
                "base": 7,
                "target": 15,
                "modulus": 41,
                "resource_budget": {"wall_seconds": 1},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "DISCRETE_LOGARITHM_TIMEOUT"
    assert result.artifact_uris == ()
    assert result.completeness.status is CapabilityCompletenessStatus.NOT_APPLICABLE
    assert observed["timeout_seconds"] == 1.0
    assert observed["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=2,
        address_space_bytes=1024 * 1024 * 1024,
    )
    input_bytes = observed["input_bytes"]
    assert isinstance(input_bytes, bytes)
    worker_payload = loads_strict_json(input_bytes)
    assert worker_payload["request"]["resource_budget"] == {"wall_seconds": 1}


def test_factorization_is_complete_in_an_isolated_bounded_worker(
    tmp_path: Path,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="integer.compute.prime_factorization",
            input={
                "value": "360",
                "resource_budget": {"wall_seconds": 10},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "factors": [
            {"prime": "2", "power": 3},
            {"prime": "3", "power": 2},
            {"prime": "5", "power": 1},
        ]
    }
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED


@pytest.mark.parametrize(
    ("value", "is_powerful", "factors", "violating_primes"),
    (
        ("1", True, [], []),
        ("72", True, [{"prime": "2", "power": 3}, {"prime": "3", "power": 2}], []),
        (
            "12",
            False,
            [{"prime": "2", "power": 2}, {"prime": "3", "power": 1}],
            ["3"],
        ),
        (
            "30",
            False,
            [
                {"prime": "2", "power": 1},
                {"prime": "3", "power": 1},
                {"prime": "5", "power": 1},
            ],
            ["2", "3", "5"],
        ),
    ),
)
def test_powerful_number_decision_preserves_a_complete_factor_witness(
    tmp_path: Path,
    value: str,
    is_powerful: bool,
    factors: list[dict[str, object]],
    violating_primes: list[str],
) -> None:
    service = _service(tmp_path)
    result = service.invoke(
        CapabilityRequest(
            capability_id="integer.decide.powerful",
            input={
                "value": value,
                "resource_budget": {"wall_seconds": 10},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
        "is_powerful": is_powerful,
        "factors": factors,
        "violating_primes": violating_primes,
    }
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    input_uri, result_uri = result.artifact_uris
    assert service.store.get(result_uri).manifest.parents == (input_uri,)
    assert result.relationships[0].source_artifact_uris == (input_uri,)
    assert result.relationships[0].target_artifact_uris == (result_uri,)


@pytest.mark.parametrize("value", ["0", "-1", "-72"])
def test_powerful_number_rejects_nonpositive_input_before_artifact_writes(
    tmp_path: Path,
    value: str,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="integer.decide.powerful",
            input={"value": value},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_NUMBER_THEORY_REQUEST"


@pytest.mark.parametrize(
    "payload",
    (
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [{"prime": "2", "power": 3}],
            "violating_primes": [],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [{"prime": "2", "power": 1}],
            "violating_primes": ["2"],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [
                {"prime": "3", "power": 1},
                {"prime": "2", "power": 2},
            ],
            "violating_primes": ["3"],
        },
    ),
)
def test_powerful_number_result_rejects_inconsistent_or_noncanonical_witnesses(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PowerfulNumberResult.model_validate(payload)


@pytest.mark.parametrize(
    ("capability_id", "expected"),
    (
        ("integer.decide.squarefree", {"holds": True}),
        ("integer.compute.radical", {"value": "30"}),
    ),
)
def test_factorization_derived_operations_complete_in_the_worker(
    tmp_path: Path,
    capability_id: str,
    expected: dict[str, object],
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input={"n": 30, "resource_budget": {"wall_seconds": 10}},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == expected
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE


def test_factorization_timeout_is_an_artifact_free_non_conclusion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def timeout_worker(*args, **kwargs):
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(
        "jacobian.domains.number_theory.factorization.run_bounded_process",
        timeout_worker,
    )
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="integer.compute.divisors",
            input={
                "value": "9999999967",
                "resource_budget": {"wall_seconds": 1},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "INTEGER_FACTORIZATION_TIMEOUT"
    assert result.artifact_uris == ()
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC
    limits = observed["resource_limits"]
    assert limits.cpu_seconds == 2
    assert limits.address_space_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    ("capability_id", "payload"),
    (
        (
            "integer.decide.squarefree",
            {"n": 30, "resource_budget": {"wall_seconds": 1}},
        ),
        (
            "integer.compute.radical",
            {"n": 30, "resource_budget": {"wall_seconds": 1}},
        ),
        (
            "integer.decide.powerful",
            {"value": "72", "resource_budget": {"wall_seconds": 1}},
        ),
    ),
)
def test_factorization_derived_timeout_is_a_non_conclusion(
    tmp_path: Path,
    monkeypatch,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(
        "jacobian.domains.number_theory.factorization.run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    result = _service(tmp_path).invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "INTEGER_FACTORIZATION_TIMEOUT"
    assert result.artifact_uris == ()
    assert result.assurance.level is CapabilityAssuranceLevel.HEURISTIC


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 1_001}),
        (NonnegativeIntegerRequest, {"n": 1_001}),
        (ModularValueRequest, {"value": "2", "modulus": 10_001}),
        (FactorialValuationRequest, {"n": 1, "base": 1_000_001}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_integer_partition_enumeration_is_complete_and_canonical(
    tmp_path: Path,
) -> None:
    result = _service(tmp_path).invoke(
        CapabilityRequest(
            capability_id="combinatorics.enumerate.integer_partitions",
            input={"n": 5, "max_parts": 2},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "n": 5,
        "max_parts": 2,
        "partitions": [[5], [4, 1], [3, 2]],
    }
    assert result.completeness.status is CapabilityCompletenessStatus.COMPLETE
