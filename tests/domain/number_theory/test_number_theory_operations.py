from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.services import open_domain_services

from jacobian.bounded_process import ProcessResourceLimits
from jacobian.canonical import loads_strict_json
from jacobian.catalog.collector import CatalogOperationCollector
from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.number_theory import number_theory_operations
from jacobian.process_policy import ProcessResult, ProcessTermination


@pytest.fixture
def number_theory_service(tmp_path: Path) -> Iterator[CatalogOperationCollector]:
    with open_domain_services(tmp_path, number_theory_operations()) as services:
        yield services.core.operations


def test_jacobi_symbol_is_domain_owned_exact_computation(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="number_theory.compute.jacobi_symbol",
            input={"a": "10", "n": 21},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"a": "10", "n": 21, "jacobi": -1}


def test_even_jacobi_denominator_fails_before_artifact_writes(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="number_theory.compute.jacobi_symbol",
            input={"a": "10", "n": 20},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_NUMBER_THEORY_REQUEST"


def test_chinese_remainder_returns_canonical_exact_solution(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="modular.solve.chinese_remainder",
            input={"residues": [2, 3, 2], "moduli": [3, 5, 7]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {"residue": "23", "modulus": "105"}


def test_chinese_remainder_reports_inconsistent_system_without_artifacts(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="modular.solve.chinese_remainder",
            input={"residues": [0, 1], "moduli": [2, 2]},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "NUMBER_THEORY_OPERATION_NOT_APPLICABLE"


def test_discrete_logarithm_returns_typed_result(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="modular.compute.discrete_logarithm",
            input={
                "base": 7,
                "target": 15,
                "modulus": 41,
                "resource_budget": {"wall_seconds": 30},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == {
        "status": "SOLVED",
        "base": 7,
        "target": 15,
        "modulus": 41,
        "discrete_log": 3,
    }
    assert result.artifact_uris == ()


def test_discrete_logarithm_reports_unsolvable_without_false_witness(
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="modular.compute.discrete_logarithm",
            input={
                "base": 2,
                "target": 3,
                "modulus": 8,
                "resource_budget": {"wall_seconds": 30},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["status"] == "UNSOLVABLE"
    assert result.output["result"]["discrete_log"] is None


def test_discrete_logarithm_timeout_is_an_artifact_free_non_conclusion(
    number_theory_service: CatalogOperationCollector,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def timeout_worker(request):
        observed["timeout_seconds"] = request.timeout_seconds
        observed["resource_limits"] = request.resource_limits
        observed["input_bytes"] = request.stdin_bytes
        return ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr(
        "jacobian.domains.number_theory.discrete_logarithm.execute_process",
        timeout_worker,
    )
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="modular.compute.discrete_logarithm",
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
    number_theory_service: CatalogOperationCollector,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="integer.compute.prime_factorization",
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
    number_theory_service: CatalogOperationCollector,
    value: str,
    is_powerful: bool,
    factors: list[dict[str, object]],
    violating_primes: list[str],
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="integer.decide.powerful",
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
    assert result.artifact_uris == ()


@pytest.mark.parametrize("value", ["0", "-1", "-72"])
def test_powerful_number_rejects_nonpositive_input_before_artifact_writes(
    number_theory_service: CatalogOperationCollector,
    value: str,
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="integer.decide.powerful",
            input={"value": value},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.artifact_uris == ()
    assert result.diagnostics[0].code == "INVALID_NUMBER_THEORY_REQUEST"


@pytest.mark.parametrize(
    ("operation_id", "expected"),
    (
        ("integer.decide.squarefree", {"holds": True}),
        ("integer.compute.radical", {"value": "30"}),
    ),
)
def test_factorization_derived_operations_complete_in_the_worker(
    number_theory_service: CatalogOperationCollector,
    operation_id: str,
    expected: dict[str, object],
) -> None:
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id=operation_id,
            input={"n": 30, "resource_budget": {"wall_seconds": 10}},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"] == expected


def test_factorization_timeout_is_an_artifact_free_non_conclusion(
    number_theory_service: CatalogOperationCollector,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def timeout_worker(request):
        observed["resource_limits"] = request.resource_limits
        return ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        )

    monkeypatch.setattr(
        "jacobian.domains.number_theory.factorization.execute_process",
        timeout_worker,
    )
    result = number_theory_service.invoke(
        OperationRequest(
            operation_id="integer.compute.divisors",
            input={
                "value": "9999999967",
                "resource_budget": {"wall_seconds": 1},
            },
        )
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "INTEGER_FACTORIZATION_TIMEOUT"
    assert result.artifact_uris == ()
    limits = observed["resource_limits"]
    assert limits.cpu_seconds == 2
    assert limits.address_space_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    ("operation_id", "payload"),
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
    number_theory_service: CatalogOperationCollector,
    monkeypatch,
    operation_id: str,
    payload: dict[str, object],
) -> None:
    monkeypatch.setattr(
        "jacobian.domains.number_theory.factorization.execute_process",
        lambda *_args, **_kwargs: ProcessResult(
            termination=ProcessTermination.TIMED_OUT,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
        ),
    )

    result = number_theory_service.invoke(
        OperationRequest(operation_id=operation_id, input=payload)
    )

    assert result.execution.status is ExecutionStatus.TIMEOUT
    assert result.diagnostics[0].code == "INTEGER_FACTORIZATION_TIMEOUT"
    assert result.artifact_uris == ()
