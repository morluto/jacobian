from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jacobian.bounded_process import bounded_process_cancellation
from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.polynomial import polynomial_operations
from jacobian.polynomials import build_polynomial_operations
from tests.support.services import DomainTestServices, open_domain_services


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(tmp_path / "state", polynomial_operations()) as services:
        adapters, _contracts = build_polynomial_operations(
            services.core.store,
            services.core.schemas,
            services.core.artifacts,
            services.verification,
            services.core.checkers,
            authorize_checker=False,
        )
        for adapter in adapters:
            services.installation.register_operation(adapter)
        yield services


def _map(exponent: int) -> dict[str, object]:
    return {
        "variables": ["x"],
        "coordinates": [
            {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [exponent],
                    }
                ]
            }
        ],
    }


def _request(exponent: int) -> OperationRequest:
    return OperationRequest(
        operation_id="polynomial.map.collision.search",
        input={
            "map": _map(exponent),
            "max_abs_numerator": 1,
            "max_denominator": 1,
        },
    )


def test_collision_search_returns_first_deterministic_candidate(
    domain_services,
) -> None:

    result = domain_services.core.operations.invoke(_request(2))

    assert result.output["found"] is True
    assert result.output["grid_point_count"] == 3
    assert result.output["examined_point_count"] == 3
    assert result.output["first_point"] == [{"num": "-1", "den": "1"}]
    assert result.output["second_point"] == [{"num": "1", "den": "1"}]
    assert result.output["common_image"] == [{"num": "1", "den": "1"}]
    assert result.output["witness_uri"] in result.artifact_uris
    assert result.output["stop_reason"] == "FIRST_COLLISION"


def test_collision_search_reports_partial_grid_after_early_collision(
    domain_services,
) -> None:

    result = domain_services.core.operations.invoke(_request(0))

    assert result.output["found"] is True
    assert result.output["grid_point_count"] == 3
    assert result.output["examined_point_count"] == 2
    assert result.output["first_point"] == [{"num": "-1", "den": "1"}]
    assert result.output["second_point"] == [{"num": "0", "den": "1"}]
    assert result.output["stop_reason"] == "FIRST_COLLISION"


def test_collision_search_reports_exact_completed_not_found_scope(
    domain_services,
) -> None:

    result = domain_services.core.operations.invoke(_request(1))

    assert result.output["found"] is False
    assert result.output["examined_point_count"] == 3
    assert result.output["grid_point_count"] == 3
    assert result.output["witness_uri"] is None
    assert result.output["stop_reason"] == "GRID_EXHAUSTED"


def test_collision_search_preserves_partial_evidence_when_cancelled(
    domain_services,
) -> None:
    cancellation_event = threading.Event()
    cancellation_event.set()

    with bounded_process_cancellation(cancellation_event):
        result = domain_services.core.operations.invoke(_request(1))

    assert result.execution.status is ExecutionStatus.CANCELLED
    assert result.output == {}
    assert len(result.artifact_uris) == 1


def test_collision_search_validates_grid_bound_before_artifact_writes(
    domain_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_put_calls = 0
    original_put = domain_services.core.artifacts.put

    def recording_put(*args: Any, **kwargs: Any) -> Any:
        nonlocal artifact_put_calls
        artifact_put_calls += 1
        return original_put(*args, **kwargs)

    monkeypatch.setattr(domain_services.core.artifacts, "put", recording_put)
    variables = ["w", "x", "y", "z"]
    polynomial_map = {
        "variables": variables,
        "coordinates": [
            {
                "terms": [
                    {
                        "coefficient": {"num": "1", "den": "1"},
                        "exponents": [
                            int(variable == coordinate) for variable in variables
                        ],
                    }
                ]
            }
            for coordinate in variables
        ],
    }

    result = domain_services.core.operations.invoke(
        OperationRequest(
            operation_id="polynomial.map.collision.search",
            input={
                "map": polynomial_map,
                "max_abs_numerator": 8,
                "max_denominator": 8,
            },
        )
    )

    assert result.execution.status.value == "ERROR"
    assert result.diagnostics[0].code == "INVALID_POLYNOMIAL_COLLISION_SEARCH_REQUEST"
    assert artifact_put_calls == 0
