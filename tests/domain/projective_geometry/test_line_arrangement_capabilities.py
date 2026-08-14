from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from tests.support.exact_domain import open_exact_domain_services
from tests.support.services import DomainTestServices

from jacobian.contracts.operations import OperationRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.projective_geometry.domain_declarations import (
    projective_geometry_operations,
)


@pytest.fixture
def projective_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_exact_domain_services(
        tmp_path / "state", projective_geometry_operations()
    ) as services:
        yield services


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _result_payload(services: DomainTestServices, computed: Any) -> dict[str, Any]:
    if "result_uri" in computed.output:
        return services.core.store.get(computed.output["result_uri"]).payload
    return computed.output["result"]


def test_projective_arrangement_materializes_the_nine_line_flat_lattice(
    projective_services: DomainTestServices,
) -> None:
    coefficients = (
        (0, 1, -1),
        (0, 1, 2),
        (0, 2, 1),
        (1, -2, -1),
        (1, -1, -2),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 2),
        (2, -1, -2),
    )
    result = projective_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.materialize",
            input={
                "lines": [
                    {
                        "label": str(index),
                        "coefficients": [_q(value) for value in line],
                    }
                    for index, line in enumerate(coefficients, start=1)
                ]
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    payload = _result_payload(projective_services, result)
    assert payload["non_double_flats"] == [
        ["1", "2", "3"],
        ["1", "4", "5"],
        ["1", "6", "7"],
        ["2", "4", "6"],
        ["2", "5", "8", "9"],
        ["3", "5", "7"],
        ["3", "6", "8"],
        ["4", "7", "9"],
    ]
    assert payload["multiplicity_histogram"] == [
        {"multiplicity": 2, "flat_count": 9},
        {"multiplicity": 3, "flat_count": 7},
        {"multiplicity": 4, "flat_count": 1},
    ]
    assert payload["pair_count_total"] == 36
    verified = projective_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.verify",
            input={"result_uri": result.output["result_uri"]},
        )
    )
    assert verified.execution.status is ExecutionStatus.COMPLETED
    assert verified.output["status"] == "VERIFIED"


def test_projective_arrangement_rejects_projectively_duplicate_lines(
    projective_services: DomainTestServices,
) -> None:
    result = projective_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.materialize",
            input={
                "lines": [
                    {"label": "L1", "coefficients": [_q(1), _q(2), _q(3)]},
                    {"label": "L2", "coefficients": [_q(2), _q(4), _q(6)]},
                ]
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_PROJECTIVE_ARRANGEMENT_REQUEST"
    assert result.artifact_uris == ()


def test_arrangement_checker_rejects_schema_valid_forged_normalization(
    projective_services: DomainTestServices,
) -> None:
    computed = projective_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.materialize",
            input={
                "lines": [
                    {"label": "L1", "coefficients": [_q(1), _q(0), _q(0)]},
                    {"label": "L2", "coefficients": [_q(0), _q(1), _q(0)]},
                ]
            },
        )
    )
    stored = projective_services.core.store.get(computed.output["result_uri"])
    forged = deepcopy(stored.payload)
    forged["normalized_lines"][0]["coefficients"]["coordinates"] = ["1", "1", "0"]
    forged_uri = projective_services.core.artifacts.put(
        schema_uri=stored.manifest.schema_uri,
        semantics_uri=stored.manifest.semantics_uri,
        payload=forged,
        parents=(computed.output["input_uri"],),
        summary="schema-valid forged projective normalization",
    ).artifact_uri
    checked = projective_services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.verify",
            input={"result_uri": forged_uri},
        )
    )
    assert checked.execution.status is ExecutionStatus.COMPLETED
    assert checked.output["status"] == "REJECTED"
