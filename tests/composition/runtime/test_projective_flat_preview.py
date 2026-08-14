from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
        tmp_path / "state",
        projective_geometry_operations(),
    ) as services:
        yield services


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _invoke(
    services: DomainTestServices,
    lines: list[tuple[str, tuple[int, int, int]]],
):
    return services.core.operations.invoke(
        OperationRequest(
            operation_id="geometry.projective_line_arrangement.flats.materialize",
            input={
                "lines": [
                    {
                        "label": label,
                        "coefficients": [_q(value) for value in coefficients],
                    }
                    for label, coefficients in lines
                ]
            },
        )
    )


def test_projective_arrangement_exposes_proof_critical_flat_preview(
    projective_services: DomainTestServices,
) -> None:
    result = _invoke(
        projective_services,
        [
            ("x0", (1, 0, 0)),
            ("x1", (1, 0, -1)),
            ("y0", (0, 1, 0)),
            ("y1", (0, 1, -1)),
            ("diag_main", (1, -1, 0)),
            ("diag_anti", (1, 1, -1)),
        ],
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["preview_complete"] is False
    assert result.output["preview"] == {
        "preview_schema_version": "1",
        "line_count": 6,
        "non_double_flat_count": 4,
        "non_double_flats": [
            {
                "point": {"coordinates": ["0", "0", "1"]},
                "incident_labels": ["diag_main", "x0", "y0"],
                "multiplicity": 3,
                "pair_count": 3,
            },
            {
                "point": {"coordinates": ["0", "1", "1"]},
                "incident_labels": ["diag_anti", "x0", "y1"],
                "multiplicity": 3,
                "pair_count": 3,
            },
            {
                "point": {"coordinates": ["1", "0", "1"]},
                "incident_labels": ["diag_anti", "x1", "y0"],
                "multiplicity": 3,
                "pair_count": 3,
            },
            {
                "point": {"coordinates": ["1", "1", "1"]},
                "incident_labels": ["diag_main", "x1", "y1"],
                "multiplicity": 3,
                "pair_count": 3,
            },
        ],
        "non_double_flats_complete": True,
        "multiplicity_histogram": [
            {"multiplicity": 2, "flat_count": 3},
            {"multiplicity": 3, "flat_count": 4},
        ],
        "pair_count_total": 15,
        "artifact_completion": "COMPLETE",
        "arithmetic": "EXACT_INTEGER",
    }
    assert result.output["result_uri"] in result.artifact_uris


def test_projective_arrangement_bounds_large_flat_preview(
    projective_services: DomainTestServices,
) -> None:
    lines = [(f"x_{value}", (1, 0, -value)) for value in range(21)]
    lines.extend((f"y_{value}", (0, 1, -value)) for value in range(21))
    lines.extend((f"s_{value}", (1, 1, -value)) for value in range(22))

    result = _invoke(projective_services, lines)

    assert result.execution.status is ExecutionStatus.COMPLETED
    preview = result.output["preview"]
    assert result.output["preview_complete"] is False
    assert preview["non_double_flat_count"] > 32
    assert len(preview["non_double_flats"]) == 32
    assert preview["non_double_flats_complete"] is False
    assert preview["pair_count_total"] == 2016
