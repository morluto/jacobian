from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.finite_fields import build_finite_field_bundle
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    DirectionRankLedger,
    FiniteDimensionalSubspace,
    direction_rank_ledger,
    element,
    finite_field,
    projective_line,
)

pytestmark = pytest.mark.requires_provider("flint")


def _forged_ledger() -> DirectionRankLedger:
    presentation = finite_field(2, (1, 1, 1))
    row_axis = Axis(name="b", labels=("b1", "b2"))
    column_axis = Axis(name="y", labels=("y1",))
    one = element(presentation, (1, 0))
    subspace = FiniteDimensionalSubspace(
        presentation=presentation,
        basis_axis=Axis(name="basis", labels=("B1",)),
        basis=(
            AxisBoundMatrix(
                presentation=presentation,
                row_axis=row_axis,
                column_axis=column_axis,
                entries=((one,), (element(presentation, (0, 0)),)),
            ),
        ),
    )
    payload = direction_rank_ledger(
        subspace,
        projective_line(presentation, row_axis),
    ).model_dump(mode="json")
    payload["entries"][0]["rank"] = 1 - payload["entries"][0]["rank"]
    return DirectionRankLedger.model_validate(payload)


def test_installed_orbit_aggregation_rejects_a_forged_ledger(tmp_path: Path) -> None:
    with open_exact_domain_services(
        tmp_path,
        build_finite_field_bundle(),
    ) as services:
        result = services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="finite_field.orbit_distribution.compute",
                input={"ledger": _forged_ledger().model_dump(mode="json")},
            )
        )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_DIRECTION_RANK_LEDGER"
    assert result.artifact_uris == ()
    assert result.verification_record_uri is None
