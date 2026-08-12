from pathlib import Path

import pytest
from tests.support.exact_domain import open_exact_domain_services

from jacobian.contracts.capabilities import CapabilityRequest
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.finite_fields import build_finite_field_bundle
from jacobian.math.finite_fields import (
    FiniteMapTable,
    element,
    finite_field,
    finite_map_table,
    finite_polynomial,
    finite_polynomial_map,
)

pytestmark = pytest.mark.requires_provider("flint")


def _table(*exponents: int) -> FiniteMapTable:
    presentation = finite_field(2, (1, 1, 1))
    zero = element(presentation, (0, 0))
    one = element(presentation, (1, 0))
    coefficients = tuple(one if power in exponents else zero for power in range(4))
    return finite_map_table(
        finite_polynomial_map(finite_polynomial(presentation, coefficients))
    )


def _forged_identity_table() -> FiniteMapTable:
    identity = _table(1)
    zero = identity.entries[0][1]
    return FiniteMapTable(
        map=identity.map,
        entries=tuple((source, zero) for source, _ in identity.entries),
    )


def test_installed_consumers_reject_targets_not_produced_by_the_polynomial(
    tmp_path: Path,
) -> None:
    payload = _forged_identity_table().model_dump(mode="json")

    with open_exact_domain_services(
        tmp_path,
        build_finite_field_bundle(),
    ) as services:
        results = tuple(
            services.core.capabilities.invoke(
                CapabilityRequest(capability_id=capability_id, input={"table": payload})
            )
            for capability_id in (
                "finite_field.polynomial_map.fibers.compute",
                "finite_field.polynomial_map.collision.compute",
                "finite_field.polynomial_map.permutation.compute",
            )
        )

    for result in results:
        assert result.execution.status is ExecutionStatus.ERROR
        assert result.diagnostics[0].code == "INVALID_FINITE_MAP_TABLE"
        assert result.artifact_uris == ()
        assert result.verification_record_uri is None
        assert set(result.output) == {"error"}


def test_valid_tables_report_operation_specific_nonconclusions(tmp_path: Path) -> None:
    cases = (
        (
            "finite_field.polynomial_map.collision.compute",
            _table(1),
            "FINITE_MAP_HAS_NO_COLLISION",
        ),
        (
            "finite_field.polynomial_map.permutation.compute",
            _table(3),
            "FINITE_MAP_NOT_PERMUTATION",
        ),
    )

    with open_exact_domain_services(
        tmp_path,
        build_finite_field_bundle(),
    ) as services:
        results = tuple(
            services.core.capabilities.invoke(
                CapabilityRequest(
                    capability_id=capability_id,
                    input={"table": table.model_dump(mode="json")},
                )
            )
            for capability_id, table, _ in cases
        )

    for result, (_, _, code) in zip(results, cases, strict=True):
        assert result.execution.status is ExecutionStatus.ERROR
        assert result.diagnostics[0].code == code
        assert result.artifact_uris == ()
