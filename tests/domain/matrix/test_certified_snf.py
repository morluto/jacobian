from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import gcd
from pathlib import Path
from random import Random

import pytest
from tests.support.services import DomainTestServices, open_domain_services

from jacobian.canonical import canonicalize_json
from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains._certified_snf import smith_reduce
from jacobian.domains.certified_snf import build_certified_snf_bundle


@pytest.fixture
def domain_services(tmp_path: Path) -> Iterator[DomainTestServices]:
    with open_domain_services(
        tmp_path / "state",
        build_certified_snf_bundle(),
    ) as services:
        yield services


def _result_payload(services: DomainTestServices, result: object) -> dict[str, object]:
    result_uri = result.output["result_uri"]  # type: ignore[attr-defined]
    return services.core.store.get(result_uri).payload


def _minor_determinant(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if size == 0:
        return 1
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(size - 1):
        selected = next(
            (row for row in range(pivot_index, size) if work[row][pivot_index] != 0),
            None,
        )
        if selected is None:
            return 0
        if selected != pivot_index:
            work[pivot_index], work[selected] = work[selected], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                work[row][column] = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                ) // previous
        previous = pivot
    return sign * work[-1][-1]


def _invariant_factors_from_determinantal_divisors(
    source: list[list[int]],
) -> tuple[int, ...]:
    rows = len(source)
    columns = len(source[0])
    previous_divisor = 1
    factors: list[int] = []
    for size in range(1, min(rows, columns) + 1):
        divisor = 0
        for selected_rows in combinations(range(rows), size):
            for selected_columns in combinations(range(columns), size):
                minor = [
                    [source[row][column] for column in selected_columns]
                    for row in selected_rows
                ]
                divisor = gcd(divisor, abs(_minor_determinant(minor)))
        if divisor == 0:
            break
        factors.append(divisor // previous_divisor)
        previous_divisor = divisor
    return tuple(factors)


def test_certified_smith_materializes_both_full_basis_changes(
    domain_services: DomainTestServices,
) -> None:
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input={
                "matrix": {
                    "row_count": 2,
                    "column_count": 2,
                    "entries": [["2", "4"], ["6", "8"]],
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    certificate = _result_payload(domain_services, result)["certificate"]
    assert certificate["diagonal"]["entries"] == [["2", "0"], ["0", "4"]]
    assert certificate["invariant_factors"] == ["2", "4"]
    assert certificate["relation"] == ("DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT")
    assert certificate["left_determinant"] in {"-1", "1"}
    assert certificate["right_determinant"] in {"-1", "1"}
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert len(result.artifact_uris) == 2


def test_elementary_reduction_matches_determinantal_divisors() -> None:
    random = Random(20260730)
    for _ in range(500):
        rows = random.randint(1, 6)
        columns = random.randint(1, 6)
        source = [
            [random.randint(-20, 20) for _ in range(columns)] for _ in range(rows)
        ]

        reduction = smith_reduce(source)

        assert reduction.invariant_factors == (
            _invariant_factors_from_determinantal_divisors(source)
        )


def test_maximum_certified_smith_payload_stays_within_artifact_budget(
    domain_services: DomainTestServices,
) -> None:
    factor = "1" + "0" * 31
    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input={
                "matrix": {
                    "row_count": 16,
                    "column_count": 16,
                    "entries": [
                        [factor if row == column else "0" for column in range(16)]
                        for row in range(16)
                    ],
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    artifacts = [domain_services.core.store.get(uri) for uri in result.artifact_uris]
    sizes = [len(canonicalize_json(artifact.payload)) for artifact in artifacts]
    assert all(size < 10 * 1024 * 1024 for size in sizes)
    assert sum(sizes) < 8 * 1024 * 1024


def test_dense_bounded_input_can_materialize_large_basis_changes(
    domain_services: DomainTestServices,
) -> None:
    random = Random(2)
    entries = [
        [str(random.randrange(-(10**31), 10**31)) for _column in range(16)]
        for _row in range(16)
    ]

    result = domain_services.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="matrix.normal_form.smith.certified.compute",
            input={
                "matrix": {
                    "row_count": 16,
                    "column_count": 16,
                    "entries": entries,
                }
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    certificate = _result_payload(domain_services, result)["certificate"]
    result_integers = [
        value
        for matrix_name in (
            "diagonal",
            "left_transformation",
            "right_transformation",
        )
        for row in certificate[matrix_name]["entries"]
        for value in row
    ]
    assert max(len(value.lstrip("-")) for value in result_integers) > 4_096
    artifacts = [domain_services.core.store.get(uri) for uri in result.artifact_uris]
    assert all(
        len(canonicalize_json(artifact.payload)) < 10 * 1024 * 1024
        for artifact in artifacts
    )
