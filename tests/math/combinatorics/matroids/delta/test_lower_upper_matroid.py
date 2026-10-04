"""Exact lower/upper matroid conversions for finite delta-matroids."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta import operations as delta_operations
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidLowerMatroidRequest,
    DeltaMatroidUpperMatroidRequest,
)
from jacobian.math.combinatorics.matroids.delta._tools import TOOLS
from jacobian.math.combinatorics.matroids.delta.operations import (
    lower_matroid,
    upper_matroid,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
)
from jacobian.math.combinatorics.matroids.values import FiniteBasisMatroid


def _symmetric_exchange(rows: tuple[tuple[int, ...], ...]) -> bool:
    feasible = {frozenset(row) for row in rows}
    for left in feasible:
        for right in feasible:
            difference = left ^ right
            for element in difference:
                if not any(
                    (left ^ {element, candidate}) in feasible
                    for candidate in difference
                ):
                    return False
    return True


def _basis_exchange(bases: tuple[tuple[int, ...], ...]) -> bool:
    basis_sets = {frozenset(row) for row in bases}
    for left in basis_sets:
        for right in basis_sets:
            for removed in left - right:
                if not any(
                    (left - {removed}) | {added} in basis_sets for added in right - left
                ):
                    return False
    return True


def _all_ground_subsets(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            row
            for cardinality in range(size + 1)
            for row in combinations(range(size), cardinality)
        )
    )


@pytest.mark.parametrize("ground_size", range(4))
def test_exhaustive_small_delta_matroids_match_independent_extrema(
    ground_size: int,
) -> None:
    """Enumerate every nonempty feasible family through a three-element ground."""
    all_rows = _all_ground_subsets(ground_size)
    for family_mask in range(1, 1 << len(all_rows)):
        rows = tuple(
            row for index, row in enumerate(all_rows) if family_mask & (1 << index)
        )
        if not _symmetric_exchange(rows):
            continue

        source = FiniteDeltaMatroid(
            ground=tuple(f"e{i}" for i in range(ground_size)), feasible=rows
        )
        lower = lower_matroid(source)
        upper = upper_matroid(source)
        expected_lower_indices = tuple(
            index for index, row in enumerate(rows) if len(row) == min(map(len, rows))
        )
        expected_upper_indices = tuple(
            index for index, row in enumerate(rows) if len(row) == max(map(len, rows))
        )

        assert isinstance(lower, FiniteBasisMatroid)
        assert lower.bases == tuple(rows[index] for index in expected_lower_indices)
        assert lower.ground == source.ground
        assert _basis_exchange(lower.bases)
        assert isinstance(upper, FiniteBasisMatroid)
        assert upper.bases == tuple(rows[index] for index in expected_upper_indices)
        assert upper.ground == source.ground
        assert _basis_exchange(upper.bases)


def test_extremal_conversions_return_canonical_matroids() -> None:
    ids = {tool.operation_id for tool in TOOLS}
    assert "delta_matroid.lower_matroid.compute" not in ids
    assert "delta_matroid.upper_matroid.compute" not in ids
    source = FiniteDeltaMatroid(
        ground=("a", "b"),
        feasible=((), (0,), (0, 1), (1,)),
    )
    for result in (lower_matroid(source), upper_matroid(source)):
        assert isinstance(result, FiniteBasisMatroid)
        restored = FiniteBasisMatroid.model_validate(result.model_dump(mode="json"))
        assert restored == result


def test_target_ground_over_bound_is_a_resource_refusal() -> None:
    source = FiniteDeltaMatroid(
        ground=tuple(f"e{i}" for i in range(65)), feasible=((),)
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        lower_matroid(source)
    assert (
        exc_info.value.errors()[0]["type"] == "finite_basis_matroid.ground_size_bound"
    )


def test_multibyte_output_label_over_byte_limit_is_resource_refusal() -> None:
    source = FiniteDeltaMatroid(ground=("😀" * 300,), feasible=((),))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        lower_matroid(source)
    assert (
        exc_info.value.errors()[0]["type"] == "finite_basis_matroid.ground_label_bound"
    )


def test_invalid_utf8_source_label_is_a_domain_rejection() -> None:
    source = FiniteDeltaMatroid(ground=("\ud800",), feasible=((),))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        lower_matroid(source)
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_request_raw_preflight_bounds_large_nested_source() -> None:
    raw = {
        "delta_matroid": {
            "ground": ["a", "b"],
            "feasible": [[0] for _ in range(16_385)],
        }
    }
    with pytest.raises(ValidationError):
        DeltaMatroidLowerMatroidRequest.model_validate(raw)
    with pytest.raises(ValidationError):
        DeltaMatroidUpperMatroidRequest.model_validate(raw)


def test_request_preflight_rejects_long_label_before_encoding() -> None:
    raw = {
        "delta_matroid": {
            "ground": ["x" * 100_000],
            "feasible": [[]],
        }
    }
    with pytest.raises(ValidationError):
        DeltaMatroidLowerMatroidRequest.model_validate(raw)


def test_native_forged_missing_ground_is_domain_rejection() -> None:
    source = FiniteDeltaMatroid.model_construct(feasible=((),))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        lower_matroid(source)
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.source_not_valid"


def test_native_forged_oversize_source_is_admitted_before_system_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FiniteDeltaMatroid.model_construct(
        ground=("a",), feasible=((0,) * 16_385,)
    )

    def system_copy_must_not_run(**_: object) -> None:
        pytest.fail("oversized source reached structural carrier reconstruction")

    monkeypatch.setattr(
        delta_operations, "FiniteFeasibleSetSystem", system_copy_must_not_run
    )
    with pytest.raises(OperationResourceAdmissionError):
        lower_matroid(source)


@pytest.mark.parametrize("operation", [lower_matroid, upper_matroid])
def test_oversize_ground_axis_is_bounded_before_label_validation(
    operation: Callable[[FiniteDeltaMatroid], FiniteBasisMatroid],
) -> None:
    source = FiniteDeltaMatroid.model_construct(
        ground=(object(),) * (MAX_DELTA_MEMBERSHIPS + 2),
        feasible=((),),
    )
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        operation(source)
    assert exc_info.value.errors()[0]["type"] == "delta_matroid.ground_size_exceeded"


def test_output_exposes_complete_bounds() -> None:
    schema = DeltaMatroidLowerMatroidRequest.model_json_schema()
    limits = schema["admission_limits"]
    assert limits["max_output_ground_elements"] == 64
    assert limits["max_output_ground_label_utf8_bytes_each"] == 1_024
    assert limits["max_output_ground_label_utf8_bytes_total"] == 16_384
    assert limits["max_output_basis_rows"] == 4_096
    assert limits["max_source_feasible_set_memberships"] == 16_384


def test_extremal_requests_accept_json_array_wire_containers() -> None:
    from jacobian.math.combinatorics.matroids.delta._models import (
        DeltaMatroidLowerMatroidRequest,
        DeltaMatroidUpperMatroidRequest,
    )

    raw = {"delta_matroid": {"ground": ["a", "b"], "feasible": [[], [0], [1]]}}
    lower = DeltaMatroidLowerMatroidRequest.model_validate(raw)
    upper = DeltaMatroidUpperMatroidRequest.model_validate(raw)

    assert lower.delta_matroid.ground == ("a", "b")
    assert upper.delta_matroid.feasible == ((), (0,), (1,))
