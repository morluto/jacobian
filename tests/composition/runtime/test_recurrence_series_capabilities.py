from __future__ import annotations

import json
from pathlib import Path

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.combinatorics import build_combinatorics_bundle

_RECURRENCE_CONVENTION = "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
_P_RECURSIVE_CONVENTION = "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
_PUBLIC_TASKS = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "public-reproductions-v1"
)
_OVERLAP_CASES = [
    {
        "query": json.loads(
            (_PUBLIC_TASKS / slug / "environment" / "input.json").read_text()
        )["query"],
        "expected_first": json.loads(
            (_PUBLIC_TASKS / slug / "tests" / "expected.json").read_text()
        )["expected_first"],
    }
    for slug in (
        "recurrence-fibonacci",
        "recurrence-linear-eval",
        "recurrence-lucas",
        "recurrence-rational-series",
    )
]


def _q(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _q_string(numerator: str, denominator: str = "1") -> dict[str, str]:
    return {"num": numerator, "den": denominator}


def _recurrence_payload(
    *,
    scope: str = "PREFIX",
    term_count: int | None = 8,
    indices: list[int] | None = None,
) -> dict[str, object]:
    return {
        "coefficients": [_q(1), _q(1)],
        "initial_values": [_q(0), _q(1)],
        "coefficient_convention": _RECURRENCE_CONVENTION,
        "scope": scope,
        "term_count": term_count,
        "indices": indices or [],
    }


def _series_payload() -> dict[str, object]:
    return {
        "numerator": [_q(1)],
        "denominator": [_q(1), _q(-1), _q(-1)],
        "coefficient_convention": "ASCENDING_POWERS_OF_X",
        "expansion_point": "0",
        "truncation_order": 8,
    }


def test_combinatorics_bundle_preserves_named_sequences_and_adds_atomic_generic_ops(
    fresh_complete_runtime,
) -> None:
    ids = {
        operation.capability_id
        for operation in build_combinatorics_bundle().capabilities
    }
    assert {
        "combinatorics.compute.fibonacci",
        "combinatorics.compute.fibonacci_pair",
        "combinatorics.compute.lucas",
        "combinatorics.recurrence.linear.evaluate",
        "combinatorics.recurrence.p_recursive.evaluate",
        "combinatorics.generating_function.coefficients.compute",
    }.issubset(ids)
    catalog_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert ids.issubset(catalog_ids)


@pytest.mark.parametrize(
    ("query", "expected"),
    tuple((case["query"], case["expected_first"]) for case in _OVERLAP_CASES),
)
def test_discovery_overlap_keeps_named_and_generic_intents_distinct(
    fresh_complete_runtime,
    query: str,
    expected: str,
) -> None:
    discovered = fresh_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(query=query, limit=5)
    )
    assert discovered.matches[0].capability_id == expected


def test_linear_recurrence_exposes_requested_values_and_complete_replay(
    fresh_complete_runtime,
) -> None:
    prefix = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.evaluate",
            input=_recurrence_payload(),
        )
    )
    sparse = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.evaluate",
            input=_recurrence_payload(
                scope="INDICES",
                term_count=None,
                indices=[0, 2, 7],
            ),
        )
    )
    assert prefix.execution.status is ExecutionStatus.COMPLETED
    assert prefix.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert [item["value"]["num"] for item in prefix.output["result"]["values"]] == [
        "0",
        "1",
        "1",
        "2",
        "3",
        "5",
        "8",
        "13",
    ]
    assert sparse.output["result"]["replay_scope_end"] == 7
    assert [item["index"] for item in sparse.output["result"]["values"]] == [0, 2, 7]
    assert len(sparse.output["result"]["replay_prefix"]) == 8
    assert sparse.artifact_uris == ()
    assert sparse.relationships == ()


def test_polynomial_coefficient_recurrence_exposes_exact_terms_and_residuals(
    fresh_complete_runtime,
) -> None:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.p_recursive.evaluate",
            input={
                "coefficient_polynomials": [
                    [_q(1), _q(1)],
                    [_q(2), _q(-6)],
                    [_q(-13), _q(9)],
                    [_q(-4)],
                    [_q(16), _q(-4)],
                ],
                "initial_values": [_q(1), _q(1), _q(1), _q(2)],
                "coefficient_convention": _P_RECURSIVE_CONVENTION,
                "polynomial_convention": "ASCENDING_POWERS_OF_N",
                "scope": "PREFIX",
                "term_count": 17,
                "indices": [],
            },
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    assert [item["value"]["num"] for item in result.output["result"]["values"]] == [
        "1",
        "1",
        "1",
        "2",
        "5",
        "14",
        "41",
        "123",
        "375",
        "1158",
        "3615",
        "11393",
        "36209",
        "115940",
        "373709",
        "1211740",
        "3949969",
    ]
    assert {item["value"]["num"] for item in result.output["result"]["residuals"]} == {
        "0"
    }


def test_polynomial_coefficient_recurrence_rejects_singular_required_step(
    fresh_complete_runtime,
) -> None:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.p_recursive.evaluate",
            input={
                "coefficient_polynomials": [[_q(-2), _q(1)], [_q(-1)]],
                "initial_values": [_q(1)],
                "coefficient_convention": _P_RECURSIVE_CONVENTION,
                "polynomial_convention": "ASCENDING_POWERS_OF_N",
                "scope": "PREFIX",
                "term_count": 4,
                "indices": [],
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.output["error"]["code"] == "INVALID_COMBINATORICS_REQUEST"


def test_large_rational_results_cross_the_python_conversion_limit_safely(
    fresh_complete_runtime,
) -> None:
    large = "9" * 64
    computed = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.recurrence.linear.evaluate",
            input={
                "coefficients": [_q_string(large)],
                "initial_values": [_q_string(large)],
                "coefficient_convention": _RECURRENCE_CONVENTION,
                "scope": "INDICES",
                "term_count": None,
                "indices": [68],
            },
        )
    )

    value = computed.output["result"]["values"][0]["value"]["num"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert len(value) > 4_300
    assert len(value) <= 32_768


@pytest.mark.parametrize(
    ("capability_id", "payload"),
    (
        (
            "combinatorics.recurrence.linear.evaluate",
            {
                "coefficients": [_q_string("9" * 64)],
                "initial_values": [_q_string("9" * 64)],
                "coefficient_convention": _RECURRENCE_CONVENTION,
                "scope": "PREFIX",
                "term_count": 513,
                "indices": [],
            },
        ),
        (
            "combinatorics.recurrence.linear.evaluate",
            {
                "coefficients": [_q_string("9" * 64)],
                "initial_values": [_q(1)],
                "coefficient_convention": _RECURRENCE_CONVENTION,
                "scope": "PREFIX",
                "term_count": 512,
                "indices": [],
            },
        ),
        (
            "combinatorics.generating_function.coefficients.compute",
            {
                "numerator": [_q(1)],
                "denominator": [_q_string("1", "9" * 64), _q_string("9" * 64)],
                "coefficient_convention": "ASCENDING_POWERS_OF_X",
                "expansion_point": "0",
                "truncation_order": 257,
            },
        ),
    ),
)
def test_oversized_derived_results_are_rejected_before_artifact_writes(
    fresh_complete_runtime,
    capability_id: str,
    payload: dict[str, object],
) -> None:
    with fresh_complete_runtime.core.store.connection() as connection:
        before = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_COMBINATORICS_REQUEST"
    assert result.artifact_uris == ()
    with fresh_complete_runtime.core.store.connection() as connection:
        after = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    assert after == before


def test_rational_generating_function_exposes_exact_residual_congruence(
    fresh_complete_runtime,
) -> None:
    computed = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="combinatorics.generating_function.coefficients.compute",
            input=_series_payload(),
        )
    )
    result = computed.output["result"]
    assert computed.execution.status is ExecutionStatus.COMPLETED
    assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert [coefficient["num"] for coefficient in result["coefficients"]] == [
        "1",
        "1",
        "2",
        "3",
        "5",
        "8",
        "13",
        "21",
    ]
    assert all(item == _q(0) for item in result["residual_coefficients"])
    assert result["residual_congruence"].endswith("ZERO_MOD_X_TO_ORDER")


@pytest.mark.parametrize(
    "payload",
    (
        {
            **_recurrence_payload(),
            "initial_values": [_q(0)],
        },
        {
            **_recurrence_payload(scope="INDICES", term_count=None, indices=[2, 2]),
        },
        {
            **_series_payload(),
            "denominator": [_q(0), _q(1)],
        },
        {
            **_series_payload(),
            "numerator": [_q(1), _q(0)],
        },
    ),
)
def test_cross_field_validation_precedes_artifact_writes(
    fresh_complete_runtime,
    payload: dict[str, object],
) -> None:
    with fresh_complete_runtime.core.store.connection() as connection:
        before = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    capability_id = (
        "combinatorics.recurrence.linear.evaluate"
        if "initial_values" in payload
        else "combinatorics.generating_function.coefficients.compute"
    )
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_COMBINATORICS_REQUEST"
    assert result.artifact_uris == ()
    with fresh_complete_runtime.core.store.connection() as connection:
        after = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    assert after == before
