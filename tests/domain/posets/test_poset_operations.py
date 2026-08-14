from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus

_DIAMOND = {
    "elements": ["0", "a", "b", "1"],
    "relation": [
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ],
    "interpretation": "COVER_EDGES",
}


def _result_payload(poset_services, result) -> dict[str, Any]:
    if "result_uri" in result.output:
        return poset_services.core.store.get(result.output["result_uri"]).payload
    return result.output["result"]


def _materialize(poset_services, presentation: dict[str, Any]) -> dict[str, Any]:
    result = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.finite.compute",
            input=presentation,
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return _result_payload(poset_services, result)["poset"]


def _invoke(poset_services, operation_id: str, poset: dict[str, Any], **extra: Any):
    return poset_services.core.operations.invoke(
        OperationRequest(
            operation_id=operation_id,
            input={"poset": poset, **extra},
        )
    )


def test_materialization_is_canonical_complete_and_inline(
    poset_services,
) -> None:
    result = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.finite.compute",
            input=_DIAMOND,
        )
    )
    payload = _result_payload(poset_services, result)
    poset = payload["poset"]
    assert poset["elements"] == ["0", "1", "a", "b"]
    assert poset["strict_order_pairs"] == [
        {"lower": "0", "upper": "1"},
        {"lower": "0", "upper": "a"},
        {"lower": "0", "upper": "b"},
        {"lower": "a", "upper": "1"},
        {"lower": "b", "upper": "1"},
    ]
    assert poset["cover_relations"] == _DIAMOND["relation"]
    assert poset["incomparable_pairs"] == [{"left": "a", "right": "b"}]
    assert poset["graded"] is True
    assert result.artifact_uris == ()


def test_canonical_poset_is_directly_consumable_by_width(
    poset_services,
) -> None:
    materialized = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.finite.compute",
            input=_DIAMOND,
        )
    )
    result = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.width.compute",
            input={"poset": _result_payload(poset_services, materialized)["poset"]},
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["width"] == 2
    assert result.artifact_uris == ()
    descriptors = {
        descriptor.operation_id: descriptor
        for descriptor in poset_services.core.operations.snapshot().operations
    }
    assert descriptors["poset.width.compute"].accepted_artifact_types == ()


def test_width_rejects_artifact_uri_input_at_the_contract_boundary(
    poset_services,
) -> None:
    result = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.width.compute",
            input={"poset_artifact_uri": "artifact://sha256/" + "a" * 64},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_POSET_REQUEST"
    assert result.artifact_uris == ()


@pytest.mark.parametrize(
    ("presentation", "expected_width", "expected_count"),
    (
        (
            {
                "elements": [],
                "relation": [],
                "interpretation": "COVER_EDGES",
            },
            0,
            1,
        ),
        (
            {
                "elements": ["a"],
                "relation": [],
                "interpretation": "COVER_EDGES",
            },
            1,
            1,
        ),
        (
            {
                "elements": ["a", "b", "c", "d"],
                "relation": [],
                "interpretation": "COVER_EDGES",
            },
            4,
            24,
        ),
        (
            {
                "elements": ["a", "b", "c", "d"],
                "relation": [
                    {"lower": "a", "upper": "b"},
                    {"lower": "b", "upper": "c"},
                    {"lower": "c", "upper": "d"},
                ],
                "interpretation": "COVER_EDGES",
            },
            1,
            1,
        ),
        (_DIAMOND, 2, 2),
    ),
)
def test_width_and_linear_extension_reference_cases(
    poset_services,
    presentation: dict[str, Any],
    expected_width: int,
    expected_count: int,
) -> None:
    poset = _materialize(poset_services, presentation)
    width_result = _invoke(poset_services, "poset.width.compute", poset)
    width = _result_payload(poset_services, width_result)
    count_result = _invoke(
        poset_services,
        "poset.linear_extensions.count",
        poset,
    )
    count = _result_payload(poset_services, count_result)
    assert width["width"] == expected_width
    assert len(width["maximum_antichain"]) == expected_width
    assert len(width["minimum_chain_cover"]) == expected_width
    assert count["count"] == expected_count
    assert count["state_count"] == len(count["states"])
    assert count["completeness"] == "COMPLETE"


def test_width_dual_certificate_covers_diamond_exactly(poset_services) -> None:
    poset = _materialize(poset_services, _DIAMOND)
    result = _result_payload(
        poset_services,
        _invoke(poset_services, "poset.width.compute", poset),
    )
    assert result["maximum_antichain"] == ["a", "b"]
    assert (
        sorted(
            element
            for chain in result["minimum_chain_cover"]
            for element in chain["elements"]
        )
        == poset["elements"]
    )
    assert result["matching_size"] + result["width"] == len(poset["elements"])
    assert result["certificate"] == "DILWORTH_ANTICHAIN_CHAIN_COVER"


def test_mobius_complete_and_selected_scopes_are_distinct(
    poset_services,
) -> None:
    poset = _materialize(poset_services, _DIAMOND)
    complete_result = _invoke(
        poset_services,
        "poset.mobius_function.compute",
        poset,
        scope="COMPLETE_MATRIX",
        intervals=[],
    )
    complete = _result_payload(poset_services, complete_result)
    selected_result = _invoke(
        poset_services,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "0", "upper": "1"}],
    )
    selected = _result_payload(poset_services, selected_result)
    values = {
        (item["lower"], item["upper"]): item["value"] for item in complete["values"]
    }
    assert values[("0", "1")] == 1
    assert complete["completeness"] == "COMPLETE_MATRIX"
    assert selected["completeness"] == "SELECTED_INTERVALS"
    assert selected["intervals"] == [{"lower": "0", "upper": "1"}]
    assert len(selected["values"]) == 1


def test_mobius_ledger_is_canonical_across_branching_topological_orders(
    poset_services,
) -> None:
    poset = _materialize(
        poset_services,
        {
            "elements": ["a", "b", "c", "d", "e"],
            "relation": [
                {"lower": "a", "upper": "b"},
                {"lower": "a", "upper": "d"},
                {"lower": "b", "upper": "c"},
                {"lower": "c", "upper": "e"},
                {"lower": "d", "upper": "e"},
            ],
            "interpretation": "COVER_EDGES",
        },
    )
    computed = _invoke(
        poset_services,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "a", "upper": "e"}],
    )
    result = _result_payload(poset_services, computed)
    value = result["values"][0]
    assert value["value"] == 1
    assert value["recurrence_contributions"] is None

    ledger = _invoke(
        poset_services,
        "poset.mobius_function.recurrence.materialize",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "a", "upper": "e"}],
    )
    ledger_result = _result_payload(poset_services, ledger)
    assert [
        item["intermediate"]
        for item in ledger_result["values"][0]["recurrence_contributions"]
    ] == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_non_graded_poset_omits_ranks(poset_services) -> None:
    poset = _materialize(
        poset_services,
        {
            "elements": ["a", "b", "c", "d"],
            "relation": [
                {"lower": "a", "upper": "d"},
                {"lower": "b", "upper": "c"},
                {"lower": "c", "upper": "d"},
            ],
            "interpretation": "COVER_EDGES",
        },
    )
    assert poset["graded"] is False
    assert poset["ranks"] is None


def test_embedded_graded_poset_cannot_omit_canonical_ranks(
    poset_services,
) -> None:
    poset = _materialize(
        poset_services,
        {
            "elements": ["a", "b"],
            "relation": [{"lower": "a", "upper": "b"}],
            "interpretation": "COVER_EDGES",
        },
    )
    poset["graded"] = False
    poset["ranks"] = None
    from jacobian.contracts.posets import FinitePoset

    with pytest.raises(ValidationError, match="graded metadata"):
        FinitePoset.model_validate(poset)

    result = _invoke(poset_services, "poset.width.compute", poset)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_POSET_REQUEST"
    assert result.artifact_uris == ()


def test_relabeling_preserves_scalar_poset_outcomes(poset_services) -> None:
    first = _materialize(poset_services, _DIAMOND)
    second = _materialize(
        poset_services,
        {
            "elements": ["z", "x", "y", "w"],
            "relation": [
                {"lower": "w", "upper": "x"},
                {"lower": "w", "upper": "y"},
                {"lower": "x", "upper": "z"},
                {"lower": "y", "upper": "z"},
            ],
            "interpretation": "COVER_EDGES",
        },
    )
    first_width = _result_payload(
        poset_services,
        _invoke(poset_services, "poset.width.compute", first),
    )
    second_width = _result_payload(
        poset_services,
        _invoke(poset_services, "poset.width.compute", second),
    )
    assert first_width["width"] == second_width["width"]
    first_count = _result_payload(
        poset_services,
        _invoke(poset_services, "poset.linear_extensions.count", first),
    )
    second_count = _result_payload(
        poset_services,
        _invoke(poset_services, "poset.linear_extensions.count", second),
    )
    assert first_count["count"] == second_count["count"]


def test_invalid_poset_request_fails_before_artifact_writes(
    poset_services,
) -> None:
    result = poset_services.core.operations.invoke(
        OperationRequest(
            operation_id="poset.finite.compute",
            input={
                "elements": ["a", "b"],
                "relation": [
                    {"lower": "a", "upper": "b"},
                    {"lower": "b", "upper": "a"},
                ],
                "interpretation": "COVER_EDGES",
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_POSET_REQUEST"
    assert result.artifact_uris == ()
