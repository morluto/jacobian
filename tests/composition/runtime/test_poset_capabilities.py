from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.posets import build_finite_poset_bundle

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


def _result_payload(fresh_complete_runtime, result) -> dict[str, Any]:
    if "result_uri" in result.output:
        return fresh_complete_runtime.core.store.get(
            result.output["result_uri"]
        ).payload
    return result.output["result"]


def _materialize(
    fresh_complete_runtime, presentation: dict[str, Any]
) -> dict[str, Any]:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=presentation,
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return _result_payload(fresh_complete_runtime, result)["poset"]


def _invoke(
    fresh_complete_runtime, capability_id: str, poset: dict[str, Any], **extra: Any
):
    return fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input={"poset": poset, **extra},
        )
    )


def test_poset_bundle_exposes_four_atomic_capabilities(fresh_complete_runtime) -> None:
    ids = tuple(
        operation.capability_id
        for operation in build_finite_poset_bundle().capabilities
    )
    assert ids == (
        "poset.finite.materialize",
        "poset.width.compute",
        "poset.linear_extensions.count",
        "poset.mobius_function.compute",
    )
    assert "poset" in fresh_complete_runtime.portfolio.domain_bundles
    catalog_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert set(ids).issubset(catalog_ids)


def test_antichain_chain_decomposition_intent_finds_width(
    fresh_complete_runtime,
) -> None:
    discovered = fresh_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="maximum antichain and minimum chain decomposition of a finite poset",
            limit=5,
        )
    )
    assert discovered.matches[0].capability_id == "poset.width.compute"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"


def test_partially_ordered_set_intent_finds_width(
    fresh_complete_runtime,
) -> None:
    discovered = fresh_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="compute the width of a finite partially ordered set",
            limit=5,
        )
    )

    assert discovered.matches[0].capability_id == "poset.width.compute"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"


def test_materialization_is_canonical_complete_and_inline(
    fresh_complete_runtime,
) -> None:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=_DIAMOND,
        )
    )
    payload = _result_payload(fresh_complete_runtime, result)
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
    assert result.assurance.level is CapabilityAssuranceLevel.COMPUTED
    assert result.artifact_uris == ()


def test_canonical_poset_is_directly_consumable_by_width(
    fresh_complete_runtime,
) -> None:
    materialized = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=_DIAMOND,
        )
    )
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input={
                "poset": _result_payload(fresh_complete_runtime, materialized)["poset"]
            },
        )
    )

    assert result.execution.status is ExecutionStatus.COMPLETED
    assert result.output["result"]["width"] == 2
    assert result.artifact_uris == ()
    descriptors = {
        descriptor.capability_id: descriptor
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert descriptors["poset.width.compute"].accepted_artifact_types == ()


def test_width_rejects_artifact_uri_input_at_the_contract_boundary(
    fresh_complete_runtime,
) -> None:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.width.compute",
            input={"poset_artifact_uri": "artifact://sha256/" + "a" * 64},
        )
    )

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_REQUEST"
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
    fresh_complete_runtime,
    presentation: dict[str, Any],
    expected_width: int,
    expected_count: int,
) -> None:
    poset = _materialize(fresh_complete_runtime, presentation)
    width_result = _invoke(fresh_complete_runtime, "poset.width.compute", poset)
    width = _result_payload(fresh_complete_runtime, width_result)
    count_result = _invoke(
        fresh_complete_runtime,
        "poset.linear_extensions.count",
        poset,
    )
    count = _result_payload(fresh_complete_runtime, count_result)
    assert width["width"] == expected_width
    assert len(width["maximum_antichain"]) == expected_width
    assert len(width["minimum_chain_cover"]) == expected_width
    assert count["count"] == expected_count
    assert count["state_count"] == len(count["states"])
    assert count["completeness"] == "COMPLETE"


def test_width_dual_certificate_covers_diamond_exactly(fresh_complete_runtime) -> None:
    poset = _materialize(fresh_complete_runtime, _DIAMOND)
    result = _result_payload(
        fresh_complete_runtime,
        _invoke(fresh_complete_runtime, "poset.width.compute", poset),
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
    fresh_complete_runtime,
) -> None:
    poset = _materialize(fresh_complete_runtime, _DIAMOND)
    complete_result = _invoke(
        fresh_complete_runtime,
        "poset.mobius_function.compute",
        poset,
        scope="COMPLETE_MATRIX",
        intervals=[],
    )
    complete = _result_payload(fresh_complete_runtime, complete_result)
    selected_result = _invoke(
        fresh_complete_runtime,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "0", "upper": "1"}],
    )
    selected = _result_payload(fresh_complete_runtime, selected_result)
    values = {
        (item["lower"], item["upper"]): item["value"] for item in complete["values"]
    }
    assert values[("0", "1")] == 1
    assert complete["completeness"] == "COMPLETE_MATRIX"
    assert selected["completeness"] == "SELECTED_INTERVALS"
    assert selected["intervals"] == [{"lower": "0", "upper": "1"}]
    assert len(selected["values"]) == 1


def test_mobius_ledger_is_canonical_across_branching_topological_orders(
    fresh_complete_runtime,
) -> None:
    poset = _materialize(
        fresh_complete_runtime,
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
        fresh_complete_runtime,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "a", "upper": "e"}],
    )
    result = _result_payload(fresh_complete_runtime, computed)
    value = result["values"][0]
    assert value["value"] == 1
    assert [item["intermediate"] for item in value["recurrence_contributions"]] == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_non_graded_poset_omits_ranks(fresh_complete_runtime) -> None:
    poset = _materialize(
        fresh_complete_runtime,
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
    fresh_complete_runtime,
) -> None:
    poset = _materialize(
        fresh_complete_runtime,
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

    result = _invoke(fresh_complete_runtime, "poset.width.compute", poset)

    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code == "INVALID_FINITE_POSET_REQUEST"
    assert result.artifact_uris == ()


def test_relabeling_preserves_scalar_poset_outcomes(fresh_complete_runtime) -> None:
    first = _materialize(fresh_complete_runtime, _DIAMOND)
    second = _materialize(
        fresh_complete_runtime,
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
        fresh_complete_runtime,
        _invoke(fresh_complete_runtime, "poset.width.compute", first),
    )
    second_width = _result_payload(
        fresh_complete_runtime,
        _invoke(fresh_complete_runtime, "poset.width.compute", second),
    )
    assert first_width["width"] == second_width["width"]
    first_count = _result_payload(
        fresh_complete_runtime,
        _invoke(fresh_complete_runtime, "poset.linear_extensions.count", first),
    )
    second_count = _result_payload(
        fresh_complete_runtime,
        _invoke(fresh_complete_runtime, "poset.linear_extensions.count", second),
    )
    assert first_count["count"] == second_count["count"]


def test_invalid_poset_request_fails_before_artifact_writes(
    fresh_complete_runtime,
) -> None:
    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
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


def test_artifact_put_is_hidden_from_discovery_but_remains_dispatchable(
    fresh_complete_runtime,
) -> None:
    catalog_ids = {
        descriptor.capability_id
        for descriptor in fresh_complete_runtime.core.capabilities.catalog().capabilities
    }
    assert "artifact.put" in catalog_ids

    discovered = fresh_complete_runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(query="store artifact", limit=20)
    )
    assert not any(
        match.capability_id == "artifact.put" for match in discovered.matches
    )

    result = fresh_complete_runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="artifact.put",
            input={
                "schema_uri": "artifact://sha256/" + "0" * 64,
                "semantics_uri": "artifact://sha256/" + "0" * 64,
                "payload": {},
            },
        )
    )
    assert result.execution.status is ExecutionStatus.ERROR
    assert result.diagnostics[0].code != "UNKNOWN_CAPABILITY"
