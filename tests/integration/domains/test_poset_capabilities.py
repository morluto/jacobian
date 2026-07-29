from __future__ import annotations

from typing import Any

import pytest

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityDiscoveryRequest,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.posets import FINITE_POSET_BUNDLE

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


def _materialize(runtime, presentation: dict[str, Any]) -> dict[str, Any]:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=presentation,
        )
    )
    assert result.execution.status is ExecutionStatus.COMPLETED
    return result.output["result"]["poset"]


def _invoke(runtime, capability_id: str, poset: dict[str, Any], **extra: Any):
    return runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id=capability_id,
            input={"poset": poset, **extra},
        )
    )


def test_poset_bundle_exposes_four_atomic_capabilities(runtime) -> None:
    ids = tuple(
        operation.capability_id for operation in FINITE_POSET_BUNDLE.capabilities
    )
    assert ids == (
        "poset.finite.materialize",
        "poset.width.compute",
        "poset.linear_extensions.count",
        "poset.mobius_function.compute",
    )
    assert "poset" in runtime.portfolio.domain_bundles
    catalog_ids = {
        descriptor.capability_id
        for descriptor in runtime.core.capabilities.catalog().capabilities
    }
    assert set(ids).issubset(catalog_ids)


def test_antichain_chain_decomposition_intent_finds_width(runtime) -> None:
    discovered = runtime.core.capabilities.discover(
        CapabilityDiscoveryRequest(
            query="maximum antichain and minimum chain decomposition of a finite poset",
            limit=5,
        )
    )
    assert discovered.matches[0].capability_id == "poset.width.compute"
    assert discovered.matches[0].lexical_fit == "STRONG_CANDIDATE"


def test_materialization_is_canonical_complete_and_artifact_backed(runtime) -> None:
    result = runtime.core.capabilities.invoke(
        CapabilityRequest(
            capability_id="poset.finite.materialize",
            input=_DIAMOND,
        )
    )
    poset = result.output["result"]["poset"]
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
    assert len(result.artifact_uris) == 2
    assert (
        runtime.core.store.get(result.output["result_uri"]).payload
        == (result.output["result"])
    )


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
    runtime,
    presentation: dict[str, Any],
    expected_width: int,
    expected_count: int,
) -> None:
    poset = _materialize(runtime, presentation)
    width = _invoke(runtime, "poset.width.compute", poset).output["result"]
    count = _invoke(
        runtime,
        "poset.linear_extensions.count",
        poset,
    ).output["result"]
    assert width["width"] == expected_width
    assert len(width["maximum_antichain"]) == expected_width
    assert len(width["minimum_chain_cover"]) == expected_width
    assert count["count"] == expected_count
    assert count["state_count"] == len(count["states"])
    assert count["completeness"] == "COMPLETE"


def test_width_dual_certificate_covers_diamond_exactly(runtime) -> None:
    poset = _materialize(runtime, _DIAMOND)
    result = _invoke(runtime, "poset.width.compute", poset).output["result"]
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


def test_mobius_complete_and_selected_scopes_are_distinct(runtime) -> None:
    poset = _materialize(runtime, _DIAMOND)
    complete = _invoke(
        runtime,
        "poset.mobius_function.compute",
        poset,
        scope="COMPLETE_MATRIX",
        intervals=[],
    ).output["result"]
    selected = _invoke(
        runtime,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "0", "upper": "1"}],
    ).output["result"]
    values = {
        (item["lower"], item["upper"]): item["value"] for item in complete["values"]
    }
    assert values[("0", "1")] == 1
    assert complete["completeness"] == "COMPLETE_MATRIX"
    assert selected["completeness"] == "SELECTED_INTERVALS"
    assert selected["intervals"] == [{"lower": "0", "upper": "1"}]
    assert len(selected["values"]) == 1


def test_mobius_ledger_is_canonical_across_branching_topological_orders(
    runtime,
) -> None:
    poset = _materialize(
        runtime,
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
    result = _invoke(
        runtime,
        "poset.mobius_function.compute",
        poset,
        scope="SELECTED_INTERVALS",
        intervals=[{"lower": "a", "upper": "e"}],
    ).output["result"]
    value = result["values"][0]
    assert value["value"] == 1
    assert [item["intermediate"] for item in value["recurrence_contributions"]] == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_non_graded_poset_omits_ranks(runtime) -> None:
    poset = _materialize(
        runtime,
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


def test_relabeling_preserves_scalar_poset_outcomes(runtime) -> None:
    first = _materialize(runtime, _DIAMOND)
    second = _materialize(
        runtime,
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
    assert (
        _invoke(runtime, "poset.width.compute", first).output["result"]["width"]
        == (_invoke(runtime, "poset.width.compute", second).output["result"]["width"])
    )
    assert (
        _invoke(runtime, "poset.linear_extensions.count", first).output["result"][
            "count"
        ]
        == _invoke(runtime, "poset.linear_extensions.count", second).output["result"][
            "count"
        ]
    )


def test_invalid_poset_request_fails_before_artifact_writes(runtime) -> None:
    result = runtime.core.capabilities.invoke(
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
