"""Dispatch boundaries for exact coherent-configuration analysis."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def _complete_graph_k3() -> dict[str, object]:
    return {
        "points": ["a", "b", "c"],
        "relation_ids": ["diagonal", "edge"],
        "relation_matrix": [
            ["diagonal", "edge", "edge"],
            ["edge", "diagonal", "edge"],
            ["edge", "edge", "diagonal"],
        ],
    }


def _escaped_thin_four_point_configuration(escaped_character: str) -> dict[str, object]:
    relation_ids = [f"{index:02d}" + escaped_character * 30 for index in range(16)]
    return {
        "points": ["a", "b", "c", "d"],
        "relation_ids": relation_ids,
        "relation_matrix": [
            [relation_ids[4 * left + right] for right in range(4)] for left in range(4)
        ],
    }


def test_dispatch_returns_the_typed_source_bound_result() -> None:
    result = invoke_operation(
        "coherent_configuration.analyze.compute",
        {"configuration": _complete_graph_k3()},
        Catalog.open(),
    )

    assert result.output is not None
    assert result.output["status"] == "COHERENT_CONFIGURATION"
    assert len(result.output["intersection_numbers"]) == 8


def test_dispatch_rejects_malformed_pair_partition() -> None:
    payload = {"configuration": _complete_graph_k3()}
    payload["configuration"]["relation_matrix"] = [["diagonal"]]

    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(
            "coherent_configuration.analyze.compute", payload, Catalog.open()
        )
    assert "square on points" in str(error.value.errors())


def test_dispatch_rejects_oversized_utf8_relation_label() -> None:
    relation_id = "😀" * 9

    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(
            "coherent_configuration.analyze.compute",
            {
                "configuration": {
                    "points": ["a"],
                    "relation_ids": [relation_id],
                    "relation_matrix": [[relation_id]],
                }
            },
            Catalog.open(),
        )
    assert "relation_ids must not exceed" in str(error.value.errors())


@pytest.mark.parametrize("escaped_character", ('"', "\x00"), ids=("quote", "nul"))
def test_dispatch_rejects_escaped_result_over_budget(
    escaped_character: str,
) -> None:
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(
            "coherent_configuration.analyze.compute",
            {
                "configuration": _escaped_thin_four_point_configuration(
                    escaped_character
                )
            },
            Catalog.open(),
        )
    assert "result exceeds the byte budget" in str(error.value.errors())
