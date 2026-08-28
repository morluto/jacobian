"""Exact, adversarial, and boundary tests for coherent configurations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.designs.coherent_configurations._models import (
    CoherentConfigurationAnalyzeRequest,
    CoherentConfigurationAnalyzeResult,
)
from jacobian.math.combinatorics.designs.coherent_configurations._operations import (
    compute_analyze,
)
from jacobian.math.combinatorics.designs.coherent_configurations._tools import TOOLS
from jacobian.math.combinatorics.designs.coherent_configurations.values import (
    MAX_POINT_LABEL_BYTES,
    MAX_RELATION_ID_BYTES,
)


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


def _path_four_relation_partition() -> dict[str, object]:
    return {
        "points": ["a", "b", "c", "d"],
        "relation_ids": ["diagonal", "edge", "nonedge"],
        "relation_matrix": [
            ["diagonal", "edge", "nonedge", "nonedge"],
            ["edge", "diagonal", "edge", "nonedge"],
            ["nonedge", "edge", "diagonal", "edge"],
            ["nonedge", "nonedge", "edge", "diagonal"],
        ],
    }


def _request(configuration: dict[str, object]) -> CoherentConfigurationAnalyzeRequest:
    return CoherentConfigurationAnalyzeRequest.model_validate(
        {"configuration": configuration}
    )


def _thin_four_point_configuration(
    *, max_relation_id_bytes: bool = False
) -> dict[str, object]:
    points = ["a", "b", "c", "d"]
    relation_ids = [
        f"r{left}{right}".ljust(32, "x") if max_relation_id_bytes else f"r{left}{right}"
        for left in range(4)
        for right in range(4)
    ]
    return {
        "points": points,
        "relation_ids": relation_ids,
        "relation_matrix": [
            [
                f"r{left}{right}".ljust(32, "x")
                if max_relation_id_bytes
                else f"r{left}{right}"
                for right in range(4)
            ]
            for left in range(4)
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


def _cyclic_twelve_configuration() -> dict[str, object]:
    points = [f"p{index:02d}" for index in range(12)]
    relation_ids = [f"d{index:02d}" for index in range(12)]
    return {
        "points": points,
        "relation_ids": relation_ids,
        "relation_matrix": [
            [f"d{(right - left) % 12:02d}" for right in range(12)] for left in range(12)
        ],
    }


def _unicode_thin_four_point_configuration() -> dict[str, object]:
    relation_ids = ["😀" * 7 + f"{index:02d}" for index in range(16)]
    return {
        "points": ["😀" * 15 + f"{index:02d}" for index in range(4)],
        "relation_ids": relation_ids,
        "relation_matrix": [
            [relation_ids[4 * left + right] for right in range(4)] for left in range(4)
        ],
    }


def test_catalog_declares_one_foundational_operation() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "coherent_configuration.analyze.compute"
    }


def test_complete_graph_rank_two_is_a_coherent_configuration() -> None:
    result = compute_analyze(_request(_complete_graph_k3()))

    assert result.status == "COHERENT_CONFIGURATION"
    assert result.coherent_configuration is not None
    assert result.fibers[0].relation_id == "diagonal"
    assert result.fibers[0].points == ("a", "b", "c")
    assert tuple(
        (entry.relation_id, entry.transpose_relation_id)
        for entry in result.transpose_map
    ) == (("diagonal", "diagonal"), ("edge", "edge"))
    assert len(result.intersection_numbers) == 8
    assert result.obstruction is None


def test_thin_configuration_with_four_fibers_is_coherent() -> None:
    result = compute_analyze(_request(_thin_four_point_configuration()))

    assert result.status == "COHERENT_CONFIGURATION"
    assert result.coherent_configuration is not None
    assert tuple(fiber.points for fiber in result.fibers) == (
        ("a",),
        ("b",),
        ("c",),
        ("d",),
    )
    assert len(result.intersection_numbers) == 16**3


def test_diagonal_relation_mixed_obstruction_is_concrete_and_deterministic() -> None:
    result = compute_analyze(
        _request(
            {
                "points": ["a", "b"],
                "relation_ids": ["mixed", "other"],
                "relation_matrix": [["mixed", "mixed"], ["other", "other"]],
            }
        )
    )

    assert result.status == "NOT_COHERENT"
    assert result.coherent_configuration is None
    assert result.obstruction is not None
    assert result.obstruction.kind == "DIAGONAL_RELATION_MIXED"
    assert result.obstruction.relation_id == "mixed"
    assert result.obstruction.first_pair == ("a", "a")
    assert result.obstruction.second_pair == ("a", "b")


def test_transpose_obstruction_is_concrete_and_deterministic() -> None:
    result = compute_analyze(
        _request(
            {
                "points": ["a", "b", "c"],
                "relation_ids": ["diagonal", "r", "s", "u"],
                "relation_matrix": [
                    ["diagonal", "r", "u"],
                    ["s", "diagonal", "u"],
                    ["s", "u", "diagonal"],
                ],
            }
        )
    )

    assert result.status == "NOT_COHERENT"
    assert result.obstruction is not None
    assert result.obstruction.kind == "TRANSPOSE_RELATION_MISMATCH"
    assert result.obstruction.relation_id == "r"
    assert result.obstruction.transpose_relation_id == "s"
    assert result.obstruction.first_pair == ("c", "a")


def test_path_partition_reports_nonconstant_intersection_numbers() -> None:
    result = compute_analyze(_request(_path_four_relation_partition()))

    assert result.status == "NOT_COHERENT"
    assert result.obstruction is not None
    assert result.obstruction.kind == "NONCONSTANT_INTERSECTION_NUMBER"
    assert result.obstruction.left_relation_id == "edge"
    assert result.obstruction.right_relation_id == "edge"
    assert result.obstruction.target_relation_id == "diagonal"
    assert result.obstruction.first_pair == ("a", "a")
    assert result.obstruction.second_pair == ("b", "b")
    assert result.obstruction.first_count == 1
    assert result.obstruction.second_count == 2


def test_malformed_partition_is_request_invalid_not_a_negative_conclusion() -> None:
    payload = _complete_graph_k3()
    payload["relation_matrix"] = [["diagonal", "edge"], ["edge", "diagonal"]]

    with pytest.raises(ValidationError) as exc_info:
        _request(payload)
    assert exc_info.value.errors()[0]["type"] == "coherent_configuration.matrix_square"


def test_relation_order_and_matrix_entries_must_be_canonical_and_complete() -> None:
    payload = _complete_graph_k3()
    payload["relation_ids"] = ["edge", "diagonal"]
    with pytest.raises(ValidationError) as exc_info:
        _request(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "coherent_configuration.relation_ids_canonical"
    )

    payload = _complete_graph_k3()
    payload["relation_ids"] = ["diagonal", "edge", "unused"]
    with pytest.raises(ValidationError) as exc_info:
        _request(payload)
    assert (
        exc_info.value.errors()[0]["type"] == "coherent_configuration.relation_ids_used"
    )


def test_utf8_label_byte_bounds_apply_to_direct_request() -> None:
    point = "😀" * (MAX_POINT_LABEL_BYTES // len("😀".encode()))
    relation_id = "😀" * (MAX_RELATION_ID_BYTES // len("😀".encode()))
    request = _request(
        {
            "points": [point],
            "relation_ids": [relation_id],
            "relation_matrix": [[relation_id]],
        }
    )

    assert compute_analyze(request).status == "COHERENT_CONFIGURATION"

    oversized_point = point + "😀"
    with pytest.raises(ValidationError) as exc_info:
        _request(
            {
                "points": [oversized_point],
                "relation_ids": ["diagonal"],
                "relation_matrix": [["diagonal"]],
            }
        )
    assert (
        exc_info.value.errors()[0]["type"] == "coherent_configuration.point_label_bytes"
    )

    oversized_relation_id = relation_id + "😀"
    with pytest.raises(ValidationError) as exc_info:
        _request(
            {
                "points": ["a"],
                "relation_ids": [oversized_relation_id],
                "relation_matrix": [[oversized_relation_id]],
            }
        )
    assert (
        exc_info.value.errors()[0]["type"] == "coherent_configuration.relation_id_bytes"
    )


@pytest.mark.parametrize("escaped_character", ('"', "\x00"), ids=("quote", "nul"))
def test_escaped_result_over_budget_is_rejected_by_native_admission(
    escaped_character: str,
) -> None:
    payload = _escaped_thin_four_point_configuration(escaped_character)
    request = _request(payload)

    with pytest.raises(ValueError, match="result exceeds the byte budget"):
        compute_analyze(request)


def test_unicode_label_tensor_stays_inside_admitted_result_envelope() -> None:
    result = compute_analyze(_request(_unicode_thin_four_point_configuration()))

    assert result.status == "COHERENT_CONFIGURATION"
    assert len(result.intersection_numbers) == 4_096
    assert len(result.model_dump_json().encode("utf-8")) <= 1_048_576


def test_maximum_relation_tensor_stays_inside_admitted_result_envelope() -> None:
    result = compute_analyze(
        _request(_thin_four_point_configuration(max_relation_id_bytes=True))
    )

    assert result.status == "COHERENT_CONFIGURATION"
    assert len(result.intersection_numbers) == 4_096
    assert len(result.model_dump_json().encode("utf-8")) <= 1_048_576


def test_maximum_point_count_translation_configuration_is_admitted() -> None:
    result = compute_analyze(_request(_cyclic_twelve_configuration()))

    assert result.status == "COHERENT_CONFIGURATION"
    assert len(result.intersection_numbers) == 12**3


def test_produced_results_round_trip() -> None:
    """Both trusted outcome shapes remain valid canonical wire values."""

    coherent = compute_analyze(_request(_complete_graph_k3()))
    noncoherent = compute_analyze(_request(_path_four_relation_partition()))

    restored_coherent = CoherentConfigurationAnalyzeResult.model_validate(
        coherent.model_dump(mode="json")
    )
    restored_noncoherent = CoherentConfigurationAnalyzeResult.model_validate(
        noncoherent.model_dump(mode="json")
    )
    assert restored_coherent == coherent
    assert restored_noncoherent == noncoherent


def test_over_relation_bound_is_rejected_before_analysis() -> None:
    payload = _thin_four_point_configuration()
    payload["relation_ids"] = [f"r{index:02d}" for index in range(17)]

    with pytest.raises(ValidationError):
        _request(payload)
