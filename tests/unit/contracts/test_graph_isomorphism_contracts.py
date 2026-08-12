from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_isomorphism import (
    GraphIsomorphismVerifyOutput,
    GraphIsomorphismViolation,
)

_ARTIFACT_URI = "artifact://sha256/" + "a" * 64
_CHECKER_URI = "checker://sha256/" + "b" * 64


def _output(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "is_isomorphism": None,
        "conclusion": "UNKNOWN",
        "left_graph_uri": _ARTIFACT_URI,
        "right_graph_uri": _ARTIFACT_URI,
        "graph_pair_uri": _ARTIFACT_URI,
        "mapping_uri": _ARTIFACT_URI,
        "claim_uri": _ARTIFACT_URI,
        "certificate_uri": _ARTIFACT_URI,
        "checker_id": _CHECKER_URI,
        "coverage": "UNKNOWN",
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    "updates",
    [
        {"is_isomorphism": True},
        {"conclusion": "TRUE"},
        {"coverage": "EXHAUSTIVE"},
        {"verification_record_uri": _ARTIFACT_URI},
    ],
)
def test_isomorphism_output_rejects_inconsistent_truth_or_verification(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        GraphIsomorphismVerifyOutput.model_validate(_output(**updates))


def test_isomorphism_output_accepts_decisive_exhaustive_verification() -> None:
    output = GraphIsomorphismVerifyOutput.model_validate(
        _output(
            is_isomorphism=False,
            conclusion="FALSE",
            coverage="EXHAUSTIVE",
            verification_record_uri=_ARTIFACT_URI,
            first_violation={
                "kind": "ADJACENCY_MISMATCH",
                "source_vertices": ["a", "b"],
                "mapped_vertices": ["x", "y"],
                "source_adjacent": True,
                "target_adjacent": False,
            },
        )
    )
    assert output.is_isomorphism is False


def test_negative_isomorphism_requires_a_typed_first_violation() -> None:
    with pytest.raises(ValidationError, match="first violation"):
        GraphIsomorphismVerifyOutput.model_validate(
            _output(
                is_isomorphism=False,
                conclusion="FALSE",
                coverage="EXHAUSTIVE",
                verification_record_uri=_ARTIFACT_URI,
            )
        )


def test_target_bijection_violation_can_identify_only_a_missing_target() -> None:
    violation = GraphIsomorphismViolation(
        kind="TARGET_BIJECTION_MISMATCH",
        mapped_vertex="isolated-target",
    )

    assert violation.vertex is None
    assert violation.mapped_vertex == "isolated-target"
