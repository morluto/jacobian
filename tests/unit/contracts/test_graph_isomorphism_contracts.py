from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.graph_isomorphism import GraphIsomorphismVerifyOutput

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
def test_isomorphism_output_rejects_inconsistent_truth_or_assurance(
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
        )
    )
    assert output.is_isomorphism is False
