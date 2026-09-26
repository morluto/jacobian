"""Native relabelling admission checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.delta.relabel import (
    MAX_DELTA_RELABEL_GROUND,
    DeltaMatroidRelabelRequest,
    relabel,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
)


def test_oversized_target_is_rejected_before_scanning_labels() -> None:
    class NonIterableTuple(tuple[str, ...]):
        def __iter__(self):
            raise AssertionError("oversized target labels must not be scanned")

    target = NonIterableTuple("" for _ in range(MAX_DELTA_RELABEL_GROUND + 1))
    source = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))

    with pytest.raises(OperationDomainValidationError) as error:
        relabel(source, target, (0,))

    assert error.value.errors()[0]["type"] == "delta_matroid.relabel_ground_limit"


def test_relabel_request_preflights_oversized_source_ground() -> None:
    payload = {
        "delta_matroid": {
            "ground": [f"e{index}" for index in range(MAX_DELTA_RELABEL_GROUND + 1)],
            "feasible": [[]],
        },
        "target_ground": [],
        "target_to_source": [],
    }
    with pytest.raises(ValidationError) as error:
        DeltaMatroidRelabelRequest.model_validate(payload)
    assert error.value.errors()[0]["type"] == "delta_matroid.relabel_ground_limit"


def test_relabel_request_preflights_source_memberships() -> None:
    payload = {
        "delta_matroid": {
            "ground": ["e0"],
            "feasible": [[0] * (MAX_DELTA_MEMBERSHIPS + 1)],
        },
        "target_ground": ["e0"],
        "target_to_source": [0],
    }
    with pytest.raises(ValidationError) as error:
        DeltaMatroidRelabelRequest.model_validate(payload)
    assert error.value.errors()[0]["type"] == (
        "delta_matroid.relabel_source_membership_limit"
    )
