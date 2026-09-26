"""Native relabelling admission checks."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids.delta.relabel import (
    MAX_DELTA_RELABEL_GROUND,
    relabel,
)
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def test_oversized_target_is_rejected_before_scanning_labels() -> None:
    class NonIterableTuple(tuple[str, ...]):
        def __iter__(self):
            raise AssertionError("oversized target labels must not be scanned")

    target = NonIterableTuple("" for _ in range(MAX_DELTA_RELABEL_GROUND + 1))
    source = FiniteDeltaMatroid(ground=("a",), feasible=((), (0,)))

    with pytest.raises(OperationDomainValidationError) as error:
        relabel(source, target, (0,))

    assert error.value.errors()[0]["type"] == "delta_matroid.relabel_ground_limit"
