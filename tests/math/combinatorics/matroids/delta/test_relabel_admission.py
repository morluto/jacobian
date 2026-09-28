"""Native relabelling distinguishes resource limits from malformed axes."""

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta.relabel import relabel
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def _source() -> FiniteDeltaMatroid:
    return FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,), (0, 1), (1,)))


def test_aggregate_target_label_bytes_are_admitted_at_the_boundary() -> None:
    source = _source()
    first = "λ" * 512
    second = "β" * 512
    result = relabel(source, (first, second), (0, 1))
    assert result.relabelled.ground == (first, second)

    with pytest.raises(OperationResourceAdmissionError) as error:
        relabel(source, (first + "λ", second), (0, 1))
    assert error.value.errors()[0]["type"] == "delta_matroid.relabel_target_bytes"


def test_invalid_utf8_target_remains_a_domain_error() -> None:
    with pytest.raises(OperationDomainValidationError) as error:
        relabel(_source(), ("\ud800", "b"), (0, 1))
    assert error.value.errors()[0]["type"] == "delta_matroid.relabel_request"


def test_result_binding_rejects_wrong_ground_or_feasible_transport() -> None:
    from pydantic import ValidationError

    from jacobian.math.combinatorics.matroids.delta.relabel import (
        DeltaMatroidRelabelling,
    )

    source = FiniteDeltaMatroid(ground=("a", "b"), feasible=((), (0,)))
    valid = relabel(source, ("B", "A"), (1, 0))
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelling.model_validate(
            {
                **valid.model_dump(),
                "relabelled": {
                    **valid.relabelled.model_dump(),
                    "ground": ["A", "B"],
                },
            }
        )
    with pytest.raises(ValidationError):
        DeltaMatroidRelabelling.model_validate(
            {
                **valid.model_dump(),
                "relabelled": {
                    **valid.relabelled.model_dump(),
                    "feasible": [[], [0]],
                },
            }
        )
