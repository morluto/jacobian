"""Native relabelling distinguishes resource limits from malformed axes."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids.delta.relabel import (
    MAX_DELTA_RELABEL_GROUND,
    DeltaMatroidRelabelRequest,
    relabel,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_MEMBERSHIPS,
    FiniteDeltaMatroid,
)


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
def test_result_binding_rejects_wrong_ground_or_feasible_transport() -> None:
    from pydantic import ValidationError
    from jacobian.math.combinatorics.matroids.delta.relabel import (
        DeltaMatroidRelabelling,
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
                    "feasible": [[], [0]],
