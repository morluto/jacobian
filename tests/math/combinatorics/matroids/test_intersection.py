from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids import LinearMatroid
from jacobian.math.combinatorics.matroids._models import (
    MatroidIntersectionRequest,
    MatroidIntersectionResult,
)
from jacobian.math.combinatorics.matroids.intersection import matroid_intersection
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _zero_matroid(columns: int) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=((0,) * columns,), columns=columns)
    )


def test_intersection_uses_bounded_exchange_work_at_twenty_elements() -> None:
    result = matroid_intersection(_zero_matroid(20), _zero_matroid(20))
    assert result.common_independent == ()
    assert result.witness.equality == 0


def test_intersection_schema_discloses_derived_work_envelope() -> None:
    schema = MatroidIntersectionRequest.model_json_schema()
    description = schema["description"]
    assert "256" in description
    assert "50,000,000" in description


def test_serialized_result_does_not_replay_finite_field_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.matrices.finite_fields.linear_algebra as linear_algebra

    result = matroid_intersection(_zero_matroid(3), _zero_matroid(3))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("result validation must not replay matrix rank")

    monkeypatch.setattr(linear_algebra, "rank", fail)

    assert (
        MatroidIntersectionResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_intersection_native_rejects_non_carriers() -> None:
    with pytest.raises(OperationDomainValidationError):
        matroid_intersection(None, _zero_matroid(1))  # type: ignore[arg-type]
