from __future__ import annotations

from itertools import combinations, product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids import LinearMatroid
from jacobian.math.combinatorics.matroids._models import (
    MatroidIntersectionRequest,
    MatroidIntersectionResult,
)
from jacobian.math.combinatorics.matroids.intersection import (
    matroid_intersection,
    replay_intersection_result,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _zero_matroid(columns: int) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=((0,) * columns,), columns=columns)
    )


def test_intersection_uses_bounded_exchange_work_at_twenty_elements() -> None:
    result = matroid_intersection(_zero_matroid(20), _zero_matroid(20))
    assert result.common_independent == ()
    assert result.witness.equality == 0


def test_intersection_accepts_last_rank_work_boundary() -> None:
    # For one-row representations, 83 elements sit just below the admitted
    # 50M rank-work envelope; the next size exceeds it.
    result = matroid_intersection(_zero_matroid(83), _zero_matroid(83))
    assert result.common_independent == ()


def test_intersection_rejects_rank_work_before_calling_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    def unexpected_rank(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank oracle ran before aggregate admission")

    monkeypatch.setattr(intersection, "pf_rank", unexpected_rank)
    with pytest.raises(OperationResourceAdmissionError) as error:
        matroid_intersection(_zero_matroid(84), _zero_matroid(84))
    assert error.value.errors()[0]["type"] == "matroid.intersection.work_bound"


def test_intersection_rejects_large_retained_axis_before_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    huge_label = "x" * 700_000
    large_axis = LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=((0,),), columns=1),
        ground_labels=(huge_label,),
    )

    def unexpected_rank(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank oracle ran before output admission")

    monkeypatch.setattr(intersection, "pf_rank", unexpected_rank)
    with pytest.raises(OperationResourceAdmissionError) as error:
        matroid_intersection(large_axis, large_axis)
    assert error.value.errors()[0]["type"] == "matroid.intersection.work_bound"


def test_intersection_admits_output_bound_at_its_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    matroid = _zero_matroid(2)
    monkeypatch.setattr(
        intersection,
        "MAX_INTERSECTION_OUTPUT_BYTES",
        intersection._intersection_output_bound_bytes(matroid, matroid),
    )
    result = matroid_intersection(matroid, matroid)
    assert result.common_independent == ()


def test_intersection_schema_discloses_derived_work_envelope() -> None:
    schema = MatroidIntersectionRequest.model_json_schema()
    description = schema["description"]
    assert "256" in description
    assert "50,000,000" in description


def _small_matroids() -> tuple[LinearMatroid, LinearMatroid]:
    first = LinearMatroid(
        matrix=PrimeFieldMatrix(
            prime=2,
            entries=((1, 0, 1, 1), (0, 1, 1, 0), (0, 0, 0, 1)),
            columns=4,
        ),
        ground_labels=("a", "b", "c", "d"),
    )
    second = LinearMatroid(
        matrix=PrimeFieldMatrix(
            prime=2,
            entries=((1, 0, 0, 1), (0, 1, 1, 1)),
            columns=4,
        ),
        ground_labels=("a", "b", "c", "d"),
    )
    return first, second


def _independent_by_coefficients(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Independent small oracle: enumerate every coefficient vector over GF(p)."""

    prime = matroid.matrix.prime
    for coefficients in product(range(prime), repeat=len(subset)):
        if not any(coefficients):
            continue
        if all(
            sum(
                coefficients[index] * row[column] for index, column in enumerate(subset)
            )
            % prime
            == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def test_intersection_cardinality_matches_exhaustive_common_independent_sets() -> None:
    first, second = _small_matroids()
    result = matroid_intersection(first, second)
    feasible_sizes = [
        len(subset)
        for size in range(first.ground_size + 1)
        for subset in combinations(range(first.ground_size), size)
        if _independent_by_coefficients(first, subset)
        and _independent_by_coefficients(second, subset)
    ]

    assert result.cardinality == max(feasible_sizes) == 2
    assert result.rank_first_common == result.rank_second_common == result.cardinality
    assert _independent_by_coefficients(first, result.common_independent)
    assert _independent_by_coefficients(second, result.common_independent)


def test_serialized_result_decoding_does_not_replay_computed_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    first, second = _small_matroids()
    result = matroid_intersection(first, second)
    calls = 0
    original_rank = intersection.pf_rank

    def count_rank(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return original_rank(*args, **kwargs)

    monkeypatch.setattr(intersection, "pf_rank", count_rank)

    assert (
        MatroidIntersectionResult.model_validate_json(result.model_dump_json())
        == result
    )
    assert calls == 0


def test_forged_common_set_and_minmax_claims_require_explicit_replay() -> None:
    first, second = _small_matroids()
    result = matroid_intersection(first, second)

    forged_common = result.model_dump(mode="python")
    forged_common["common_independent"] = (1, 2)
    forged_common["rank_first_common"] = 2
    forged_common["rank_second_common"] = 2
    decoded_common = MatroidIntersectionResult.model_validate(forged_common)
    with pytest.raises(ValueError, match="common-set rank claims"):
        replay_intersection_result(decoded_common)

    forged_witness = result.model_dump(mode="python")
    forged_witness["witness"] = {
        "subset": (0, 1, 2, 3),
        "rank_first": 2,
        "rank_second_complement": 0,
        "equality": 2,
    }
    decoded_witness = MatroidIntersectionResult.model_validate(forged_witness)
    with pytest.raises(ValueError, match="min-max rank claims"):
        replay_intersection_result(decoded_witness)


def test_intersection_native_rejects_non_carriers() -> None:
    with pytest.raises(OperationDomainValidationError):
        matroid_intersection(None, _zero_matroid(1))  # type: ignore[arg-type]
