from __future__ import annotations

from itertools import combinations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidRankMultiplier,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionRankCertificateResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    verify_weighted_intersection_rank_certificate,
    weighted_intersection_certificate,
    weighted_intersection_rank_certificate,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.matrices.finite_fields.linear_algebra import rank as field_rank


def _matroid(
    rows: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def _weights(labels: tuple[str, ...], values: tuple[int, ...]) -> MatroidWeightFunction:
    return MatroidWeightFunction(ground_axis=labels, values=values)


def _rank_one_arguments() -> tuple[
    LinearMatroid,
    LinearMatroid,
    MatroidWeightFunction,
    tuple[int, ...],
    tuple[MatroidRankMultiplier, ...],
    tuple[MatroidRankMultiplier, ...],
]:
    labels = ("a", "b")
    first = _matroid(((1, 1),), labels)
    second = _matroid(((1, 1),), labels)
    return (
        first,
        second,
        _weights(labels, (5, 3)),
        (0,),
        (MatroidRankMultiplier(subset=(0, 1), multiplier=5),),
        (),
    )


def test_rank_dual_round_trips_and_composes_with_weight_split_checker() -> None:
    first, second, weight_function, candidate, first_terms, second_terms = (
        _rank_one_arguments()
    )
    result = weighted_intersection_rank_certificate(
        first, second, weight_function, candidate, first_terms, second_terms
    )

    assert result.total_weight == 5
    oracle = max(
        sum(weight_function.values[index] for index in feasible)
        for size in range(3)
        for feasible in combinations(range(2), size)
        if _subset_rank(first, feasible)
        == _subset_rank(second, feasible)
        == len(feasible)
    )
    assert oracle == result.total_weight
    assert result.first_split.values == (5, 3)
    assert result.second_split.values == (0, 0)
    decoded = MatroidWeightedIntersectionRankCertificateResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_weighted_intersection_rank_certificate(decoded)
    split_result = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=decoded.first,
            second=decoded.second,
            weight_function=decoded.weight_function,
            common_independent=decoded.common_independent,
            first_split=decoded.first_split,
            second_split=decoded.second_split,
        )
    )
    assert split_result.total_weight == decoded.total_weight


def test_empty_rank_certificate_rejects_composite_field_characteristic() -> None:
    labels = ("a", "b")
    source = LinearMatroid(
        matrix=PrimeFieldMatrix(prime=4, entries=((1, 0), (0, 1)), columns=2),
        ground_labels=labels,
    )
    with pytest.raises(OperationDomainValidationError):
        weighted_intersection_rank_certificate(
            source,
            source,
            _weights(labels, (-2, -5)),
            (),
            (),
            (),
        )


def test_all_negative_objective_certifies_empty_optimum_and_round_trips() -> None:
    labels = ("a", "b")
    free = _matroid(((1, 0), (0, 1)), labels)
    result = weighted_intersection_rank_certificate(
        free,
        free,
        _weights(labels, (-2, -5)),
        (),
        (),
        (),
    )

    assert result.total_weight == 0
    assert result.first_split.values == (0, 0)
    assert result.second_split.values == (-2, -5)
    decoded = MatroidWeightedIntersectionRankCertificateResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_weighted_intersection_rank_certificate(decoded)
    composed = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=decoded.first,
            second=decoded.second,
            weight_function=decoded.weight_function,
            common_independent=(),
            first_split=decoded.first_split,
            second_split=decoded.second_split,
        )
    )
    assert composed.total_weight == 0


def test_zero_rank_loop_term_covers_positive_loop_weight() -> None:
    labels = ("a", "loop")
    first = _matroid(((1, 0),), labels)
    second = _matroid(((1, 0), (0, 1)), labels)
    result = weighted_intersection_rank_certificate(
        first,
        second,
        _weights(labels, (4, 3)),
        (0,),
        (
            MatroidRankMultiplier(subset=(1,), multiplier=3),
            MatroidRankMultiplier(subset=(0, 1), multiplier=4),
        ),
        (),
    )

    assert result.total_weight == 4
    assert result.first_split.values == (4, 3)
    assert verify_weighted_intersection_rank_certificate(result)
    composed = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=result.first,
            second=result.second,
            weight_function=result.weight_function,
            common_independent=result.common_independent,
            first_split=result.first_split,
            second_split=result.second_split,
        )
    )
    assert composed.total_weight == result.total_weight


def test_disjoint_singleton_supports_certify_empty_optimum_with_two_loops() -> None:
    labels = ("a", "b")
    first = _matroid(((1, 0),), labels)
    second = _matroid(((0, 1),), labels)
    weights = _weights(labels, (1, 1))
    rank_result = weighted_intersection_rank_certificate(
        first,
        second,
        weights,
        (),
        (MatroidRankMultiplier(subset=(1,), multiplier=1),),
        (MatroidRankMultiplier(subset=(0,), multiplier=1),),
    )

    assert rank_result.total_weight == 0
    assert rank_result.first_split.values == (0, 1)
    assert rank_result.second_split.values == (1, 0)
    assert verify_weighted_intersection_rank_certificate(rank_result)
    composed = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=first,
            second=second,
            weight_function=weights,
            common_independent=(),
            first_split=rank_result.first_split,
            second_split=rank_result.second_split,
        )
    )
    assert composed.total_weight == 0
    with pytest.raises(OperationDomainValidationError, match="does not certify"):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=first,
                second=second,
                weight_function=weights,
                common_independent=(),
                first_split=_weights(labels, (1, 1)),
                second_split=_weights(labels, (0, 0)),
            )
        )


def test_owner_manifest_example_runs_through_declared_types() -> None:
    from jacobian.math.combinatorics.matroids._tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "matroid.intersection.weighted_rank_certificate.check"
    )
    request = tool.request_type.model_validate(tool.examples[0].input)
    result = tool.run(request)

    assert result.total_weight == 5
    assert result.first_split.values == (5, 3)


def test_forged_dual_value_fails_after_serialization() -> None:
    result = weighted_intersection_rank_certificate(*_rank_one_arguments())
    raw = result.model_dump(mode="python")
    raw["first_rank_terms"][0]["multiplier"] = 4
    forged = MatroidWeightedIntersectionRankCertificateResult.model_validate(raw)

    assert not verify_weighted_intersection_rank_certificate(forged)


def test_nonoptimal_candidate_rejected_by_dual_objective_equality() -> None:
    first, second, weight_function, _, first_terms, second_terms = _rank_one_arguments()
    with pytest.raises(OperationDomainValidationError, match="dual objective"):
        weighted_intersection_rank_certificate(
            first, second, weight_function, (1,), first_terms, second_terms
        )


def test_admission_rejects_many_expensive_ranks_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    n = 256
    labels = tuple(f"e{i}" for i in range(n))
    rows = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    source = _matroid(rows, labels)
    all_indices = tuple(range(n))
    prefix = tuple(range(n - 1))
    family = (
        MatroidRankMultiplier(subset=prefix, multiplier=1),
        MatroidRankMultiplier(subset=all_indices, multiplier=1),
    )

    def unexpected(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank kernel ran before aggregate admission")

    monkeypatch.setattr(intersection, "_weighted_rank", unexpected)
    with pytest.raises(OperationResourceAdmissionError, match="rank-dual certificate"):
        weighted_intersection_rank_certificate(
            source,
            source,
            _weights(labels, (0,) * n),
            (),
            family,
            family,
        )


def test_256_element_request_is_accepted_inside_rank_work_envelope() -> None:
    n = 256
    rank_bound = 128
    labels = tuple(f"e{i}" for i in range(n))
    rows = tuple(tuple(int(i == j) for j in range(n)) for i in range(rank_bound))
    source = _matroid(rows, labels)
    candidate = tuple(range(rank_bound))
    result = weighted_intersection_rank_certificate(
        source,
        source,
        _weights(labels, (1,) * rank_bound + (0,) * rank_bound),
        candidate,
        (MatroidRankMultiplier(subset=tuple(range(n)), multiplier=1),),
        (),
    )

    assert result.total_weight == rank_bound
    assert result.common_independent == candidate


def test_output_size_rejects_long_repeated_labels_before_rank_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    labels = ("x" * 300_000,)
    source = _matroid(((1,),), labels)

    def unexpected(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank kernel ran before output-size admission")

    monkeypatch.setattr(intersection, "_weighted_rank", unexpected)
    with pytest.raises(OperationResourceAdmissionError) as error:
        weighted_intersection_rank_certificate(
            source,
            source,
            _weights(labels, (1,)),
            (),
            (),
            (),
        )

    assert error.value.errors()[0]["type"] == (
        "matroid.weighted_intersection.rank_dual.work_bound"
    )


def _subset_rank(matroid: LinearMatroid, subset: tuple[int, ...]) -> int:
    if not subset:
        return 0
    entries = tuple(
        tuple(row[index] for index in subset) for row in matroid.matrix.entries
    )
    return field_rank(
        PrimeFieldMatrix(
            prime=matroid.matrix.prime,
            entries=entries,
            columns=len(subset),
        )
    )
