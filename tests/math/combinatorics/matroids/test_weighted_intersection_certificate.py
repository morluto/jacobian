from __future__ import annotations

from itertools import combinations, product

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    verify_weighted_intersection_result,
    weighted_intersection_certificate,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(
    rows: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def _weight(labels: tuple[str, ...], values: tuple[int, ...]) -> MatroidWeightFunction:
    return MatroidWeightFunction(ground_axis=labels, values=values)


def _independent_by_coefficients(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Independent oracle by exhaustive GF(2) coefficient vectors."""
    for coefficients in product(range(2), repeat=len(subset)):
        if not any(coefficients):
            continue
        if all(
            sum(
                coefficients[index] * row[column] for index, column in enumerate(subset)
            )
            % 2
            == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def _all_independent(matroid: LinearMatroid) -> tuple[tuple[int, ...], ...]:
    n = matroid.ground_size
    return tuple(
        subset
        for size in range(n + 1)
        for subset in combinations(range(n), size)
        if _independent_by_coefficients(matroid, subset)
    )


def test_supplied_split_certificate_round_trips_and_recomputes() -> None:
    labels = ("a", "b")
    matroid = _matroid(((1, 1),), labels)
    objective = _weight(labels, (5, 3))
    zero = _weight(labels, (0, 0))
    request = MatroidWeightedIntersectionCertificateRequest(
        first=matroid,
        second=matroid,
        weight_function=objective,
        common_independent=(0,),
        first_split=objective,
        second_split=zero,
    )

    result = weighted_intersection_certificate(request)

    assert result.total_weight == 5
    assert result.first_maximizer.total_weight == 5
    assert result.second_maximizer.total_weight == 0
    assert result.first_maximizer.independent_set == (0,)
    decoded = MatroidWeightedIntersectionResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_weighted_intersection_result(decoded)


def test_all_negative_weights_certify_empty_optimum_and_round_trip() -> None:
    labels = ("a", "b", "c")
    first = _matroid(((1, 1, 0),), labels)
    second = _matroid(((0, 1, 1),), labels)
    objective = _weight(labels, (-1, -2, -3))
    zero = _weight(labels, (0, 0, 0))
    result = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=first,
            second=second,
            weight_function=objective,
            common_independent=(),
            first_split=objective,
            second_split=zero,
        )
    )

    assert result.total_weight == 0
    assert result.first_maximizer.independent_set == ()
    assert result.second_maximizer.independent_set == ()
    decoded = MatroidWeightedIntersectionResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_weighted_intersection_result(decoded)


def test_owner_manifest_example_executes_through_its_declared_types() -> None:
    from jacobian.math.combinatorics.matroids._tools import TOOLS

    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "matroid.intersection.weighted_certificate.check"
    )
    example = tool.examples[0]
    request = tool.request_type.model_validate(example.input)
    result = tool.run(request)

    assert result.total_weight == 5
    assert (
        result.first_maximizer.total_weight + result.second_maximizer.total_weight == 5
    )


def test_certificate_matches_exhaustive_common_independent_set_oracle() -> None:
    labels = ("a", "b", "c")
    matroids = (
        _matroid(((1, 1, 1),), labels),
        _matroid(((1, 0, 1), (0, 1, 1)), labels),
        _matroid(((1, 0, 0), (0, 1, 0)), labels),
    )
    weights = (2, 1, -1)
    objective = _weight(labels, weights)
    families = tuple(_all_independent(matroid) for matroid in matroids)

    for first in matroids:
        for second in matroids:
            common = tuple(
                subset
                for subset in families[matroids.index(first)]
                if subset in families[matroids.index(second)]
            )
            maximum = max(sum(weights[i] for i in subset) for subset in common)
            candidate = min(
                subset
                for subset in common
                if sum(weights[i] for i in subset) == maximum
            )

            split: tuple[tuple[int, ...], tuple[int, ...]] | None = None
            first_family = _all_independent(first)
            second_family = _all_independent(second)
            for first_values in product(range(-4, 5), repeat=3):
                second_values = tuple(
                    weights[index] - first_values[index] for index in range(3)
                )
                first_max = max(
                    sum(first_values[index] for index in subset)
                    for subset in first_family
                )
                second_max = max(
                    sum(second_values[index] for index in subset)
                    for subset in second_family
                )
                if first_max + second_max == maximum:
                    split = (first_values, second_values)
                    break
            assert split is not None
            result = weighted_intersection_certificate(
                MatroidWeightedIntersectionCertificateRequest(
                    first=first,
                    second=second,
                    weight_function=objective,
                    common_independent=candidate,
                    first_split=_weight(labels, split[0]),
                    second_split=_weight(labels, split[1]),
                )
            )
            assert result.total_weight == maximum


def test_certificate_rejects_inexact_split() -> None:
    labels = ("a", "b")
    matroid = _matroid(((1, 0), (0, 1)), labels)
    with pytest.raises(OperationDomainValidationError, match="sum coordinatewise"):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=matroid,
                second=matroid,
                weight_function=_weight(labels, (4, 2)),
                common_independent=(0,),
                first_split=_weight(labels, (4, 2)),
                second_split=_weight(labels, (0, 1)),
            )
        )


def test_split_witness_supports_bounded_extra_digits_for_cancellation() -> None:
    labels = ("loop",)
    loop = _matroid(((0,),), labels)
    certificate = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=loop,
            second=loop,
            weight_function=_weight(labels, (0,)),
            common_independent=(),
            first_split=_weight(labels, (10**12 + 1,)),
            second_split=_weight(labels, (-(10**12 + 1),)),
        )
    )

    assert certificate.total_weight == 0
    assert certificate.first_maximizer.total_weight == 0
    assert certificate.second_maximizer.total_weight == 0
    decoded = MatroidWeightedIntersectionResult.model_validate_json(
        certificate.model_dump_json()
    )
    assert verify_weighted_intersection_result(decoded)
    from jacobian.math.combinatorics.matroids.operations import (
        verify_maximum_weight_independent_set,
    )

    assert verify_maximum_weight_independent_set(decoded.first_maximizer)
    replayed = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=decoded.first,
            second=decoded.second,
            weight_function=decoded.weight_function,
            common_independent=decoded.common_independent,
            first_split=decoded.first_maximizer.weight_function,
            second_split=decoded.second_maximizer.weight_function,
        )
    )
    assert replayed == decoded


def test_certificate_rejects_a_feasible_but_nonoptimal_candidate() -> None:
    labels = ("a", "b")
    matroid = _matroid(((1, 0), (0, 1)), labels)
    weights = _weight(labels, (4, 2))
    zero = _weight(labels, (0, 0))
    with pytest.raises(OperationDomainValidationError, match="does not certify"):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=matroid,
                second=matroid,
                weight_function=weights,
                common_independent=(1,),
                first_split=weights,
                second_split=zero,
            )
        )


def test_certificate_rejects_candidate_not_independent_in_both_sources() -> None:
    labels = ("a", "b")
    parallel = _matroid(((1, 1),), labels)
    free = _matroid(((1, 0), (0, 1)), labels)
    weights = _weight(labels, (2, 1))
    zero = _weight(labels, (0, 0))
    with pytest.raises(OperationDomainValidationError, match="independent in both"):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=parallel,
                second=free,
                weight_function=weights,
                common_independent=(0, 1),
                first_split=weights,
                second_split=zero,
            )
        )


def test_explicit_checker_rejects_forged_source_maximum_claims() -> None:
    labels = ("a", "b")
    matroid = _matroid(((1, 1),), labels)
    weights = _weight(labels, (5, 3))
    zero = _weight(labels, (0, 0))
    result = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=matroid,
            second=matroid,
            weight_function=weights,
            common_independent=(0,),
            first_split=weights,
            second_split=zero,
        )
    )
    raw = result.model_dump(mode="python")
    raw["first_maximizer"]["total_weight"] = 4
    raw["second_maximizer"]["total_weight"] = 1
    forged = MatroidWeightedIntersectionResult.model_validate(raw)

    assert forged.total_weight == (
        forged.first_maximizer.total_weight + forged.second_maximizer.total_weight
    )
    assert not verify_weighted_intersection_result(forged)


def test_aggregate_work_rejects_before_any_greedy_rank_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    n = 256
    labels = tuple(f"e{i}" for i in range(n))
    rows = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    matroid = _matroid(rows, labels)
    all_positive = _weight(labels, (1,) * n)
    zero = _weight(labels, (0,) * n)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("rank kernel ran before aggregate admission")

    monkeypatch.setattr(
        intersection, "_maximum_weight_independent_set_admitted", unexpected
    )
    with pytest.raises(OperationResourceAdmissionError, match="work or result output"):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=matroid,
                second=matroid,
                weight_function=all_positive,
                common_independent=(),
                first_split=all_positive,
                second_split=zero,
            )
        )


def test_output_size_rejects_long_repeated_axis_labels_before_rank_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    labels = ("x" * 300_000,)
    matroid = _matroid(((1,),), labels)
    objective = _weight(labels, (1,))
    zero = _weight(labels, (0,))

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("rank kernel ran before output-size admission")

    monkeypatch.setattr(
        intersection, "_maximum_weight_independent_set_admitted", unexpected
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="codepoint allocation bound"
    ):
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=matroid,
                second=matroid,
                weight_function=objective,
                common_independent=(0,),
                first_split=objective,
                second_split=zero,
            )
        )
