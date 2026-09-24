from __future__ import annotations

import json
from itertools import combinations, product

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidRankMultiplier,
    MatroidWeightedIntersectionCertificateRequest,
    MatroidWeightedIntersectionOptimizationRequest,
    MatroidWeightedIntersectionOptimizationResult,
    MatroidWeightedIntersectionRankCertificateRequest,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    maximum_weight_matroid_intersection,
    verify_weighted_intersection_rank_certificate,
    verify_weighted_intersection_result,
    weighted_intersection_certificate,
    weighted_intersection_rank_certificate,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(
    columns: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    row_count = len(columns[0]) if columns else 0
    rows = tuple(
        tuple(columns[column][row] for column in range(len(columns)))
        for row in range(row_count)
    )
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def _independent_by_coefficients(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Independent exact oracle: no nonzero coefficient vector kills the columns."""
    for coefficients in product(range(matroid.matrix.prime), repeat=len(subset)):
        if not any(coefficients):
            continue
        if all(
            sum(
                coefficients[index] * row[column] for index, column in enumerate(subset)
            )
            % matroid.matrix.prime
            == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def _independent_family(matroid: LinearMatroid) -> tuple[tuple[int, ...], ...]:
    n = matroid.ground_size
    return tuple(
        subset
        for size in range(n + 1)
        for subset in combinations(range(n), size)
        if _independent_by_coefficients(matroid, subset)
    )


def _rank_for_test(matroid: LinearMatroid, subset: tuple[int, ...]) -> int:
    return max(
        (
            len(candidate)
            for size in range(len(subset) + 1)
            for candidate in combinations(subset, size)
            if _independent_by_coefficients(matroid, candidate)
        ),
        default=0,
    )


def _request(
    first: LinearMatroid, second: LinearMatroid, weights: tuple[int, ...]
) -> MatroidWeightedIntersectionOptimizationRequest:
    return MatroidWeightedIntersectionOptimizationRequest(
        first=first,
        second=second,
        weight_function=MatroidWeightFunction(
            ground_axis=first.ground_axis, values=weights
        ),
    )


def test_catalog_optimizer_example_returns_replayable_checker_result() -> None:
    catalog = Catalog.open()
    operation = catalog.operation("matroid.intersection.maximum_weight.compute")
    assert operation is not None
    example = operation.examples[0]

    result = invoke_operation(operation.operation_id, example.input, catalog)
    decoded = operation.result_type.model_validate_json(json.dumps(result.output))

    assert decoded.common_independent == (0,)
    assert (
        decoded.first_maximizer.total_weight + decoded.second_maximizer.total_weight
        == 5
    )
    assert verify_weighted_intersection_result(decoded)


def test_weighted_intersection_matches_exhaustive_gf2_instances() -> None:
    """Compare all small represented matroid pairs with a coefficient oracle."""
    labels = ("a", "b", "c")
    representatives: dict[tuple[tuple[int, ...], ...], LinearMatroid] = {}
    for flat in product(range(2), repeat=2 * len(labels)):
        columns = tuple(
            (flat[column], flat[len(labels) + column]) for column in range(len(labels))
        )
        matroid = _matroid(columns, labels)
        representatives.setdefault(_independent_family(matroid), matroid)
    families = tuple(representatives)
    matroids = tuple(representatives.values())
    objectives = tuple(product((-1, 0, 1), repeat=len(labels)))

    for first_index, first in enumerate(matroids):
        for second_index, second in enumerate(matroids):
            common = tuple(
                subset
                for subset in families[first_index]
                if subset in set(families[second_index])
            )
            for weights in objectives:
                expected = min(
                    common,
                    key=lambda subset: (
                        -sum(weights[index] for index in subset),
                        len(subset),
                        subset,
                    ),
                )
                result = maximum_weight_matroid_intersection(
                    _request(first, second, weights)
                )
                assert result.common_independent == expected
                assert result.total_weight == sum(weights[index] for index in expected)


def test_finite_unit_slack_reweights_before_stopping() -> None:
    labels = ("a", "b")
    # In the first source, a and b are parallel. In the second, a is a loop
    # and b is the sole nonloop. The initial source maximizer is a, so the
    # source slack is exactly one and must trigger a dual adjustment to reach b.
    first = _matroid(((1,), (1,)), labels)
    second = _matroid(((0,), (1,)), labels)

    result = maximum_weight_matroid_intersection(_request(first, second, (2, 1)))

    assert result.common_independent == (1,)
    assert result.total_weight == 1
    decoded = MatroidWeightedIntersectionOptimizationResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result
    assert verify_weighted_intersection_result(result)


def test_loop_counterexample_has_rank_dual_even_when_terminal_split_does_not() -> None:
    labels = ("a", "b")
    first = _matroid(((1,), (0,)), labels)
    second = _matroid(((0,), (1,)), labels)
    weight_function = MatroidWeightFunction(
        ground_axis=labels,
        values=(1, 1),
    )
    optimum = maximum_weight_matroid_intersection(
        _request(first, second, weight_function.values)
    )
    with pytest.raises(OperationDomainValidationError) as split_error:
        weighted_intersection_certificate(
            MatroidWeightedIntersectionCertificateRequest(
                first=first,
                second=second,
                weight_function=weight_function,
                common_independent=optimum.common_independent,
                first_split=weight_function,
                second_split=MatroidWeightFunction(ground_axis=labels, values=(0, 0)),
            )
        )
    certificate = weighted_intersection_rank_certificate(
        MatroidWeightedIntersectionRankCertificateRequest(
            first=first,
            second=second,
            weight_function=weight_function,
            common_independent=optimum.common_independent,
            first_rank_terms=(MatroidRankMultiplier(subset=(1,), multiplier=1),),
            second_rank_terms=(MatroidRankMultiplier(subset=(0,), multiplier=1),),
        )
    )

    assert optimum.common_independent == ()
    assert (
        optimum.first_maximizer.total_weight + optimum.second_maximizer.total_weight
        == 0
    )
    assert split_error.value.errors()[0]["type"] == (
        "matroid.weighted_intersection.optimality"
    )
    supplied = weighted_intersection_certificate(
        MatroidWeightedIntersectionCertificateRequest(
            first=first,
            second=second,
            weight_function=weight_function,
            common_independent=optimum.common_independent,
            first_split=optimum.first_maximizer.weight_function,
            second_split=optimum.second_maximizer.weight_function,
        )
    )
    assert supplied == optimum
    assert verify_weighted_intersection_result(optimum)
    assert certificate.total_weight == 0
    assert verify_weighted_intersection_rank_certificate(certificate)


def test_optimizer_validates_shared_prime_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    labels = ("a", "b", "c")
    first = _matroid(((1, 0), (1, 1), (0, 1)), labels)
    second = _matroid(((1,), (0,), (1,)), labels)
    calls = 0

    def count_prime_checks(_prime: int) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(intersection, "_admit_prime", count_prime_checks)
    maximum_weight_matroid_intersection(_request(first, second, (3, 2, 1)))

    assert calls == 1


def _all_two_element_dual_chains() -> tuple[tuple[MatroidRankMultiplier, ...], ...]:
    subsets = ((0,), (1,), (0, 1))
    chains: list[tuple[MatroidRankMultiplier, ...]] = [()]
    for subset in subsets:
        for multiplier in (1, 2):
            chains.append(
                (MatroidRankMultiplier(subset=subset, multiplier=multiplier),)
            )
    for smaller, larger in ((subsets[0], subsets[2]), (subsets[1], subsets[2])):
        for first_multiplier, second_multiplier in product((1, 2), repeat=2):
            chains.append(
                (
                    MatroidRankMultiplier(subset=smaller, multiplier=first_multiplier),
                    MatroidRankMultiplier(subset=larger, multiplier=second_multiplier),
                )
            )
    return tuple(chains)


def _find_tiny_rank_dual(
    first: LinearMatroid,
    second: LinearMatroid,
    weights: tuple[int, ...],
    optimum: int,
    chains: tuple[tuple[MatroidRankMultiplier, ...], ...],
) -> tuple[tuple[MatroidRankMultiplier, ...], tuple[MatroidRankMultiplier, ...]] | None:
    for first_terms in chains:
        first_cover = [0, 0]
        first_value = 0
        for term in first_terms:
            first_value += term.multiplier * _rank_for_test(first, term.subset)
            for index in term.subset:
                first_cover[index] += term.multiplier
        if first_value > optimum:
            continue
        for second_terms in chains:
            dual_value = first_value
            cover = list(first_cover)
            for term in second_terms:
                dual_value += term.multiplier * _rank_for_test(second, term.subset)
                for index in term.subset:
                    cover[index] += term.multiplier
            if dual_value == optimum and all(
                cover[index] >= weights[index] for index in range(2)
            ):
                return first_terms, second_terms
    return None


def test_every_two_element_gf2_optimum_has_an_exhaustively_found_rank_dual() -> None:
    labels = ("a", "b")
    vectors = tuple(product(range(2), repeat=2))
    represented: dict[tuple[tuple[int, ...], ...], LinearMatroid] = {}
    for columns in product(vectors, repeat=2):
        matroid = _matroid(columns, labels)
        represented.setdefault(_independent_family(matroid), matroid)

    chains = _all_two_element_dual_chains()
    objectives = tuple(product((-1, 0, 1), repeat=2))
    for first in represented.values():
        for second in represented.values():
            for weights in objectives:
                optimum = maximum_weight_matroid_intersection(
                    _request(first, second, weights)
                )
                split_certificate = weighted_intersection_certificate(
                    MatroidWeightedIntersectionCertificateRequest(
                        first=first,
                        second=second,
                        weight_function=MatroidWeightFunction(
                            ground_axis=labels, values=weights
                        ),
                        common_independent=optimum.common_independent,
                        first_split=optimum.first_maximizer.weight_function,
                        second_split=optimum.second_maximizer.weight_function,
                    )
                )
                assert split_certificate == optimum
                matching_terms = _find_tiny_rank_dual(
                    first, second, weights, optimum.total_weight, chains
                )
                assert matching_terms is not None
                certificate = weighted_intersection_rank_certificate(
                    MatroidWeightedIntersectionRankCertificateRequest(
                        first=first,
                        second=second,
                        weight_function=MatroidWeightFunction(
                            ground_axis=labels, values=weights
                        ),
                        common_independent=optimum.common_independent,
                        first_rank_terms=matching_terms[0],
                        second_rank_terms=matching_terms[1],
                    )
                )
                assert certificate.total_weight == optimum.total_weight
                assert verify_weighted_intersection_rank_certificate(certificate)


def test_empty_common_set_wins_when_all_common_weights_are_nonpositive() -> None:
    labels = ("a", "b")
    first = _matroid(((1,), (1,)), labels)
    second = _matroid(((1,), (1,)), labels)

    result = maximum_weight_matroid_intersection(_request(first, second, (-4, 0)))

    assert result.common_independent == ()
    assert result.total_weight == 0


def test_exact_near_work_boundary_is_accepted() -> None:
    n = 19
    labels = tuple(f"e{index}" for index in range(n))
    rank_one = _matroid(tuple((1,) for _ in labels), labels)
    weights = (99_999_999_999,) * n

    result = maximum_weight_matroid_intersection(_request(rank_one, rank_one, weights))

    assert result.common_independent == (0,)
    assert result.total_weight == weights[0]


def test_resource_admission_precedes_any_rank_kernel_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    n = 20
    labels = tuple(f"e{index}" for index in range(n))
    dense = _matroid(
        tuple(tuple(int(i == j) for i in range(n)) for j in range(n)), labels
    )
    request = _request(dense, dense, (99_999_999_999,) * n)

    def unexpected_rank(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank backend was called before weighted admission")

    monkeypatch.setattr(intersection, "pf_rank", unexpected_rank)
    with pytest.raises(OperationResourceAdmissionError) as error:
        maximum_weight_matroid_intersection(request)

    assert error.value.errors()[0]["type"] == (
        "matroid.weighted_intersection.optimize.work_bound"
    )
