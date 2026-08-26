"""Independent replay tests for numerical-semigroup correctness repairs."""

from __future__ import annotations

from itertools import combinations, pairwise
from math import lcm

from tests.math.numerical_semigroups._support import numerical_semigroup_error

from jacobian.math.numerical_semigroups._element_invariant_models import (
    ElementCatenaryDegreeRequest,
    ElementDeltaSetRequest,
    ElementElasticityRequest,
)
from jacobian.math.numerical_semigroups._element_invariant_operations import (
    compute_element_catenary_degree,
)
from jacobian.math.numerical_semigroups._factorization_models import (
    FactorizationComputeRequest,
    FactorizationGraphComputeRequest,
)
from jacobian.math.numerical_semigroups._global_invariant_models import (
    BettiElementsRequest,
    CatenaryDegreeRequest,
    DeltaSetRequest,
    ElasticityRequest,
)
from jacobian.math.numerical_semigroups._global_invariant_operations import (
    compute_betti_elements,
    compute_catenary_degree,
    compute_delta_set,
    compute_elasticity,
)
from jacobian.math.numerical_semigroups._presentation_models import (
    MinimalPresentationRequest,
    MinimalPresentationResult,
    PresentationBinomialsRequest,
)
from jacobian.math.numerical_semigroups._presentation_operations import (
    compute_minimal_presentation,
    compute_presentation_binomials,
)


def _brute_factorizations(
    generators: tuple[int, ...], value: int
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    ranges = tuple(range(value // generator + 1) for generator in generators)

    def visit(index: int, coordinates: tuple[int, ...], degree: int) -> None:
        if index == len(generators):
            if degree == value:
                result.append(coordinates)
            return
        for coordinate in ranges[index]:
            candidate = degree + coordinate * generators[index]
            if candidate > value:
                break
            visit(index + 1, (*coordinates, coordinate), candidate)

    visit(0, (), 0)
    return tuple(result)


def _r_components(
    factorizations: tuple[tuple[int, ...], ...],
) -> tuple[frozenset[int], ...]:
    unseen = set(range(len(factorizations)))
    components: list[frozenset[int]] = []
    while unseen:
        root = unseen.pop()
        component = {root}
        pending = [root]
        while pending:
            left = pending.pop()
            neighbors = {
                right
                for right in unseen
                if any(
                    a > 0 and b > 0
                    for a, b in zip(
                        factorizations[left], factorizations[right], strict=True
                    )
                )
            }
            unseen.difference_update(neighbors)
            component.update(neighbors)
            pending.extend(neighbors)
        components.append(frozenset(component))
    return tuple(components)


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    common = tuple(min(a, b) for a, b in zip(left, right, strict=True))
    return max(sum(left) - sum(common), sum(right) - sum(common))


def _brute_catenary(factorizations: tuple[tuple[int, ...], ...]) -> int:
    if len(factorizations) <= 1:
        return 0
    distances = sorted(
        {
            _distance(factorizations[left], factorizations[right])
            for left, right in combinations(range(len(factorizations)), 2)
        }
    )
    for threshold in distances:
        unseen = set(range(len(factorizations)))
        pending = [unseen.pop()]
        while pending:
            left = pending.pop()
            neighbors = {
                right
                for right in unseen
                if _distance(factorizations[left], factorizations[right]) <= threshold
            }
            unseen.difference_update(neighbors)
            pending.extend(neighbors)
        if not unseen:
            return threshold
    raise AssertionError("finite complete graph did not connect")


def test_betti_and_minimal_presentation_replay_independently() -> None:
    generators = (6, 10, 15)
    betti = compute_betti_elements(
        BettiElementsRequest(generators=tuple(map(str, generators)))
    )
    presentation = compute_minimal_presentation(
        MinimalPresentationRequest(generators=tuple(map(str, generators)))
    )
    brute_betti: list[int] = []
    required_relations = 0
    for value in range(1, max(map(int, betti.betti_elements)) + 1):
        factorizations = _brute_factorizations(generators, value)
        if len(factorizations) > 1:
            components = _r_components(factorizations)
            if len(components) > 1:
                brute_betti.append(value)
                required_relations += len(components) - 1
    assert tuple(map(int, betti.betti_elements)) == tuple(brute_betti)
    assert len(presentation.relations) == required_relations == 2
    for relation in presentation.relations:
        assert sum(a * b for a, b in zip(relation.first, generators, strict=True)) == 30
        assert (
            sum(a * b for a, b in zip(relation.second, generators, strict=True)) == 30
        )


def test_minimal_presentation_rejects_relations_inside_one_r_class() -> None:
    with numerical_semigroup_error():
        MinimalPresentationResult.model_validate(
            {
                "minimal_generators": ["4", "10", "15"],
                "betti_elements": ["20", "30"],
                "relations": [
                    {"first": [5, 0, 0], "second": [0, 2, 0]},
                    {"first": [0, 3, 0], "second": [5, 1, 0]},
                ],
            }
        )


def test_element_and_global_catenary_replay_independently() -> None:
    generators = (3, 8, 10)
    for value in (16, 18, 20, 30):
        expected = _brute_catenary(_brute_factorizations(generators, value))
        observed = compute_element_catenary_degree(
            ElementCatenaryDegreeRequest(
                generators=tuple(map(str, generators)), value=str(value)
            )
        )
        assert observed.catenary_degree == expected
    global_result = compute_catenary_degree(
        CatenaryDegreeRequest(generators=tuple(map(str, generators)))
    )
    assert global_result.catenary_degree == 6
    assert global_result.witness_betti_elements == ("18",)


def test_global_catenary_includes_distances_inside_betti_r_classes() -> None:
    generators = (4, 10, 15)
    result = compute_catenary_degree(
        CatenaryDegreeRequest(generators=tuple(map(str, generators)))
    )
    expected = max(
        _brute_catenary(_brute_factorizations(generators, int(record.betti_element)))
        for record in result.betti_degrees
    )
    assert result.catenary_degree == expected == 5
    assert {
        int(record.betti_element): record.catenary_degree
        for record in result.betti_degrees
    } == {20: 5, 30: 5}


def test_global_delta_replays_every_element_through_theorem_bound() -> None:
    generators = (3, 8, 10)
    result = compute_delta_set(DeltaSetRequest(generators=tuple(map(str, generators))))
    observed: set[int] = set()
    for value in range(result.checked_through + 1):
        lengths = sorted(
            {
                sum(factorization)
                for factorization in _brute_factorizations(generators, value)
            }
        )
        observed.update(right - left for left, right in pairwise(lengths))
    assert tuple(sorted(observed)) == result.delta_set == (1, 2, 3, 4)


def test_global_elasticity_has_an_exact_attaining_witness() -> None:
    generators = (4, 6, 9)
    result = compute_elasticity(
        ElasticityRequest(generators=tuple(map(str, generators)))
    )
    witness = lcm(generators[0], generators[-1])
    lengths = tuple(
        sum(factorization)
        for factorization in _brute_factorizations(generators, witness)
    )
    assert result.elasticity == f"{max(lengths)}/{min(lengths)}" == "9/4"
    assert (result.smallest_generator, result.largest_generator) == ("4", "9")


def test_presentation_binomials_replay_to_zero() -> None:
    generators = (4, 6, 9)
    presentation = compute_minimal_presentation(
        MinimalPresentationRequest(generators=tuple(map(str, generators)))
    )
    result = compute_presentation_binomials(
        PresentationBinomialsRequest(
            generators=tuple(map(str, generators)),
            relations=presentation.relations,
        )
    )
    for binomial in result.binomials:
        left_degree = sum(
            exponent * generator
            for exponent, generator in zip(
                binomial.left_exponents, generators, strict=True
            )
        )
        right_degree = sum(
            exponent * generator
            for exponent, generator in zip(
                binomial.right_exponents, generators, strict=True
            )
        )
        assert (binomial.left_coefficient, binomial.right_coefficient) == ("1", "-1")
        assert left_degree == right_degree


def test_degenerate_free_semigroup_has_empty_relations_and_invariants() -> None:
    assert (
        compute_betti_elements(BettiElementsRequest(generators=("1",))).betti_elements
        == ()
    )
    assert (
        compute_minimal_presentation(
            MinimalPresentationRequest(generators=("1",))
        ).relations
        == ()
    )
    assert compute_delta_set(DeltaSetRequest(generators=("1",))).delta_set == ()
    assert (
        compute_catenary_degree(
            CatenaryDegreeRequest(generators=("1",))
        ).catenary_degree
        == 0
    )


def test_nonmember_and_undefined_element_invariants_fail_closed() -> None:
    with numerical_semigroup_error():
        ElementDeltaSetRequest(generators=("3", "5"), value="7")
    with numerical_semigroup_error():
        ElementElasticityRequest(generators=("3", "5"), value="0")
    with numerical_semigroup_error():
        ElementCatenaryDegreeRequest(generators=("3", "5"), value="7")


def test_completeness_boundaries_reject_unmaterializable_claims() -> None:
    with numerical_semigroup_error():
        FactorizationComputeRequest(
            generators=("6", "7", "8", "9", "10", "11"), value="220"
        )
    with numerical_semigroup_error():
        FactorizationGraphComputeRequest(
            generators=("6", "7", "8", "9", "10", "11"), value="200"
        )
    with numerical_semigroup_error():
        BettiElementsRequest(generators=("499", "500"))
    with numerical_semigroup_error():
        DeltaSetRequest(generators=("10", "11", "34", "35"))
