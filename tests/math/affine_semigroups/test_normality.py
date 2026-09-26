from __future__ import annotations

import json
from collections.abc import Callable
from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    AffineSemigroupNormality,
    PositiveAffineSemigroup,
    normality,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineSemigroupNormalityRequest,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _semigroup(vectors: tuple[tuple[int, int], ...]) -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=tuple(f"g{index}" for index in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(2)),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=(
            CanonicalRational.from_fraction(Fraction(1)),
            CanonicalRational.from_fraction(Fraction(1)),
        ),
    )


def _brute_member(
    vectors: tuple[tuple[int, int], ...], target: tuple[int, int]
) -> bool:
    """Independent coefficient-box oracle for these small nonnegative vectors."""
    limits = tuple(
        min(target[row] // vector[row] for row in range(2) if vector[row] > 0)
        for vector in vectors
    )

    def visit(index: int, point: tuple[int, int]) -> bool:
        if index == len(vectors):
            return point == target
        vector = vectors[index]
        return any(
            visit(
                index + 1, (point[0] + count * vector[0], point[1] + count * vector[1])
            )
            for count in range(limits[index] + 1)
        )

    return visit(0, (0, 0))


def _brute_in_cone_and_lattice(
    vectors: tuple[tuple[int, int], ...], point: tuple[int, int]
) -> bool:
    rays = sorted(
        set(vectors),
        key=lambda vector: Fraction(vector[1], vector[0]),
    )
    lower, upper = rays[0], rays[-1]
    in_cone = (
        lower[0] * point[1] - lower[1] * point[0] >= 0
        and point[0] * upper[1] - point[1] * upper[0] >= 0
    )

    def minor_gcd(generators: tuple[tuple[int, int], ...]) -> int:
        value = 0
        for left, right in combinations(generators, 2):
            value = gcd(value, abs(left[0] * right[1] - left[1] * right[0]))
        return value

    return in_cone and minor_gcd((*vectors, point)) == minor_gcd(vectors)


def test_nonnormality_returns_a_hole_against_independent_factorization_oracle() -> None:
    vectors = ((1, 0), (1, 2), (2, 1))
    result = normality(_semigroup(vectors))

    assert not result.normal
    assert result.hole == (1, 1)
    assert _brute_in_cone_and_lattice(vectors, result.hole)
    assert not _brute_member(vectors, result.hole)


def test_redundant_even_quadrant_generators_are_normal_in_their_group_lattice() -> None:
    # The generated group is 2Z^2, so the saturation in the correct lattice is
    # the even quadrant. The third generator is redundant and the hole is empty.
    vectors = ((2, 0), (0, 2), (2, 2))
    result = normality(_semigroup(vectors))

    assert result.normal
    assert result.hole is None
    assert result.semigroup.configuration.columns_vectors == vectors


def test_scaled_quadrant_admits_ray_orders_in_its_generated_lattice() -> None:
    result = normality(_semigroup(((20, 0), (0, 20))))

    assert result.normal
    assert result.hole is None


def test_rank_deficient_semigroup_has_structured_native_error() -> None:
    with pytest.raises(OperationDomainValidationError, match="full-rank"):
        normality(_semigroup(((1, 0), (2, 0))))


def test_normality_admits_the_semigroup_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import jacobian.math.affine_semigroups.semigroup as semigroup_module

    original = semigroup_module._admit_semigroup
    calls = 0

    def count_admission(value: object) -> PositiveAffineSemigroup:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(semigroup_module, "_admit_semigroup", count_admission)
    normality(_semigroup(((1, 0), (0, 1))))

    assert calls == 1


def test_duplicate_generators_are_deduplicated_for_admission_and_membership() -> None:
    vectors = ((1, 0),) * 5 + ((0, 1),) * 5
    result = normality(_semigroup(vectors))

    assert result.normal
    assert result.hole is None
    # Internal pricing/search deduplicates vector values while the public
    # result retains the caller's labelled generator axis.
    assert result.semigroup.configuration.columns_vectors == vectors


def test_normality_manifest_example_round_trips() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "affine_semigroup.normality.compute"
    )
    request = AffineSemigroupNormalityRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    run_normality = cast(
        Callable[[AffineSemigroupNormalityRequest], AffineSemigroupNormality],
        tool.run,
    )
    result = run_normality(request)

    assert result.normal is False
    assert result.hole == (1, 1)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_normality_bounds_all_membership_candidates_before_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.affine_semigroups.semigroup as semigroup_module

    monkeypatch.setattr(semigroup_module, "MAX_AFFINE_NORMALITY_WORK", 0)
    monkeypatch.setattr(
        semigroup_module,
        "_fiber_has_member",
        lambda *_args: pytest.fail("candidate search started before batch admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="normality membership"):
        normality(_semigroup(((2, 0), (0, 2), (2, 2))))


def test_normality_admits_combined_work_before_hilbert_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.affine_semigroups.semigroup as semigroup_module

    monkeypatch.setattr(semigroup_module, "MAX_AFFINE_NORMALITY_WORK", 0)
    monkeypatch.setattr(
        semigroup_module,
        "_hilbert_basis_admitted",
        lambda *_args: pytest.fail("Hilbert normalization expanded before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="normality membership"):
        normality(_semigroup(((2, 0), (0, 2), (2, 2))))


def test_preflight_prices_full_axis_for_possible_hole_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.affine_semigroups.semigroup as semigroup_module

    # Unique-vector fiber searches fit below this budget. Charging the one
    # possible witness replay against all ten source columns pushes it above.
    monkeypatch.setattr(semigroup_module, "MAX_AFFINE_NORMALITY_WORK", 200)
    monkeypatch.setattr(
        semigroup_module,
        "_hilbert_basis_admitted",
        lambda *_args: pytest.fail("normalization started before work admission"),
    )
    vectors = ((1, 0),) * 5 + ((0, 1),) * 5
    with pytest.raises(OperationResourceAdmissionError, match="normality membership"):
        normality(_semigroup(vectors))
