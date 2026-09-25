from __future__ import annotations

import importlib
import itertools
import json
from collections import deque
from collections.abc import Callable
from fractions import Fraction
from functools import cmp_to_key
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
    AffineSemigroupFundamentalHoles,
    PositiveAffineSemigroup,
    fundamental_holes,
)
from jacobian.math.affine_semigroups.fundamental_holes import (
    AffineSemigroupFundamentalHolesRequest,
    _iter_parallelogram_lattice_points,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _semigroup(vectors: tuple[tuple[int, int], ...]) -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=tuple(f"g{i}" for i in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(2)),
    )
    grading = tuple(CanonicalRational.from_fraction(Fraction(1)) for _ in range(2))
    return PositiveAffineSemigroup(configuration=configuration, grading=grading)


def _det(left: tuple[int, int], right: tuple[int, int]) -> int:
    return left[0] * right[1] - left[1] * right[0]


def _compare_rays(left: tuple[int, int], right: tuple[int, int]) -> int:
    determinant = _det(left, right)
    return -1 if determinant > 0 else 1 if determinant < 0 else 0


def _lattice_index(vectors: tuple[tuple[int, int], ...]) -> int:
    return gcd(
        *(
            abs(_det(vectors[i], vectors[j]))
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        )
    )


def _oracle_fundamental_holes(
    vectors: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Use determinant indices and direct semigroup closure as an independent oracle."""
    rays = sorted(set(vectors), key=cmp_to_key(_compare_rays))
    lower, upper = rays[0], rays[-1]
    lattice_index = _lattice_index(vectors)
    assert lattice_index > 0

    def in_saturation(point: tuple[int, int]) -> bool:
        first_numerator = _det(point, upper)
        second_numerator = _det(lower, point)
        if first_numerator < 0 or second_numerator < 0:
            return False
        enlarged_index = gcd(
            lattice_index,
            *(abs(_det(vector, point)) for vector in vectors),
        )
        return enlarged_index == lattice_index

    # In the first quadrant and with grading x+y, every fundamental point has
    # degree less than the sum of one source generator on each extreme ray.
    bound = 2 * max(x + y for x, y in vectors)
    semigroup_points = {(0, 0)}
    pending = deque([(0, 0)])
    while pending:
        point = pending.popleft()
        for generator in vectors:
            neighbor = (point[0] + generator[0], point[1] + generator[1])
            if sum(neighbor) <= bound and neighbor not in semigroup_points:
                semigroup_points.add(neighbor)
                pending.append(neighbor)

    result = []
    for x in range(bound + 1):
        for y in range(bound + 1 - x):
            point = (x, y)
            if point == (0, 0) or not in_saturation(point) or point in semigroup_points:
                continue
            is_fundamental = not any(
                saturation_predecessor(point, semigroup_point, in_saturation)
                for semigroup_point in semigroup_points
                if semigroup_point != (0, 0) and sum(semigroup_point) <= sum(point)
            )
            if is_fundamental:
                result.append(point)
    return tuple(result)


def saturation_predecessor(
    point: tuple[int, int],
    semigroup_point: tuple[int, int],
    in_saturation: Callable[[tuple[int, int]], bool],
) -> bool:
    return in_saturation((point[0] - semigroup_point[0], point[1] - semigroup_point[1]))


def test_parity_holes_reduce_to_one_fundamental_hole() -> None:
    result = fundamental_holes(_semigroup(((2, 0), (0, 2), (1, 1), (1, 0))))

    assert result.holes == ((0, 1),)


def test_numerical_semigroup_product_matches_fundamental_hole_example() -> None:
    # The first coordinate is <3,5,7>; the second freely translates it.
    result = fundamental_holes(_semigroup(((3, 0), (5, 0), (7, 0), (0, 1))))

    assert result.holes == ((1, 0), (2, 0))


def test_nontrivial_generated_lattice_and_normal_edge_case() -> None:
    singular = fundamental_holes(_semigroup(((4, 0), (0, 2), (2, 2))))
    normal = fundamental_holes(_semigroup(((1, 0), (0, 1))))

    assert singular.holes == ((2, 0),)
    assert normal.holes == ()


def test_skinny_index_two_parallelogram_is_enumerated_by_lattice_index() -> None:
    vectors = ((10_000, 9_999), (10_002, 10_001), (10_001, 10_000))
    result = fundamental_holes(_semigroup(vectors))

    # The extreme-ray determinant is 2, though the axis box of their
    # parallelogram contains about 100 million integer points. The independent
    # unimodular map T(x,y)=(x-y, 9999x-10000y) sends the columns to
    # (1,0),(1,-2),(1,-1). With degree x, all saturation points of degree <2
    # are 0,(1,0),(1,-1),(1,-2), already generated. Every fundamental hole has
    # degree <2, so the complete fundamental-hole set is empty.
    transformed = tuple((x - y, 9_999 * x - 10_000 * y) for x, y in vectors)
    assert _det((1, -1), (9_999, -10_000)) == -1
    assert abs(_det(vectors[0], vectors[1])) == 2
    assert transformed == ((1, 0), (1, -2), (1, -1))
    assert (
        gcd(
            *(
                _det(transformed[i], transformed[j])
                for i in range(3)
                for j in range(i + 1, 3)
            )
        )
        == 1
    )
    # The rays are (1,-2),(1,0), with determinant 2. Their half-open
    # parallelogram has exactly two lattice points: zero and their midpoint
    # (1,-1), both source-semigroup elements.
    assert _det((1, -2), (1, 0)) == 2
    assert (1, -1) in transformed
    assert set(_iter_parallelogram_lattice_points((1, -2), (1, 0))) == {
        (0, 0),
        (1, -1),
    }
    assert result.holes == ()


def test_exhaustive_small_generators_match_independent_saturation_oracle() -> None:
    choices = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1))
    for size in range(2, 5):
        for vectors in itertools.combinations_with_replacement(choices, size):
            if _lattice_index(vectors) == 0:
                continue
            actual = fundamental_holes(_semigroup(vectors)).holes
            assert actual == _oracle_fundamental_holes(vectors), vectors


def test_candidate_box_is_admitted_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holes_module = importlib.import_module(
        "jacobian.math.affine_semigroups.fundamental_holes"
    )

    monkeypatch.setattr(holes_module, "MAX_AFFINE_HOLE_CANDIDATES", 0)
    monkeypatch.setattr(
        holes_module,
        "_iter_parallelogram_lattice_points",
        lambda *_args: pytest.fail("parallelogram enumerated before admission"),
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="parallelogram lattice point count"
    ):
        fundamental_holes(_semigroup(((3, 0), (5, 0), (0, 1))))


def test_degenerate_cone_is_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="full-rank"):
        tool = next(
            item
            for item in TOOLS
            if item.operation_id == "affine_semigroup.fundamental_holes.compute"
        )
        request = AffineSemigroupFundamentalHolesRequest(
            semigroup=_semigroup(((1, 0), (2, 0)))
        )
        run = cast(
            Callable[
                [AffineSemigroupFundamentalHolesRequest],
                AffineSemigroupFundamentalHoles,
            ],
            tool.run,
        )
        run(request)


def test_catalog_example_and_json_round_trip() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.fundamental_holes.compute"
    )
    request = AffineSemigroupFundamentalHolesRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    run = cast(
        Callable[
            [AffineSemigroupFundamentalHolesRequest],
            AffineSemigroupFundamentalHoles,
        ],
        tool.run,
    )
    result = run(request)

    assert result.holes == ((0, 1),)
    assert (
        AffineSemigroupFundamentalHoles.model_validate_json(result.model_dump_json())
        == result
    )
