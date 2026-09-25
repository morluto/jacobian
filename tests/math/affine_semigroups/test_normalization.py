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
    AffineSemigroupNormalization,
    PositiveAffineSemigroup,
    normalization,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineSemigroupNormalizationRequest,
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


def _gcd_of_minors(vectors: tuple[tuple[int, int], ...]) -> int:
    divisor = 0
    for left, right in combinations(vectors, 2):
        divisor = gcd(divisor, abs(left[0] * right[1] - left[1] * right[0]))
    return divisor


def _brute_normalization_oracle(
    vectors: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Enumerate cone-lattice points in a ray parallelogram and test atoms.

    Group-lattice membership is checked by the invariant index gcd of all
    2-by-2 minors: adjoining a vector preserves the index exactly when the
    vector was already in the generated lattice. This oracle uses no HNF or
    production Hilbert-basis helper.
    """
    ordered_by_slope = sorted(
        vectors,
        key=lambda vector: (
            vector[0] == 0,
            Fraction(vector[1], vector[0]) if vector[0] else Fraction(0),
        ),
    )
    lower_ray, upper_ray = ordered_by_slope[0], ordered_by_slope[-1]
    if lower_ray[0] * upper_ray[1] - lower_ray[1] * upper_ray[0] == 0:
        raise ValueError("oracle requires a full-rank pointed cone")

    lattice_index = _gcd_of_minors(vectors)

    def in_group_lattice(point: tuple[int, int]) -> bool:
        return _gcd_of_minors((*vectors, point)) == lattice_index

    def in_cone(point: tuple[int, int]) -> bool:
        return (
            lower_ray[0] * point[1] - lower_ray[1] * point[0] >= 0
            and point[0] * upper_ray[1] - point[1] * upper_ray[0] >= 0
        )

    # The lattice primitive points on the extreme rays are no longer than
    # the raw input ray vectors. Their semi-open parallelogram is contained
    # in this integer coordinate box, which therefore includes every atom.
    bound_x = lower_ray[0] + upper_ray[0]
    bound_y = lower_ray[1] + upper_ray[1]
    lattice_points = {
        (x, y)
        for x in range(bound_x + 1)
        for y in range(bound_y + 1)
        if (x, y) != (0, 0) and in_cone((x, y)) and in_group_lattice((x, y))
    }

    atoms = []
    for point in lattice_points:
        decomposable = any(
            other != point
            and (point[0] - other[0], point[1] - other[1]) in lattice_points
            for other in lattice_points
        )
        if not decomposable:
            atoms.append(point)
    return tuple(sorted(atoms))


def test_normalization_matches_small_exact_lattice_oracle() -> None:
    # In the first case gp(S)=2Z^2, so the normalization is not the Hilbert
    # basis in the ambient integer grid. In the second, gp(S)=Z^2 and the
    # missing lattice point (1,1) is an explicit normalization generator.
    cases = (
        (
            ((2, 0), (0, 2), (2, 2)),
            ((0, 2), (2, 0)),
        ),
        (
            ((1, 0), (1, 2), (2, 1)),
            ((1, 0), (1, 1), (1, 2)),
        ),
    )
    for vectors, expected in cases:
        result = normalization(_semigroup(vectors))
        assert result.generators == expected
        assert result.semigroup.configuration.columns_vectors == vectors


@pytest.mark.parametrize(
    "vectors",
    (
        ((1, 0),),
        ((1, 0), (2, 0)),
        ((1, 1), (2, 2)),
        ((1, 1), (2, 2), (3, 3)),
    ),
)
def test_normalization_reports_expected_native_domain_failures(
    vectors: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        normalization(_semigroup(vectors))
    assert exc_info.value.errors()[0]["type"] == "affine_semigroup.normalization_domain"


def test_normalization_matches_exhaustive_small_pointed_semigroups() -> None:
    # Exhaust every two- and three-generator subset of this small positive
    # quadrant grid. The brute oracle checks lattice membership by minor gcds
    # and decomposability by direct addition in a complete finite ray box.
    points = tuple((x, y) for x in range(3) for y in range(3) if (x, y) != (0, 0))
    checked = 0
    for size in (2, 3):
        for vectors in combinations(points, size):
            if _gcd_of_minors(vectors) == 0:
                continue
            expected = _brute_normalization_oracle(vectors)
            actual = normalization(_semigroup(vectors)).generators
            assert actual == expected, vectors
            checked += 1
    assert checked == 81


def test_normalization_catalog_contract_and_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "affine_semigroup.normalization.compute"
    )
    request = AffineSemigroupNormalizationRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    run_normalization = cast(
        Callable[
            [AffineSemigroupNormalizationRequest],
            AffineSemigroupNormalization,
        ],
        tool.run,
    )
    result = run_normalization(request)
    assert result.generators == ((0, 2), (2, 0))


def test_normalization_preserves_full_source_through_model_and_json_round_trips() -> (
    None
):
    source = PositiveAffineSemigroup(
        configuration=AffineConfiguration(
            row_labels=("degree-x", "degree-y"),
            generator_labels=("edge", "interior", "vertical"),
            entries=((1, 2, 1), (0, 1, 2)),
        ),
        grading=(
            CanonicalRational.from_fraction(Fraction(3, 2)),
            CanonicalRational.from_fraction(Fraction(5, 3)),
        ),
    )
    result = normalization(source)
    assert result.semigroup == source
    assert result.semigroup.model_dump(mode="python") == source.model_dump(
        mode="python"
    )

    wire = result.model_dump_json()
    decoded = type(result).model_validate_json(wire)
    assert decoded == result
    assert decoded.semigroup.model_dump_json() == source.model_dump_json()


def test_normalization_requires_full_rank_and_bounded_hilbert_search() -> None:
    with pytest.raises(OperationDomainValidationError, match="full-rank"):
        normalization(_semigroup(((1, 0), (2, 0))))

    # The group lattice is Z^2 due to the interior vector, while the cone rays
    # have determinant 1,001. The request is rejected before Hilbert enumeration.
    with pytest.raises(OperationResourceAdmissionError, match="determinant"):
        normalization(_semigroup(((1, 0), (1, 1001), (2, 1))))
