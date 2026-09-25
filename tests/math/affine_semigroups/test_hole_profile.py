from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from fractions import Fraction
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
    AffineSemigroupHoleProfile,
    PositiveAffineSemigroup,
    holes_through_degree,
)
from jacobian.math.affine_semigroups.holes import AffineSemigroupHolesRequest
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


_PARITY_SEMIGROUP = _semigroup(((2, 0), (0, 2), (1, 1), (1, 0)))


def _brute_member(
    vectors: tuple[tuple[int, int], ...], target: tuple[int, int]
) -> bool:
    if min(target) < 0:
        return False
    maxima = tuple(sum(target) // sum(vector) for vector in vectors)
    return any(
        tuple(
            sum(counts[column] * vectors[column][row] for column in range(len(vectors)))
            for row in range(2)
        )
        == target
        for counts in itertools.product(*(range(maximum + 1) for maximum in maxima))
    )


def _brute_holes(
    vectors: tuple[tuple[int, int], ...], degree: int
) -> tuple[tuple[int, int], ...]:
    """Enumerate the N^2 cone slice and use an independent factorization box."""
    return tuple(
        (x, y)
        for x in range(degree + 1)
        for y in range(degree + 1 - x)
        if gcd(
            *(
                abs(
                    vectors[left][0] * vectors[right][1]
                    - vectors[left][1] * vectors[right][0]
                )
                for left in range(len(vectors))
                for right in range(left + 1, len(vectors))
            )
        )
        == 1
        and not _brute_member(vectors, (x, y))
    )


def test_parity_family_has_exact_holes_through_degree() -> None:
    # The generated lattice is Z^2. For x>0 every point is generated; at x=0,
    # only (0,2) can contribute, so exactly the odd vertical points are holes.
    result = holes_through_degree(_PARITY_SEMIGROUP, 4)

    assert result.holes == ((0, 1), (0, 3))
    assert result.semigroup.configuration.columns_vectors == (
        (2, 0),
        (0, 2),
        (1, 1),
        (1, 0),
    )


def test_small_profiles_match_independent_factorization_oracle() -> None:
    vectors = ((2, 0), (0, 2), (1, 1), (1, 0))
    for degree in range(7):
        assert holes_through_degree(_PARITY_SEMIGROUP, degree).holes == _brute_holes(
            vectors, degree
        )


def test_even_quadrant_respects_its_proper_generated_lattice() -> None:
    semigroup = _semigroup(((2, 0), (0, 2), (2, 2)))

    assert holes_through_degree(semigroup, 12).holes == ()


@pytest.mark.parametrize("degree", [-2, -1, 0])
def test_negative_and_zero_degree_profiles_are_empty(degree: int) -> None:
    assert holes_through_degree(_PARITY_SEMIGROUP, degree).holes == ()


def test_degenerate_cone_is_rejected_at_native_and_catalog_boundaries() -> None:
    semigroup = _semigroup(((1, 0), (2, 0)))
    with pytest.raises(OperationDomainValidationError) as error:
        holes_through_degree(semigroup, 3)
    assert error.value.errors()[0]["type"] == "affine_semigroup.hole_profile_cone"

    request = AffineSemigroupHolesRequest(semigroup=semigroup, max_degree=3)
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.holes_through_degree.compute"
    )
    with pytest.raises(OperationDomainValidationError):
        tool.run(request)


def test_candidate_box_is_admitted_before_lattice_point_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.affine_semigroups.holes as holes_module

    monkeypatch.setattr(holes_module, "MAX_AFFINE_HOLE_CANDIDATES", 0)
    monkeypatch.setattr(
        holes_module,
        "_iter_candidate_coordinates",
        lambda *_args: pytest.fail("candidate points enumerated before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError, match="candidate box"):
        holes_through_degree(_PARITY_SEMIGROUP, 2)


def test_catalog_example_and_serialized_profile_round_trip() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "affine_semigroup.holes_through_degree.compute"
    )
    request = AffineSemigroupHolesRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    run = cast(
        Callable[
            [AffineSemigroupHolesRequest],
            AffineSemigroupHoleProfile,
        ],
        tool.run,
    )
    result = run(request)

    assert isinstance(result, AffineSemigroupHoleProfile)
    assert result.holes == ((0, 1),)
    assert (
        AffineSemigroupHoleProfile.model_validate_json(result.model_dump_json())
        == result
    )


def test_forged_typed_axes_are_bounded_before_revalidation_copy() -> None:
    semigroup = _PARITY_SEMIGROUP.model_construct(
        configuration=AffineConfiguration.model_construct(
            row_labels=("x", "y"),
            generator_labels=tuple(f"g{index}" for index in range(100_000)),
            entries=((1,), (1,)),
        ),
        grading=_PARITY_SEMIGROUP.grading,
    )

    with pytest.raises(OperationDomainValidationError, match="configuration"):
        holes_through_degree(semigroup, 1)


def test_forged_grading_scalars_are_rejected_before_rational_validation() -> None:
    forged_rational = CanonicalRational.model_construct(num="1", den=1)
    semigroup = _PARITY_SEMIGROUP.model_construct(
        configuration=_PARITY_SEMIGROUP.configuration,
        grading=(forged_rational, _PARITY_SEMIGROUP.grading[1]),
    )

    with pytest.raises(OperationDomainValidationError, match="exact integers"):
        holes_through_degree(semigroup, 1)
