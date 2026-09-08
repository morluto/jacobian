"""Independent finite enumeration and serialized composition of coset fibres."""

from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.finite import (
    PrimeFieldVectorSpace,
    subspace_compute,
    subspace_membership,
)
from jacobian.math.geometry.finite._models import LinearSubspace
from jacobian.math.geometry.finite.cosets import (
    CosetIntersection,
    CosetIntersectionProfile,
    coset_intersection_profile,
)


def _span_elements(subspace: LinearSubspace) -> set[tuple[int, ...]]:
    return {
        tuple(
            sum(c * row[j] for c, row in zip(coefficients, subspace.basis, strict=True))
            % subspace.space.field_order
            for j in range(len(subspace.space.axis))
        )
        for coefficients in product(
            range(subspace.space.field_order), repeat=subspace.dimension
        )
    }


def _oracle(result: CosetIntersectionProfile) -> None:
    space = result.space
    elements = _span_elements(result.subspace)
    remaining = set(result.subset)
    expected = set()
    while remaining:
        a = min(remaining)
        coset = {
            tuple((x + y) % space.field_order for x, y in zip(a, h, strict=True))
            for h in elements
        }
        members = remaining & coset
        expected.add((min(coset), tuple(sorted(members))))
        remaining -= members
    assert {(r.representative, r.members) for r in result.rows} == expected
    assert [r.representative for r in result.rows] == sorted(
        r.representative for r in result.rows
    )
    assert all(r.cardinality == len(r.members) for r in result.rows)


def test_all_subsets_of_binary_three_space_and_all_subspaces() -> None:
    space = PrimeFieldVectorSpace(field_order=2, axis=("x", "y", "z"))
    vectors = tuple(product(range(2), repeat=3))
    bases = {
        subspace_compute(space, generators).subspace.basis
        for size in range(4)
        for generators in combinations(vectors, size)
    }
    for basis in bases:
        subspace = LinearSubspace(space=space, basis=basis)
        for mask in range(256):
            subset = tuple(v for i, v in enumerate(vectors) if mask & (1 << i))
            _oracle(coset_intersection_profile(space, subspace, subset))


def test_odd_field_noncoordinate_pivots_and_translation() -> None:
    space = PrimeFieldVectorSpace(field_order=5, axis=("a", "b", "c", "d"))
    subspace = subspace_compute(space, ((0, 2, 3, 1), (0, 0, 0, 4))).subspace
    subset = ((0, 0, 0, 0), (0, 1, 4, 0), (0, 2, 3, 1), (1, 1, 1, 1))
    result = coset_intersection_profile(space, subspace, subset)
    _oracle(result)
    translation = (2, 3, 4, 1)
    shifted = tuple(
        sorted(
            tuple((a + b) % 5 for a, b in zip(vector, translation, strict=True))
            for vector in subset
        )
    )
    other = coset_intersection_profile(space, subspace, shifted)
    _oracle(other)
    assert sorted(r.cardinality for r in result.rows) == sorted(
        r.cardinality for r in other.rows
    )


def test_serialized_span_to_partition_to_membership() -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("u", "v"))
    produced = subspace_compute(space, ((2, 1),)).subspace
    subspace = LinearSubspace.model_validate_json(produced.model_dump_json())
    result = coset_intersection_profile(space, subspace, ((0, 0), (1, 2), (2, 1)))
    restored = CosetIntersectionProfile.model_validate_json(result.model_dump_json())
    for row in restored.rows:
        for member in row.members:
            difference = tuple(
                (a - b) % 3 for a, b in zip(member, row.representative, strict=True)
            )
            assert subspace_membership(restored.subspace, difference).is_member
    assert (
        coset_intersection_profile(restored.space, restored.subspace, restored.subset)
        == restored
    )


@pytest.mark.parametrize("subset", [(), ((),)])
def test_zero_dimensional_parent(subset: tuple[tuple[int, ...], ...]) -> None:
    space = PrimeFieldVectorSpace(field_order=9973, axis=())
    result = coset_intersection_profile(
        space, LinearSubspace(space=space, basis=()), subset
    )
    assert len(result.rows) == len(subset)
    assert result.space == space


@pytest.mark.parametrize(
    "basis", [((0, 0),), ((2, 0),), ((0, 1), (1, 0)), ((1, 1), (0, 1))]
)
def test_authored_non_rref_claims_are_rejected_even_for_empty_subset(
    basis: tuple[tuple[int, ...], ...],
) -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("x", "y"))
    authored = LinearSubspace(space=space, basis=basis)
    with pytest.raises(OperationDomainValidationError, match="row echelon"):
        coset_intersection_profile(space, authored, ())


def test_composite_field_is_rejected_even_when_subset_empty() -> None:
    space = PrimeFieldVectorSpace(field_order=9, axis=("x",))
    with pytest.raises(OperationDomainValidationError, match="prime"):
        coset_intersection_profile(space, LinearSubspace(space=space, basis=()), ())


def test_intersection_row_rejects_cardinality_mismatch() -> None:
    with pytest.raises(ValidationError, match="cardinality"):
        CosetIntersection(representative=(0,), members=((0,),), cardinality=2)


def test_intersection_row_requires_canonical_member_order() -> None:
    with pytest.raises(ValidationError, match="lexicographically"):
        CosetIntersection(representative=(0,), members=((1,), (0,)), cardinality=2)
    with pytest.raises(ValidationError, match="lexicographically"):
        CosetIntersection(representative=(0,), members=((0,), (0,)), cardinality=2)


def test_profile_rows_must_cover_the_retained_subset() -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("x",))
    with pytest.raises(ValidationError, match="complete retained subset"):
        CosetIntersectionProfile(
            space=space,
            subspace=LinearSubspace(space=space, basis=()),
            subset=((0,), (1,)),
            rows=(
                CosetIntersection(representative=(0,), members=((0,),), cardinality=1),
            ),
        )


@pytest.mark.parametrize("subset", [((0,), (0,)), ((1,), (0,)), ((3,),), ((0, 0),)])
def test_noncanonical_subset_is_rejected(subset: tuple[tuple[int, ...], ...]) -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("x",))
    with pytest.raises(ValidationError):
        coset_intersection_profile(space, LinearSubspace(space=space, basis=()), subset)


def test_parent_mismatch() -> None:
    space = PrimeFieldVectorSpace(field_order=3, axis=("x",))
    other = PrimeFieldVectorSpace(field_order=3, axis=("y",))
    with pytest.raises(ValidationError, match="ordered axis"):
        coset_intersection_profile(space, LinearSubspace(space=other, basis=()), ())


def test_many_occupied_cosets_in_large_ambient_space_are_accepted() -> None:
    space = PrimeFieldVectorSpace(
        field_order=9973, axis=tuple(f"x{i}" for i in range(32))
    )
    basis = ((1, *([0] * 31)),)
    subset = tuple((0, i, *([0] * 30)) for i in range(8192))
    result = coset_intersection_profile(
        space, LinearSubspace(space=space, basis=basis), subset
    )
    assert tuple(r.representative for r in result.rows) == subset
    assert all(r.members == (r.representative,) for r in result.rows)


def test_excessive_coordinate_allocation_is_rejected() -> None:
    space = PrimeFieldVectorSpace(field_order=2, axis=tuple(f"x{i}" for i in range(32)))
    subset = tuple((*v, *([0] * 18)) for v in product(range(2), repeat=14))
    with pytest.raises(OperationResourceAdmissionError, match="allocation"):
        coset_intersection_profile(space, LinearSubspace(space=space, basis=()), subset)


def test_full_space_uses_one_occupied_row_at_large_subset_boundary() -> None:
    space = PrimeFieldVectorSpace(field_order=2, axis=tuple(f"x{i}" for i in range(32)))
    subspace = LinearSubspace(
        space=space,
        basis=tuple(tuple(int(i == j) for j in range(32)) for i in range(32)),
    )
    subset = tuple((*v, *([0] * 18)) for v in product(range(2), repeat=14))[:16000]
    result = coset_intersection_profile(space, subspace, subset)
    assert len(result.rows) == 1
    assert result.rows[0].representative == (0,) * 32
    assert result.rows[0].members == subset
    assert result.rows[0].cardinality == 16000
