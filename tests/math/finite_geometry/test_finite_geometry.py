"""Tests for finite geometry operations."""

import pytest
from pydantic import ValidationError

from jacobian.math import finite_geometry
from jacobian.math.finite_fields import (
    Axis,
    AxisBoundMatrix,
    FiniteDimensionalSubspace,
    element,
    finite_field,
    restrict_scalars,
)
from jacobian.math.finite_geometry._models import (
    GrassmannianCountRequest,
    LinearSubspace,
    ProjectivePointCanonicalizeRequest,
    ProjectivePointEqualRequest,
    ProjectiveSpaceEnumerateRequest,
    ProjectiveSpaceEnumerateResult,
    SubspaceComputeRequest,
    SubspaceIntersectionRequest,
    SubspaceMembershipRequest,
    SubspaceSpanRequest,
)
from jacobian.math.finite_geometry._operations import (
    compute_grassmannian_count,
    compute_projective_point_canonicalize,
    compute_projective_point_equal,
    compute_projective_space_enumerate,
    compute_subspace_compute,
    compute_subspace_intersection,
    compute_subspace_membership,
    compute_subspace_span,
)
from jacobian.math.finite_geometry._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "finite_geometry.grassmannian.count",
        "finite_geometry.projective_point.canonicalize",
        "finite_geometry.projective_point.equal.decide",
        "finite_geometry.projective_space.enumerate_points",
        "finite_geometry.subspace.compute",
        "finite_geometry.subspace.intersection.compute",
        "finite_geometry.subspace.membership.decide",
        "finite_geometry.subspace.span.compute",
    }


def test_projective_point_canonicalize_scales_to_one() -> None:
    request = ProjectivePointCanonicalizeRequest(
        space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
    )
    result = compute_projective_point_canonicalize(request)
    assert result.point.coordinates[0] == 1
    assert result.point.space == request.space
    assert result.scale == 2


def test_projective_point_canonicalize_rejects_zero() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(0, 0)
        )
    assert error.value.errors()[0]["type"] == "finite_geometry.projective_vector_zero"


def test_projective_point_equal_same_point() -> None:
    point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
        )
    ).point
    request = ProjectivePointEqualRequest(point_a=point, point_b=point)
    result = compute_projective_point_equal(request)
    assert result.equal is True


def test_projective_point_equal_different_points() -> None:
    space = {"field_order": 5, "axis": ("x", "y")}
    request = ProjectivePointEqualRequest(
        point_a={"space": space, "coordinates": (1, 0)},
        point_b={"space": space, "coordinates": (0, 1)},
    )
    result = compute_projective_point_equal(request)
    assert result.equal is False


@pytest.mark.requires_backend("flint")
def test_projective_point_embeds_into_finite_field_restrict_scalars() -> None:
    """A finite-geometry producer composes through an explicit field extension."""
    geometry_point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 2, "axis": ("x", "y")}, vector=(1, 1)
        )
    ).point
    presentation = finite_field(2, (1, 1, 1))
    row_axis = Axis(name="coordinate directions", labels=("x", "y"))

    direction = finite_geometry.embed_projective_point_in_finite_field(
        geometry_point,
        presentation,
        row_axis,
    )
    one = element(presentation, (1, 0))
    zero = element(presentation, (0, 0))
    subspace = FiniteDimensionalSubspace(
        presentation=presentation,
        basis_axis=Axis(name="matrix basis", labels=("B",)),
        basis=(
            AxisBoundMatrix(
                presentation=presentation,
                row_axis=row_axis,
                column_axis=Axis(name="target", labels=("c",)),
                entries=((one,), (zero,)),
            ),
        ),
    )

    restricted = restrict_scalars(subspace, direction)

    assert tuple(value.coordinates for value in direction.coordinates) == (
        (1, 0),
        (1, 0),
    )
    assert restricted.matrix.entries == ((1,), (0,))
    assert (
        type(direction).model_validate(direction.model_dump(mode="json")) == direction
    )


def test_projective_point_embedding_requires_explicit_compatible_target() -> None:
    point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 2, "axis": ("x", "y")}, vector=(1, 1)
        )
    ).point
    target = finite_field(2, (1, 1, 1))

    with pytest.raises(ValueError, match="axis labels"):
        finite_geometry.embed_projective_point_in_finite_field(
            point,
            target,
            Axis(name="different coordinates", labels=("y", "x")),
        )

    with pytest.raises(ValueError, match="characteristic"):
        finite_geometry.embed_projective_point_in_finite_field(
            point,
            finite_field(3, (1, 0, 1)),
            Axis(name="coordinate directions", labels=("x", "y")),
        )


def test_public_api_constructs_and_embeds_without_private_imports() -> None:
    """The documented surface constructs the conversion's source value."""
    assert finite_geometry.__all__ == [
        "PrimeFieldVectorSpace",
        "ProjectivePoint",
        "ProjectivePointSequence",
        "embed_projective_point_in_finite_field",
        "projective_point",
    ]

    space = finite_geometry.PrimeFieldVectorSpace(field_order=2, axis=("x", "y"))
    point = finite_geometry.projective_point(space, (1, 1))
    assert isinstance(point, finite_geometry.ProjectivePoint)
    assert point.coordinates == (1, 1)

    direction = finite_geometry.embed_projective_point_in_finite_field(
        point,
        finite_field(2, (1, 1, 1)),
        Axis(name="coordinate directions", labels=("x", "y")),
    )
    assert tuple(value.coordinates for value in direction.coordinates) == (
        (1, 0),
        (1, 0),
    )


def test_enumeration_sequence_composes_into_embeddings_directly() -> None:
    """Enumerated points compose into the consumer as typed values, without
    manual reconstruction from coordinate tuples and a parent space."""
    result = compute_projective_space_enumerate(
        ProjectiveSpaceEnumerateRequest(space={"field_order": 2, "axis": ("x", "y")})
    )

    presentation = finite_field(2, (1, 1, 1))
    row_axis = Axis(name="coordinate directions", labels=("x", "y"))
    directions = [
        finite_geometry.embed_projective_point_in_finite_field(
            point, presentation, row_axis
        )
        for point in result.sequence
    ]

    assert [tuple(v.coordinates for v in d.coordinates) for d in directions] == [
        ((0, 0), (1, 0)),
        ((1, 0), (0, 0)),
        ((1, 0), (1, 0)),
    ]
    for point in result.sequence:
        assert isinstance(point, finite_geometry.ProjectivePoint)
        assert point.space.axis == ("x", "y")


def test_subspace_compute_basic() -> None:
    request = SubspaceComputeRequest(
        space={"field_order": 3, "axis": ("x", "y", "z")},
        vectors=((1, 0, 0), (0, 1, 0)),
    )
    result = compute_subspace_compute(request)
    assert result.subspace.dimension == 2
    assert result.subspace.space == request.space


def test_subspace_membership_member() -> None:
    subspace = compute_subspace_compute(
        SubspaceComputeRequest(
            space={"field_order": 3, "axis": ("x", "y", "z")},
            vectors=((1, 0, 0), (0, 1, 0)),
        )
    ).subspace
    request = SubspaceMembershipRequest(subspace=subspace, vector=(1, 1, 0))
    result = compute_subspace_membership(request)
    assert result.is_member is True


def test_subspace_membership_nonmember() -> None:
    subspace = LinearSubspace(
        space={"field_order": 3, "axis": ("x", "y", "z")},
        basis=((1, 0, 0), (0, 1, 0)),
    )
    request = SubspaceMembershipRequest(subspace=subspace, vector=(1, 1, 1))
    result = compute_subspace_membership(request)
    assert result.is_member is False


def test_subspace_span_dependent() -> None:
    request = SubspaceSpanRequest(
        space={"field_order": 2, "axis": ("x", "y")},
        vectors=((1, 0), (1, 0)),
        subspaces=(),
    )
    result = compute_subspace_span(request)
    assert result.subspace.dimension == 1


def test_subspace_intersection_trivial() -> None:
    space = {"field_order": 2, "axis": ("x", "y")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0),)},
        subspace_b={"space": space, "basis": ((0, 1),)},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 0


def test_subspace_intersection_identical() -> None:
    """Two identical subspaces should intersect at full dimension."""
    space = {"field_order": 2, "axis": ("x", "y")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0),)},
        subspace_b={"space": space, "basis": ((1, 0),)},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 1


def test_subspace_intersection_overlapping() -> None:
    """Two planes in F_3^3 meeting in a line."""
    space = {"field_order": 3, "axis": ("x", "y", "z")}
    request = SubspaceIntersectionRequest(
        subspace_a={"space": space, "basis": ((1, 0, 0), (0, 1, 0))},
        subspace_b={"space": space, "basis": ((0, 1, 0), (0, 0, 1))},
    )
    result = compute_subspace_intersection(request)
    assert result.subspace.dimension == 1


def test_projective_point_equal_reports_scale() -> None:
    """Scale should be the actual scalar relating the two vectors."""
    point = compute_projective_point_canonicalize(
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 5, "axis": ("x", "y")}, vector=(2, 3)
        )
    ).point
    request = ProjectivePointEqualRequest(point_a=point, point_b=point)
    result = compute_projective_point_equal(request)
    assert result.equal is True
    assert result.point_a == result.point_b


def test_grassmannian_count_lines_in_pg_2_2() -> None:
    request = GrassmannianCountRequest(
        field_order=2, ambient_dimension=3, subspace_dimension=1
    )
    result = compute_grassmannian_count(request)
    assert result.count == "7"


def test_grassmannian_count_planes_in_f2_4() -> None:
    request = GrassmannianCountRequest(
        field_order=2, ambient_dimension=4, subspace_dimension=2
    )
    result = compute_grassmannian_count(request)
    assert result.count == "35"


def test_grassmannian_count_exact_past_json_integer_range() -> None:
    """The exact Gaussian-binomial value stays a canonical decimal string."""

    result = compute_grassmannian_count(
        GrassmannianCountRequest(
            field_order=2,
            ambient_dimension=15,
            subspace_dimension=7,
        )
    )
    assert result.count == "246614610741341843"
    assert int(result.count) > (1 << 53) - 1


def test_projective_space_enumerate_pg1_f2() -> None:
    request = ProjectiveSpaceEnumerateRequest(
        space={"field_order": 2, "axis": ("x", "y")}
    )
    result = compute_projective_space_enumerate(request)
    assert len(result.sequence) == 3
    assert result.sequence.coordinates == ((0, 1), (1, 0), (1, 1))
    assert [point.coordinates for point in result.sequence.points] == [
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_enumeration_wire_form_stays_compact_and_typed_natively() -> None:
    """The wire form stores the parent space once plus bare coordinate
    tuples, while native iteration yields parent-bound typed points."""
    result = compute_projective_space_enumerate(
        ProjectiveSpaceEnumerateRequest(space={"field_order": 3, "axis": ("x", "y")})
    )

    wire = result.model_dump(mode="json")
    assert set(wire) == {"sequence", "method"}
    assert set(wire["sequence"]) == {"space", "coordinates"}
    assert wire["sequence"]["coordinates"] == [[0, 1], [1, 0], [1, 1], [1, 2]]
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_enumerate_admission_rejects_results_beyond_the_transport_budget() -> None:
    """A pathological axis label cannot smuggle an untransportable complete
    result past admission: the serialized-result bound fires before any
    enumeration runs."""
    with pytest.raises(ValidationError) as error:
        ProjectiveSpaceEnumerateRequest(
            space={"field_order": 2, "axis": ("x", "y" * (9 * 1024 * 1024))}
        )
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.projective_enumeration_result_too_large"
    )


def test_enumerate_admission_estimates_normalized_label_encoding() -> None:
    """The serialized-result bound measures NFC-normalized labels.

    Canonical JSON normalizes string values, so combining-character
    spellings can double in encoded length after normalization; a label
    that fits the budget only before normalization must still be rejected
    before any enumeration runs.
    """
    label = "x" + "\u0344" * (2_600_000)
    with pytest.raises(ValidationError) as error:
        ProjectiveSpaceEnumerateRequest(space={"field_order": 2, "axis": ("x", label)})
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.projective_enumeration_result_too_large"
    )


def test_enumeration_replay_rejects_unnormalized_representatives() -> None:
    """Sequence coordinates stay bound to the canonical representative
    invariant of the declared parent space."""
    result = compute_projective_space_enumerate(
        ProjectiveSpaceEnumerateRequest(space={"field_order": 3, "axis": ("x", "y")})
    )
    assert result.sequence.coordinates == ((0, 1), (1, 0), (1, 1), (1, 2))

    payload = result.model_dump()
    payload["sequence"]["coordinates"] = (
        (2, 1),
        *payload["sequence"]["coordinates"][1:],
    )
    with pytest.raises(ValidationError) as error:
        ProjectiveSpaceEnumerateResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.projective_coordinates_not_normalized"
    )


def test_enumeration_sequence_replay_rejects_duplicates_and_wrong_counts() -> None:
    """The sequence value itself certifies uniqueness and completeness."""
    result = compute_projective_space_enumerate(
        ProjectiveSpaceEnumerateRequest(space={"field_order": 2, "axis": ("x", "y")})
    )

    payload = result.model_dump()
    payload["sequence"]["coordinates"] = ((0, 1), (0, 1), (1, 0))
    with pytest.raises(ValidationError) as error:
        ProjectiveSpaceEnumerateResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.point_sequence_points_not_unique"
    )

    payload = result.model_dump()
    payload["sequence"]["coordinates"] = ((0, 1), (1, 0))
    with pytest.raises(ValidationError) as error:
        ProjectiveSpaceEnumerateResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.point_sequence_count_mismatch"
    )


def test_request_rejects_nonprime_field() -> None:
    with pytest.raises(ValidationError) as error:
        ProjectivePointCanonicalizeRequest(
            space={"field_order": 4, "axis": ("x", "y")}, vector=(1, 2)
        )
    assert error.value.errors()[0]["type"] == "finite_geometry.field_order_not_prime"


def test_canonical_values_compose_and_reject_different_parents() -> None:
    space = finite_geometry.PrimeFieldVectorSpace(field_order=3, axis=("x", "y"))
    computed = compute_subspace_compute(
        SubspaceComputeRequest(space=space, vectors=((1, 0),))
    ).subspace
    assert compute_subspace_membership(
        SubspaceMembershipRequest(subspace=computed, vector=(2, 0))
    ).is_member
    assert (
        compute_subspace_span(
            SubspaceSpanRequest(space=space, vectors=(), subspaces=(computed,))
        ).subspace
        == computed
    )

    other = LinearSubspace(
        space={"field_order": 5, "axis": ("x", "y")}, basis=((1, 0),)
    )
    with pytest.raises(ValidationError) as error:
        SubspaceIntersectionRequest(subspace_a=computed, subspace_b=other)
    assert (
        error.value.errors()[0]["type"]
        == "finite_geometry.intersection_parent_mismatch"
    )


def test_axis_identity_is_part_of_the_parent() -> None:
    point_x = {"space": {"field_order": 3, "axis": ("x", "y")}, "coordinates": (1, 0)}
    point_y = {"space": {"field_order": 3, "axis": ("y", "x")}, "coordinates": (1, 0)}
    with pytest.raises(ValidationError) as error:
        ProjectivePointEqualRequest(point_a=point_x, point_b=point_y)
    assert (
        error.value.errors()[0]["type"] == "finite_geometry.projective_parent_mismatch"
    )


def test_source_bound_results_reject_forged_values() -> None:
    result = compute_subspace_compute(
        SubspaceComputeRequest(
            space={"field_order": 3, "axis": ("x", "y")},
            vectors=((1, 0),),
        )
    )
    payload = result.model_dump()
    payload["subspace"]["basis"] = ((0, 1),)
    with pytest.raises(ValidationError) as error:
        type(result).model_validate(payload)
    assert error.value.errors()[0]["type"] == "finite_geometry.subspace_replay_mismatch"

    count = compute_grassmannian_count(
        GrassmannianCountRequest(
            field_order=2, ambient_dimension=3, subspace_dimension=1
        )
    )
    payload = count.model_dump()
    payload["count"] = "8"
    with pytest.raises(ValidationError) as error:
        type(count).model_validate(payload)
    assert (
        error.value.errors()[0]["type"] == "finite_geometry.grassmannian_count_mismatch"
    )
