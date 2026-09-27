"""Independent correctness tests for exact cellular-sheaf section spaces."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    SheafField,
    SheafStalk,
    from_cover_maps,
    restrict_sections,
    sections,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FromCoverMapsRequest,
    SheafCochainCoordinate,
    SheafSectionRestriction,
    SheafSectionSpace,
)
from jacobian.math.topology.cellular_sheaves._tools import TOOLS as SHEAF_TOOLS


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _constant_interval(
    *,
    field: SheafField = SheafField.PRIME_FIELD,
    prime: int = 2,
    scalar=None,
):
    if scalar is None:
        scalar = 1 if field is SheafField.PRIME_FIELD else _q(1)
    complex_value = canonical_complex(("a", "b"), (("a", "b"),))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_value.faces_by_dimension
        for face in group.faces
    )
    maps = tuple(
        CoverRestrictionMatrix(
            source=source,
            target=("a", "b"),
            entries=((scalar,),),
        )
        for source in (("a",), ("b",))
    )
    request = FromCoverMapsRequest(
        complex=complex_value,
        coefficient_field=field,
        prime=prime if field is SheafField.PRIME_FIELD else None,
        stalks=stalks,
        cover_maps=maps,
    )
    built = from_cover_maps(
        request.complex,
        request.coefficient_field,
        request.prime,
        request.stalks,
        request.cover_maps,
    )
    assert built.sheaf is not None
    return built.sheaf


def _matrix_vector_mod2(matrix, vector):
    return tuple(
        sum(int(entry) * value for entry, value in zip(row, vector, strict=True)) % 2
        for row in matrix
    )


def test_section_space_matches_all_globally_compatible_interval_assignments() -> None:
    sheaf = _constant_interval()
    space = sections(sheaf)
    assert space.dimension == 1
    assert space.ambient_basis == tuple(
        SheafCochainCoordinate(simplex=stalk.simplex, basis_label=label)
        for stalk in sheaf.stalks
        for label in stalk.basis
    )
    assert len(space.compatibility_matrix) == 2
    assert all(len(row) == 3 for row in space.compatibility_matrix)
    assert len(space.compatibility_row_axes) == 2

    compatible = set()
    for assignment in product(range(2), repeat=3):
        vertex_a, vertex_b, edge = assignment
        if vertex_a == edge and vertex_b == edge:
            compatible.add(assignment)
    assert compatible == {(0, 0, 0), (1, 1, 1)}
    basis = tuple(tuple(map(int, vector)) for vector in space.basis_coordinates)
    assert all(
        _matrix_vector_mod2(space.compatibility_matrix, vector) == (0, 0)
        for vector in basis
    )
    generated = {
        tuple(
            sum(
                coefficient * basis[index][column]
                for index, coefficient in enumerate(coefficients)
            )
            % 2
            for column in range(3)
        )
        for coefficients in product(range(2), repeat=space.dimension)
    }
    assert generated == compatible

    for stalk_index, evaluation in enumerate(space.evaluations):
        ambient_position = stalk_index
        assert tuple(int(row[0]) for row in evaluation.entries) == (
            int(space.basis_coordinates[0][ambient_position]),
        )
    assert SheafSectionSpace.model_validate_json(space.model_dump_json()) == space


def test_zero_stalks_and_no_compatibility_rows_are_canonical() -> None:
    complex_value = canonical_complex(("a",), (("a",),))
    result = from_cover_maps(
        complex_value,
        SheafField.RATIONAL,
        None,
        (SheafStalk(simplex=("a",), basis=()),),
        (),
    )
    assert result.sheaf is not None
    space = sections(result.sheaf)
    assert space.dimension == 0
    assert space.ambient_basis == ()
    assert space.compatibility_matrix == ()
    assert space.basis_coordinates == ()
    assert space.evaluations[0].entries == ()
    assert SheafSectionSpace.model_validate_json(space.model_dump_json()) == space


def test_rational_section_basis_and_stalk_evaluations_are_exact() -> None:
    space = sections(_constant_interval(field=SheafField.RATIONAL, scalar=_q("2/3")))
    vector = tuple(
        Fraction(value.num, value.den) for value in space.basis_coordinates[0]
    )
    assert space.dimension == 1
    assert tuple(
        sum(
            Fraction(entry.num, entry.den) * value
            for entry, value in zip(row, vector, strict=True)
        )
        for row in space.compatibility_matrix
    ) == (Fraction(0), Fraction(0))
    for coordinate, evaluation in enumerate(space.evaluations):
        value = evaluation.entries[0][0]
        assert Fraction(value.num, value.den) == vector[coordinate]


def test_section_restriction_is_the_exact_inclusion_induced_map() -> None:
    sheaf = _constant_interval(field=SheafField.RATIONAL, scalar=_q("2"))
    vertices = canonical_complex(("a", "b"), (("a",), ("b",)))
    result = restrict_sections(sheaf, vertices)
    assert result.source.dimension == 1
    assert result.target.dimension == 2
    assert result.entries == ((_q("1/2"),), (_q("1/2"),))
    # Independent oracle: source assignments satisfy 2*a=edge=2*b, so their
    # restrictions to the two isolated vertices are precisely the diagonal.
    assert tuple(
        tuple(Fraction(value.num, value.den) for value in row) for row in result.entries
    ) == (
        (Fraction(1, 2),),
        (Fraction(1, 2),),
    )
    assert result.source.sheaf == sheaf
    assert result.target.sheaf.complex == vertices
    assert (
        SheafSectionRestriction.model_validate_json(result.model_dump_json()) == result
    )


def test_published_section_restriction_example_runs() -> None:
    tool = next(
        item
        for item in SHEAF_TOOLS
        if item.operation_id == "cellular_sheaf.sections.restrict"
    )
    assert len(tool.examples) == 1
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert isinstance(result, SheafSectionRestriction)
    assert result.entries == ((_q("1"),), (_q("1"),))


def test_full_matrix_includes_derived_comparable_restrictions() -> None:
    complex_value = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    cells = tuple(
        face for group in complex_value.faces_by_dimension for face in group.faces
    )
    known = set(cells)
    cover_maps = tuple(
        CoverRestrictionMatrix(source=source, target=target, entries=((1,),))
        for target in cells
        if len(target) > 1
        for source in (
            target[:index] + target[index + 1 :] for index in range(len(target))
        )
        if source in known
    )
    constructed = from_cover_maps(
        complex_value,
        SheafField.PRIME_FIELD,
        2,
        tuple(SheafStalk(simplex=cell, basis=("x",)) for cell in cells),
        cover_maps,
    )
    assert constructed.sheaf is not None
    space = sections(constructed.sheaf)
    assert len(space.compatibility_row_axes) == 12
    assert any(
        len(axis.target) - len(axis.source) == 2
        for axis in space.compatibility_row_axes
    )
    compatible_assignments = {
        assignment
        for assignment in product(range(2), repeat=7)
        if len(set(assignment)) == 1
    }
    basis = tuple(tuple(map(int, vector)) for vector in space.basis_coordinates)
    span = {
        tuple(
            sum(
                coefficient * basis[index][coordinate]
                for index, coefficient in enumerate(coefficients)
            )
            % 2
            for coordinate in range(7)
        )
        for coefficients in product(range(2), repeat=space.dimension)
    }
    assert span == compatible_assignments


def test_full_compatibility_matrix_is_admitted_before_dense_construction() -> None:
    complex_value = canonical_complex(tuple("abcde"), (("a", "b", "c", "d", "e"),))
    all_faces = tuple(
        face for group in complex_value.faces_by_dimension for face in group.faces
    )
    stalks = tuple(
        SheafStalk(simplex=face, basis=tuple(f"b{index}" for index in range(8)))
        for face in all_faces
    )
    cover_maps = tuple(
        CoverRestrictionMatrix(
            source=face[:position] + face[position + 1 :],
            target=face,
            entries=tuple(
                tuple(_q("1") if row == column else _q("0") for column in range(8))
                for row in range(8)
            ),
        )
        for face in all_faces
        if len(face) > 1
        for position in range(len(face))
    )
    constructed = from_cover_maps(
        complex_value, SheafField.RATIONAL, None, stalks, cover_maps
    )
    assert constructed.sheaf is not None
    with pytest.raises(
        OperationResourceAdmissionError, match="full compatibility matrix"
    ):
        sections(constructed.sheaf)


def test_catalog_section_operation_exposes_a_reusable_target_space() -> None:
    tool = next(
        item
        for item in SHEAF_TOOLS
        if item.operation_id == "cellular_sheaf.sections.compute"
    )
    request = tool.request_type.model_validate({"sheaf": _constant_interval()})
    result = tool.run(request)
    assert isinstance(result, SheafSectionSpace)
    assert result.dimension == 1
    assert tuple(item.simplex for item in result.evaluations) == (
        ("a",),
        ("b",),
        ("a", "b"),
    )
    assert any(
        tool.operation_id == "cellular_sheaf.sections.compute" for tool in SHEAF_TOOLS
    )
