from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology._models import FiniteSimplicialComplex, canonical_complex
from jacobian.math.topology._structural import (
    InducedSubcomplexRequest,
    compute_induced_subcomplex,
)
from jacobian.math.topology._tools import TOOLS


def test_induced_subcomplex_and_source_face_transport() -> None:
    source = canonical_complex(("a", "b", "c", "d"), (("a", "b", "c"), ("c", "d")))
    result = compute_induced_subcomplex(
        InducedSubcomplexRequest(complex=source, selected_vertices=("a", "c", "d"))
    )

    assert result.induced_complex == canonical_complex(
        ("a", "c", "d"), (("a", "c"), ("c", "d"))
    )
    source_face_set = {
        face for dimension in source.faces_by_dimension for face in dimension.faces
    }
    retained = {face for face in source_face_set if set(face).issubset({"a", "c", "d"})}
    assert {entry.source_face for entry in result.face_images} == source_face_set
    assert {
        entry.source_face
        for entry in result.face_images
        if entry.induced_face is not None
    } == retained
    assert all(
        entry.induced_face == entry.source_face
        for entry in result.face_images
        if entry.induced_face is not None
    )
    assert result.model_validate(result.model_dump()) == result


def test_induced_subcomplex_composes_as_vertex_restriction() -> None:
    source = canonical_complex(("a", "b", "c", "d"), (("a", "b", "c"), ("c", "d")))
    once = compute_induced_subcomplex(
        InducedSubcomplexRequest(complex=source, selected_vertices=("a", "b", "c"))
    ).induced_complex
    twice = compute_induced_subcomplex(
        InducedSubcomplexRequest(complex=once, selected_vertices=("a", "c"))
    ).induced_complex
    direct = compute_induced_subcomplex(
        InducedSubcomplexRequest(complex=source, selected_vertices=("a", "c"))
    ).induced_complex
    assert twice == direct == canonical_complex(("a", "c"), (("a", "c"),))


def test_induced_subcomplex_rejects_repeated_or_absent_vertices() -> None:
    source = canonical_complex(("a", "b"), (("a", "b"),))
    for selected in (("a", "a"), ("missing",)):
        with pytest.raises(OperationDomainValidationError):
            compute_induced_subcomplex(
                InducedSubcomplexRequest(complex=source, selected_vertices=selected)
            )


def test_induced_subcomplex_rejects_forged_canonical_vertex_axis() -> None:
    valid = canonical_complex(("a", "b"), (("a", "b"),))
    forged = FiniteSimplicialComplex.model_construct(
        **{
            **{
                field: getattr(valid, field)
                for field in FiniteSimplicialComplex.model_fields
            },
            "vertices": ("a", "b", "orphan"),
        }
    )
    request = InducedSubcomplexRequest.model_construct(
        complex=forged, selected_vertices=("orphan",)
    )
    with pytest.raises(OperationDomainValidationError):
        compute_induced_subcomplex(request)


def test_induced_subcomplex_is_discoverable() -> None:
    assert any(
        tool.operation_id == "topology.simplicial_complex.induced_subcomplex.compute"
        for tool in TOOLS
    )
