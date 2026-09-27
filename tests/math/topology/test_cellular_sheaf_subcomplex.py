"""Tests for restriction of cellular sheaves to subcomplexes."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafField,
    SheafStalk,
    from_cover_maps,
    restrict_to_subcomplex,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    SheafSubcomplexRequest,
)
from jacobian.math.topology.cellular_sheaves._tools import TOOLS as SHEAF_TOOLS


def _q(value: str | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _triangle_sheaf() -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = []
    for target in cells:
        for i in range(len(target)):
            source = target[:i] + target[i + 1 :]
            if source in cells:
                # Distinct exact edge values make the independently checked
                # composite to the triangle's top cell observable.
                value = _q("2") if source == ("a",) else _q("1")
                covers.append(
                    CoverRestrictionMatrix(
                        source=source, target=target, entries=((value,),)
                    )
                )
    result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(SheafStalk(simplex=cell, basis=(f"b{len(cell)}",)) for cell in cells),
        tuple(covers),
    )
    assert result.sheaf is not None
    return result.sheaf


def test_restriction_filters_cells_and_keeps_derived_exact_maps() -> None:
    sheaf = _triangle_sheaf()
    edge = canonical_complex(("a", "b"), (("a", "b"),))
    result = restrict_to_subcomplex(sheaf, edge)
    target = result.subcomplex
    assert target.complex == edge
    assert target.coefficient_field is SheafField.RATIONAL
    assert [stalk.simplex for stalk in target.stalks] == [
        ("a",),
        ("b",),
        ("a", "b"),
    ]
    assert all(stalk.basis == (f"b{len(stalk.simplex)}",) for stalk in target.stalks)
    assert {(r.source, r.target) for r in target.cover_restrictions} == {
        (("a",), ("a", "b")),
        (("b",), ("a", "b")),
    }
    direct_derived = next(
        r
        for r in sheaf.derived_restrictions
        if r.source == ("a",) and r.target == ("a", "b", "c")
    )
    assert direct_derived.entries == ((_q("2"),),)
    full = restrict_to_subcomplex(sheaf, sheaf.complex).subcomplex
    assert full.derived_restrictions == sheaf.derived_restrictions


def test_restriction_composes_through_nested_subcomplexes() -> None:
    sheaf = _triangle_sheaf()
    edge = restrict_to_subcomplex(
        sheaf, canonical_complex(("a", "b"), (("a", "b"),))
    ).subcomplex
    vertex_through_edge = restrict_to_subcomplex(
        edge, canonical_complex(("a",), (("a",),))
    ).subcomplex
    vertex_direct = restrict_to_subcomplex(
        sheaf, canonical_complex(("a",), (("a",),))
    ).subcomplex
    assert vertex_through_edge.model_dump(mode="json") == vertex_direct.model_dump(
        mode="json"
    )


def test_nonincluded_face_is_rejected() -> None:
    sheaf = _triangle_sheaf()
    outside = canonical_complex(("a", "z"), (("a", "z"),))
    with pytest.raises(OperationDomainValidationError, match="belong"):
        restrict_to_subcomplex(sheaf, outside)


def test_authored_subcomplex_face_closure_is_admitted() -> None:
    sheaf = _triangle_sheaf()
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    edge_group = complex_.faces_by_dimension[1]
    incomplete_group = edge_group.model_copy(update={"faces": edge_group.faces[:-1]})
    incomplete = complex_.model_copy(
        update={
            "faces_by_dimension": (
                complex_.faces_by_dimension[0],
                incomplete_group,
                complex_.faces_by_dimension[2],
            ),
            "f_vector": (3, 2, 1),
            "closure_size": 6,
        }
    )
    with pytest.raises(OperationDomainValidationError, match="canonical"):
        restrict_to_subcomplex(sheaf, incomplete)


def test_subcomplex_rebuilds_forged_derived_restrictions() -> None:
    sheaf = _triangle_sheaf()
    derived = tuple(
        item.model_copy(update={"entries": ((_q("1"),),)})
        if item.source == ("a",) and item.target == ("a", "b", "c")
        else item
        for item in sheaf.derived_restrictions
    )
    forged = sheaf.model_copy(update={"derived_restrictions": derived})
    with pytest.raises(OperationDomainValidationError, match="cover-map composite"):
        restrict_to_subcomplex(forged, sheaf.complex)


def test_subcomplex_rejects_stalks_outside_the_rank_envelope() -> None:
    complex_ = canonical_complex(("a",), (("a",),))
    oversized = FiniteCellularSheaf.model_construct(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        prime=None,
        stalks=(SheafStalk(simplex=("a",), basis=tuple(f"b{i}" for i in range(9))),),
        cover_restrictions=(),
        derived_restrictions=(),
        diamonds=0,
        comparable_pairs=0,
    )
    with pytest.raises(OperationResourceAdmissionError, match="stalk"):
        restrict_to_subcomplex(oversized, complex_)


def test_catalog_publishes_subcomplex_restriction() -> None:
    tool = next(
        item
        for item in SHEAF_TOOLS
        if item.operation_id == "cellular_sheaf.subcomplex.restrict"
    )
    assert tool.request_type is SheafSubcomplexRequest
    request = SheafSubcomplexRequest(
        sheaf=_triangle_sheaf(),
        subcomplex=canonical_complex(("a",), (("a",),)),
    )
    assert tool.run(request).subcomplex.complex == request.subcomplex
