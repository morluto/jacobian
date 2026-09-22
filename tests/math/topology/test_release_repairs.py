"""Regression coverage for repaired finite topology release boundaries."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves._models import (
    FiniteCellularSheaf,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.extensions import morphism
from jacobian.math.topology.cubical_complexes._models import CubicalCell
from jacobian.math.topology.cubical_complexes.extensions import (
    CubicalTriangulationRequest,
    RelativeCubicalHomologyRequest,
    relative_homology,
    triangulate,
)
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)
from jacobian.math.topology.edge_paths.presentation_maps import direct_relator_match
from jacobian.math.topology.release import (
    CliqueRequest,
    FacePosetRequest,
    OrientabilityRequest,
    clique_complex,
    face_poset,
    orientability,
)
from jacobian.math.topology.simplicial_sets.maps import normalized_chains
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def test_standard_simplex_normalized_prefix_has_square_zero_boundary() -> None:
    simplex = standard_simplex(1, 2)
    normalized = normalized_chains(simplex)
    assert normalized.differential_squared_zero is True
    assert normalized.nondegenerate_counts == (2, 1, 0)


def test_face_poset_and_clique_reconstruct_small_triangle() -> None:
    request = FacePosetRequest(
        complex={"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]}
    )
    poset = face_poset(request)
    assert poset.order_complex.closure_size == 25
    clique = clique_complex(
        CliqueRequest(complex={"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]})
    )
    assert clique.clique_facets == (("a", "b", "c"),)


def test_zero_dimensional_orientability_has_independent_facets() -> None:
    result = orientability(
        OrientabilityRequest(
            complex={"vertices": ["a", "b", "c"], "facets": [["a"], ["b"], ["c"]]}
        )
    )
    assert result.orientable is True
    assert result.facet_signs == (1, 1, 1)
    assert result.obstruction_ridge is None


def test_relative_homology_rejects_composite_modulus_before_rank() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    edge = CubicalCell(intervals=((0, 0), (0, 1)))
    with pytest.raises(OperationDomainValidationError, match="prime field"):
        relative_homology(
            RelativeCubicalHomologyRequest(
                cells=(square,), subcomplex_cells=(edge,), prime=4
            )
        )


def test_triangulation_rejects_factorial_output_before_materialization() -> None:
    cube = CubicalCell(intervals=tuple((0, 1) for _ in range(10)))
    with pytest.raises(OperationResourceAdmissionError, match="simplex output"):
        triangulate(CubicalTriangulationRequest(cells=(cube,)))


def _rank_one_interval_sheaf() -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_.faces_by_dimension
        for face in group.faces
    )
    covers = tuple(
        (face, ("a", "b"))
        for face in (("a",), ("b",))
    )
    return FiniteCellularSheaf(
        complex=complex_,
        coefficient_field="QQ",
        stalks=stalks,
        cover_restrictions=tuple(
            # A deliberately incomplete value is accepted by the wire model;
            # the morphism operation must establish completeness itself.
            {
                "source": source,
                "target": target,
                "row_basis": ("x",),
                "column_basis": ("x",),
                "entries": (("1",),),
                "cover_path": (source, target),
            }
            for source, target in covers[:1]
        ),
    )


def test_sheaf_morphism_rejects_incomplete_forged_diagram() -> None:
    sheaf = _rank_one_interval_sheaf()
    components = (("a", (("1",),)), ("b", (("1",),)), ("a.b", (("1",),)))
    with pytest.raises(OperationDomainValidationError, match="every canonical"):
        morphism(sheaf, sheaf, components)


def test_sheaf_morphism_translates_malformed_scalar_to_owner_error() -> None:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_.faces_by_dimension
        for face in group.faces
    )
    sheaf = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field="QQ",
        stalks=stalks,
        cover_restrictions=tuple(
            {
                "source": source,
                "target": ("a", "b"),
                "row_basis": ("x",),
                "column_basis": ("x",),
                "entries": (("1",),),
                "cover_path": (source, ("a", "b")),
            }
            for source in (("a",), ("b",))
        ),
    )
    components = (("a", (("bad",),)), ("b", (("1",),)), ("a.b", (("1",),)))
    with pytest.raises(OperationDomainValidationError, match="exact scalars"):
        morphism(sheaf, sheaf, components)


def test_presentation_release_is_only_a_direct_relator_witness() -> None:
    source = FiniteGroupPresentation(
        generators=("x",),
        relators=(FiniteGroupWord(letters=(WordLetter(generator=0, exponent=1),)),),
    )
    target = FiniteGroupPresentation(
        generators=("a",),
        relators=(
            FiniteGroupWord(
                letters=(
                    WordLetter(generator=0, exponent=1),
                    WordLetter(generator=0, exponent=1),
                )
            ),
        ),
    )
    image = FiniteGroupWord(
        letters=tuple(WordLetter(generator=0, exponent=1) for _ in range(4))
    )
    result = direct_relator_match(source, target, (image,))
    assert result.direct_relators_matched is False
    assert not hasattr(result, "relators_preserved")
