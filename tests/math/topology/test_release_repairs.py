"""Regression coverage for repaired finite topology release boundaries."""

from __future__ import annotations

from typing import Any

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import (
    SimplicialComplexRequest,
    canonical_complex,
)
from jacobian.math.topology.cellular_sheaves._models import (
    FiniteCellularSheaf,
    SheafField,
    SheafRestriction,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.extensions import morphism
from jacobian.math.topology.chain_complexes._filtered_models import (
    FilteredSubspace,
    FiltrationLevel,
)
from jacobian.math.topology.chain_complexes.filtered_extensions import (
    FilteredChainMapRequest,
    filtered_map,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)
from jacobian.math.topology.cubical_complexes._models import CubicalCell
from jacobian.math.topology.cubical_complexes.extensions import (
    CubicalTriangulationRequest,
    RelativeCubicalHomologyRequest,
    boundary,
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
    HomologyManifoldRequest,
    OrientabilityRequest,
    clique_complex,
    face_poset,
    homology_manifold,
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
        complex=SimplicialComplexRequest.model_validate(
            {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]}
        )
    )
    poset = face_poset(request)
    assert poset.order_complex.closure_size == 25
    clique = clique_complex(
        CliqueRequest(
            complex=SimplicialComplexRequest.model_validate(
                {"vertices": ["a", "b", "c"], "facets": [["a", "b", "c"]]}
            )
        )
    )
    assert clique.clique_facets == (("a", "b", "c"),)


def test_clique_complex_expands_graph_beyond_source_dimension() -> None:
    graph = CliqueRequest(
        complex=SimplicialComplexRequest.model_validate(
            {
                "vertices": ["a", "b", "c", "d"],
                "facets": [
                    ["a", "b"],
                    ["a", "c"],
                    ["a", "d"],
                    ["b", "c"],
                    ["b", "d"],
                    ["c", "d"],
                ],
            }
        )
    )
    result = clique_complex(graph)
    assert result.source.dimension == 1
    assert result.clique_facets == (("a", "b", "c", "d"),)
    assert result.clique_complex.dimension == 3


def test_zero_dimensional_orientability_has_independent_facets() -> None:
    result = orientability(
        OrientabilityRequest(
            complex=SimplicialComplexRequest.model_validate(
                {
                    "vertices": ["a", "b", "c"],
                    "facets": [["a"], ["b"], ["c"]],
                }
            )
        )
    )
    assert result.orientable is True
    assert result.facet_signs == (1, 1, 1)
    assert result.obstruction_ridge is None


def test_homology_manifold_rejects_impure_empty_links_and_composite_fields() -> None:
    impure = homology_manifold(
        HomologyManifoldRequest(
            complex=SimplicialComplexRequest.model_validate(
                {
                    "vertices": ["a", "b", "c", "x"],
                    "facets": [["a", "b", "c"], ["x"]],
                }
            )
        )
    )
    assert impure.homology_manifold is False
    with pytest.raises(OperationDomainValidationError, match="prime field"):
        homology_manifold(
            HomologyManifoldRequest(
                complex=SimplicialComplexRequest.model_validate(
                    {"vertices": ["x"], "facets": [["x"]]}
                ),
                prime=4,
            )
        )


def test_boundary_includes_lower_dimensional_maximal_cells() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    disjoint_edge = CubicalCell(intervals=((3, 4), (0, 0)))

    result = boundary((square, disjoint_edge))

    assert result.maximal_cells == (square, disjoint_edge)
    edge_terms = tuple(term for term in result.terms if term.source == disjoint_edge)
    assert tuple((term.face, term.coefficient) for term in edge_terms) == (
        (CubicalCell(intervals=((4, 4), (0, 0))), 1),
        (CubicalCell(intervals=((3, 3), (0, 0))), -1),
    )
    assert {term.face for term in edge_terms}.issubset(result.boundary_cells)


def test_relative_homology_rejects_composite_modulus_before_rank() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    edge = CubicalCell(intervals=((0, 0), (0, 1)))
    with pytest.raises(OperationDomainValidationError, match="prime field"):
        relative_homology(
            RelativeCubicalHomologyRequest(
                cells=(square,), subcomplex_cells=(edge,), prime=4
            )
        )


def test_triangulation_retains_source_axis_through_serialization() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    result = triangulate(CubicalTriangulationRequest(cells=(square,)))
    restored = type(result).model_validate(result.model_dump(mode="json"))
    assert restored.source_cells == (square,)
    assert len(restored.complex.cells) == 9
    assert len(restored.simplices_by_cell) == len(restored.source_cells)


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
    covers = tuple((face, ("a", "b")) for face in (("a",), ("b",)))
    return FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        stalks=stalks,
        cover_restrictions=tuple(
            # A deliberately incomplete value is accepted by the wire model;
            # the morphism operation must establish completeness itself.
            SheafRestriction(
                source=source,
                target=target,
                row_basis=("x",),
                column_basis=("x",),
                entries=(("1",),),
                cover_path=(source, target),
            )
            for source, target in covers[:1]
        ),
    )


def test_sheaf_morphism_requires_a_proved_prime_field() -> None:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_.faces_by_dimension
        for face in group.faces
    )
    restrictions = tuple(
        SheafRestriction(
            source=source,
            target=("a", "b"),
            row_basis=("x",),
            column_basis=("x",),
            entries=(("1",),),
            cover_path=(source, ("a", "b")),
        )
        for source in (("a",), ("b",))
    )
    # The structural carrier accepts the declared modulus; the arithmetic
    # consumer must establish that GF(p) is actually a field.
    sheaf = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.PRIME_FIELD,
        prime=4,
        stalks=stalks,
        cover_restrictions=restrictions,
    )
    components = tuple((face, (("1",),)) for face in sheaf.canonical_face_order)
    with pytest.raises(OperationDomainValidationError, match="prime"):
        morphism(sheaf, sheaf, components)
    # Neither a serialized carrier nor a native bypass carries trusted field
    # provenance: both consumer boundaries must establish the same fact.
    restored = type(sheaf).model_validate_json(sheaf.model_dump_json())
    forged = type(sheaf).model_construct(**sheaf.model_dump())
    for authored in (restored, forged):
        with pytest.raises(OperationDomainValidationError, match="prime"):
            morphism(authored, authored, components)


def test_sheaf_morphism_uses_modular_arithmetic_and_tuple_axes() -> None:
    complex_ = canonical_complex(("a", "b", "a.b"), (("a", "b"), ("a.b",)))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_.faces_by_dimension
        for face in group.faces
    )
    restrictions = tuple(
        SheafRestriction(
            source=source,
            target=("a", "b"),
            row_basis=("x",),
            column_basis=("x",),
            entries=(("2",),),
            cover_path=(source, ("a", "b")),
        )
        for source in (("a",), ("b",))
    )
    sheaf = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.PRIME_FIELD,
        prime=2,
        stalks=stalks,
        cover_restrictions=restrictions,
        comparable_pairs=2,
    )
    components = tuple((face, (("2",),)) for face in sheaf.canonical_face_order)
    result = morphism(sheaf, sheaf, components)
    assert result.natural is True
    assert all(matrix == (("0",),) for _key, matrix in result.components)


def test_sheaf_morphism_preserves_width_through_zero_stalk() -> None:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    source_stalks = tuple(
        SheafStalk(simplex=face, basis=() if len(face) == 2 else ("x",))
        for face in faces
    )
    target_stalks = tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces)
    source = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        stalks=source_stalks,
        cover_restrictions=tuple(
            SheafRestriction(
                source=vertex,
                target=("a", "b"),
                row_basis=(),
                column_basis=("x",),
                entries=(),
                cover_path=(vertex, ("a", "b")),
            )
            for vertex in (("a",), ("b",))
        ),
    )
    target = FiniteCellularSheaf(
        complex=complex_,
        coefficient_field=SheafField.RATIONAL,
        stalks=target_stalks,
        cover_restrictions=tuple(
            SheafRestriction(
                source=vertex,
                target=("a", "b"),
                row_basis=("x",),
                column_basis=("x",),
                entries=(("0",),),
                cover_path=(vertex, ("a", "b")),
            )
            for vertex in (("a",), ("b",))
        ),
    )
    components = (
        (("a",), (("0",),)),
        (("b",), (("0",),)),
        (("a", "b"), ((),)),
    )

    result = morphism(source, target, components)

    assert result.natural is True


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
        coefficient_field=SheafField.RATIONAL,
        stalks=stalks,
        cover_restrictions=tuple(
            SheafRestriction(
                source=source,
                target=("a", "b"),
                row_basis=("x",),
                column_basis=("x",),
                entries=(("1",),),
                cover_path=(source, ("a", "b")),
            )
            for source in (("a",), ("b",))
        ),
    )
    components = (("a", (("bad",),)), ("b", (("1",),)), ("a.b", (("1",),)))
    with pytest.raises(OperationDomainValidationError, match="exact scalars"):
        morphism(sheaf, sheaf, components)
    malformed: Any = (("a", ((None,),)), ("b", (("1",),)), ("a.b", (("1",),)))
    with pytest.raises(OperationDomainValidationError, match="exact scalars"):
        morphism(sheaf, sheaf, malformed)


def test_relative_homology_admits_group_and_matrix_bounds_before_dense_work() -> None:
    cube = CubicalCell(intervals=((0, 1),) * 10)
    vertex = CubicalCell(intervals=((0, 0),) * 10)
    with pytest.raises(OperationResourceAdmissionError, match="chain group"):
        relative_homology(
            RelativeCubicalHomologyRequest(cells=(cube,), subcomplex_cells=(vertex,))
        )


def test_filtered_chain_map_reduces_compositions_and_map_output() -> None:
    complex_ = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=2,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=((("1",),),),
    )
    filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=(("1",),)),
                FilteredSubspace(vectors=(("1",),)),
            )
        ),
    )
    result = filtered_map(
        FilteredChainMapRequest(
            source=complex_,
            source_filtration=filtration,
            target=complex_,
            target_filtration=filtration,
            maps=((("0",),), (("2",),)),
        )
    )
    assert result.chain_map is True
    assert result.maps == ((("0",),), (("0",),))


def test_filtered_chain_map_preserves_width_through_zero_chain_group() -> None:
    source = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(0, 1),
        differential_matrices=((),),
    )
    target = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=((("0",),),),
    )
    source_filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=()),
                FilteredSubspace(vectors=(("1",),)),
            )
        ),
    )
    target_filtration = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=(("1",),)),
                FilteredSubspace(vectors=(("1",),)),
            )
        ),
    )

    result = filtered_map(
        FilteredChainMapRequest(
            source=source,
            source_filtration=source_filtration,
            target=target,
            target_filtration=target_filtration,
            maps=(((),), (("0",),)),
        )
    )

    assert result.chain_map is True
    assert result.filtration_preserving is True


def test_filtered_chain_map_rejects_non_nested_non_exhaustive_filtration() -> None:
    complex_ = ChainComplexValue(
        coefficient_ring=CoefficientRing.PRIME_FIELD,
        prime=2,
        degree_min=0,
        degree_max=1,
        basis_sizes=(1, 1),
        differential_matrices=((("1",),),),
    )
    malformed = (
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=(("1",),)),
                FilteredSubspace(vectors=(("1",),)),
            )
        ),
        FiltrationLevel(
            subspaces=(
                FilteredSubspace(vectors=(("0",),)),
                FilteredSubspace(vectors=(("0",),)),
            )
        ),
    )
    with pytest.raises(OperationDomainValidationError, match="not contained"):
        filtered_map(
            FilteredChainMapRequest(
                source=complex_,
                source_filtration=malformed,
                target=complex_,
                target_filtration=malformed,
                maps=((("1",),), (("1",),)),
            )
        )


def test_standard_simplex_rejects_derived_carrier_overflow_before_tables() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="32 simplices"):
        standard_simplex(4, 2)
    with pytest.raises(OperationResourceAdmissionError, match="32 simplices"):
        standard_simplex(3, 3)


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
