"""Boundary regressions for the finite topology extension operations."""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology._models import SimplicialComplexRequest, canonical_complex
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    FiniteCellularSheaf,
    SheafField,
    SheafStalk,
)
from jacobian.math.topology.cellular_sheaves.extensions import morphism
from jacobian.math.topology.cellular_sheaves.operations import from_cover_maps
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
    relative_homology,
    triangulate,
)
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)
from jacobian.math.topology.edge_paths.presentation_maps import homomorphism
from jacobian.math.topology.release import (
    CliqueRequest,
    FacePosetRequest,
    OrientabilityRequest,
    clique_complex,
    face_poset,
    orientability,
)
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.maps import (
    SimplicialMapRequest,
    simplicial_map,
)


def _complex(*facets: tuple[str, ...]) -> SimplicialComplexRequest:
    vertices = tuple(sorted({vertex for facet in facets for vertex in facet}))
    return SimplicialComplexRequest(vertices=vertices, facets=facets)


def test_zero_manifold_orientability_has_independent_signs() -> None:
    result = orientability(
        OrientabilityRequest(complex=_complex(("a",), ("b",), ("c",)))
    )
    assert result.orientable is True
    assert result.facet_signs == (1, 1, 1)
    assert result.obstruction_ridge is None


def test_face_poset_rejects_barycentric_output_before_enumeration() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        face_poset(
            FacePosetRequest(
                complex=_complex(("a", "b", "c", "d", "e", "f"),)
            )
        )


def test_clique_complex_handles_many_isolated_vertices() -> None:
    result = clique_complex(
        CliqueRequest(complex=_complex(*((str(i),) for i in range(20))))
    )
    assert len(result.clique_facets) == 20


def test_relative_cubical_homology_rejects_composite_modulus() -> None:
    square = CubicalCell(intervals=((0, 1), (0, 1)))
    with pytest.raises(OperationDomainValidationError):
        relative_homology(
            RelativeCubicalHomologyRequest(
                cells=(square,), subcomplex_cells=(square,), prime=4
            )
        )


def test_cubical_triangulation_rejects_factorial_materialization() -> None:
    cube = CubicalCell(intervals=tuple((0, 1) for _ in range(10)))
    with pytest.raises(OperationResourceAdmissionError):
        triangulate(CubicalTriangulationRequest(cells=(cube,)))


def test_sheaf_morphism_rejects_forged_incomplete_diagram() -> None:
    complex_ = canonical_complex(("a", "b"), (("a", "b"),))
    stalks = tuple(
        SheafStalk(simplex=face, basis=("x",))
        for group in complex_.faces_by_dimension
        for face in group.faces
    )
    covers = tuple(
        CoverRestrictionMatrix(source=face, target=("a", "b"), entries=(("1",),))
        for face in (("a",), ("b",))
    )
    complete = from_cover_maps(
        complex_, SheafField.RATIONAL, None, stalks, covers
    ).sheaf
    assert complete is not None
    forged = FiniteCellularSheaf.model_construct(
        complex=complete.complex,
        coefficient_field=complete.coefficient_field,
        prime=complete.prime,
        stalks=complete.stalks,
        cover_restrictions=complete.cover_restrictions[:-1],
        derived_restrictions=complete.derived_restrictions,
        diamonds=complete.diamonds,
        comparable_pairs=complete.comparable_pairs,
    )
    components = tuple(
        (".".join(stalk.simplex), (("1",),)) for stalk in stalks
    )
    with pytest.raises(OperationDomainValidationError):
        morphism(forged, complete, components)


def test_filtered_map_requires_exhaustive_filtration() -> None:
    complex_ = ChainComplexValue(
        coefficient_ring=CoefficientRing.RATIONAL,
        basis_sizes=(1,),
        degree_min=0,
        degree_max=0,
        differential_matrices=(),
    )
    empty = (FiltrationLevel(subspaces=(FilteredSubspace(vectors=()),)),)
    request = FilteredChainMapRequest(
        source=complex_, target=complex_, source_filtration=empty,
        target_filtration=empty, maps=(((),),),
    )
    with pytest.raises(OperationDomainValidationError):
        filtered_map(request)


def _word(*letters: tuple[int, int]) -> FiniteGroupWord:
    return FiniteGroupWord(
        letters=tuple(WordLetter(generator=i, exponent=e) for i, e in letters)
    )


def test_presentation_map_proves_short_normal_closure_witness() -> None:
    source = FiniteGroupPresentation(generators=("x",), relators=(_word((0, 1)),))
    target = FiniteGroupPresentation(
        generators=("a",), relators=(_word((0, 1), (0, 1)),)
    )
    result = homomorphism(source, target, (_word((0, 1), (0, 1), (0, 1), (0, 1)),))
    assert result.relators_preserved is True
    assert result.normal_closure_status == "PROVED"


def test_simplicial_map_reestablishes_forged_identity_metadata() -> None:
    forged = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=2,
        sets=(("v0", "v1"), ("e",), ("t",)),
        face_maps=(((0,), (1,)), ((0,), (0,), (0,))),
        degeneracy_maps=(((0, 0),), ((0,), (0,))),
        total_simplices=4,
        checked_identities=0,
    )
    with pytest.raises(OperationDomainValidationError):
        simplicial_map(
            SimplicialMapRequest(
                source=forged,
                target=forged,
                maps=((0, 0), (0,), (0,)),
            )
        )
