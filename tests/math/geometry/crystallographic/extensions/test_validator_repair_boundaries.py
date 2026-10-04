"""Regressions for the crystallographic extension trust boundaries.

Each result validator must refuse a decoded payload that contradicts the
source it retains, and the collections must be bounded before they are
materialized.
"""

from __future__ import annotations

import json
import time
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingRequest,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.face_orbits import (
    quotient_face_orbit_complex,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    pair_crystallographic_polytope_facets,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori._models import (
    BieberbachTranslationTorusChains,
)
from jacobian.math.geometry.polytopes._models import (
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)


def _polygon_request(*, klein: bool = False) -> CrystallographicPolytopePairingRequest:
    extension = FiniteLatticeExtension(
        multiplication_table=((0, 1), (1, 0)),
        action_matrices=(((1, 0), (0, 1)), ((1, 0), (0, -1))),
        factor_set=(((0, 0), (0, 0)), ((0, 0), (1, 0))),
    )
    points: tuple[tuple[Fraction, Fraction], ...] = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 2), Fraction(0)),
        (Fraction(1, 2), Fraction(1)),
    )
    return CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(extension),
        polytope=RationalVPolytope(
            space=RationalCoordinateSpace(axes=("x", "y")),
            vertices=tuple(
                RationalPolytopeVertex(
                    vertex_id=f"v{index}",
                    coordinates=tuple(
                        CanonicalRational.from_fraction(value) for value in point
                    ),
                )
                for index, point in enumerate(points)
            ),
        ),
        lattice_axes=("x", "y"),
        pairings=(
            PolytopeFacetPairing(
                source_facet_index=0,
                target_facet_index=3,
                lattice_translation=(0, 1),
                holonomy_element=1,
            ),
            PolytopeFacetPairing(
                source_facet_index=1,
                target_facet_index=2,
                lattice_translation=(0, 1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=2,
                target_facet_index=1,
                lattice_translation=(0, -1),
                holonomy_element=0,
            ),
            PolytopeFacetPairing(
                source_facet_index=3,
                target_facet_index=0,
                lattice_translation=(-1, 1),
                holonomy_element=1,
            ),
        ),
    )


def _real_complex() -> BieberbachFaceOrbitComplex:
    request = _polygon_request()
    pairing = pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )
    return quotient_face_orbit_complex(
        check_crystallographic_fundamental_domain(pairing)
    )


def _klein_domain() -> CrystallographicFundamentalDomainResult:
    request = _polygon_request(klein=True)
    pairing = pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )
    return check_crystallographic_fundamental_domain(pairing)


# --- an orbit map must land on the endpoint the pairing sends it to ---------


def test_swapped_target_vertex_is_refused() -> None:
    """The other endpoint of the same target edge is not the affine image."""
    original = _real_complex()
    payload = original.model_dump()
    facets = original.source.source.facet_profile.facets
    for item in payload["orbit_maps"]:
        others = [
            vertex
            for vertex in facets[item["target_facet_index"]].source_vertex_indices
            if vertex != item["target_vertex_index"]
        ]
        if others:
            item["target_vertex_index"] = others[0]
            break
    with pytest.raises(ValueError):
        BieberbachFaceOrbitComplex.model_validate(payload)


# --- the differentials must augment the retained incidences -----------------


def test_forged_differential_is_refused() -> None:
    """A changed d2 column reports homology the incidences do not support."""
    original = _real_complex()
    payload = original.model_dump()
    matrices = payload["quotient_chain_complex"]["differential_matrices"]
    d2 = [list(row) for row in matrices[-1]]
    d2[0][0] = 1
    payload["quotient_chain_complex"]["differential_matrices"] = [
        *matrices[:-1],
        tuple(tuple(row) for row in d2),
    ]
    with pytest.raises(ValueError):
        BieberbachFaceOrbitComplex.model_validate(payload)


# --- oversized collections are bounded before they are materialized --------


def test_oversized_orbit_maps_are_bounded_by_the_field() -> None:
    original = _real_complex()
    payload = original.model_dump()
    payload["orbit_maps"] = [dict(payload["orbit_maps"][0]) for _ in range(200_000)]
    start = time.perf_counter()
    with pytest.raises(ValueError):
        BieberbachFaceOrbitComplex.model_validate(payload)
    # A field-level bound refuses without building every nested map first.
    assert time.perf_counter() - start < 0.5


def test_oversized_vertex_orbit_row_is_bounded() -> None:
    original = _real_complex()
    payload = original.model_dump()
    payload["vertex_orbits"] = [tuple(range(10_000))]
    with pytest.raises(ValueError):
        BieberbachFaceOrbitComplex.model_validate(payload)


def test_collections_declare_field_level_bounds() -> None:
    fields = BieberbachFaceOrbitComplex.model_fields
    for name in (
        "vertex_orbits",
        "edge_orbit_representatives",
        "orbit_maps",
        "boundary_1_to_0",
        "boundary_2_to_1",
    ):
        metadata = fields[name].metadata
        assert any(
            getattr(item, "max_length", None) is not None for item in metadata
        ), f"{name} has no field-level maximum length"


# --- a translation torus needs trivial holonomy and real directions ---------


def _forged_translation_torus_payload() -> dict[str, object]:
    def rational(value: int) -> CanonicalRational:
        return CanonicalRational(num=value, den=1)

    return {
        "source": _klein_domain().model_dump(),
        "circle_directions": ((rational(1), rational(2)), (rational(0), rational(1))),
        "quotient_chain_complex": {
            "coefficient_ring": "ZZ",
            "prime": None,
            "degree_min": 0,
            "degree_max": 2,
            "basis_sizes": (1, 2, 1),
            "differential_matrices": (((0, 0),), ((0,), (0,))),
        },
    }


def test_klein_bottle_source_cannot_carry_a_translation_torus() -> None:
    """A checked Klein-bottle source has nontrivial holonomy."""
    with pytest.raises(ValueError):
        BieberbachTranslationTorusChains.model_validate(
            _forged_translation_torus_payload()
        )


# --- negative controls -----------------------------------------------------


def test_real_payload_round_trips() -> None:
    original = _real_complex()
    again = BieberbachFaceOrbitComplex.model_validate(original.model_dump())
    assert len(again.orbit_maps) == len(original.orbit_maps)
    assert again.quotient_chain_complex == original.quotient_chain_complex


def test_real_payload_homology_is_unchanged() -> None:
    """The augmented differentials still produce the same homology."""
    from jacobian.math.topology.chain_complexes.operations import homology_groups

    original = _real_complex()
    from jacobian.math.topology.chain_complexes.values import IntegralHomologyGroupValue

    decoded = type(original).model_validate_json(original.model_dump_json())
    groups = homology_groups(decoded.quotient_chain_complex).homology_groups
    integral = []
    for group in groups:
        assert isinstance(group, IntegralHomologyGroupValue)
        integral.append(group)
    assert tuple(group.free_rank for group in integral) == (1, 1, 0)
    assert integral[1].torsion_invariant_factors == (2,)


def test_a_non_realization_is_still_refused_by_the_operation() -> None:
    """A wrong affine realization is still a domain rejection."""
    from jacobian.catalog.models import OperationDomainValidationError

    request = _polygon_request()
    wrong = request.affine_realization.model_copy(
        update={"section_maps": request.affine_realization.section_maps[:1] * 2}
    )
    with pytest.raises(OperationDomainValidationError):
        pair_crystallographic_polytope_facets(
            wrong,
            request.polytope,
            request.lattice_axes,
            request.pairings,
        )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [("holonomy_element", 2), ("lattice_translation", [0]), ("target_facet_index", 4)],
)
def test_pairing_source_axes_are_validated_before_endpoint_arithmetic(
    field: str, value: object, wire: bool
) -> None:
    payload = _real_complex().model_dump(mode="json" if wire else "python")
    if field == "lattice_translation":
        value = ["0"] if wire else (0,)
    # Preserve the duplicated pairing/map claim so the source-relative shape
    # check, rather than disagreement between two copies, must reject it.
    for pairing in payload["source"]["source"]["pairings"]:
        pairing[field] = value
    for orbit_map in payload["orbit_maps"]:
        orbit_map[field] = value
    with pytest.raises(ValidationError) as error:
        if wire:
            BieberbachFaceOrbitComplex.model_validate_json(json.dumps(payload))
        else:
            BieberbachFaceOrbitComplex.model_validate(payload)
    assert any(
        item["type"] == "crystallographic.extension.polytope_pairing_result"
        for item in error.value.errors()
    )
