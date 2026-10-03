"""Exact product-cell oracle for checked translation tori."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairingRequest,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    check_crystallographic_fundamental_domain,
    pair_crystallographic_polytope_facets,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori.operations import (
    translation_torus_quotient_chains,
)
from jacobian.math.geometry.polytopes._models import (
    RationalVPolytope,
    _canonical_v_polytope_vertices,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    IntegralHomologyGroupValue,
)


def _checked_translation_torus(
    generators: tuple[tuple[int, ...], ...], *, expected_fundamental: bool = True
) -> CrystallographicFundamentalDomainResult:
    dimension = len(generators)
    extension = FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(
            tuple(
                tuple(int(row == column) for column in range(dimension))
                for row in range(dimension)
            ),
        ),
        factor_set=((tuple(0 for _ in range(dimension)),),),
    )
    points_by_mask = {
        tuple(
            Fraction(
                sum(
                    generators[axis][coordinate]
                    for axis in range(dimension)
                    if mask & (1 << axis)
                )
            )
            for coordinate in range(dimension)
        ): mask
        for mask in range(1 << dimension)
    }
    points = tuple(points_by_mask)
    polytope = RationalVPolytope.model_validate(
        {
            "space": {"axes": tuple(f"x{axis}" for axis in range(dimension))},
            "vertices": tuple(
                {
                    "vertex_id": f"v{index:02d}",
                    "coordinates": tuple(
                        CanonicalRational.from_fraction(value) for value in point
                    ),
                }
                for index, point in enumerate(points)
            ),
        }
    )
    profile = facet_incidence(
        _canonical_v_polytope_vertices(polytope), dimension_bound=dimension
    )
    facets_by_bit: dict[tuple[int, int], int] = {}
    for index, facet in enumerate(profile.facets):
        masks = {
            points_by_mask[
                tuple(
                    value.as_fraction()
                    for value in profile.vertices[vertex].coordinates
                )
            ]
            for vertex in facet.source_vertex_indices
        }
        fixed_bits = [
            axis
            for axis in range(dimension)
            if all(mask & (1 << axis) for mask in masks)
            or all(not mask & (1 << axis) for mask in masks)
        ]
        assert len(fixed_bits) == 1
        fixed_bit = fixed_bits[0]
        fixed_value = int(all(mask & (1 << fixed_bit) for mask in masks))
        facets_by_bit[(fixed_bit, fixed_value)] = index
    pairings: list[PolytopeFacetPairing] = []
    for axis in range(dimension):
        low = facets_by_bit[(axis, 0)]
        high = facets_by_bit[(axis, 1)]
        direction = generators[axis]
        pairings.extend(
            (
                PolytopeFacetPairing(
                    source_facet_index=low,
                    target_facet_index=high,
                    lattice_translation=direction,
                    holonomy_element=0,
                ),
                PolytopeFacetPairing(
                    source_facet_index=high,
                    target_facet_index=low,
                    lattice_translation=tuple(-value for value in direction),
                    holonomy_element=0,
                ),
            )
        )
    request = CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(extension),
        polytope=polytope,
        lattice_axes=tuple(f"x{axis}" for axis in range(dimension)),
        pairings=tuple(pairings),
    )
    pairing = pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )
    checked = check_crystallographic_fundamental_domain(pairing)
    assert checked.is_fundamental_domain is expected_fundamental
    return checked


def _checked_cube() -> CrystallographicFundamentalDomainResult:
    return _checked_translation_torus(((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def _free_ranks(chain: ChainComplexValue) -> tuple[int, ...]:
    ranks = []
    for group in homology_groups(chain).homology_groups:
        assert isinstance(group, IntegralHomologyGroupValue)
        ranks.append(group.free_rank)
    return tuple(ranks)


def test_checked_cube_gives_integral_torus_product_chains() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    chain = result.quotient_chain_complex

    assert chain.basis_sizes == (1, 3, 3, 1)
    assert chain.differential_matrices == (
        ((0, 0, 0),),
        ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ((0,), (0,), (0,)),
    )
    assert _free_ranks(chain) == (
        1,
        3,
        3,
        1,
    )


def test_serialized_source_composes_without_losing_exact_cell_data() -> None:
    source = _checked_cube()
    decoded = type(source).model_validate_json(source.model_dump_json())
    result = translation_torus_quotient_chains(decoded)

    assert tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in result.circle_directions
    ) == tuple(
        tuple(Fraction(value) for value in row)
        for row in ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    )
    assert result.source == decoded


def test_result_value_rejects_nonfundamental_source() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    payload = result.model_dump(mode="python")
    payload["source"]["is_fundamental_domain"] = False

    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_result_value_rejects_forged_circle_directions() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    payload = result.model_dump(mode="python")
    payload["circle_directions"] = ((0, 0, 0),) * 3

    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_result_value_rejects_nonzero_differential() -> None:
    result = translation_torus_quotient_chains(_checked_cube())
    payload = result.model_dump(mode="python")
    chain = payload["quotient_chain_complex"]
    chain["differential_matrices"] = (
        ((1, 0, 0),),
        *chain["differential_matrices"][1:],
    )

    with pytest.raises(ValidationError):
        type(result).model_validate(payload)


def test_checked_sheared_parallelepiped_recovers_integral_directions() -> None:
    generators = ((1, 1, 0), (0, 1, 1), (0, 0, 1))
    result = translation_torus_quotient_chains(_checked_translation_torus(generators))

    assert tuple(
        tuple(value.as_fraction() for value in vector)
        for vector in result.circle_directions
    ) == tuple(
        tuple(Fraction(value) for value in row)
        for row in ((0, 0, 1), (0, 1, 1), (1, 1, 0))
    )
    assert _free_ranks(result.quotient_chain_complex) == (1, 3, 3, 1)


def test_four_dimensional_translation_torus_chains() -> None:
    result = translation_torus_quotient_chains(
        _checked_translation_torus(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )
    )

    result = type(result).model_validate_json(result.model_dump_json())
    assert result.quotient_chain_complex.basis_sizes == (1, 4, 6, 4, 1)
    assert _free_ranks(result.quotient_chain_complex) == (1, 4, 6, 4, 1)


@pytest.mark.parametrize(
    ("generators", "expected_ranks"),
    (
        (((1,),), (1, 1)),
        (((-1,),), (1, 1)),
        (((1, 0), (0, 1)), (1, 2, 1)),
        (((1, -1), (0, -1)), (1, 2, 1)),
    ),
)
def test_lower_rank_translation_tori(
    generators: tuple[tuple[int, ...], ...], expected_ranks: tuple[int, ...]
) -> None:
    result = translation_torus_quotient_chains(_checked_translation_torus(generators))
    result = type(result).model_validate_json(result.model_dump_json())

    assert _free_ranks(result.quotient_chain_complex) == expected_ranks


@pytest.mark.parametrize("wire", [False, True])
def test_coherent_negative_source_cannot_claim_translation_torus(wire: bool) -> None:
    result = translation_torus_quotient_chains(_checked_translation_torus(((1,),)))
    # [0,2] really fails for the unit translation lattice. Its retained volumes
    # and overlap witness are coherent, so the source model itself must accept it.
    negative = _checked_translation_torus(((2,),), expected_fundamental=False)
    assert negative.polytope_volume.as_fraction() == 2
    assert negative.quotient_covolume.as_fraction() == 1
    assert negative.overlap_translation is not None
    mode = "json" if wire else "python"
    payload = result.model_dump(mode=mode)
    payload["source"] = negative.model_dump(mode=mode)
    payload["circle_directions"] = (
        (CanonicalRational(num=2, den=1).model_dump(mode=mode),),
    )
    with pytest.raises(ValidationError) as error:
        if wire:
            type(result).model_validate_json(json.dumps(payload))
        else:
            type(result).model_validate(payload)
    assert any(
        item["type"] == "crystallographic.translation_torus_chain"
        for item in error.value.errors()
    )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "scales", [(Fraction(2), Fraction(1, 2)), (Fraction(2), Fraction(1))]
)
def test_torus_directions_must_generate_the_retained_unit_lattice(
    scales: tuple[Fraction, Fraction], wire: bool
) -> None:
    result = translation_torus_quotient_chains(
        _checked_translation_torus(((1, 0), (0, 1)))
    )
    mode = "json" if wire else "python"
    payload = result.model_dump(mode=mode)
    pairing = payload["source"]["source"]
    coordinate_rows = [
        *(vertex["coordinates"] for vertex in pairing["polytope"]["vertices"]),
        *(vertex["coordinates"] for vertex in pairing["facet_profile"]["vertices"]),
        *payload["circle_directions"],
    ]
    for row in coordinate_rows:
        for axis, coordinate in enumerate(row):
            value = (
                Fraction(int(coordinate["num"]), int(coordinate["den"])) * scales[axis]
            )
            coordinate.update(
                CanonicalRational.from_fraction(value).model_dump(mode=mode)
            )
    with pytest.raises(ValidationError) as exc_info:
        if wire:
            type(result).model_validate_json(json.dumps(payload))
        else:
            type(result).model_validate(payload)
    assert (
        exc_info.value.errors()[0]["type"]
        == "crystallographic.translation_torus_directions"
    )


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize(
    "claim", ["factor_set", "section_shift", "pairing_translation", "volume"]
)
def test_torus_source_relations_are_checked_without_replaying_geometry(
    claim: str, wire: bool
) -> None:
    result = translation_torus_quotient_chains(_checked_translation_torus(((1,),)))
    mode = "json" if wire else "python"
    payload = result.model_dump(mode=mode)
    pairing = payload["source"]["source"]
    realization = pairing["affine_realization"]
    if claim == "volume":
        for field in ("polytope_volume", "quotient_covolume"):
            payload["source"][field] = CanonicalRational(num=2, den=1).model_dump(
                mode=mode
            )
    elif claim == "factor_set":
        realization["source"]["factor_set"] = ((("1" if wire else 1,),),)
    elif claim == "section_shift":
        realization["section_maps"][0]["section_shift"][0].update(
            CanonicalRational(num=1, den=1).model_dump(mode=mode)
        )
    else:
        for side in pairing["pairings"]:
            side["lattice_translation"] = tuple(
                str(2 * int(value)) if wire else 2 * value
                for value in side["lattice_translation"]
            )
    with pytest.raises(ValidationError, match="translation_torus"):
        if wire:
            type(result).model_validate_json(json.dumps(payload))
        else:
            type(result).model_validate(payload)
