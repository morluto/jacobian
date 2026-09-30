"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/geometry/crystallographic/extensions/test_polytope_facet_pairings.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicPolytopePairingRequest,
    FiniteLatticeExtension,
    PolytopeFacetPairing,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    affine_section_realization,
    pair_crystallographic_polytope_facets,
)
from jacobian.math.geometry.polytopes._models import (
    RationalVPolytope,
)


def _request() -> CrystallographicPolytopePairingRequest:
    source = FiniteLatticeExtension(
        multiplication_table=((0,),),
        action_matrices=(((1, 0), (0, 1)),),
        factor_set=(((0, 0),),),
    )
    polytope = RationalVPolytope(
        space={"axes": ("x", "y")},
        vertices=tuple(
            {
                "vertex_id": f"v{index}",
                "coordinates": tuple(
                    CanonicalRational.from_fraction(Fraction(value)) for value in point
                ),
            }
            for index, point in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
        ),
    )
    return CrystallographicPolytopePairingRequest(
        affine_realization=affine_section_realization(source),
        polytope=polytope,
        lattice_axes=("x", "y"),
        pairings=(
            PolytopeFacetPairing(
                source_facet_index=0,
                target_facet_index=3,
                lattice_translation=(1, 0),
                holonomy_element=0,
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
                lattice_translation=(-1, 0),
                holonomy_element=0,
            ),
        ),
    )


def _pair(request: CrystallographicPolytopePairingRequest):
    """Call the native pairing function with unpacked domain arguments."""
    return pair_crystallographic_polytope_facets(
        request.affine_realization,
        request.polytope,
        request.lattice_axes,
        request.pairings,
    )


def test_pairing_operation_is_published_with_square_example() -> None:
    tool = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id
        == "crystallographic.extension.polytope_facet_pairings.compute"
    )

    assert tool.examples[0].name == "unit_square_translation_pairings"
    # ``dispatch.parse_operation_input`` is exactly this projection, inlined so a
    # math test does not import the product dispatch boundary.
    example_request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    assert tool.run(example_request).facet_profile.dimension == 2
