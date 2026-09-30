"""Focused regressions for native trust boundaries and decoding contracts."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.topology._models import canonical_complex
from jacobian.math.topology.cellular_sheaves import (
    FiniteCellularSheaf,
    SheafField,
    SheafStalk,
    from_cover_maps,
    morphism,
    sheaf_cochain_complex,
)
from jacobian.math.topology.cellular_sheaves._models import (
    CoverRestrictionMatrix,
    SheafCochainComplex,
)
from jacobian.math.topology.cellular_sheaves.morphism_image import (
    SheafMorphismImageResult,
    image_of_morphism,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalProductResult,
    CubicalSkeletonResult,
)
from jacobian.math.topology.cubical_complexes.operations import (
    product as cubical_product,
)
from jacobian.math.topology.cubical_complexes.operations import (
    skeleton as cubical_skeleton,
)
from jacobian.math.topology.release import OrderComplexRequest, order_complex
from jacobian.math.topology.simplicial_sets import (
    FiniteTruncatedSimplicialSet,
    TruncatedSimplicialMap,
)
from jacobian.math.topology.simplicial_sets.degenerate_submodule import (
    DegenerateSubmoduleRequest,
    degenerate_submodule,
)
from jacobian.math.topology.simplicial_sets.map_preimage import (
    SimplicialMapPreimageRequest,
    simplicial_map_preimage,
)
from jacobian.math.topology.simplicial_sets.skeleton import (
    SimplicialSetSkeletonResult,
    simplicial_set_skeleton,
)
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
from jacobian.math.topology.simplicial_sets.subset_models import SimplicialSubsetPrefix


def _q(value: int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _triangle_sheaf() -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    faces = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    covers = tuple(
        (face, coface)
        for coface in faces
        for face in faces
        if len(coface) == len(face) + 1 and set(face) < set(coface)
    )
    result = from_cover_maps(
        complex_,
        SheafField.RATIONAL,
        None,
        tuple(SheafStalk(simplex=face, basis=("x",)) for face in faces),
        tuple(
            CoverRestrictionMatrix(source=face, target=coface, entries=((_q(1),),))
            for face, coface in covers
        ),
    )
    assert result.sheaf is not None
    return result.sheaf


def _constant_triangle() -> FiniteCellularSheaf:
    complex_ = canonical_complex(("a", "b", "c"), (("a", "b", "c"),))
    cells = tuple(face for group in complex_.faces_by_dimension for face in group.faces)
    stalks = tuple(SheafStalk(simplex=face, basis=("x",)) for face in cells)
    covers = tuple(
        CoverRestrictionMatrix(source=face, target=coface, entries=((_q(1),),))
        for coface in cells
        if len(coface) > 1
        for face in (
            coface[:position] + coface[position + 1 :]
            for position in range(len(coface))
        )
    )
    result = from_cover_maps(complex_, SheafField.RATIONAL, None, stalks, covers)
    assert result.sheaf is not None
    return result.sheaf


def test_decoded_skeleton_rechecks_inclusion_naturality() -> None:
    """A matching serialized prefix does not establish the face squares."""
    source = standard_simplex(1, 1)
    result = simplicial_set_skeleton(source, 1)
    payload: dict[str, Any] = result.model_dump()
    skeleton_payload: dict[str, Any] = payload["skeleton"]
    face_maps: tuple[Any, ...] = skeleton_payload["face_maps"]
    swapped: tuple[Any, ...] = (tuple(reversed(face_maps[0])),)
    skeleton_payload["face_maps"] = swapped
    inclusion_payload: dict[str, Any] = payload["inclusion"]
    inclusion_payload["source"] = skeleton_payload

    with pytest.raises(ValidationError, match="canonical simplicial sets"):
        SimplicialSetSkeletonResult.model_validate(payload)


def test_native_skeleton_rejects_a_carrier_with_a_missing_field() -> None:
    """A forged model without required carrier fields gets a domain error."""
    forged: FiniteTruncatedSimplicialSet = FiniteTruncatedSimplicialSet.model_construct(
        **{
            key: value
            for key, value in standard_simplex(1, 1).__dict__.items()
            if key != "sets"
        }
    )

    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_set_skeleton(forged, 1)

    assert error.value.errors()[0]["type"] == ("simplicial_set.skeleton_source_invalid")


def test_forged_target_inclusion_is_rejected_before_lookup() -> None:
    """A non-injective inclusion row must not reach the preimage lookup."""
    target = standard_simplex(1, 1)
    forged_inclusion = TruncatedSimplicialMap.model_construct(
        source=target,
        target=target,
        maps=((0, 0), (0, 0, 0)),
    )
    forged_subset = SimplicialSubsetPrefix.model_construct(inclusion=forged_inclusion)
    identity = TruncatedSimplicialMap(
        source=target,
        target=target,
        maps=tuple(tuple(range(len(level))) for level in target.sets),
    )
    request = SimplicialMapPreimageRequest.model_construct(
        simplicial_map=identity, target_subset=forged_subset
    )

    with pytest.raises(OperationDomainValidationError) as error:
        simplicial_map_preimage(request)

    assert error.value.errors()[0]["type"] == ("simplicial_map.preimage_subset_invalid")


def test_raw_prime_field_context_is_canonicalized_before_admission() -> None:
    """A retained scalar-context string must not bypass primality admission."""
    request = DegenerateSubmoduleRequest.model_construct(
        simplicial_set=standard_simplex(1, 1),
        coefficient_ring="GF_p",
        prime=4,
    )

    with pytest.raises(OperationDomainValidationError) as error:
        degenerate_submodule(request)

    assert error.value.errors()[0]["type"] == (
        "simplicial_set.degenerate_submodule_scalar_context"
    )


def test_malformed_coboundary_shape_is_rejected_before_size_preflight() -> None:
    """A non-list matrix cannot defer the aggregate cell admission check."""
    result = sheaf_cochain_complex(_constant_triangle())
    payload: dict[str, Any] = result.model_dump()
    payload["coboundary_matrices"] = ["not-a-matrix"]

    with pytest.raises(ValidationError) as error:
        SheafCochainComplex.model_validate(payload)

    assert error.value.errors()[0]["type"] == (
        "topology.cellular_sheaf.coboundary_shape_invalid"
    )


def test_image_result_derives_diagram_counters_from_its_complex() -> None:
    """Consistently forged counters do not establish the parent diagram."""
    sheaf = _triangle_sheaf()
    axis = sheaf.canonical_face_order
    result = image_of_morphism(
        morphism(sheaf, sheaf, tuple((cell, ((_q(1),),)) for cell in axis))
    )
    payload: dict[str, Any] = result.model_dump()
    parents: list[dict[str, Any]] = [
        payload["image"],
        payload["morphism"]["source"],
        payload["morphism"]["target"],
        payload["inclusion"]["source"],
        payload["inclusion"]["target"],
        payload["factor"]["source"],
        payload["factor"]["target"],
    ]
    for parent in parents:
        parent["diamonds"] += 1
        parent["comparable_pairs"] += 1

    with pytest.raises(ValidationError, match="diagram counters"):
        SheafMorphismImageResult.model_validate(payload)


def test_oversized_poset_containers_reject_before_serialization() -> None:
    """An over-limit forged poset must not be copied before admission."""
    poset = FinitePoset.model_construct(
        elements=tuple(f"element-{index}" for index in range(65)),
        strict_order_pairs=(),
        cover_relations=(),
        incomparable_pairs=(),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        order_complex(OrderComplexRequest.model_construct(poset=poset))

    assert error.value.errors()[0]["type"] == (
        "topology.order_complex.poset_container_budget"
    )


def test_dependent_product_contract_rejects_an_empty_complex() -> None:
    """Void-complex support does not extend to a product of nonempty factors."""
    result = cubical_product(
        (CubicalCell(intervals=((0, 1),)),),
        (CubicalCell(intervals=((5, 5),)),),
    )
    payload: dict[str, Any] = result.model_dump()
    complex_payload: dict[str, Any] = payload["complex"]
    complex_payload["cells"] = ()

    with pytest.raises(ValidationError) as error:
        CubicalProductResult.model_validate(payload)

    assert error.value.errors()[0]["type"] == ("cubical_complex.product_empty_complex")


def test_dependent_skeleton_contract_rejects_an_empty_complex() -> None:
    """Void-complex support does not extend to a nonempty source skeleton."""
    result = cubical_skeleton((CubicalCell(intervals=((0, 1),)),), 1)
    payload: dict[str, Any] = result.model_dump()
    complex_payload: dict[str, Any] = payload["complex"]
    complex_payload["cells"] = ()
    skeleton_payload: dict[str, Any] = payload["skeleton"]
    skeleton_payload["cells"] = ()

    with pytest.raises(ValidationError) as error:
        CubicalSkeletonResult.model_validate(payload)

    assert error.value.errors()[0]["type"] == ("cubical_complex.skeleton_empty_complex")
