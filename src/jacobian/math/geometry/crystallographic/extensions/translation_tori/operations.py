"""Exact quotient chains for parallelepiped translation tori."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb
from typing import NoReturn

from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic.extensions._models import (
    CrystallographicFundamentalDomainResult,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    check_crystallographic_fundamental_domain,
)
from jacobian.math.geometry.crystallographic.extensions.translation_tori._models import (
    BieberbachTranslationTorusChains,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

MAX_TORUS_SOURCE_BYTES = 256_000
MAX_TORUS_RESULT_BYTES = 600_000
_Vector = tuple[Fraction, ...]


def _domain(reason: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("source",),
        code=f"crystallographic.translation_torus.{reason}",
        message=message,
    )


def _resource(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("source",),
        code=f"crystallographic.translation_torus.{reason}",
        message=message,
    )


def _parallelepiped_directions(
    vertices: tuple[_Vector, ...],
) -> tuple[_Vector, ...] | None:
    """Find canonical edge vectors whose subset sums are the vertex set."""
    points = frozenset(vertices)
    dimension = len(vertices[0])
    base = min(points)
    candidates = tuple(sorted(point for point in points if point != base))
    for endpoints in combinations(candidates, dimension):
        vectors = tuple(
            tuple(endpoints[index][axis] - base[axis] for axis in range(dimension))
            for index in range(dimension)
        )
        sums = frozenset(
            tuple(
                base[axis]
                + sum(
                    (vectors[j][axis] for j in range(dimension) if mask & (1 << j)),
                    Fraction(0),
                )
                for axis in range(dimension)
            )
            for mask in range(1 << dimension)
        )
        if sums == points:
            return tuple(sorted(vectors))
    return None


def translation_torus_quotient_chains(
    source: CrystallographicFundamentalDomainResult,
) -> BieberbachTranslationTorusChains:
    """Build integral quotient chains for a verified translation torus.

    This operation is intentionally restricted to rank-one through rank-four
    pure translation actions with a parallelepiped fundamental domain. It
    returns the product cell structure on the torus, not a general
    Bieberbach face-orbit complex.
    """
    try:
        checked = CrystallographicFundamentalDomainResult.model_validate(
            source.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="crystallographic.translation_torus.source_shape",
            message="source does not satisfy the fundamental-domain value contract",
        ) from exc

    pairing = checked.source
    extension = pairing.affine_realization.source
    profile = pairing.facet_profile
    dimension = len(extension.action_matrices[0])
    # This strict shape check occurs before the geometric source is rebuilt.
    if (
        not 1 <= dimension <= 4
        or len(extension.multiplication_table) != 1
        or len(profile.vertices) != 1 << dimension
        or len(profile.facets) != 2 * dimension
        or len(pairing.pairings) != 2 * dimension
    ):
        _resource(
            "shape_bound",
            "input must have rank one through four with 2^rank vertices and 2*rank facets and directed pairings",
        )
    input_bytes = len(checked.model_dump_json().encode("utf-8"))
    if input_bytes > MAX_TORUS_SOURCE_BYTES:
        _resource("source_bound", "fundamental-domain source exceeds its byte envelope")
    predicted_result_bytes = input_bytes + 4096
    if predicted_result_bytes > MAX_TORUS_RESULT_BYTES:
        _resource("result_bound", "quotient-chain result exceeds its byte envelope")

    identity = tuple(
        tuple(int(row == column) for column in range(dimension))
        for row in range(dimension)
    )
    if (
        extension.multiplication_table != ((0,),)
        or extension.action_matrices != (identity,)
        or extension.factor_set != ((tuple(0 for _ in range(dimension)),),)
    ):
        _domain(
            "not_pure_translation",
            "the translation-torus slice requires trivial holonomy and zero factor set",
        )
    if not checked.is_fundamental_domain:
        _domain("source_not_fundamental", "source is not a checked fundamental domain")

    recomputed = check_crystallographic_fundamental_domain(pairing)
    if recomputed != checked or not recomputed.is_fundamental_domain:
        _domain("stale_source", "source must be a freshly checked fundamental domain")

    vertices = tuple(
        tuple(value.as_fraction() for value in vertex.coordinates)
        for vertex in profile.vertices
    )
    directions = _parallelepiped_directions(vertices)
    if directions is None:
        _domain("not_parallelepiped", "the source vertices are not a parallelepiped")
    if any(value.denominator != 1 for vector in directions for value in vector):
        _domain(
            "nonintegral_translation_cell",
            "parallelepiped edge vectors must be integral in the source lattice axes",
        )

    signed_directions = {
        tuple(int(value) * sign for value in vector)
        for vector in directions
        for sign in (-1, 1)
    }
    seen_facets: set[int] = set()
    direction_counts: dict[tuple[int, ...], int] = {}
    for side in pairing.pairings:
        if (
            side.holonomy_element != 0
            or side.source_facet_index == side.target_facet_index
            or side.source_facet_index in seen_facets
            or side.lattice_translation not in signed_directions
        ):
            _domain(
                "side_pairing_shape",
                "each facet must be paired once to its opposite by a signed edge translation",
            )
        seen_facets.add(side.source_facet_index)
        translated = side.lattice_translation
        direction = min(
            translated,
            tuple(-coordinate for coordinate in translated),
        )
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    if set(direction_counts.values()) != {2} or len(direction_counts) != dimension:
        _domain(
            "side_pairing_directions",
            "each circle direction must pair exactly one opposite facet pair",
        )
    pairing_by_source = {side.source_facet_index: side for side in pairing.pairings}
    if any(
        pairing_by_source[side.target_facet_index].target_facet_index
        != side.source_facet_index
        or tuple(
            -coordinate
            for coordinate in pairing_by_source[
                side.target_facet_index
            ].lattice_translation
        )
        != side.lattice_translation
        for side in pairing.pairings
    ):
        _domain(
            "side_pairing_inverse",
            "opposite facet pairings must be inverse translations",
        )

    canonical_directions = tuple(
        tuple(CanonicalRational.from_fraction(value) for value in vector)
        for vector in directions
    )
    basis_sizes = tuple(comb(dimension, degree) for degree in range(dimension + 1))
    zero_chain = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        prime=None,
        degree_min=0,
        degree_max=dimension,
        basis_sizes=basis_sizes,
        differential_matrices=tuple(
            tuple(
                tuple(0 for _ in range(basis_sizes[degree]))
                for _ in range(basis_sizes[degree - 1])
            )
            for degree in range(1, dimension + 1)
        ),
    )
    result = BieberbachTranslationTorusChains(
        source=checked,
        circle_directions=canonical_directions,
        quotient_chain_complex=zero_chain,
    )
    return result


__all__ = ["translation_torus_quotient_chains"]
