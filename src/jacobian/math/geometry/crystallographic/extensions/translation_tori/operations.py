"""Exact quotient chains for parallelepiped translation tori."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
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
_Vector3 = tuple[Fraction, Fraction, Fraction]


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
    vertices: tuple[_Vector3, ...],
) -> tuple[_Vector3, _Vector3, _Vector3] | None:
    """Find the canonical edge triple whose subset sums are the vertex set."""
    points = frozenset(vertices)
    base = min(points)
    candidates = tuple(sorted(point for point in points if point != base))
    for endpoints in combinations(candidates, 3):
        vectors: tuple[_Vector3, _Vector3, _Vector3] = (
            (
                endpoints[0][0] - base[0],
                endpoints[0][1] - base[1],
                endpoints[0][2] - base[2],
            ),
            (
                endpoints[1][0] - base[0],
                endpoints[1][1] - base[1],
                endpoints[1][2] - base[2],
            ),
            (
                endpoints[2][0] - base[0],
                endpoints[2][1] - base[1],
                endpoints[2][2] - base[2],
            ),
        )
        sums = frozenset(
            tuple(
                base[axis]
                + sum(
                    (vectors[j][axis] for j in range(3) if mask & (1 << j)), Fraction(0)
                )
                for axis in range(3)
            )
            for mask in range(8)
        )
        if sums == points:
            ordered = tuple(sorted(vectors))
            return (ordered[0], ordered[1], ordered[2])
    return None


def translation_torus_quotient_chains(
    source: CrystallographicFundamentalDomainResult,
) -> BieberbachTranslationTorusChains:
    """Build integral quotient chains for a verified parallelepiped 3-torus.

    This operation is intentionally restricted to a rank-three, pure
    translation action with six opposite facet pairings. It returns the
    product cell structure on ``(S^1)^3``, not a general three-dimensional
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
    # This strict shape check occurs before the geometric source is rebuilt.
    if (
        len(extension.action_matrices[0]) != 3
        or len(extension.multiplication_table) != 1
        or len(profile.vertices) != 8
        or len(profile.facets) != 6
        or len(pairing.pairings) != 6
    ):
        _domain(
            "source_shape",
            "input must have rank three, eight vertices, six facets, and six directed pairings",
        )
    input_bytes = len(checked.model_dump_json().encode("utf-8"))
    if input_bytes > MAX_TORUS_SOURCE_BYTES:
        _resource("source_bound", "fundamental-domain source exceeds its byte envelope")
    predicted_result_bytes = input_bytes + 4096
    if predicted_result_bytes > MAX_TORUS_RESULT_BYTES:
        _resource("result_bound", "quotient-chain result exceeds its byte envelope")

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    if (
        extension.multiplication_table != ((0,),)
        or extension.action_matrices != (identity,)
        or extension.factor_set != (((0, 0, 0),),)
    ):
        _domain(
            "not_pure_translation",
            "the 3-torus slice requires trivial holonomy and zero factor set",
        )
    if not checked.is_fundamental_domain:
        _domain("source_not_fundamental", "source is not a checked fundamental domain")

    recomputed = check_crystallographic_fundamental_domain(pairing)
    if recomputed != checked or not recomputed.is_fundamental_domain:
        _domain("stale_source", "source must be a freshly checked fundamental domain")

    vertices = tuple(
        (
            vertex.coordinates[0].as_fraction(),
            vertex.coordinates[1].as_fraction(),
            vertex.coordinates[2].as_fraction(),
        )
        for vertex in profile.vertices
    )
    directions = _parallelepiped_directions(vertices)
    if directions is None:
        _domain(
            "not_parallelepiped", "the eight source vertices are not a parallelepiped"
        )
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
    direction_counts: dict[tuple[int, int, int], int] = {}
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
        translated = (
            side.lattice_translation[0],
            side.lattice_translation[1],
            side.lattice_translation[2],
        )
        direction = min(
            translated,
            (-translated[0], -translated[1], -translated[2]),
        )
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    if set(direction_counts.values()) != {2} or len(direction_counts) != 3:
        _domain(
            "side_pairing_directions",
            "each of the three circle directions must pair exactly one opposite facet pair",
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

    canonical_directions = (
        (
            CanonicalRational.from_fraction(directions[0][0]),
            CanonicalRational.from_fraction(directions[0][1]),
            CanonicalRational.from_fraction(directions[0][2]),
        ),
        (
            CanonicalRational.from_fraction(directions[1][0]),
            CanonicalRational.from_fraction(directions[1][1]),
            CanonicalRational.from_fraction(directions[1][2]),
        ),
        (
            CanonicalRational.from_fraction(directions[2][0]),
            CanonicalRational.from_fraction(directions[2][1]),
            CanonicalRational.from_fraction(directions[2][2]),
        ),
    )
    zero_chain = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        prime=None,
        degree_min=0,
        degree_max=3,
        basis_sizes=(1, 3, 3, 1),
        differential_matrices=(
            ((0, 0, 0),),
            ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ((0,), (0,), (0,)),
        ),
    )
    result = BieberbachTranslationTorusChains(
        source=checked,
        circle_directions=canonical_directions,
        quotient_chain_complex=zero_chain,
    )
    return result


__all__ = ["translation_torus_quotient_chains"]
