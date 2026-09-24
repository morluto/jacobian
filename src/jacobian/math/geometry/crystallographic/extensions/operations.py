"""Exact finite-extension validation and torsion-freeness decisions."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product as cartesian_product
from math import ceil, comb, floor, log10
from typing import NoReturn

from pydantic import ValidationError

from jacobian._exact import (
    MAX_CANONICAL_INTEGER_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
    require_bounded_rational,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic.extensions._models import (
    MAX_COCYCLE_ENTRY_DIGITS,
    MAX_EXTENSION_GROUP_ORDER,
    MAX_EXTENSION_LATTICE_RANK,
    MAX_EXTENSION_PAIRING_DIGITS,
    MAX_EXTENSION_TORSION_RESULT_BYTES,
    MAX_EXTENSION_TORSION_VECTOR_DIGITS,
    CrystallographicAffineRealization,
    CrystallographicAffineSectionMap,
    CrystallographicExtensionTorsionResult,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairing,
    CrystallographicPolytopePairingRequest,
    CrystallographicPolytopePairingResult,
    FiniteLatticeExtension,
    NonTorsionLiftObstruction,
    PolytopeFacetPairing,
    TorsionLiftWitness,
)
from jacobian.math.geometry.polytopes._models import (
    MAX_COMPUTED_FACETS,
    MAX_FACET_COORDINATE_DIGITS,
    MAX_FACET_INCIDENCES,
    MAX_VERTICES,
    FacetIncidenceResult,
    RationalVPolytope,
    _canonical_v_polytope_vertices,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    MAX_DD_RAY_BOUND,
    _component_digit_bound,
    dd_work_bound,
    halfspaces_to_generators,
    require_dd_height_admissible,
    require_dd_weighted_work_admissible,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence, polytope_volume
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_INPUT_DIGITS,
    MAX_CERTIFIED_SNF_INPUT_DIMENSION,
    SmithNormalFormCertificate,
)
from jacobian.math.matrices.values import IntegerMatrix

_IntMatrix = tuple[tuple[int, ...], ...]
_IntVector = tuple[int, ...]

MAX_AFFINE_REALIZATION_WORK = 10_000
MAX_AFFINE_REALIZATION_RESULT_BYTES = 100_000
MAX_POLYTOPE_PAIRING_WORK = 2_000_000_000
MAX_POLYTOPE_PAIRING_RESULT_BYTES = 8_000_000
MAX_FUNDAMENTAL_DOMAIN_CANDIDATES = 20_000
MAX_FUNDAMENTAL_DOMAIN_WORK = 2_000_000_000
MAX_FUNDAMENTAL_DOMAIN_RESULT_BYTES = 16_000_000


def _domain(
    reason: str, message: str, location: tuple[str | int, ...] = ()
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"crystallographic.extension.{reason}",
        message=message,
    )


def _resource(
    reason: str, message: str, location: tuple[str | int, ...] = ()
) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=location,
        code=f"crystallographic.extension.{reason}",
        message=message,
    )


def _identity(rank: int) -> _IntMatrix:
    return tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )


def _matmul(left: _IntMatrix, right: _IntMatrix) -> _IntMatrix:
    return tuple(
        tuple(
            sum(left[row][i] * right[i][column] for i in range(len(right)))
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matvec(matrix: _IntMatrix, vector: _IntVector) -> _IntVector:
    return tuple(
        sum(matrix[row][i] * vector[i] for i in range(len(vector)))
        for row in range(len(matrix))
    )


def _determinant(matrix: _IntMatrix) -> int:
    """Bareiss determinant for the admitted matrices of rank at most four."""
    work = [list(row) for row in matrix]
    n = len(work)
    if n == 1:
        return work[0][0]
    sign = 1
    previous = 1
    for pivot in range(n - 1):
        if work[pivot][pivot] == 0:
            swap = next((r for r in range(pivot + 1, n) if work[r][pivot]), None)
            if swap is None:
                return 0
            work[pivot], work[swap] = work[swap], work[pivot]
            sign = -sign
        value = work[pivot][pivot]
        for row in range(pivot + 1, n):
            for column in range(pivot + 1, n):
                numerator = (
                    work[row][column] * value - work[row][pivot] * work[pivot][column]
                )
                if numerator % previous:
                    raise ArithmeticError("fraction-free determinant lost exactness")
                work[row][column] = numerator // previous
        for row in range(pivot + 1, n):
            work[row][pivot] = 0
        previous = value
    return sign * work[-1][-1]


def _check_group_table(source: FiniteLatticeExtension) -> int:
    table = source.multiplication_table
    order = len(table)
    for a, row in enumerate(table):
        for b, product in enumerate(row):
            if not 0 <= product < order:
                _domain(
                    "group_table",
                    "multiplication table entry is out of range",
                    ("multiplication_table", a, b),
                )
    if any(table[0][g] != g or table[g][0] != g for g in range(order)):
        _domain(
            "identity",
            "table index 0 must be the two-sided identity",
            ("multiplication_table",),
        )
    for g in range(order):
        if not any(table[g][h] == 0 and table[h][g] == 0 for h in range(order)):
            _domain(
                "inverse",
                "every table element must have a two-sided inverse",
                ("multiplication_table", g),
            )
    for g in range(order):
        for h in range(order):
            gh = table[g][h]
            for k in range(order):
                if table[gh][k] != table[g][table[h][k]]:
                    _domain(
                        "associativity",
                        "multiplication table is not associative",
                        ("multiplication_table", g, h, k),
                    )
    return order


def _validate_action(
    source: FiniteLatticeExtension, order: int, rank: int
) -> tuple[_IntMatrix, ...]:
    actions = tuple(
        tuple(tuple(int(x) for x in row) for row in matrix)
        for matrix in source.action_matrices
    )
    if actions[0] != _identity(rank):
        _domain(
            "action_identity",
            "rho(0) must be the identity matrix",
            ("action_matrices", 0),
        )
    for g, matrix in enumerate(actions):
        if abs(_determinant(matrix)) != 1:
            _domain(
                "action_unimodular",
                "each holonomy matrix must be unimodular",
                ("action_matrices", g),
            )
    for g in range(order):
        for h in range(order):
            if (
                _matmul(actions[g], actions[h])
                != actions[source.multiplication_table[g][h]]
            ):
                _domain(
                    "action_homomorphism",
                    "action matrices do not represent the group multiplication",
                    ("action_matrices", g, h),
                )
    if len(set(actions)) != order:
        _domain(
            "action_not_faithful",
            "linear holonomy action must be faithful",
            ("action_matrices",),
        )
    return actions


def _validate_cocycle(
    source: FiniteLatticeExtension,
    order: int,
    rank: int,
    actions: tuple[_IntMatrix, ...],
) -> None:
    factor_set = source.factor_set
    for g in range(order):
        if any(factor_set[g][0]) or any(factor_set[0][g]):
            _domain(
                "cocycle_normalization",
                "factor set must be normalized at identity 0",
                ("factor_set", g),
            )
    for g in range(order):
        for h in range(order):
            gh = source.multiplication_table[g][h]
            for k in range(order):
                hk = source.multiplication_table[h][k]
                lhs = tuple(
                    factor_set[g][h][i] + factor_set[gh][k][i] for i in range(rank)
                )
                rho_f = _matvec(actions[g], factor_set[h][k])
                rhs = tuple(rho_f[i] + factor_set[g][hk][i] for i in range(rank))
                if lhs != rhs:
                    _domain(
                        "cocycle_identity",
                        "factor set fails the left 2-cocycle identity",
                        ("factor_set", g, h, k),
                    )


def _canonicalize_extension(source: object) -> FiniteLatticeExtension:
    if not isinstance(source, FiniteLatticeExtension):
        _domain("source_type", "source must be a canonical finite lattice extension")
    try:
        checked = FiniteLatticeExtension.model_validate(
            source.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="crystallographic.extension.source_shape",
            message="source does not satisfy the finite-extension wire contract",
        ) from exc

    return checked


def _validate_extension(checked: FiniteLatticeExtension) -> tuple[int, int]:
    order = _check_group_table(checked)
    rank = len(checked.action_matrices[0])
    actions = _validate_action(checked, order, rank)
    _validate_cocycle(checked, order, rank, actions)
    return order, rank


def _admit_and_validate(source: object) -> tuple[FiniteLatticeExtension, int, int]:
    checked = _canonicalize_extension(source)
    order, rank = _validate_extension(checked)
    return checked, order, rank


def _admit_affine_realization(order: int, rank: int) -> None:
    """Bound averaging, cocycle replay, and source-bound result construction."""
    if (
        not 1 <= order <= MAX_EXTENSION_GROUP_ORDER
        or not 1 <= rank <= MAX_EXTENSION_LATTICE_RANK
    ):
        _resource(
            "affine_realization_shape",
            "finite extension exceeds the affine-realization envelope",
        )
    work = (
        order**3
        + order * order * rank**3
        + order**3 * rank
        + order * rank**3
        + order * order * rank * rank
        + order * order * rank
    )
    if work > MAX_AFFINE_REALIZATION_WORK:
        _resource(
            "affine_realization_work",
            "affine realization exceeds its exact arithmetic work envelope",
        )
    # q_g is an average of |G| cocycle entries. Bound the retained source and
    # affine maps using their maximum scalar widths before constructing them.
    q_digits = MAX_COCYCLE_ENTRY_DIGITS + len(str(order))
    source_scalars = order * order + order * rank * rank + order * order * rank
    map_scalars = order * (rank * rank + rank)
    predicted_bytes = (
        (source_scalars + map_scalars) * (2 * q_digits + 32) + order * order * 16 + 1024
    )
    if predicted_bytes > MAX_AFFINE_REALIZATION_RESULT_BYTES:
        _resource(
            "affine_realization_output",
            "affine realization exceeds its exact result byte envelope",
        )


def affine_section_realization(
    source: FiniteLatticeExtension,
) -> CrystallographicAffineRealization:
    """Realize the extension's chosen finite-holonomy section by affine maps.

    The canonical rational shift is
    ``q_g = (1/|G|) sum_h f(g,h)``. It satisfies
    ``q_g + rho(g)q_h - q_(gh) = f(g,h)``. Thus the section maps obey
    ``A_g o A_h = T_(f(g,h)) o A_(gh)``; the full extension action
    ``(v,g) -> T_v o A_g`` composes according to the extension multiplication.
    """
    checked = _canonicalize_extension(source)
    order = len(checked.multiplication_table)
    rank = len(checked.action_matrices[0])
    _admit_affine_realization(order, rank)
    _validate_extension(checked)

    shifts = tuple(
        tuple(
            Fraction(
                sum(checked.factor_set[g][h][coordinate] for h in range(order)),
                order,
            )
            for coordinate in range(rank)
        )
        for g in range(order)
    )
    for g in range(order):
        for h in range(order):
            gh = checked.multiplication_table[g][h]
            for coordinate in range(rank):
                composed_shift = shifts[g][coordinate] + sum(
                    checked.action_matrices[g][coordinate][j] * shifts[h][j]
                    for j in range(rank)
                )
                if (
                    composed_shift - shifts[gh][coordinate]
                    != checked.factor_set[g][h][coordinate]
                ):
                    raise ArithmeticError(
                        "averaged affine section failed the extension cocycle identity"
                    )
    return CrystallographicAffineRealization(
        source=checked,
        section_maps=tuple(
            CrystallographicAffineSectionMap(
                holonomy_element=g,
                linear_part=checked.action_matrices[g],
                section_shift=tuple(
                    CanonicalRational.from_fraction(value) for value in shifts[g]
                ),
            )
            for g in range(order)
        ),
    )


def _extension_product(
    source: FiniteLatticeExtension,
    left: tuple[tuple[int, ...], int],
    right: tuple[tuple[int, ...], int],
) -> tuple[tuple[int, ...], int]:
    left_translation, left_holonomy = left
    right_translation, right_holonomy = right
    action = source.action_matrices[left_holonomy]
    translated = tuple(
        left_translation[i]
        + sum(
            action[i][j] * right_translation[j] for j in range(len(right_translation))
        )
        + source.factor_set[left_holonomy][right_holonomy][i]
        for i in range(len(left_translation))
    )
    return translated, source.multiplication_table[left_holonomy][right_holonomy]


def _inverse_extension_element(
    source: FiniteLatticeExtension,
    element: tuple[tuple[int, ...], int],
) -> tuple[tuple[int, ...], int]:
    """Invert one canonical lattice-extension element exactly."""
    translation, holonomy = element
    inverse_holonomy = next(
        candidate
        for candidate in range(len(source.multiplication_table))
        if source.multiplication_table[holonomy][candidate] == 0
        and source.multiplication_table[candidate][holonomy] == 0
    )
    action_inverse = _unimodular_inverse(source.action_matrices[holonomy])
    offset = tuple(
        translation[axis] + source.factor_set[holonomy][inverse_holonomy][axis]
        for axis in range(len(translation))
    )
    inverse_translation = tuple(-value for value in _matvec(action_inverse, offset))
    return inverse_translation, inverse_holonomy


def _admit_polytope_pairing(polytope: RationalVPolytope, pairing_count: int) -> None:
    """Preflight exact pairing work/output before the full facet enumeration."""
    dimension = len(polytope.space.axes)
    vertices = len(polytope.vertices)
    if dimension > MAX_EXTENSION_LATTICE_RANK or vertices > MAX_VERTICES:
        _resource("polytope_pairing_shape", "polytope exceeds the pairing envelope")
    # The full facet profile is itself bounded by the polytope kernel. Pairing
    # maps add at most rank affine-coordinate evaluations for every source
    # incidence; use the profile's global incidence ceiling before computing it.
    try:
        for vertex in polytope.vertices:
            for coordinate in vertex.coordinates:
                require_bounded_rational(
                    coordinate,
                    max_digits=MAX_FACET_COORDINATE_DIGITS,
                    label="facet-pairing polytope coordinate",
                )
    except ValueError:
        _domain(
            "polytope_pairing_coordinate",
            "polytope coordinates exceed the facet-profile digit bound",
            ("polytope", "vertices"),
        )
    coordinate_digits = max(
        canonical_rational_component_digits(value)
        for vertex in polytope.vertices
        for value in vertex.coordinates
    )
    scalar_digits = 2 * dimension * coordinate_digits + 64
    facet_bound = min(
        MAX_COMPUTED_FACETS, _facet_count_upper_bound(vertices, dimension)
    )
    incidence_bound = min(MAX_FACET_INCIDENCES, facet_bound * vertices)
    pairing_work = incidence_bound * dimension * dimension * scalar_digits**2
    if pairing_work > MAX_POLYTOPE_PAIRING_WORK:
        _resource(
            "polytope_pairing_work",
            "facet pairing maps exceed the exact arithmetic work envelope",
        )
    # Full profile worst-case: 256 primitive facets and 16,384 incidences.
    # Supporting coefficients use the facet kernel's determinant height bound.
    facet_digits = 2 * dimension * dimension * MAX_FACET_COORDINATE_DIGITS + 32
    output_bytes = (
        MAX_COMPUTED_FACETS * (dimension + 1) * (facet_digits + 8)
        + MAX_FACET_INCIDENCES * 8
        + pairing_count * 256
        + vertices * dimension * (coordinate_digits * 2 + 8)
        + 32_768
    )
    if output_bytes > min(MAX_POLYTOPE_PAIRING_RESULT_BYTES, 10_000_000):
        _resource(
            "polytope_pairing_output",
            "facet profile and pairing value exceed the result byte envelope",
        )


def _facet_count_upper_bound(vertex_count: int, dimension: int) -> int:
    """A simple upper-bound-theorem estimate for facets of a convex polytope."""
    if dimension == 1:
        return 2
    if dimension == 2:
        return vertex_count
    if dimension == 3:
        return max(0, 2 * vertex_count - 4)
    if dimension == 4:
        return 2 * comb(vertex_count, 2)
    raise ValueError("crystallographic polytope pairing supports dimensions 1..4")


def pair_crystallographic_polytope_facets(
    request: CrystallographicPolytopePairingRequest,
) -> CrystallographicPolytopePairingResult:
    """Validate a complete exact side-pairing ledger on a bounded V-polytope.

    This operation establishes only that every computed facet is paired once,
    paired by an extension affine map onto the target facet, and paired back by
    the exact inverse extension element. It does not establish a tiling,
    quotient cell structure, torsion-freeness, or a resolution.
    """
    try:
        checked_request = CrystallographicPolytopePairingRequest.model_validate(
            request.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="crystallographic.extension.polytope_pairing_shape",
            message="request does not satisfy the facet-pairing wire contract",
        ) from exc
    polytope = checked_request.polytope
    realization = checked_request.affine_realization
    source, order, rank = _admit_and_validate(realization.source)
    if (
        checked_request.lattice_axes != polytope.space.axes
        or len(checked_request.lattice_axes) != rank
    ):
        _domain(
            "polytope_pairing_axes",
            "lattice coordinate axes must equal the ordered polytope axes",
            ("lattice_axes",),
        )
    if len(checked_request.pairings) > MAX_COMPUTED_FACETS:
        _resource("polytope_pairing_facets", "too many proposed facet pairings")
    _admit_polytope_pairing(polytope, len(checked_request.pairings))

    # Affine realizations are caller-supplied claims; recompute their canonical
    # section maps before relying on them to map any facet.
    expected_realization = affine_section_realization(source)
    if realization != expected_realization:
        _domain(
            "polytope_pairing_realization",
            "affine realization does not match its finite extension source",
            ("affine_realization",),
        )

    vertices = _canonical_v_polytope_vertices(polytope)
    try:
        profile = facet_incidence(vertices, dimension_bound=rank)
    except OperationDomainValidationError:
        raise
    facet_count = len(profile.facets)
    if len(checked_request.pairings) != facet_count:
        _domain(
            "polytope_pairing_complete",
            "pairing ledger must contain one entry per computed facet",
            ("pairings",),
        )
    normalized = _validate_pairing_ledger(
        source=source,
        realization=realization,
        profile=profile,
        vertices=vertices,
        pairings=checked_request.pairings,
        order=order,
        rank=rank,
    )
    return CrystallographicPolytopePairingResult(
        affine_realization=expected_realization,
        polytope=polytope,
        lattice_axes=checked_request.lattice_axes,
        facet_profile=profile,
        pairings=normalized,
    )


def check_crystallographic_fundamental_domain(
    result: CrystallographicPolytopePairingResult,
) -> CrystallographicFundamentalDomainResult:
    """Check full-dimensional translate overlaps and exact quotient covolume.

    For the admitted faithful cocompact affine action, locally finite translates
    with disjoint interiors and the quotient covolume cover the ambient space.
    Facet pairings alone are never treated as evidence of a tiling.
    """
    try:
        checked = CrystallographicPolytopePairingResult.model_validate(
            result.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="crystallographic.extension.fundamental_domain_source",
            message="input does not satisfy the retained side-pairing contract",
        ) from exc
    # Recompute every retained claim that will be used below. This catches a
    # hand-authored/stale facet profile or affine realization in a decoded value.
    rebuilt = pair_crystallographic_polytope_facets(
        CrystallographicPolytopePairingRequest(
            affine_realization=checked.affine_realization,
            polytope=checked.polytope,
            lattice_axes=checked.lattice_axes,
            pairings=tuple(
                PolytopeFacetPairing.model_validate(
                    item.model_dump(mode="python", warnings=False), strict=True
                )
                for item in checked.pairings
            ),
        )
    )
    if rebuilt != checked:
        _domain(
            "fundamental_domain_stale_source",
            "facet profile or side-pairing ledger differs from its retained source",
            ("source",),
        )
    _, order, rank = _admit_and_validate(checked.affine_realization.source)
    vertices = _canonical_v_polytope_vertices(checked.polytope)
    profile = rebuilt.facet_profile
    # H-to-V has a fixed 64-row envelope. Reject before candidate enumeration
    # where the two full facet lists would exceed it.
    if len(profile.facets) * 2 > 64:
        _resource(
            "fundamental_domain_intersection_rows",
            "intersection exceeds the 64-facet conversion envelope",
        )
    rows = tuple(
        (
            tuple(c.as_fraction() for c in facet.halfspace.coefficients),
            facet.halfspace.offset.as_fraction(),
        )
        for facet in profile.facets
    )
    coordinates = tuple(
        tuple(c.as_fraction() for c in vertex.coordinates) for vertex in vertices
    )
    volume_result = polytope_volume(vertices, None, rank)
    volume = volume_result.volume.as_fraction()
    covolume = Fraction(1, order)
    candidate_ranges, inverse_matrices = _translation_candidate_ranges(
        checked.affine_realization, coordinates, rank
    )
    candidate_count = sum(_range_product_count(bounds) for bounds in candidate_ranges)
    candidate_count -= 1  # exclude (v=0,g=identity)
    if candidate_count > MAX_FUNDAMENTAL_DOMAIN_CANDIDATES:
        _resource(
            "fundamental_domain_candidates",
            "translation overlap candidate count exceeds the exact bound",
        )
    _admit_fundamental_domain_intersections(
        profile,
        rows,
        checked.affine_realization,
        candidate_ranges,
        inverse_matrices,
        rank,
        candidate_count,
        checked,
    )
    witness: tuple[tuple[int, ...], int] | None = None
    for holonomy, bounds in enumerate(candidate_ranges):
        amap = checked.affine_realization.section_maps[holonomy]
        inverse = inverse_matrices[holonomy]
        shift = tuple(c.as_fraction() for c in amap.section_shift)
        for translation in _integer_vectors(bounds):
            if holonomy == 0 and translation == (0,) * rank:
                continue
            # x in T_v A_g(P) iff A_g^{-1}(x-v) lies in P.
            translated_rows = []
            for normal, offset in rows:
                transformed_normal = tuple(
                    sum(normal[k] * Fraction(inverse[k][i]) for k in range(rank))
                    for i in range(rank)
                )
                affine_offset = sum(
                    transformed_normal[i] * (shift[i] + translation[i])
                    for i in range(rank)
                )
                translated_rows.append((transformed_normal, offset + affine_offset))
            intersection = halfspaces_to_generators((*rows, *translated_rows), rank)
            if not intersection.empty and intersection.affine_dimension == rank:
                witness = (translation, holonomy)
                break
        if witness is not None:
            break
    good = witness is None and volume == covolume
    return CrystallographicFundamentalDomainResult(
        source=rebuilt,
        is_fundamental_domain=good,
        polytope_volume=CanonicalRational.from_fraction(volume),
        quotient_covolume=CanonicalRational.from_fraction(covolume),
        overlap_translation=None if witness is None else witness[0],
        overlap_holonomy_element=None if witness is None else witness[1],
    )


def _translation_candidate_ranges(
    realization: CrystallographicAffineRealization,
    coordinates: tuple[tuple[Fraction, ...], ...],
    rank: int,
) -> tuple[list[tuple[range, ...]], list[_IntMatrix]]:
    candidate_ranges = []
    inverse_matrices = []
    polytope_lows = tuple(min(point[i] for point in coordinates) for i in range(rank))
    polytope_highs = tuple(max(point[i] for point in coordinates) for i in range(rank))
    for amap in realization.section_maps:
        matrix = tuple(tuple(int(value) for value in row) for row in amap.linear_part)
        inverse_matrices.append(_unimodular_inverse(matrix))
        transformed = tuple(
            tuple(
                sum(Fraction(amap.linear_part[i][j]) * point[j] for j in range(rank))
                + amap.section_shift[i].as_fraction()
                for i in range(rank)
            )
            for point in coordinates
        )
        transformed_lows = tuple(
            min(point[i] for point in transformed) for i in range(rank)
        )
        transformed_highs = tuple(
            max(point[i] for point in transformed) for i in range(rank)
        )
        candidate_ranges.append(
            tuple(
                range(
                    ceil(polytope_lows[i] - transformed_highs[i]),
                    floor(polytope_highs[i] - transformed_lows[i]) + 1,
                )
                for i in range(rank)
            )
        )
    return candidate_ranges, inverse_matrices


def _admit_fundamental_domain_intersections(
    profile: FacetIncidenceResult,
    rows: tuple[tuple[tuple[Fraction, ...], Fraction], ...],
    realization: CrystallographicAffineRealization,
    candidate_ranges: list[tuple[range, ...]],
    inverse_matrices: list[_IntMatrix],
    rank: int,
    candidate_count: int,
    checked: CrystallographicPolytopePairingResult,
) -> None:
    """Preflight every possible H-to-V call with the converter's own bounds."""
    admitted_rows: list[tuple[Fraction, ...]] = [
        (*normal, offset) for normal, offset in rows
    ]
    for bounds, inverse, amap in zip(
        candidate_ranges, inverse_matrices, realization.section_maps, strict=True
    ):
        if any(not axis for axis in bounds):
            continue
        shift = tuple(c.as_fraction() for c in amap.section_shift)
        endpoints = tuple(
            tuple(dict.fromkeys((axis.start, axis.stop - 1))) for axis in bounds
        )
        for translation in cartesian_product(*endpoints):
            for normal, offset in rows:
                transformed_normal = tuple(
                    sum(normal[k] * inverse[k][i] for k in range(rank))
                    for i in range(rank)
                )
                affine_offset = sum(
                    transformed_normal[i] * (shift[i] + translation[i])
                    for i in range(rank)
                )
                admitted_rows.append((*transformed_normal, offset + affine_offset))
    component_digits = _component_digit_bound(admitted_rows)
    dd_rows = 2 * len(profile.facets) + 1
    dd_rays, dd_pairs = dd_work_bound(dd_rows, rank + 1)
    if dd_rays > MAX_DD_RAY_BOUND:
        _resource(
            "fundamental_domain_conversion_rays",
            "per-intersection ray bound exceeds the H-to-V kernel envelope",
        )
    try:
        minor_digits = require_dd_height_admissible(
            component_digits, rank, affine_halfspaces=True
        )
        require_dd_weighted_work_admissible(
            dd_rows, rank + 1, minor_digits, candidate_pairs=dd_pairs
        )
    except ValueError:
        _resource(
            "fundamental_domain_conversion_height",
            "per-intersection converter height or weighted work exceeds its kernel bound",
        )
    aggregate_work = candidate_count * max(1, dd_pairs) * minor_digits**2
    if aggregate_work > MAX_FUNDAMENTAL_DOMAIN_WORK:
        _resource(
            "fundamental_domain_work",
            "aggregate incremental DD pair and height work exceeds the admitted bound",
        )
    serialized_size = len(
        json.dumps(
            checked.model_dump(mode="json", warnings=False), separators=(",", ":")
        )
    )
    if serialized_size * 2 + 1024 > MAX_FUNDAMENTAL_DOMAIN_RESULT_BYTES:
        _resource(
            "fundamental_domain_output",
            "source-bound result exceeds the output byte envelope",
        )


def _range_product_count(bounds: tuple[range, ...]) -> int:
    count = 1
    for axis in bounds:
        count *= len(axis)
    return count


def _integer_vectors(bounds: tuple[range, ...]):
    if not bounds or any(not axis for axis in bounds):
        return
    from itertools import product

    yield from product(*bounds)


def _unimodular_inverse(matrix: _IntMatrix) -> _IntMatrix:
    # Integral action matrices are unimodular; the adjugate/determinant formula
    # is tiny at rank <=4 and avoids introducing floating-point arithmetic.
    determinant = _determinant(matrix)
    rank = len(matrix)
    if abs(determinant) != 1:
        _domain("fundamental_domain_action", "affine action matrix is not unimodular")
    cofactors = tuple(
        tuple(
            (-1) ** (i + j)
            * (
                1
                if rank == 1
                else _determinant(
                    tuple(
                        tuple(matrix[r][c] for c in range(rank) if c != j)
                        for r in range(rank)
                        if r != i
                    )
                )
            )
            for j in range(rank)
        )
        for i in range(rank)
    )
    return tuple(
        tuple(cofactors[j][i] // determinant for j in range(rank)) for i in range(rank)
    )


def _validate_pairing_ledger(
    *,
    source: FiniteLatticeExtension,
    realization: CrystallographicAffineRealization,
    profile: FacetIncidenceResult,
    vertices: tuple[Vertex, ...],
    pairings: tuple[CrystallographicPolytopePairing, ...],
    order: int,
    rank: int,
) -> tuple[CrystallographicPolytopePairing, ...]:
    """Check ledger completeness, exact inverses, and all facet vertex maps."""
    facet_count = len(profile.facets)
    by_source: dict[int, object] = {}
    for position, pairing in enumerate(pairings):
        if (
            pairing.source_facet_index >= facet_count
            or pairing.target_facet_index >= facet_count
        ):
            _domain(
                "polytope_pairing_facet_index",
                "pairing facet index is outside the computed facet profile",
                ("pairings", position),
            )
        if pairing.source_facet_index in by_source:
            _domain(
                "polytope_pairing_duplicate_source",
                "each source facet must occur exactly once",
                ("pairings", position, "source_facet_index"),
            )
        if (
            len(pairing.lattice_translation) != rank
            or pairing.holonomy_element >= order
        ):
            _domain(
                "polytope_pairing_element",
                "pairing element dimensions or holonomy index do not match the source",
                ("pairings", position),
            )
        by_source[pairing.source_facet_index] = pairing
    if set(by_source) != set(range(facet_count)):
        _domain("polytope_pairing_complete", "every facet must occur as a source")

    normalized: list[CrystallographicPolytopePairing] = []
    for source_index in range(facet_count):
        pairing = by_source[source_index]
        target = by_source.get(pairing.target_facet_index)
        if target is None:
            _domain(
                "polytope_pairing_reverse", "every target facet needs a reverse entry"
            )
        element = (tuple(pairing.lattice_translation), pairing.holonomy_element)
        reverse_element = (tuple(target.lattice_translation), target.holonomy_element)
        identity = ((0,) * rank, 0)
        if (
            target.target_facet_index != source_index
            or _extension_product(source, element, reverse_element) != identity
            or _extension_product(source, reverse_element, element) != identity
        ):
            _domain(
                "polytope_pairing_inverse",
                "reverse pairing must be the exact two-sided group inverse",
                ("pairings", source_index),
            )
        section_map = realization.section_maps[pairing.holonomy_element]
        source_facet = profile.facets[source_index]
        target_facet = profile.facets[pairing.target_facet_index]
        mapped: set[tuple[Fraction, ...]] = set()
        for vertex_index in source_facet.source_vertex_indices:
            point = vertices[vertex_index].coordinates
            image = tuple(
                sum(
                    section_map.linear_part[row][column]
                    * Fraction(*point[column].as_integer_ratio())
                    for column in range(rank)
                )
                + Fraction(*section_map.section_shift[row].as_integer_ratio())
                + pairing.lattice_translation[row]
                for row in range(rank)
            )
            mapped.add(image)
        target_vertices = {
            tuple(
                Fraction(*vertices[index].coordinates[axis].as_integer_ratio())
                for axis in range(rank)
            )
            for index in target_facet.source_vertex_indices
        }
        if (
            len(mapped) != len(source_facet.source_vertex_indices)
            or mapped != target_vertices
        ):
            _domain(
                "polytope_pairing_image",
                "extension affine map must biject the complete source facet vertex set onto its target",
                ("pairings", source_index),
            )
        normalized.append(
            CrystallographicPolytopePairing(
                source_facet_index=source_index,
                target_facet_index=pairing.target_facet_index,
                lattice_translation=pairing.lattice_translation,
                holonomy_element=pairing.holonomy_element,
            )
        )
    return tuple(normalized)


def _admit_result_size(order: int, rank: int) -> None:
    """Bound the serialized source and all worst-case exact witness scalars."""
    norm_digits = (
        order
        + max(0, order - 1) * ceil(log10(max(2, rank)))
        + ceil(log10(order + 1))
        + 1
    )
    offset_digits = MAX_COCYCLE_ENTRY_DIGITS + ceil(log10(order + 1)) + 1
    obstruction_bytes = (
        rank * rank * (norm_digits + 2)
        + rank * (offset_digits + 2)
        + rank * (MAX_CANONICAL_INTEGER_DIGITS + 2)
        + (MAX_CANONICAL_INTEGER_DIGITS + 2)
        + (MAX_EXTENSION_PAIRING_DIGITS + 2)
        + 512
    )
    torsion_witness_bytes = (
        rank * (MAX_EXTENSION_TORSION_VECTOR_DIGITS + 2)
        + rank * rank * (norm_digits + 2)
        + rank * (offset_digits + 2)
        + 512
    )
    source_bytes = 16_384
    predicted_bytes = source_bytes + max(
        max(0, order - 1) * obstruction_bytes, torsion_witness_bytes
    )
    if predicted_bytes > MAX_EXTENSION_TORSION_RESULT_BYTES:
        _resource(
            "result_size_bound",
            "source and exact torsion witnesses exceed the result byte envelope",
        )


def _element_order(table: tuple[tuple[int, ...], ...], element: int) -> int:
    current = 0
    for exponent in range(1, len(table) + 1):
        current = table[current][element]
        if current == 0:
            return exponent
    raise ArithmeticError("finite group table contains an element of infinite order")


def _power_offset_and_norm(
    source: FiniteLatticeExtension,
    element: int,
    holonomy_order: int,
    rank: int,
) -> tuple[_IntMatrix, _IntVector]:
    table = source.multiplication_table
    actions = tuple(
        tuple(tuple(int(x) for x in row) for row in matrix)
        for matrix in source.action_matrices
    )
    factor_set = source.factor_set
    norm = [[0] * rank for _ in range(rank)]
    power = _identity(rank)
    current_group_element = 0
    offset = [0] * rank
    for _ in range(holonomy_order):
        for row in range(rank):
            for column in range(rank):
                norm[row][column] += power[row][column]
        current_group_element = table[current_group_element][element]
        cocycle_term = factor_set[current_group_element][element]
        for i in range(rank):
            offset[i] += cocycle_term[i]
        power = _matmul(power, actions[element])
    return tuple(tuple(row) for row in norm), tuple(offset)


def _admit_smith(norm: _IntMatrix, order: int, rank: int) -> IntegerMatrix:
    """Preflight each generated norm against certified Smith's input envelope."""
    # The action wire cap bounds every entry by 9. For any power below |G|,
    # matrix multiplication gives |A^j_ik| <= rank^(j-1) 9^j. Summing |G|
    # powers yields this conservative decimal bound for all entries of N_g.
    power_count = max(1, order)
    digits = (
        power_count
        + max(0, power_count - 1) * ceil(log10(max(2, rank)))
        + ceil(log10(power_count + 1))
        + 1
    )
    if (
        rank > MAX_CERTIFIED_SNF_INPUT_DIMENSION
        or digits > MAX_CERTIFIED_SNF_INPUT_DIGITS
    ):
        _resource(
            "smith_input_bound",
            "generated norm matrix exceeds certified Smith admission",
        )
    actual_digits = max(
        (len(str(abs(value))) for row in norm for value in row), default=1
    )
    if actual_digits > MAX_CERTIFIED_SNF_INPUT_DIGITS:
        _resource(
            "smith_input_bound",
            "generated norm matrix exceeds certified Smith input digit bound",
        )
    # Existing certified Smith operates on <=16 square matrices and <=32-digit
    # entries; the exact |G|/rank/action envelope above fits that contract.
    return IntegerMatrix(entries=norm)


def _smith_rhs(
    certificate: SmithNormalFormCertificate, offset: _IntVector
) -> tuple[int, ...]:
    rhs = tuple(-value for value in offset)
    left = tuple(
        tuple(int(value) for value in row)
        for row in certificate.left_transformation.entries
    )
    return tuple(
        sum(left[row][i] * rhs[i] for i in range(len(rhs))) for row in range(len(rhs))
    )


def _obstruction_row(
    certificate: SmithNormalFormCertificate, rhs: tuple[int, ...]
) -> int | None:
    diagonal = certificate.diagonal.entries
    rank = certificate.rank
    for row in range(rank):
        divisor = int(diagonal[row][row])
        if rhs[row] % divisor:
            return row
    for row in range(rank, len(rhs)):
        if rhs[row] != 0:
            return row
    return None


def _solution_from_smith(
    certificate: SmithNormalFormCertificate, rhs: tuple[int, ...]
) -> _IntVector:
    rank = certificate.rank
    diagonal = certificate.diagonal.entries
    smith_coordinates = [0] * len(rhs)
    for row in range(rank):
        divisor = int(diagonal[row][row])
        if rhs[row] % divisor:
            raise ArithmeticError("attempted to solve an obstructed Smith system")
        smith_coordinates[row] = rhs[row] // divisor
    if any(rhs[row] != 0 for row in range(rank, len(rhs))):
        raise ArithmeticError("attempted to solve an inconsistent zero Smith row")
    right = tuple(
        tuple(int(value) for value in row)
        for row in certificate.right_transformation.entries
    )
    return tuple(
        sum(right[row][i] * smith_coordinates[i] for i in range(len(rhs)))
        for row in range(len(rhs))
    )


def decide_extension_torsion(
    source: FiniteLatticeExtension,
) -> CrystallographicExtensionTorsionResult:
    """Decide whether the finite-lattice extension is torsion-free.

    For each nonidentity ``g`` of order ``m``, lifts ``(v,g)`` satisfy
    ``(v,g)^m=(N_g v+c_g, 1)``. Thus a finite-order lift exists exactly when
    ``N_g v=-c_g`` is solvable over integers. A Smith decomposition supplies
    either a divisibility obstruction or a concrete translation vector.
    """
    checked, order, rank = _admit_and_validate(source)
    _admit_result_size(order, rank)
    certificates: list[NonTorsionLiftObstruction] = []
    for element in range(1, order):
        element_order = _element_order(checked.multiplication_table, element)
        norm, offset = _power_offset_and_norm(checked, element, element_order, rank)
        smith_input = _admit_smith(norm, order, rank)
        certificate = smith_normal_form_certificate(smith_input)
        transformed_rhs = _smith_rhs(certificate, offset)
        obstruction_row = _obstruction_row(certificate, transformed_rhs)
        if obstruction_row is None:
            translation = _solution_from_smith(certificate, transformed_rhs)
            if (
                tuple(
                    a + b
                    for a, b in zip(_matvec(norm, translation), offset, strict=True)
                )
                != (0,) * rank
            ):
                raise ArithmeticError(
                    "Smith coordinates did not reconstruct a torsion lift"
                )
            return CrystallographicExtensionTorsionResult(
                source=checked,
                torsion_free=False,
                torsion_witness=TorsionLiftWitness(
                    holonomy_element=element,
                    holonomy_order=element_order,
                    translation_part=translation,
                    norm_matrix=norm,
                    power_offset=offset,
                ),
                lift_obstructions=(),
                conclusion="HAS_TORSION",
            )
        # One row of U is enough as a compact mathematical witness. From
        # U N V = D, row_i(U) N has every entry divisible by d_i; if d_i=0,
        # it annihilates N exactly. Its pairing with -c fails that condition.
        row = tuple(
            int(value)
            for value in certificate.left_transformation.entries[obstruction_row]
        )
        modulus = int(certificate.diagonal.entries[obstruction_row][obstruction_row])
        pairing = sum(row[i] * -offset[i] for i in range(rank))
        norm_pairing = tuple(
            sum(row[i] * norm[i][column] for i in range(rank)) for column in range(rank)
        )
        if modulus == 0:
            if any(norm_pairing) or pairing == 0:
                raise ArithmeticError("Smith zero-row obstruction failed exact replay")
        elif any(value % modulus for value in norm_pairing) or pairing % modulus == 0:
            raise ArithmeticError("Smith divisibility obstruction failed exact replay")
        certificates.append(
            NonTorsionLiftObstruction(
                holonomy_element=element,
                holonomy_order=element_order,
                norm_matrix=norm,
                power_offset=offset,
                obstruction_vector=row,
                modulus=modulus,
                pairing=pairing,
            )
        )
    return CrystallographicExtensionTorsionResult(
        source=checked,
        torsion_free=True,
        torsion_witness=None,
        lift_obstructions=tuple(certificates),
        conclusion="TORSION_FREE",
    )


__all__ = ["affine_section_realization", "decide_extension_torsion"]
