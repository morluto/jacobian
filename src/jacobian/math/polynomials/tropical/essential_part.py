"""Tie-inclusive attained support from an exact coefficient-lifted hull."""

from __future__ import annotations

from fractions import Fraction
from math import comb

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    MAX_DD_PAIR_BOUND,
    MAX_DD_RAY_BOUND,
    HullConversion,
    PolyhedralConversionAdmissionError,
    dd_work_bound,
    points_to_facets,
    rational_rank,
    require_dd_height_admissible,
    require_dd_weighted_work_admissible,
    upper_bound_facets,
)
from jacobian.math.polynomials.tropical._models import EssentialPartRequest
from jacobian.math.polynomials.tropical.operations import (
    _admit_polynomial,
    _digits,
    _finite_value,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_ESSENTIAL_FACES,
    MAX_TROPICAL_ESSENTIAL_RESULT_BYTES,
    MAX_TROPICAL_ESSENTIAL_TERMS,
    MAX_TROPICAL_ESSENTIAL_VARIABLES,
    MAX_TROPICAL_ESSENTIAL_WORK,
    MAX_TROPICAL_EXPONENT,
    MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS,
    TropicalEssentialHullFace,
    TropicalEssentialLiftedFace,
    TropicalPolynomial,
    TropicalPolynomialEssentialPart,
    TropicalPolynomialTerm,
    TropicalScalar,
    TropicalSemiring,
)


def _reject(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(location=location, code=code, message=message)


def _rational(value: Fraction | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _affine_dimension(points: tuple[tuple[Fraction, ...], ...], dimension: int) -> int:
    if len(points) < 2:
        return 0
    origin = points[0]
    return rational_rank(
        [
            [point[axis] - origin[axis] for axis in range(dimension)]
            for point in points[1:]
        ],
        dimension,
    )


def _upper_bound_total_faces(vertex_count: int, dimension: int) -> int:
    """Return the cyclic-polytope upper bound, including the whole polytope."""
    if dimension == 0:
        return 1
    if dimension == 1:
        return 3 if vertex_count >= 2 else 1
    half = dimension // 2
    h = [0] * (dimension + 1)
    for index in range(half + 1):
        value = comb(vertex_count - dimension - 1 + index, index)
        h[index] = value
        h[dimension - index] = value
    proper_face_bound = sum(
        sum(
            comb(dimension - index, dimension - rank) * h[index]
            for index in range(rank + 1)
        )
        for rank in range(1, dimension + 1)
    )
    return proper_face_bound + 1


def _lifted_hull_bounds(
    term_count: int,
    ambient_dimension: int,
    affine_dimension: int,
    component_digits: int,
) -> tuple[int, int, int, int, int]:
    """Return sound hull, DD, and face-count bounds before backend expansion."""
    intrinsic_dimensions = range(1, min(ambient_dimension, term_count - 1) + 1)
    geometric_facet_bound = (
        upper_bound_facets(term_count, affine_dimension) if affine_dimension else 0
    )
    # points_to_facets currently admits against its ambient-dimension bound;
    # retain that backend bound while separately using the intrinsic bound for
    # the actual face-lattice and output estimates.
    facet_bound = max(
        geometric_facet_bound, upper_bound_facets(term_count, ambient_dimension)
    )
    ambient_rays, ambient_pairs = dd_work_bound(term_count, ambient_dimension + 1)
    rank_bounds = tuple(
        dd_work_bound(term_count, rank + 1) for rank in intrinsic_dimensions
    )
    ray_bound = max((ambient_rays, *(rays for rays, _pairs in rank_bounds)))
    pair_bound = max((ambient_pairs, *(pairs for _rays, pairs in rank_bounds)))
    if ray_bound > MAX_DD_RAY_BOUND or pair_bound > MAX_DD_PAIR_BOUND:
        raise PolyhedralConversionAdmissionError(
            "exact lifted hull exceeds the rank-aware double-description work bound "
            f"(at most {ray_bound} rays and {pair_bound} candidate pairs; "
            f"limits are {MAX_DD_RAY_BOUND} and {MAX_DD_PAIR_BOUND})"
        )
    minor_digits = require_dd_height_admissible(
        component_digits, ambient_dimension, affine_halfspaces=False
    )
    try:
        require_dd_weighted_work_admissible(
            term_count,
            ambient_dimension + 1,
            minor_digits,
            candidate_pairs=pair_bound,
        )
    except PolyhedralConversionAdmissionError as error:
        raise PolyhedralConversionAdmissionError(
            f"rank-aware lifted hull height work exceeds its envelope: {error}"
        ) from error
    face_bound = _upper_bound_total_faces(term_count, affine_dimension)
    return facet_bound, ray_bound, pair_bound, face_bound, geometric_facet_bound


def _preflight(
    poly: TropicalPolynomial,
) -> tuple[tuple[tuple[Fraction, ...], ...], int, int, int, int, int]:
    if not isinstance(poly, TropicalPolynomial):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="tropical.essential_part_polynomial_shape",
            message="essential-part input must be a tropical polynomial",
        )
    raw_semiring = getattr(poly, "semiring", None)
    raw_variables = getattr(poly, "variables", None)
    raw_terms = getattr(poly, "terms", None)
    if (
        not isinstance(raw_semiring, TropicalSemiring)
        or type(raw_variables) is not tuple
        or type(raw_terms) is not tuple
    ):
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="tropical.essential_part_polynomial_shape",
            message="essential-part input must use canonical polynomial containers",
        )
    variable_count, term_count = len(raw_variables), len(raw_terms)
    if variable_count > MAX_TROPICAL_ESSENTIAL_VARIABLES:
        _reject(
            ("polynomial", "variables"),
            "tropical.essential_part_dimension_bound",
            f"essential-part hulls admit at most {MAX_TROPICAL_ESSENTIAL_VARIABLES} variables",
        )
    if term_count > MAX_TROPICAL_ESSENTIAL_TERMS:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_term_bound",
            f"essential-part hulls admit at most {MAX_TROPICAL_ESSENTIAL_TERMS} terms",
        )
    for term in raw_terms:
        if (
            not isinstance(term, TropicalPolynomialTerm)
            or type(term.exponents) is not tuple
            or len(term.exponents) != variable_count
            or not isinstance(term.coefficient, TropicalScalar)
        ):
            raise OperationDomainValidationError(
                location=("polynomial", "terms"),
                code="tropical.essential_part_polynomial_shape",
                message="essential-part terms must match the bounded variable axis",
            )
        value = term.coefficient.value
        if value is not None and (
            not isinstance(value, CanonicalRational)
            or type(value.num) is not int
            or type(value.den) is not int
        ):
            raise OperationDomainValidationError(
                location=("polynomial", "terms", "coefficient"),
                code="tropical.essential_part_coefficient",
                message="finite coefficients must use canonical exact rationals",
            )
    _admit_polynomial(poly)
    coefficient_digits = max(
        (
            max(
                _digits(_finite_value(term.coefficient).num),
                _digits(_finite_value(term.coefficient).den),
            )
            for term in poly.terms
        ),
        default=1,
    )
    if coefficient_digits > MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS:
        _reject(
            ("polynomial", "terms", "coefficient"),
            "tropical.essential_part_height_bound",
            "coefficient heights exceed the exact lifted-hull envelope",
        )
    try:
        poly = TropicalPolynomial.model_validate(poly.model_dump(mode="python"))
    except (TypeError, ValueError, AttributeError) as error:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="tropical.essential_part_polynomial_shape",
            message="essential-part input must be a canonical tropical polynomial",
        ) from error
    dimension = variable_count + 1
    component_digits = max(coefficient_digits, len(str(MAX_TROPICAL_EXPONENT)))
    points = tuple(
        (
            *tuple(Fraction(exponent) for exponent in term.exponents),
            _finite_value(term.coefficient).as_fraction(),
        )
        for term in poly.terms
    )
    affine_dimension = _affine_dimension(points, dimension)
    try:
        (
            facet_bound,
            _ray_bound,
            _pair_bound,
            face_bound,
            geometric_facet_bound,
        ) = _lifted_hull_bounds(
            term_count, dimension, affine_dimension, component_digits
        )
    except PolyhedralConversionAdmissionError as error:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_hull_bound",
            str(error),
        )
    work_bound = face_bound * max(1, geometric_facet_bound)
    output_scalar_digits = (
        (dimension + 1) * component_digits + 8 + len(str(facet_bound))
    )
    face_row_bytes = (
        256
        + (dimension + 1) * (2 * output_scalar_digits + 4)
        + term_count * (len(str(term_count)) + 1)
    )
    output_bound = 4096 + term_count * (512 + variable_count * 64)
    output_bound += (face_bound + geometric_facet_bound) * face_row_bytes
    if work_bound > MAX_TROPICAL_ESSENTIAL_WORK:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_work_bound",
            "lifted hull incidence may exceed the admitted face-work envelope",
        )
    if output_bound > MAX_TROPICAL_ESSENTIAL_RESULT_BYTES:
        _reject(
            ("polynomial",),
            "tropical.essential_part_result_bound",
            "lifted hull incidence may exceed the admitted result envelope",
        )
    # points_to_facets applies the authoritative ray, candidate-pair, and
    # height-weighted DD admission before expanding the lifted hull.
    return (
        points,
        term_count,
        dimension,
        facet_bound,
        face_row_bytes,
        affine_dimension,
    )


def _trivial_result(
    poly: TropicalPolynomial, points: tuple[tuple[Fraction, ...], ...]
) -> TropicalPolynomialEssentialPart | None:
    if not poly.terms:
        return TropicalPolynomialEssentialPart.model_construct(
            source=poly,
            polynomial=poly,
            essential_term_indices=(),
            inessential_term_indices=(),
            variable_indices=tuple(range(len(poly.variables))),
            lifted_affine_dimension=0,
            affine_equalities=(),
            hull_facets=(),
            finite_faces=(),
            face_incidence=(),
        )
    if len(poly.terms) != 1:
        return None
    normal = (
        *(_rational(0) for _ in poly.variables),
        _rational(-1 if poly.semiring.convention == "MIN_PLUS" else 1),
    )
    coefficient = points[0][-1]
    offset = _rational(
        -coefficient if poly.semiring.convention == "MIN_PLUS" else coefficient
    )
    face = TropicalEssentialLiftedFace(
        face_index=None,
        dimension=0,
        normal=normal,
        offset=offset,
        source_term_indices=(0,),
    )
    return TropicalPolynomialEssentialPart.model_construct(
        source=poly,
        polynomial=poly,
        essential_term_indices=(0,),
        inessential_term_indices=(),
        variable_indices=tuple(range(len(poly.variables))),
        lifted_affine_dimension=0,
        affine_equalities=(),
        hull_facets=(),
        finite_faces=(face,),
        face_incidence=(
            TropicalEssentialHullFace(
                face_index=0,
                dimension=0,
                source_term_indices=(0,),
                maximal_finite_face_indices=(0,),
            ),
        ),
    )


def _source_hull_facets(
    hull: HullConversion, term_count: int, hull_dimension: int
) -> tuple[TropicalEssentialLiftedFace, ...]:
    return tuple(
        TropicalEssentialLiftedFace(
            face_index=index,
            dimension=max(0, hull_dimension - 1),
            normal=tuple(_rational(value) for value in normal),
            offset=_rational(offset),
            source_term_indices=tuple(
                source for source in range(term_count) if incidence & (1 << source)
            ),
        )
        for index, ((normal, offset), incidence) in enumerate(
            zip(hull.facets, hull.facet_incidence, strict=True)
        )
    )


def _flat_hull_faces(
    poly: TropicalPolynomial,
    points: tuple[tuple[Fraction, ...], ...],
    hull: HullConversion,
    dimension: int,
    normal: tuple[int, ...],
    offset: int,
    face_row_bytes: int,
) -> tuple[TropicalEssentialLiftedFace, tuple[TropicalEssentialHullFace, ...]]:
    sign = Fraction(normal[-1])
    oriented = tuple(Fraction(value) / sign for value in normal)
    oriented_offset = Fraction(offset) / sign
    if poly.semiring.convention == "MIN_PLUS":
        oriented = tuple(-value for value in oriented)
        oriented_offset = -oriented_offset
    whole_normal = tuple(_rational(value) for value in oriented)
    whole_offset = _rational(oriented_offset)
    pending_masks = [(1 << len(points)) - 1]
    face_masks: set[int] = set()
    face_work = 0
    while pending_masks:
        mask = pending_masks.pop()
        if mask in face_masks:
            continue
        face_masks.add(mask)
        for facet_mask in hull.facet_incidence:
            face_work += 1
            if face_work > MAX_TROPICAL_ESSENTIAL_WORK:
                _reject(
                    ("polynomial", "terms"),
                    "tropical.essential_part_face_work_bound",
                    "lower-hull face incidence exceeds the admitted work envelope",
                )
            intersection = mask & facet_mask
            if intersection and intersection != mask:
                pending_masks.append(intersection)
        if len(face_masks) > MAX_TROPICAL_ESSENTIAL_FACES:
            _reject(
                ("polynomial", "terms"),
                "tropical.essential_part_face_bound",
                "lower-hull face incidence exceeds the admitted output envelope",
            )
    ordered_masks = sorted(
        face_masks,
        key=lambda mask: (
            _affine_dimension(
                tuple(points[i] for i in range(len(points)) if mask & (1 << i)),
                dimension,
            ),
            mask,
        ),
    )
    if (
        len(ordered_masks) + len(hull.facets)
    ) * face_row_bytes > MAX_TROPICAL_ESSENTIAL_RESULT_BYTES:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_result_bound",
            "complete lower-hull face incidence exceeds the admitted result envelope",
        )
    records = []
    for face_index, mask in enumerate(ordered_masks):
        indices = tuple(i for i in range(len(points)) if mask & (1 << i))
        records.append(
            TropicalEssentialHullFace(
                face_index=face_index,
                dimension=_affine_dimension(
                    tuple(points[i] for i in indices), dimension
                ),
                source_term_indices=indices,
                maximal_finite_face_indices=(0,),
            )
        )
    finite_face = TropicalEssentialLiftedFace(
        face_index=None,
        dimension=_affine_dimension(points, dimension),
        normal=whole_normal,
        offset=whole_offset,
        source_term_indices=tuple(range(len(points))),
    )
    return (finite_face,), tuple(records)


def _selected_lower_faces(
    poly: TropicalPolynomial,
    points: tuple[tuple[Fraction, ...], ...],
    hull: HullConversion,
    hull_facets: tuple[TropicalEssentialLiftedFace, ...],
    face_row_bytes: int,
) -> tuple[
    tuple[int, ...],
    tuple[TropicalEssentialLiftedFace, ...],
    tuple[TropicalEssentialHullFace, ...],
]:
    term_count = len(points)
    lower_sign = -1 if poly.semiring.convention == "MIN_PLUS" else 1
    selected = tuple(
        index
        for index, (normal, _offset) in enumerate(hull.facets)
        if normal[-1] * lower_sign > 0
    )
    essential_bits = 0
    for index in selected:
        essential_bits |= hull.facet_incidence[index]
    essential = tuple(i for i in range(term_count) if essential_bits & (1 << i))
    pending = [(hull.facet_incidence[index], index) for index in selected]
    seen: dict[int, int] = {}
    intersections = 0
    while pending:
        incidence, parent_index = pending.pop()
        if incidence == 0 or incidence in seen:
            continue
        seen[incidence] = parent_index
        for facet_incidence in hull.facet_incidence:
            intersections += 1
            if intersections > MAX_TROPICAL_ESSENTIAL_WORK:
                _reject(
                    ("polynomial", "terms"),
                    "tropical.essential_part_face_work_bound",
                    "lower-hull face incidence exceeds the admitted work envelope",
                )
            intersection = incidence & facet_incidence
            if intersection and intersection != incidence:
                pending.append((intersection, parent_index))
        if len(seen) > MAX_TROPICAL_ESSENTIAL_FACES:
            _reject(
                ("polynomial", "terms"),
                "tropical.essential_part_face_bound",
                "lower-hull face incidence exceeds the admitted output envelope",
            )
    ordered = sorted(
        seen.items(),
        key=lambda row: (
            _affine_dimension(
                tuple(points[i] for i in range(term_count) if row[0] & (1 << i)),
                len(points[0]),
            ),
            row[0],
        ),
    )
    if (
        len(ordered) + len(hull_facets)
    ) * face_row_bytes > MAX_TROPICAL_ESSENTIAL_RESULT_BYTES:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_result_bound",
            "complete lower-hull face incidence exceeds the admitted result envelope",
        )
    finite_faces = tuple(hull_facets[index] for index in selected)
    selected_position = {
        facet_index: position for position, facet_index in enumerate(selected)
    }
    faces = []
    for face_index, (incidence, _parent_index) in enumerate(ordered):
        indices = tuple(i for i in range(term_count) if incidence & (1 << i))
        faces.append(
            TropicalEssentialHullFace(
                face_index=face_index,
                dimension=_affine_dimension(
                    tuple(points[i] for i in indices), len(points[0])
                ),
                source_term_indices=indices,
                maximal_finite_face_indices=tuple(
                    selected_position[index]
                    for index in selected
                    if incidence & hull.facet_incidence[index] == incidence
                ),
            )
        )
    return essential, finite_faces, tuple(faces)


def tropical_polynomial_essential_part(
    poly: TropicalPolynomial,
) -> TropicalPolynomialEssentialPart:
    """Return all source terms attaining the tropical value at some point.

    This issue-defined contract is tie-inclusive: a term that only ties on a
    lower-dimensional set remains in the result. It is not the unique-region
    functional normal form.
    """
    points, _term_count, dimension, facet_bound, face_row_bytes, hull_dimension = (
        _preflight(poly)
    )
    trivial = _trivial_result(poly, points)
    if trivial is not None:
        return trivial

    try:
        hull = points_to_facets(points, dimension, max_facets=facet_bound)
    except PolyhedralConversionAdmissionError as error:
        _reject(
            ("polynomial", "terms"),
            "tropical.essential_part_hull_bound",
            f"exact lifted hull exceeds its admitted envelope: {error}",
        )

    hull_facets = _source_hull_facets(hull, len(points), hull_dimension)
    equality_rows = tuple(
        tuple(_rational(value) for value in (*normal, offset))
        for normal, offset in hull.affine_equalities
    )
    flat_normal = next(
        (
            (normal, offset)
            for normal, offset in hull.affine_equalities
            if normal[-1] != 0
        ),
        None,
    )
    if flat_normal is None:
        essential, finite_faces, face_incidence = _selected_lower_faces(
            poly, points, hull, hull_facets, face_row_bytes
        )
    else:
        essential = tuple(range(len(points)))
        finite_faces, face_incidence = _flat_hull_faces(
            poly, points, hull, dimension, *flat_normal, face_row_bytes
        )

    inessential = tuple(
        index for index in range(len(points)) if index not in set(essential)
    )
    target = TropicalPolynomial.model_construct(
        semiring=poly.semiring,
        variables=poly.variables,
        terms=tuple(poly.terms[index] for index in essential),
    )
    return TropicalPolynomialEssentialPart.model_construct(
        source=poly,
        polynomial=target,
        essential_term_indices=essential,
        inessential_term_indices=inessential,
        variable_indices=tuple(range(len(poly.variables))),
        lifted_affine_dimension=hull_dimension,
        affine_equalities=equality_rows,
        hull_facets=hull_facets,
        finite_faces=finite_faces,
        face_incidence=face_incidence,
    )


def compute_essential_part(
    request: EssentialPartRequest,
) -> TropicalPolynomialEssentialPart:
    return tropical_polynomial_essential_part(request.polynomial)
