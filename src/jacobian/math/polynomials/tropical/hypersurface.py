"""Exact bounded bivariate tropical hypersurfaces."""

from __future__ import annotations

from fractions import Fraction
from math import comb, gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    MAX_DD_WEIGHTED_HEIGHT_WORK,
    PolyhedralConversionAdmissionError,
    _admit_halfspaces_to_generators,
    _halfspaces_to_generators_from_admission,
    rational_rank,
    require_dd_height_admissible,
    require_dd_weighted_work_admissible,
    require_dd_work_admissible,
)
from jacobian.math.geometry.polytopes._rational_geometry import (
    PolyhedralConversionError,
)
from jacobian.math.geometry.polytopes.values import (
    RationalAffineHalfspace,
    RationalHPolyhedron,
    RationalPolyhedronSpace,
    RationalPolyhedronVPresentation,
)
from jacobian.math.polynomials.tropical._models import BivariateHypersurfaceRequest
from jacobian.math.polynomials.tropical.regular_subdivision import (
    _lifted_points,
    _regular_subdivision_from_admission,
)
from jacobian.math.polynomials.tropical.regular_subdivision import (
    _preflight as _preflight_subdivision,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_EXPONENT,
    MAX_TROPICAL_HYPERSURFACE_CELLS,
    MAX_TROPICAL_HYPERSURFACE_RESULT_BYTES,
    MAX_TROPICAL_SUBDIVISION_RESULT_BYTES,
    TropicalHypersurface,
    TropicalHypersurfaceCell,
    TropicalPolynomial,
    TropicalRegularSubdivision,
)

MAX_TROPICAL_HYPERSURFACE_WORK = 50_000_000_000
"""Height-weighted hull and cell conversion work admitted per hypersurface."""


def _refuse(code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("polynomial",), code=code, message=message
    )


def _preflight_hypersurface(poly: TropicalPolynomial) -> tuple[int, int]:
    """Admit worst-case corner cells before constructing any hull or polyhedron."""

    term_count, coefficient_digits = _preflight_subdivision(poly)
    points = _lifted_points(poly)
    exponent_rows = [
        [point[axis] - points[0][axis] for axis in range(2)] for point in points[1:]
    ]
    if rational_rank(exponent_rows, 2) != 2:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="tropical.hypersurface_newton_dimension",
            message="the exponent support must have affine dimension two",
        )

    # A planar subdivision on n support points has at most 3n-6 edges and
    # 2n-4 two-cells. Its dual curve therefore has at most 5n-10 cells.
    max_corner_cells = 5 * term_count - 10
    if max_corner_cells > MAX_TROPICAL_HYPERSURFACE_CELLS:
        _refuse(
            "tropical.hypersurface_cell_bound",
            "the worst-case corner-cell count exceeds its exact envelope",
        )

    # A cell active on r terms needs 2(r-1) tie inequalities and at most
    # n-r dominance inequalities. The worst case is 2n-2 rows. Heights of
    # coefficient differences are bounded by the sum of two 32-digit inputs.
    max_inequalities = 2 * term_count - 2
    coefficient_difference_digits = 2 * coefficient_digits + 2
    synthetic_rows = tuple(
        (
            (MAX_TROPICAL_EXPONENT, -MAX_TROPICAL_EXPONENT),
            10**coefficient_difference_digits,
        )
        for _ in range(max_inequalities)
    )
    try:
        cell_admission = _admit_halfspaces_to_generators(
            synthetic_rows, 2, maximum_result_minor_digits=32_768
        )
        hull_minor_digits = require_dd_height_admissible(
            max(coefficient_digits, len(str(MAX_TROPICAL_EXPONENT))),
            3,
            affine_halfspaces=False,
        )
        _hull_rays, hull_pairs = require_dd_work_admissible(term_count, 4)
        require_dd_weighted_work_admissible(
            term_count, 4, hull_minor_digits, candidate_pairs=hull_pairs
        )
        require_dd_weighted_work_admissible(
            max_inequalities + 1,
            3,
            cell_admission.minor_digits,
            candidate_pairs=cell_admission.candidate_pairs,
        )
    except PolyhedralConversionAdmissionError as error:
        _refuse("tropical.hypersurface_work_bound", str(error))

    # Match the H-to-V owner's conservative result formula and include a
    # complete H presentation plus all subdivision output before either hull
    # conversion starts.
    total_vectors = cell_admission.maximum_rays + 2
    per_component_bytes = 2 * cell_admission.minor_digits + 32
    v_bytes = total_vectors * 2 * per_component_bytes + total_vectors * 4 + 4_096
    h_bytes = 2_048 + max_inequalities * 640
    cell_bytes = v_bytes + h_bytes + 512 + term_count * 16 + 16 * 8
    incidence_bytes = max_corner_cells * max_corner_cells * 16
    estimated_output_bytes = (
        MAX_TROPICAL_SUBDIVISION_RESULT_BYTES
        + max_corner_cells * cell_bytes
        + incidence_bytes
    )
    if estimated_output_bytes > MAX_TROPICAL_HYPERSURFACE_RESULT_BYTES:
        _refuse(
            "tropical.hypersurface_output_bound",
            "the complete corner complex may exceed the exact output envelope",
        )

    hull_work = hull_pairs * hull_minor_digits**2
    rank_digits = 4 * cell_admission.minor_digits + 4
    rank_rows = cell_admission.maximum_rays + 2
    cell_rank_work = rank_rows * 8 * rank_digits**2
    cell_work = max_corner_cells * (
        cell_admission.candidate_pairs * cell_admission.minor_digits**2 + cell_rank_work
    )
    plane_work = term_count * comb(term_count, 3)
    if hull_work + cell_work + plane_work > MAX_TROPICAL_HYPERSURFACE_WORK:
        _refuse(
            "tropical.hypersurface_work_bound",
            "combined lifted-hull and corner-cell work exceeds its exact envelope",
        )
    return term_count, coefficient_digits


def _as_rational(value: Fraction | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _cell_h_presentation(
    poly: TropicalPolynomial, active: tuple[int, ...]
) -> RationalHPolyhedron:
    anchor = poly.terms[active[0]]
    anchor_height = poly.terms[active[0]].coefficient.value.as_fraction()
    rows: list[RationalAffineHalfspace] = []

    def add_row(normal: tuple[int, int], bound: Fraction) -> None:
        rows.append(
            RationalAffineHalfspace(
                normal=tuple(_as_rational(value) for value in normal),
                bound=_as_rational(bound),
            )
        )

    for index in active[1:]:
        term = poly.terms[index]
        height = term.coefficient.value.as_fraction()
        normal = tuple(
            term.exponents[axis] - anchor.exponents[axis] for axis in range(2)
        )
        bound = anchor_height - height
        add_row(normal, bound)
        add_row(tuple(-value for value in normal), -bound)

    for index, term in enumerate(poly.terms):
        if index in active:
            continue
        height = term.coefficient.value.as_fraction()
        difference = tuple(
            anchor.exponents[axis] - term.exponents[axis] for axis in range(2)
        )
        if poly.semiring.convention == "MIN_PLUS":
            add_row(difference, height - anchor_height)
        else:
            add_row(tuple(-value for value in difference), anchor_height - height)

    return RationalHPolyhedron(
        space=RationalPolyhedronSpace(axes=poly.variables),
        inequalities=tuple(rows),
    )


def _admit_h_to_v(source: RationalHPolyhedron):
    rows = tuple(
        (
            tuple(component.as_fraction() for component in row.normal),
            row.bound.as_fraction(),
        )
        for row in source.inequalities
    )
    try:
        admission = _admit_halfspaces_to_generators(
            rows, len(source.space.axes), maximum_result_minor_digits=32_768
        )
    except PolyhedralConversionAdmissionError as error:
        _refuse("tropical.hypersurface_cell_conversion", str(error))

    dimension = len(source.space.axes)
    rank_digits = 2 * max(1, dimension) * admission.minor_digits + dimension + 2
    rank_rows = admission.maximum_rays + dimension
    rank_work = rank_rows * max(1, dimension) ** 3 * rank_digits**2
    if rank_work > MAX_DD_WEIGHTED_HEIGHT_WORK:
        _refuse(
            "tropical.hypersurface_cell_conversion_work",
            "corner-cell affine rank work exceeds the exact envelope",
        )
    return admission, (
        admission.candidate_pairs * admission.minor_digits**2 + rank_work
    )


def _h_to_v_expand(
    source: RationalHPolyhedron, admission
) -> RationalPolyhedronVPresentation:
    try:
        converted = _halfspaces_to_generators_from_admission(admission)
    except (PolyhedralConversionAdmissionError, PolyhedralConversionError) as error:
        _refuse("tropical.hypersurface_cell_conversion", str(error))
    if converted.empty:
        raise RuntimeError("dual subdivision produced an empty tropical cell")
    result = RationalPolyhedronVPresentation._from_kernel(
        space=source.space,
        points=tuple(
            tuple(_as_rational(value) for value in point)
            for point in converted.vertices
        ),
        rays=tuple(
            tuple(_as_rational(value) for value in ray)
            for ray in converted.recession_rays
        ),
        lineality=tuple(
            tuple(_as_rational(value) for value in basis)
            for basis in converted.lineality_basis
        ),
        empty=False,
        affine_dimension=converted.affine_dimension,
    )
    return result


def _lattice_length(poly: TropicalPolynomial, active: tuple[int, ...]) -> int:
    endpoints = [poly.terms[index].exponents for index in active]
    # A one-dimensional dual face's geometric lattice length is the gcd of
    # the coordinate differences between its extreme points, not all source
    # term spacings along the face.
    delta = max(endpoints, key=lambda point: (point[0], point[1]))
    anchor = min(endpoints, key=lambda point: (point[0], point[1]))
    length = gcd(abs(delta[0] - anchor[0]), abs(delta[1] - anchor[1]))
    if length <= 0:
        raise RuntimeError("one-dimensional dual face has zero lattice length")
    return length


def tropical_bivariate_hypersurface(poly: TropicalPolynomial) -> TropicalHypersurface:
    """Return the complete exact bivariate corner locus of one polynomial."""

    term_count, _coefficient_digits = _preflight_hypersurface(poly)
    subdivision: TropicalRegularSubdivision = _regular_subdivision_from_admission(
        poly, term_count
    )
    selected_supports = tuple(
        support for support in subdivision.face_supports if support.dimension in (1, 2)
    )
    if not 1 <= len(selected_supports) <= MAX_TROPICAL_HYPERSURFACE_CELLS:
        raise RuntimeError("subdivision dual exceeds the admitted corner-cell bound")

    h_presentations = tuple(
        _cell_h_presentation(poly, support.source_term_indices)
        for support in selected_supports
    )
    cell_ids = tuple(f"tc{index:03d}" for index in range(len(selected_supports)))
    vertex_indices = tuple(
        index
        for index, support in enumerate(selected_supports)
        if support.dimension == 2
    )
    incident_by_index: list[list[str]] = [[] for _ in selected_supports]
    for index, support in enumerate(selected_supports):
        if support.dimension != 1:
            continue
        for vertex_index in vertex_indices:
            vertex_support = selected_supports[vertex_index]
            if set(support.source_term_indices).issubset(
                vertex_support.source_term_indices
            ):
                incident_by_index[index].append(cell_ids[vertex_index])
                incident_by_index[vertex_index].append(cell_ids[index])

    # Prepare every exact H-to-V plan and aggregate its actual output/work
    # before expanding the first corner cell.
    admissions = []
    aggregate_v_bytes = 0
    aggregate_cell_work = 0
    for h_presentation in h_presentations:
        admission, work = _admit_h_to_v(h_presentation)
        admissions.append(admission)
        total_vectors = admission.maximum_rays + 2
        aggregate_v_bytes += (
            total_vectors * 2 * (2 * admission.minor_digits + 32)
            + total_vectors * 4
            + 4_096
        )
        aggregate_cell_work += work
    total_active_references = sum(
        len(support.source_term_indices) for support in selected_supports
    )
    total_dual_references = sum(
        len(support.lifted_face_indices) for support in selected_supports
    )
    total_incidence_references = sum(len(incident) for incident in incident_by_index)
    output_bound = (
        MAX_TROPICAL_SUBDIVISION_RESULT_BYTES
        + aggregate_v_bytes
        + sum(len(presentation.inequalities) * 640 for presentation in h_presentations)
        + len(selected_supports) * 4_096
        + total_active_references * 16
        + total_dual_references * 80
        + total_incidence_references * 16
    )
    if output_bound > MAX_TROPICAL_HYPERSURFACE_RESULT_BYTES:
        _refuse(
            "tropical.hypersurface_output_bound",
            "the exact corner-cell values exceed the admitted output envelope",
        )
    if aggregate_cell_work > MAX_TROPICAL_HYPERSURFACE_WORK:
        _refuse(
            "tropical.hypersurface_work_bound",
            "combined exact corner-cell conversion exceeds its work envelope",
        )

    prepared = tuple(
        _h_to_v_expand(h_presentation, admission)
        for h_presentation, admission in zip(h_presentations, admissions, strict=True)
    )

    cells = []
    for index, (support, h_presentation, v_presentation) in enumerate(
        zip(selected_supports, h_presentations, prepared, strict=True)
    ):
        dimension = 2 - support.dimension
        if v_presentation.affine_dimension != dimension:
            raise RuntimeError(
                "H-to-V corner-cell dimension disagrees with dual subdivision"
            )
        active = support.source_term_indices
        cells.append(
            TropicalHypersurfaceCell.model_construct(
                cell_id=cell_ids[index],
                dimension=dimension,
                inequalities=h_presentation,
                generators=v_presentation,
                active_term_indices=active,
                dual_face_id=support.face_id,
                dual_lifted_face_indices=support.lifted_face_indices,
                incident_cell_ids=tuple(sorted(incident_by_index[index])),
                weight=_lattice_length(poly, active) if dimension == 1 else None,
            )
        )
    return TropicalHypersurface.model_construct(
        subdivision=subdivision,
        cells=tuple(cells),
    )


def compute_bivariate_hypersurface(
    request: BivariateHypersurfaceRequest,
) -> TropicalHypersurface:
    return tropical_bivariate_hypersurface(request.polynomial)
