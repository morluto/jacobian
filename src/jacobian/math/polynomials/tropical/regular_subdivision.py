"""Bounded exact bivariate regular subdivisions of tropical polynomials."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.polytopes._models import (
    PrimitiveFacet,
    RationalCoordinateSpace,
    RationalPolytopeVertex,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes._polyhedral_conversion import (
    PolyhedralConversionAdmissionError,
    points_to_facets,
    rational_rank,
)
from jacobian.math.geometry.polytopes.complexes._models import (
    MAX_COMPLEX_CELLS,
    ComplexFace,
    SourceCellTransport,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)
from jacobian.math.geometry.polytopes.operations import facet_incidence
from jacobian.math.geometry.polytopes.values import Vertex
from jacobian.math.polynomials.tropical._models import (
    BivariateRegularSubdivisionRequest,
)
from jacobian.math.polynomials.tropical.operations import (
    _admit_polynomial,
    _digits,
    _finite_value,
)
from jacobian.math.polynomials.tropical.values import (
    MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS,
    MAX_TROPICAL_SUBDIVISION_RESULT_BYTES,
    MAX_TROPICAL_SUBDIVISION_TERMS,
    TropicalLiftedSubdivisionFace,
    TropicalPolynomial,
    TropicalRegularSubdivision,
    TropicalSubdivisionFaceSupport,
)


def _reject(location: tuple[str, ...], code: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=location,
        code=code,
        message=message,
    )


def _preflight(poly: TropicalPolynomial) -> tuple[int, int]:
    """Admit the small exact hull, projected complex, and complete output."""
    _admit_polynomial(poly)
    if len(poly.variables) != 2:
        raise OperationDomainValidationError(
            location=("polynomial", "variables"),
            code="tropical.regular_subdivision_bivariate",
            message="regular subdivision requires exactly two variables",
        )
    term_count = len(poly.terms)
    if term_count < 3:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="tropical.regular_subdivision_newton_dimension",
            message="the Newton polygon must have affine dimension two",
        )
    if term_count > MAX_TROPICAL_SUBDIVISION_TERMS:
        _reject(
            ("polynomial", "terms"),
            "tropical.regular_subdivision_term_bound",
            f"bivariate subdivision admits at most {MAX_TROPICAL_SUBDIVISION_TERMS} terms",
        )
    coefficient_digits = max(
        max(
            _digits(_finite_value(term.coefficient).num),
            _digits(_finite_value(term.coefficient).den),
        )
        for term in poly.terms
    )
    if coefficient_digits > MAX_TROPICAL_SUBDIVISION_COEFFICIENT_DIGITS:
        _reject(
            ("polynomial", "terms", "coefficient"),
            "tropical.regular_subdivision_height_bound",
            "coefficient heights exceed the exact subdivision envelope",
        )

    # A 3-polytope with n vertices has at most 2n-4 facets. This bounds
    # the top-dimensional projected cells; each planar cell with at most n
    # source points has at most 2n+2 faces including the empty face.
    maximal_cells = max(1, 2 * term_count - 4)
    if maximal_cells > MAX_COMPLEX_CELLS:
        _reject(
            ("polynomial", "terms"),
            "tropical.regular_subdivision_cell_bound",
            "lifted hull may exceed the polytopal-complex maximal-cell envelope",
        )
    face_bound = min(1024, maximal_cells * (2 * term_count + 2))
    estimated_result_bytes = (
        8_192
        + term_count * (2 * coefficient_digits + 512)
        + face_bound * (512 + 256 * term_count)
    )
    if estimated_result_bytes > MAX_TROPICAL_SUBDIVISION_RESULT_BYTES:
        _reject(
            ("polynomial",),
            "tropical.regular_subdivision_result_bound",
            "regular subdivision output may exceed the admitted byte envelope",
        )
    # The 10-term input cap bounds the affine-plane scan by C(10, 3) exact
    # candidates. points_to_facets then applies the shared DD ray, pair, and
    # coefficient-height admission before its lifted-hull conversion.
    return term_count, coefficient_digits


def _rational(value: Fraction | int) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def _lifted_points(poly: TropicalPolynomial) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        (
            Fraction(term.exponents[0]),
            Fraction(term.exponents[1]),
            _finite_value(term.coefficient).as_fraction(),
        )
        for term in poly.terms
    )


def _affine_lift_face(
    points: tuple[tuple[Fraction, ...], ...], convention: str
) -> tuple[tuple[CanonicalRational, ...], CanonicalRational]:
    """Return the convention-oriented supporting plane for a rank-two lift."""
    for first, second, third in combinations(points, 3):
        dx1, dy1 = second[0] - first[0], second[1] - first[1]
        dx2, dy2 = third[0] - first[0], third[1] - first[1]
        determinant = dx1 * dy2 - dx2 * dy1
        if determinant:
            dz1, dz2 = second[2] - first[2], third[2] - first[2]
            slope_x = (dz1 * dy2 - dz2 * dy1) / determinant
            slope_y = (dx1 * dz2 - dx2 * dz1) / determinant
            intercept = first[2] - slope_x * first[0] - slope_y * first[1]
            if convention == "MIN_PLUS":
                normal = (slope_x, slope_y, Fraction(-1))
                offset = -intercept
            else:
                normal = (-slope_x, -slope_y, Fraction(1))
                offset = intercept
            return tuple(_rational(value) for value in normal), _rational(offset)
    raise RuntimeError("full-dimensional Newton support has no noncollinear triple")


def _make_cell(
    poly: TropicalPolynomial, source_indices: tuple[int, ...]
) -> RationalVPolytope:
    vertices = tuple(
        RationalPolytopeVertex(
            vertex_id=f"v{index:03d}",
            coordinates=tuple(
                CanonicalRational.from_integer_ratio(exponent, 1)
                for exponent in poly.terms[index].exponents
            ),
        )
        for index in source_indices
    )
    return RationalVPolytope(
        space=RationalCoordinateSpace(axes=poly.variables),
        vertices=vertices,
    )


def _face_source_terms(
    *,
    poly: TropicalPolynomial,
    complex_faces: tuple[ComplexFace, ...],
    source_cell_map: tuple[SourceCellTransport, ...],
    source_indices_by_cell: tuple[tuple[int, ...], ...],
    cell_facets: tuple[tuple[PrimitiveFacet, ...], ...],
) -> tuple[TropicalSubdivisionFaceSupport, ...]:
    cell_by_id = {row.cell_id: row.source_index for row in source_cell_map}
    records = []
    for face in complex_faces:
        lifted_indices = tuple(
            sorted(cell_by_id[cell_id] for cell_id in face.maximal_cell_ids)
        )
        if face.dimension == -1:
            records.append(
                TropicalSubdivisionFaceSupport(
                    face_id=face.face_id,
                    dimension=-1,
                    source_term_indices=(),
                    lifted_face_indices=lifted_indices,
                )
            )
            continue

        per_parent_support = []
        for cell_source_index in lifted_indices:
            cell_indices = source_indices_by_cell[cell_source_index]
            cell_coordinates = tuple(
                tuple(Fraction(term.exponents[axis]) for axis in range(2))
                for term in (poly.terms[index] for index in cell_indices)
            )
            face_vertices = {
                tuple(coordinate.as_fraction() for coordinate in point.coordinates)
                for point in face.vertices
            }
            active_facets = []
            if face.dimension < 2:
                for facet in cell_facets[cell_source_index]:
                    facet_coordinates = {
                        cell_coordinates[index] for index in facet.source_vertex_indices
                    }
                    if face_vertices.issubset(facet_coordinates):
                        active_facets.append(facet.halfspace)
            face_terms = []
            for term_index in source_indices_by_cell[cell_source_index]:
                point = tuple(
                    Fraction(value) for value in poly.terms[term_index].exponents
                )
                if all(
                    sum(
                        coefficient.as_fraction() * coordinate
                        for coefficient, coordinate in zip(
                            halfspace.coefficients, point, strict=True
                        )
                    )
                    == halfspace.offset.as_fraction()
                    for halfspace in active_facets
                ):
                    face_terms.append(term_index)
            per_parent_support.append(tuple(face_terms))
        if any(support != per_parent_support[0] for support in per_parent_support[1:]):
            raise RuntimeError(
                "shared subdivision face has inconsistent lifted provenance"
            )
        records.append(
            TropicalSubdivisionFaceSupport(
                face_id=face.face_id,
                dimension=face.dimension,
                source_term_indices=per_parent_support[0],
                lifted_face_indices=lifted_indices,
            )
        )
    return tuple(records)


def tropical_bivariate_regular_subdivision(
    poly: TropicalPolynomial,
) -> TropicalRegularSubdivision:
    """Return one polynomial's exact planar regular subdivision.

    The result is the face-closed projection of lower lifted facets for
    min-plus and upper lifted facets for max-plus. It does not intersect
    tropical hypersurfaces or claim a system-level variety.
    """
    term_count, _coefficient_digits = _preflight(poly)
    return _regular_subdivision_from_admission(poly, term_count)


def _regular_subdivision_from_admission(
    poly: TropicalPolynomial, term_count: int
) -> TropicalRegularSubdivision:
    """Construct after a composing operation has admitted subdivision work."""
    points = _lifted_points(poly)
    exponent_differences = [
        [point[axis] - points[0][axis] for axis in range(2)] for point in points[1:]
    ]
    if rational_rank(exponent_differences, 2) != 2:
        raise OperationDomainValidationError(
            location=("polynomial", "terms"),
            code="tropical.regular_subdivision_newton_dimension",
            message="the exponent support must have affine dimension two",
        )
    lifted_differences = [
        [point[axis] - points[0][axis] for axis in range(3)] for point in points[1:]
    ]
    lift_rank = rational_rank(lifted_differences, 3)

    facet_rows: list[
        tuple[
            int | None,
            tuple[CanonicalRational, ...],
            CanonicalRational,
            tuple[int, ...],
        ]
    ] = []
    if lift_rank == 2:
        normal, offset = _affine_lift_face(points, poly.semiring.convention)
        facet_rows.append((None, normal, offset, tuple(range(term_count))))
    else:
        try:
            hull = points_to_facets(
                points,
                3,
                max_facets=2 * term_count - 4,
            )
        except PolyhedralConversionAdmissionError as error:
            _reject(
                ("polynomial", "terms"),
                "tropical.regular_subdivision_hull_bound",
                f"exact lifted hull exceeds its admitted envelope: {error}",
            )
        for hull_facet_index, ((normal, offset), incidence) in enumerate(
            zip(hull.facets, hull.facet_incidence, strict=True)
        ):
            vertical = normal[2]
            is_subdivision_face = (
                vertical < 0 if poly.semiring.convention == "MIN_PLUS" else vertical > 0
            )
            if not is_subdivision_face:
                continue
            source_term_indices = tuple(
                index for index in range(term_count) if incidence & (1 << index)
            )
            facet_rows.append(
                (
                    hull_facet_index,
                    tuple(_rational(value) for value in (*normal,)),
                    _rational(offset),
                    source_term_indices,
                )
            )
        if not facet_rows:
            raise RuntimeError(
                "full-dimensional lifted hull has no selected support faces"
            )

    source_indices_by_cell = tuple(row[3] for row in facet_rows)
    cells = tuple(_make_cell(poly, indices) for indices in source_indices_by_cell)
    cell_complex = polytopal_complex_closure(cells)
    cell_id_by_source = {
        row.source_index: row.cell_id for row in cell_complex.source_cell_map
    }
    lifted_faces = tuple(
        TropicalLiftedSubdivisionFace(
            lifted_face_index=index,
            source_hull_facet_index=hull_index,
            normal=normal,
            offset=offset,
            source_term_indices=indices,
            subdivision_cell_id=cell_id_by_source[index],
        )
        for index, (hull_index, normal, offset, indices) in enumerate(facet_rows)
    )

    cell_facets = []
    for indices in source_indices_by_cell:
        vertices = tuple(
            Vertex(
                coordinates=tuple(
                    CanonicalRational.from_integer_ratio(exponent, 1)
                    for exponent in poly.terms[index].exponents
                )
            )
            for index in indices
        )
        profile = facet_incidence(vertices, 2)
        cell_facets.append(profile.facets)
    face_supports = _face_source_terms(
        poly=poly,
        complex_faces=cell_complex.faces,
        source_cell_map=cell_complex.source_cell_map,
        source_indices_by_cell=source_indices_by_cell,
        cell_facets=tuple(cell_facets),
    )
    return TropicalRegularSubdivision.model_construct(
        source=poly,
        cell_complex=cell_complex,
        lifted_faces=lifted_faces,
        face_supports=face_supports,
    )


def compute_bivariate_regular_subdivision(
    request: BivariateRegularSubdivisionRequest,
) -> TropicalRegularSubdivision:
    return tropical_bivariate_regular_subdivision(request.polynomial)
