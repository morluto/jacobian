"""Exact orthogonal direct sums of rational quadratic forms."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.matrices.values import (
    RationalMatrix,
    rational_matrix_from_fractions,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_models import (
    QuadraticFormDirectSumRequest,
    QuadraticFormDirectSumResult,
    QuadraticFormRestrictionRequest,
    QuadraticFormRestrictionResult,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def quadratic_form_direct_sum(
    request: QuadraticFormDirectSumRequest,
) -> QuadraticFormDirectSumResult:
    """Return the orthogonal sum, with inclusion and projection matrices.

    Coordinates are tagged by factor and local index so equal source labels in
    different factors remain distinct. Original labels remain available in
    ``source_forms``; map matrices use the same ordered local coordinates.
    """

    forms = request.forms
    total = sum(len(form.axis) for form in forms)
    axis: list[str] = []
    diagonal: list[CanonicalRational] = []
    cross_terms: list[QuadraticCrossTerm] = []
    inclusions: list[RationalMatrix] = []
    projections: list[RationalMatrix] = []
    offset = 0
    for component, form in enumerate(forms):
        dimension = len(form.axis)
        axis.extend(f"qf{component}_{coordinate}" for coordinate in range(dimension))
        diagonal.extend(form.diagonal_coefficients)
        cross_terms.extend(
            QuadraticCrossTerm(
                left=term.left + offset,
                right=term.right + offset,
                coefficient=term.coefficient,
            )
            for term in form.cross_terms
        )
        inclusion_entries = [
            [Fraction(0) for _ in range(dimension)] for _ in range(total)
        ]
        projection_entries = [
            [Fraction(0) for _ in range(total)] for _ in range(dimension)
        ]
        for coordinate in range(dimension):
            inclusion_entries[offset + coordinate][coordinate] = Fraction(1)
            projection_entries[coordinate][offset + coordinate] = Fraction(1)
        inclusions.append(
            rational_matrix_from_fractions(inclusion_entries, column_count=dimension)
        )
        projections.append(
            rational_matrix_from_fractions(projection_entries, column_count=total)
        )
        offset += dimension

    result_form = RationalQuadraticForm(
        axis=tuple(axis),
        diagonal_coefficients=tuple(diagonal),
        cross_terms=tuple(cross_terms),
    )
    return QuadraticFormDirectSumResult._from_kernel(
        source_forms=forms,
        form=result_form,
        coordinate_inclusions=tuple(inclusions),
        coordinate_projections=tuple(projections),
    )


def quadratic_form_restrict_coordinates(
    request: QuadraticFormRestrictionRequest,
) -> QuadraticFormRestrictionResult:
    """Restrict to an ordered coordinate subset of the source axis."""

    source = request.form
    selected = request.selected_axis
    positions = {label: index for index, label in enumerate(source.axis)}
    selected_positions = tuple(positions[label] for label in selected)
    restricted_index = {
        source_index: index for index, source_index in enumerate(selected_positions)
    }
    diagonal = tuple(
        source.diagonal_coefficients[index] for index in selected_positions
    )
    cross_terms = []
    for term in source.cross_terms:
        if term.left not in restricted_index or term.right not in restricted_index:
            continue
        left, right = sorted(
            (restricted_index[term.left], restricted_index[term.right])
        )
        cross_terms.append(
            QuadraticCrossTerm(left=left, right=right, coefficient=term.coefficient)
        )
    restricted = RationalQuadraticForm(
        axis=selected,
        diagonal_coefficients=diagonal,
        cross_terms=tuple(
            sorted(cross_terms, key=lambda term: (term.left, term.right))
        ),
    )
    entries = tuple(
        tuple(
            CanonicalRational.from_integer_ratio(int(row == source_index), 1)
            for source_index in selected_positions
        )
        for row in range(len(source.axis))
    )
    inclusion = rational_matrix_from_fractions(
        tuple(tuple(value.as_fraction() for value in row) for row in entries),
        column_count=len(selected),
    )
    return QuadraticFormRestrictionResult._from_kernel(
        source_form=source,
        selected_axis=selected,
        form=restricted,
        inclusion=inclusion,
    )


__all__ = ["quadratic_form_direct_sum", "quadratic_form_restrict_coordinates"]
