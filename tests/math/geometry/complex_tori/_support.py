"""Sparse exact fixtures for lattice-presented complex tori."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.complex_tori import LatticeComplexStructure
from jacobian.math.lattices.invariant_forms import IntegralBilinearForm
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    IntegerMatrix,
)
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    embeddings,
)

QuarticCoordinates = tuple[int | Fraction, ...]
QuarticRows = tuple[tuple[QuarticCoordinates, ...], ...]


def rational(value: int | Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(value))


def quartic_matrix(
    entries: QuarticRows,
    *,
    root_index: int = 1,
) -> EmbeddedRealSimpleNumberFieldMatrix:
    """Realize power-basis coordinates in the selected root of x^4 - 2."""

    presentation = SimpleNumberFieldPresentation(
        coefficients_descending=("1", "0", "0", "0", "-2")
    )
    embedding = embeddings(presentation).records[root_index].embedding
    return EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding,
        entries=tuple(
            tuple(
                SimpleNumberFieldElement(
                    presentation=presentation,
                    coefficients_ascending=tuple(
                        rational(coordinate) for coordinate in value
                    ),
                )
                for value in row
            )
            for row in entries
        ),
    )


def quartic_rank_zero_torus() -> LatticeComplexStructure:
    """Return J = [[0,-Y0],[Y0^-1,0]] for Y0=[[1,a],[a^3,a^2]]."""

    zero = (0, 0, 0, 0)
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2", "e3", "e4"),
        complex_structure=quartic_matrix(
            (
                (zero, zero, (-1, 0, 0, 0), (0, -1, 0, 0)),
                (zero, zero, (0, 0, 0, -1), (0, 0, -1, 0)),
                ((-1, 0, -1, 0), (0, 1, 0, Fraction(1, 2)), zero, zero),
                ((0, 1, 0, 1), (-1, 0, Fraction(-1, 2), 0), zero, zero),
            )
        ),
    )


def quartic_rank_one_torus(*, root_index: int = 1) -> LatticeComplexStructure:
    """Return J = [[0,-Y1],[Y1^-1,0]] for Y1=[[1,a],[a,-a^3]]."""

    zero = (0, 0, 0, 0)
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2", "e3", "e4"),
        complex_structure=quartic_matrix(
            (
                (zero, zero, (-1, 0, 0, 0), (0, -1, 0, 0)),
                (zero, zero, (0, -1, 0, 0), (0, 0, 0, 1)),
                (
                    (2, -1, 1, -1),
                    (1, -1, 1, Fraction(-1, 2)),
                    zero,
                    zero,
                ),
                (
                    (1, -1, 1, Fraction(-1, 2)),
                    (1, -1, Fraction(1, 2), Fraction(-1, 2)),
                    zero,
                    zero,
                ),
            ),
            root_index=root_index,
        ),
    )


def standard_alternating_form(
    torus: LatticeComplexStructure,
) -> IntegralBilinearForm:
    return IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(
            entries=(
                ("0", "0", "1", "0"),
                ("0", "0", "0", "1"),
                ("-1", "0", "0", "0"),
                ("0", "-1", "0", "0"),
            )
        ),
    )


def quartic_index_six_torus() -> LatticeComplexStructure:
    """Return P^-1 J P for the explicit determinant-minus-six P in its test."""

    zero = (0, 0, 0, 0)
    return LatticeComplexStructure(
        coordinate_axis=("e1", "e2", "e3", "e4"),
        complex_structure=quartic_matrix(
            (
                (
                    zero,
                    (Fraction(1, 6), Fraction(-1, 6), Fraction(1, 6), Fraction(-1, 12)),
                    zero,
                    (
                        Fraction(-1, 6),
                        Fraction(1, 6),
                        Fraction(-1, 12),
                        Fraction(1, 12),
                    ),
                ),
                ((0, -6, 0, 0), zero, (-1, 0, 0, 0), zero),
                (
                    zero,
                    (2, -1, 1, -1),
                    zero,
                    (-1, 1, -1, Fraction(1, 2)),
                ),
                ((0, 0, 0, -6), zero, (0, 1, 0, 0), zero),
            )
        ),
    )


def index_six_alternating_form(
    torus: LatticeComplexStructure,
) -> IntegralBilinearForm:
    return IntegralBilinearForm(
        coordinate_axis=torus.coordinate_axis,
        kind="ALTERNATING",
        matrix=IntegerMatrix(
            entries=(
                ("0", "0", "0", "6"),
                ("0", "0", "1", "0"),
                ("0", "-1", "0", "0"),
                ("-6", "0", "0", "0"),
            )
        ),
    )


__all__ = [
    "index_six_alternating_form",
    "quartic_index_six_torus",
    "quartic_matrix",
    "quartic_rank_one_torus",
    "quartic_rank_zero_torus",
    "rational",
    "standard_alternating_form",
]
