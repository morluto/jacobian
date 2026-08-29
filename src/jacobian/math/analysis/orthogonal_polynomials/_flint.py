"""Private FLINT adapters for exact moment matrices."""

from fractions import Fraction


def determinant_and_rank(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, int]:
    """Return the exact determinant and rank of a rational matrix."""

    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in matrix]
    )
    determinant = backend.det()
    return (
        Fraction(int(determinant.numerator), int(determinant.denominator)),
        int(backend.rank()),
    )


__all__ = ["determinant_and_rank"]
