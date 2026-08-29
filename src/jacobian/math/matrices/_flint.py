"""Private python-flint adapters for dense exact matrices."""

from fractions import Fraction


def rational_determinant(entries: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Return the exact determinant through FLINT's dense rational kernel."""

    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in entries]
    )
    result = backend.det()
    return Fraction(int(result.numerator), int(result.denominator))


__all__ = ["rational_determinant"]
