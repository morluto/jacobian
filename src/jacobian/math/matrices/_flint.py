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


def rational_characteristic_polynomial(
    entries: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    """Return monic ``det(λI - A)`` coefficients, highest degree first."""

    from flint import fmpq, fmpq_mat

    order = len(entries)
    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in entries]
    )
    polynomial = backend.charpoly()
    return tuple(
        Fraction(int(polynomial[index].p), int(polynomial[index].q))
        for index in range(order, -1, -1)
    )


__all__ = ["rational_characteristic_polynomial", "rational_determinant"]
