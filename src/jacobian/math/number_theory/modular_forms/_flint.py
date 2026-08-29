"""Private FLINT adapters for exact finite modular-form series."""

from fractions import Fraction


def delta_coefficients(
    e4: tuple[Fraction, ...], e6: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Return ``(E4^3 - E6^2) / 1728`` through the input precision.

    ``fmpq_series`` multiplication consults FLINT's process-global ``ctx.cap``
    even when its operands carry explicit precision.  Exact polynomial
    multiplication followed by explicit truncation computes the same finite
    prefix without mutating or depending on that ambient context.
    """

    from flint import fmpq, fmpq_poly

    precision = len(e4)
    e4_poly = fmpq_poly([fmpq(value.numerator, value.denominator) for value in e4])
    e6_poly = fmpq_poly([fmpq(value.numerator, value.denominator) for value in e6])
    e4_cubed = ((e4_poly * e4_poly).truncate(precision) * e4_poly).truncate(precision)
    e6_squared = (e6_poly * e6_poly).truncate(precision)
    delta = (e4_cubed - e6_squared) / 1728
    return tuple(
        Fraction(int(delta[index].numerator), int(delta[index].denominator))
        for index in range(precision)
    )


__all__ = ["delta_coefficients"]
