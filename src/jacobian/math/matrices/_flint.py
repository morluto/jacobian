"""Private python-flint adapters for dense exact matrices."""

from fractions import Fraction

RationalEntries = tuple[tuple[Fraction, ...], ...]
IntegerEntries = tuple[tuple[int, ...], ...]


def rational_determinant(entries: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Return the exact determinant through FLINT's dense rational kernel."""

    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in entries]
    )
    result = backend.det()
    return Fraction(int(result.numerator), int(result.denominator))


def rational_rref(entries: RationalEntries) -> tuple[RationalEntries, int]:
    """Return the exact rectangular RREF and rank through FLINT."""

    from flint import fmpq, fmpq_mat

    backend = fmpq_mat(
        [[fmpq(value.numerator, value.denominator) for value in row] for row in entries]
    )
    reduced, rank = backend.rref()
    return (
        tuple(
            tuple(
                Fraction(
                    int(reduced[row, column].numerator),
                    int(reduced[row, column].denominator),
                )
                for column in range(reduced.ncols())
            )
            for row in range(reduced.nrows())
        ),
        int(rank),
    )


def integer_smith_normal_form(entries: IntegerEntries) -> IntegerEntries:
    """Return the exact rectangular Smith diagonal through FLINT."""

    from flint import fmpz_mat

    reduced = fmpz_mat([list(row) for row in entries]).snf()
    return tuple(
        tuple(int(reduced[row, column]) for column in range(reduced.ncols()))
        for row in range(reduced.nrows())
    )


__all__ = ["integer_smith_normal_form", "rational_determinant", "rational_rref"]
