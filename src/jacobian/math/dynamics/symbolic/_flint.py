"""Private FLINT adapters for exact symbolic-dynamics matrices."""


def characteristic_coefficients(
    matrix: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Return coefficients of ``det(lambda I - A)`` in descending order."""

    from flint import fmpz_mat

    polynomial = fmpz_mat(matrix).charpoly()
    return tuple(int(polynomial[index]) for index in range(len(matrix), -1, -1))


def matrix_power_traces(
    matrix: tuple[tuple[int, ...], ...], max_period: int
) -> tuple[int, ...]:
    """Return ``trace(A^n)`` for every ``1 <= n <= max_period``."""

    from flint import fmpz_mat

    adjacency = fmpz_mat(matrix)
    power = adjacency
    size = len(matrix)
    traces: list[int] = []
    for period in range(1, max_period + 1):
        traces.append(sum(int(power[index, index]) for index in range(size)))
        if period < max_period:
            power *= adjacency
    return tuple(traces)


__all__ = ["characteristic_coefficients", "matrix_power_traces"]
