"""Private FLINT adapters for exact symbolic-dynamics matrices."""


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


__all__ = ["matrix_power_traces"]
