"""Private FLINT adapters for exact conductance systems."""

from fractions import Fraction


def solve_potentials(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
    source: int,
    sink: int,
    *,
    fixed: int,
) -> tuple[Fraction, ...]:
    """Solve the reduced rational Laplacian with one fixed zero gauge."""

    from flint import fmpq, fmpq_mat

    laplacian = [[fmpq(0) for _ in range(vertex_count)] for _ in range(vertex_count)]
    for left, right, value in edges:
        conductance = fmpq(value.numerator, value.denominator)
        laplacian[left][left] += conductance
        laplacian[right][right] += conductance
        laplacian[left][right] -= conductance
        laplacian[right][left] -= conductance
    free = tuple(node for node in range(vertex_count) if node != fixed)
    reduced = fmpq_mat([[laplacian[row][column] for column in free] for row in free])
    rhs = fmpq_mat([[int(node == source) - int(node == sink)] for node in free])
    solution = reduced.solve(rhs)
    potentials = [Fraction(0)] * vertex_count
    for index, node in enumerate(free):
        backend_value = solution[index, 0]
        potentials[node] = Fraction(
            int(backend_value.numerator), int(backend_value.denominator)
        )
    return tuple(potentials)


__all__ = ["solve_potentials"]
