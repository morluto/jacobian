"""Deterministic exact quotient preprocessing inside the supervised worker."""

from __future__ import annotations

import time

from jacobian._execution import OperationExecutionTimeoutError, request_checkpoint


def _column_hermite(matrix: list[list[int]], determinant: int) -> list[list[int]]:
    """Lower column HNF via the pinned deterministic modular algorithm.

    SymPy 1.14's hermite_normal_form(D=delta, check_rank=False) uses
    Cohen Algorithm 2.4.8, not the unbounded randomized FLINT HNF dispatcher.
    The source is nonsingular and delta is its determinant; unit elimination
    preserves delta. Simultaneously reversing rows/columns changes SymPy's
    upper column convention to the lower convention used by this adapter.

    With B=max(delta, max|entry|), b=bit_length(B), the pinned algorithm has
    <=n(n+1)/2 extended-gcd calls and O(n^3) scalar updates. Euclid takes
    <=2b+1 quotient steps. A updates are reduced modulo R<=delta, so temporary
    products are <=2B^2. W column reductions have at most n updates per column,
    each multiplying its magnitude by <=1+delta: |W|<=delta*(1+delta)^n.
    Thus 32*n^3+32*n^2*(b+1) integer-ring operations and
    (n+2)*(b+1) bits per scalar conservatively cover this HNF, including
    Euclidean arithmetic. The original graph determinant is <=50^48;
    all subsequent matrices and reduced divisors have entries <=delta.
    No randomized retries or estimated Smith branches enter this bound.

    https://docs.sympy.org/latest/modules/matrices/normalforms.html
    """
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    n = len(matrix)
    reverse = Matrix([row[::-1] for row in matrix[::-1]])
    hnf = hermite_normal_form(reverse, D=determinant, check_rank=False)
    request_checkpoint("after determinant-modular Hermite form")
    return [[int(hnf[n - 1 - i, n - 1 - j]) for j in range(n)] for i in range(n)]


def prepare_smith_coordinates(
    matrix: list[list[int]], divisor: list[int], deadline: float
) -> tuple[list[list[int]], list[int], int]:
    """Compute each reduction once and admit only the remaining Smith work."""
    from flint import fmpz_mat

    from jacobian.math.graphs.chip_firing._smith_bounds import admit_smith_residual

    request_checkpoint("before Hermite quotient preprocessing")
    # Exact determinant, not a probabilistic estimate. The public owner
    # established a connected graph, so this reduced Laplacian is nonsingular.
    determinant = int(fmpz_mat(matrix).det())
    request_checkpoint("after the reduced-Laplacian determinant")
    hnf = _column_hermite(matrix, determinant)
    residual, image = _reduce_hermite_units(hnf, divisor, determinant, deadline)
    request_checkpoint("after Hermite quotient preprocessing")
    admit_smith_residual(residual, determinant)
    return residual, image, len(matrix) - len(residual)


def _reduce_hermite_units(
    matrix: list[list[int]], divisor: list[int], determinant: int, deadline: float
) -> tuple[list[list[int]], list[int]]:
    """Remove trivial quotient directions in both Hermite orientations.

    Every repeated pass removes at least one coordinate, so an n-square
    input uses at most 2n Hermite forms and n nonsingular rational solves.
    The matrices entering each pass are column HNF, with entries <=delta,
    and divisor entries are reduced modulo delta. Here delta<=50**48 for
    admitted graphs. Row HNF R=U*C has the same entry bound. We transport
    only the divisor, R*(C^-1*d), never materialize U. Fraction-free solve
    stored entries are bounded minors of [C|d]: Hadamard gives
    H=(sqrt(n)*delta)**n. Pre-division products are bounded by 2*H**2,
    and the final product adds at most a factor n*delta to H.
    Thus these bounded reductions scale with n and log(delta), not a
    recursively inflated estimate for hypothetical Smith transformations.

    SymPy owns determinant-modular HNF and FLINT owns fraction-free
    elimination; the adapter only removes
    unit blocks and carries their exact quotient maps. Backend domains:
    https://python-flint.readthedocs.io/en/latest/fmpz_mat.html
    https://python-flint.readthedocs.io/en/latest/fmpq_mat.html
    """
    from flint import fmpq_mat

    while matrix:
        request_checkpoint("during Hermite quotient reduction")
        if time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(
                "request deadline expired during Hermite quotient reduction"
            )
        n = len(matrix)
        units = [i for i in range(n) if matrix[i][i] == 1]
        retained = [i for i in range(n) if matrix[i][i] != 1]
        # In column HNF the unit rows are elementary: after simultaneous
        # reordering C=[[I,0],[B,C1]], so d1=d_tail-B*d_head.
        # delta*Z^n lies in C*Z^n by the adjugate identity.
        residues = [value % determinant for value in divisor]
        divisor = [
            (residues[i] - sum(matrix[i][j] * residues[j] for j in units)) % determinant
            for i in retained
        ]
        matrix = [[matrix[i][j] for j in retained] for i in retained]
        n = len(matrix)
        if n <= 1:
            break

        transposed = [list(row) for row in zip(*matrix, strict=True)]
        row_hnf = [
            list(row)
            for row in zip(*_column_hermite(transposed, determinant), strict=True)
        ]
        request_checkpoint("after row-Hermite quotient reduction")
        retained = [i for i in range(n) if row_hnf[i][i] != 1]
        if len(retained) == n:
            break
        # Row HNF has elementary unit columns: R=[[I,B],[0,C1]].
        # Column elimination of B leaves the transported d_tail unchanged.
        solution = fmpq_mat(matrix).solve(  # type: ignore[call-arg]  # python-flint 0.9 stubs omit algorithm.
            fmpq_mat([[v] for v in divisor]), algorithm="fflu"
        )
        transported = fmpq_mat(row_hnf) * solution
        request_checkpoint("after transporting the Hermite divisor")
        # R*C^-1 is unimodular, so every transported coordinate is integral.
        if any(transported[i, 0].denominator != 1 for i in retained):
            raise RuntimeError("Hermite quotient transport was not integral")
        divisor = [int(transported[i, 0].numerator) % determinant for i in retained]
        if not retained:
            return [], []
        smaller = [[row_hnf[i][j] for j in retained] for i in retained]
        matrix = _column_hermite(smaller, determinant)
        # The next pass has strictly fewer coordinates, not a retry of the
        # same Smith request or a replay of a completed decomposition.
    return matrix, divisor
