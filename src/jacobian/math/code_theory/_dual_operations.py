"""Dual code and syndrome operations for coding theory."""

from jacobian.math.code_theory._models import (
    DualCodeRequest,
    DualCodeResult,
    SyndromeRequest,
    SyndromeResult,
)


def compute_dual_code(request: DualCodeRequest) -> DualCodeResult:
    """Compute the dual code (parity check matrix) from a generator matrix.

    Uses SymPy's null space computation over GF(p) to find the parity
    check matrix H such that G * H^T = 0.
    """
    from sympy import Matrix

    p = request.field_order
    rows = request.generator_matrix
    k = len(rows)
    n = len(rows[0])

    mat = Matrix(rows)

    # Compute null space over GF(p)
    null_space = mat.nullspace()

    # Convert null space vectors to rows of H
    if not null_space:
        # Null space is trivial - shouldn't happen for k < n
        parity_check: tuple[tuple[int, ...], ...] = ()
    else:
        # Convert each null space vector to a tuple of residues mod p
        parity_rows = []
        for vec in null_space:
            row = []
            for entry in vec:
                val = int(entry) % p
                row.append(val)
            parity_rows.append(tuple(row))
        parity_check = tuple(parity_rows)

    return DualCodeResult(
        field_order=p,
        parity_check_matrix=parity_check,
        code_dimension=k,
        code_length=n,
        dual_dimension=len(parity_check),
    )


def compute_syndrome(request: SyndromeRequest) -> SyndromeResult:
    """Compute the syndrome H * r^T mod p for a received word."""
    p = request.field_order
    h = request.parity_check_matrix
    r = request.received_word
    num_rows = len(h)
    num_cols = len(r)

    syndrome = []
    for i in range(num_rows):
        s = sum(h[i][j] * r[j] for j in range(num_cols)) % p
        syndrome.append(s)

    return SyndromeResult(
        field_order=p,
        syndrome=tuple(syndrome),
    )
