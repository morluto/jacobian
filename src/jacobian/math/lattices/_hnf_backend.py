"""Deterministic SymPy 1.14 modular row HNF with a unimodular lift.

Use the public DomainMatrix FF_dense algorithm, rather than an automatic
matrix-kernel selector, and the public HNF algorithm with an explicit modulus.
Admission is derived from the pinned implementations in
``sympy.polys.matrices.dense.ddm_irref_den`` and
``sympy.polys.matrices.normalforms._hermite_normal_form_modulo_D``.
Their loops and coefficient updates are bounded in ``_hnf_bounds``.
"""

from typing import Any

from jacobian.math.lattices._hnf_bounds import HNFAdmission


def modular_row_hnf(
    entries: list[list[int]], admission: HNFAdmission
) -> tuple[Any, Any]:
    """Compute [H|U] from one elimination and one square modular HNF.

    For G=[A|I], let B consist of the first independent columns of G.
    Fraction-free Gauss-Jordan gives N/den=B^-1 G, with |den|=|det B|;
    the identity suffix guarantees full row rank even for singular A.
    SymPy 1.14 FF_dense performs no primitive-content cancellation, so
    its last pivot denominator is the signed pivot minor, not just a
    common denominator of the rational RREF.

    If K is the row HNF of B, then K*N/den is the row HNF of G: pivot
    columns are K, and each preceding nonpivot column depends only on
    preceding pivots. Its suffix U=K*B^-1 is unimodular. The same product
    establishes both H and U without a second solve or verification pass.
    """
    from flint import fmpz_mat
    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import hermite_normal_form

    rows, columns = admission.rows, admission.columns
    augmented = [
        [ZZ(value) for value in row] + [ZZ(int(i == j)) for j in range(rows)]
        for i, row in enumerate(entries)
    ]
    matrix = DomainMatrix(augmented, (rows, columns + rows), ZZ)
    reduced, denominator, pivots = matrix.rref_den(method="FF_dense")
    numerator = reduced.to_list()

    # SymPy produces upper column HNF. Reverse-transpose before and after
    # to obtain upper row HNF with positive pivots and reduced entries above.
    pivot_transpose = [
        [augmented[rows - 1 - j][pivots[rows - 1 - i]] for j in range(rows)]
        for i in range(rows)
    ]
    column_hnf = hermite_normal_form(
        DomainMatrix(pivot_transpose, (rows, rows), ZZ), D=abs(denominator)
    ).to_list()
    pivot_hnf = [
        [column_hnf[rows - 1 - j][rows - 1 - i] for j in range(rows)]
        for i in range(rows)
    ]
    # Scalar products fix the counted lift path; no automatic matrix product
    # or inverse kernel can expand the admitted computation.
    lifted = [
        [
            ZZ.exquo(
                sum((pivot_hnf[i][k] * numerator[k][j] for k in range(rows)), ZZ.zero),
                denominator,
            )
            for j in range(columns + rows)
        ]
        for i in range(rows)
    ]
    # Preserve the established native matrix return type. FLINT only stores
    # these already computed entries; it runs no matrix algorithm here.
    return (
        fmpz_mat([row[:columns] for row in lifted]),
        fmpz_mat([row[columns:] for row in lifted]),
    )
