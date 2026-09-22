"""Exact structural kernels for general quadratic forms."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH
from jacobian.math.matrices.values import (
    RationalMatrix,
    RationalVectorSpaceBasis,
    rational_matrix_from_fractions,
    rational_vector_space_basis_from_fractions,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_QUADRATIC_PULLBACK_AXIS,
    MAX_QUADRATIC_PULLBACK_OUTPUT_ENTRIES,
    MAX_QUADRATIC_PULLBACK_WORK,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    coefficient_matrix_entries,
    require_coefficient_matrix_budget,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    RationalQuadraticForm,
)


def _matrix(form: RationalQuadraticForm) -> tuple[tuple[Fraction, ...], ...]:
    return coefficient_matrix_entries(form)


def quadratic_signature(form: RationalQuadraticForm) -> tuple[int, int, int]:
    require_coefficient_matrix_budget(form)
    try:
        return symmetric_inertia(_matrix(form))
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("form",), code="quadratic_form.signature_domain", message=str(exc)
        ) from exc


def quadratic_radical(
    form: RationalQuadraticForm,
) -> tuple[int, RationalVectorSpaceBasis]:
    matrix = _matrix(form)
    n = len(matrix)
    try:
        import sympy

        null = sympy.Matrix(
            [
                [sympy.Rational(x.numerator, x.denominator) for x in row]
                for row in matrix
            ]
        ).nullspace()
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("form",), code="quadratic_form.radical_domain", message=str(exc)
        ) from exc
    vectors = tuple(tuple(Fraction(value) for value in vector) for vector in null)
    rank = n - len(vectors)
    return rank, rational_vector_space_basis_from_fractions(
        vectors, ambient_dimension=n
    )


def require_pullback_budget(
    form: RationalQuadraticForm,
    matrix: RationalMatrix,
    target_axis: tuple[str, ...],
) -> None:
    """Admit source, target, work, and dense output before pullback allocation."""

    require_coefficient_matrix_budget(form)
    source_dimension = len(form.axis)
    target_dimension = len(target_axis)
    if matrix.row_count != source_dimension or matrix.column_count != target_dimension:
        raise OperationDomainValidationError(
            location=("matrix",),
            code="quadratic_form.pullback_shape",
            message="pullback matrix dimensions must map target to source",
        )
    if target_dimension > MAX_QUADRATIC_PULLBACK_AXIS:
        raise OperationResourceAdmissionError(
            location=("target_axis",),
            code="quadratic_form.pullback_axis_bound",
            message="pullback target axis exceeds the owner envelope",
        )
    if (
        any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > MAX_OPAQUE_LABEL_LENGTH
            for label in target_axis
        )
        or len(set(target_axis)) != target_dimension
    ):
        raise OperationDomainValidationError(
            location=("target_axis",),
            code="quadratic_form.pullback_axis_labels",
            message="pullback target labels must be bounded, nonempty, trimmed, and unique",
        )
    work = source_dimension * source_dimension * target_dimension * target_dimension
    if work > MAX_QUADRATIC_PULLBACK_WORK:
        raise OperationResourceAdmissionError(
            location=("matrix", "entries"),
            code="quadratic_form.pullback_work_bound",
            message="pullback arithmetic exceeds the owner work envelope",
        )
    output_entries = target_dimension * target_dimension
    if output_entries > MAX_QUADRATIC_PULLBACK_OUTPUT_ENTRIES:
        raise OperationResourceAdmissionError(
            location=("target_axis",),
            code="quadratic_form.pullback_output_bound",
            message="pullback output exceeds the owner envelope",
        )
    _admit_pullback_coefficient_growth(form, matrix)


def _factor_growth(value: int) -> int:
    """Return a safe decimal-product growth charge; units cost nothing."""

    return 0 if abs(value) <= 1 else len(str(abs(value)))


def _rational_product_bound(value: Fraction) -> tuple[int, int]:
    return _factor_growth(value.numerator), _factor_growth(value.denominator)


def _admit_pullback_coefficient_growth(
    form: RationalQuadraticForm, matrix: RationalMatrix
) -> None:
    """Bound every dense M^T A M coefficient before Fraction products."""

    source = _matrix(form)
    source_bounds = tuple(
        _rational_product_bound(value) for row in source for value in row
    )
    matrix_bounds = tuple(
        _rational_product_bound(value.as_fraction())
        for row in matrix.entries
        for value in row
    )
    if not source_bounds or not matrix_bounds:
        return
    max_source_numerator = max(bound[0] for bound in source_bounds)
    max_source_denominator = max(bound[1] for bound in source_bounds)
    max_matrix_numerator = max(bound[0] for bound in matrix_bounds)
    max_matrix_denominator = max(bound[1] for bound in matrix_bounds)
    term_count = len(form.axis) ** 2
    term_numerator = max_source_numerator + 2 * max_matrix_numerator
    term_denominator = max_source_denominator + 2 * max_matrix_denominator
    denominator_growth = term_count * term_denominator
    numerator_growth = (
        term_numerator
        + (term_count - 1) * term_denominator
        + len(str(max(term_count, 1)))
        + 1
    )
    component_digits = 1 + max(denominator_growth, numerator_growth)
    if component_digits > MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("matrix", "entries"),
            code="quadratic_form.pullback_coefficient_growth",
            message=(
                "pullback coefficients exceed the rational quadratic-form output "
                "envelope"
            ),
        )


def quadratic_pullback(
    form: RationalQuadraticForm,
    matrix: RationalMatrix,
    target_axis: tuple[str, ...],
) -> RationalQuadraticForm:
    require_pullback_budget(form, matrix, target_axis)
    a = _matrix(form)
    m = tuple(tuple(x.as_fraction() for x in row) for row in matrix.entries)
    n = len(target_axis)
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(
                sum(
                    (
                        m[k][i] * a[k][ell] * m[ell][j]
                        for k in range(len(a))
                        for ell in range(len(a))
                    ),
                    Fraction(0),
                )
            )
        out.append(row)
    diag = tuple(CanonicalRational.from_fraction(out[i][i]) for i in range(n))
    crosses = []
    from jacobian.math.number_theory.quadratic_forms.general.values import (
        QuadraticCrossTerm,
    )

    for i in range(n):
        for j in range(i + 1, n):
            value = out[i][j] + out[j][i]
            if value:
                crosses.append(
                    QuadraticCrossTerm(
                        left=i,
                        right=j,
                        coefficient=CanonicalRational.from_fraction(value),
                    )
                )
    return RationalQuadraticForm(
        axis=tuple(target_axis), diagonal_coefficients=diag, cross_terms=tuple(crosses)
    )


def quadratic_diagonalization(  # noqa: C901
    form: RationalQuadraticForm,
) -> tuple[tuple[Fraction, ...], RationalMatrix]:
    """Congruence diagonalization with an exact tracked change of basis.

    The working matrix is always ``P.T * A * P``.  In particular, a pair
    pivot is first changed to two nonzero diagonal pivots and then passed
    through the same elimination loop as an ordinary pivot.  Skipping that
    second phase leaves cross-terms with the untouched coordinates.
    """
    a = [list(row) for row in _matrix(form)]
    n = len(a)
    p = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]

    def swap_axis(left: int, right: int) -> None:
        if left == right:
            return
        a[left], a[right] = a[right], a[left]
        for row in a:
            row[left], row[right] = row[right], row[left]
        for row in p:
            row[left], row[right] = row[right], row[left]

    def pair_change(index: int) -> None:
        """Apply P <- P*T and A <- T.T*A*T for T=[[1,1],[1,-1]]."""
        old_a = [row[:] for row in a]
        old_p = [row[:] for row in p]
        for r in range(n):
            p[r][index] = old_p[r][index] + old_p[r][index + 1]
            p[r][index + 1] = old_p[r][index] - old_p[r][index + 1]
        # Build T explicitly; this keeps the congruence relation obvious and
        # handles pair/untouched cross terms as well as the pair block.
        transformed = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        t = [[Fraction(int(r == c)) for c in range(n)] for r in range(n)]
        t[index][index] = t[index + 1][index] = Fraction(1)
        t[index][index + 1] = Fraction(1)
        t[index + 1][index + 1] = Fraction(-1)
        for r in range(n):
            for c in range(n):
                transformed[r][c] = sum(
                    (
                        t[u][r] * old_a[u][v] * t[v][c]
                        for u in range(n)
                        for v in range(n)
                    ),
                    Fraction(),
                )
        for r in range(n):
            a[r][:] = transformed[r]

    k = 0
    diagonal: list[Fraction] = []
    while k < n:
        pivot = next((i for i in range(k, n) if a[i][i]), None)
        if pivot is None:
            pair = next(
                ((i, j) for i in range(k, n) for j in range(i + 1, n) if a[i][j]),
                None,
            )
            if pair is None:
                diagonal.extend(Fraction(0) for _ in range(k, n))
                break
            i, j = pair
            swap_axis(i, k)
            if j == k:
                j = i
            swap_axis(j, k + 1)
            pair_change(k)
            # The pair now has nonzero diagonal entries.  Do not skip it:
            # ordinary pivots below eliminate all remaining cross-terms.
            continue
        if pivot != k:
            swap_axis(k, pivot)
        q = a[k][k]
        diagonal.append(q)
        # C_j <- C_j - (A_kj/q) C_k, together with the matching row
        # operation.  This is a symmetric congruence, not just a column edit.
        for j in range(k + 1, n):
            factor = a[k][j] / q
            for r in range(n):
                p[r][j] -= factor * p[r][k]
            for r in range(n):
                a[r][j] -= factor * a[r][k]
            for c in range(n):
                a[j][c] -= factor * a[k][c]
        k += 1
    if len(diagonal) < n:
        diagonal.extend(Fraction(0) for _ in range(n - len(diagonal)))
    if any(a[i][j] for i in range(n) for j in range(n) if i != j):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.diagonalization_failed",
            message="exact congruence elimination did not produce a diagonal matrix",
        )
    return tuple(diagonal), rational_matrix_from_fractions(
        tuple(tuple(row) for row in p)
    )


def modular_histogram(
    form: RationalQuadraticForm, modulus: int
) -> tuple[tuple[int, ...], int]:
    if any(c.den != 1 for c in form.diagonal_coefficients) or any(
        t.coefficient.den != 1 for t in form.cross_terms
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.integral_required",
            message="modular profiles require integral coefficients",
        )
    states = modulus ** len(form.axis)
    if states > 2_000_000:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="quadratic_form.modular_bound",
            message="residue domain exceeds admitted bound",
        )
    hist = [0] * modulus
    diag = tuple(c.num for c in form.diagonal_coefficients)
    for point in product(range(modulus), repeat=len(form.axis)):
        value = sum(diag[i] * point[i] * point[i] for i in range(len(point)))
        value += sum(
            t.coefficient.num * point[t.left] * point[t.right] for t in form.cross_terms
        )
        hist[value % modulus] += 1
    return tuple(hist), states


__all__ = [
    "modular_histogram",
    "quadratic_diagonalization",
    "quadratic_pullback",
    "quadratic_radical",
    "quadratic_signature",
    "require_pullback_budget",
]
