"""Exact structural kernels for general quadratic forms."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd

from sympy import Poly, cyclotomic_poly, symbols

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math._exact_linear_algebra import symmetric_inertia
from jacobian.math._labels import MAX_OPAQUE_LABEL_LENGTH
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_SCALAR_DIGITS,
    RationalMatrix,
    RationalVectorSpaceBasis,
    rational_matrix_from_fractions,
    rational_vector_space_basis_from_fractions,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    MAX_QUADRATIC_DIAGONALIZATION_AXIS,
    MAX_QUADRATIC_DIAGONALIZATION_INTERMEDIATE_DIGITS,
    MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS,
    MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_TOTAL_DIGITS,
    MAX_QUADRATIC_DIAGONALIZATION_WORK,
    MAX_QUADRATIC_GAUSS_STATES,
    MAX_QUADRATIC_GAUSS_SUPPORT_TERMS,
    MAX_QUADRATIC_GAUSS_WORK,
    MAX_QUADRATIC_PULLBACK_AXIS,
    MAX_QUADRATIC_PULLBACK_OUTPUT_ENTRIES,
    MAX_QUADRATIC_PULLBACK_WORK,
    FiniteGaussSumRequest,
    FiniteGaussSumResult,
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


def _ceil_log10_positive(value: int) -> int:
    """Return the exact integer ceiling of log10(value), for value >= 1."""
    if value <= 1:
        return 0
    return len(str(value - 1))


def require_diagonalization_budget(form: RationalQuadraticForm) -> None:
    """Admit dense work and a Hadamard bound before rational elimination.

    Clearing all denominators gives an integral matrix C. The stored cross
    coefficients contribute half-coefficients to C's symmetric matrix, so
    their reduced denominators are counted twice. Pair pivots apply the
    integer matrix [[1, 1], [1, -1]]; each transformed entry is a sum of at
    most four entries of C. Every later Schur-complement entry and change
    coefficient is a ratio of minors of this transformed integral matrix.
    Hadamard's bound, enlarged for the unreduced Fraction products in one
    elimination update, therefore bounds every exact intermediate.
    """
    n = len(form.axis)
    if n > MAX_QUADRATIC_DIAGONALIZATION_AXIS:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.diagonalization_axis_bound",
            message="diagonalization dimension exceeds the dense exact envelope",
        )
    work = n * n * n
    if work > MAX_QUADRATIC_DIAGONALIZATION_WORK:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.diagonalization_work_bound",
            message="diagonalization exceeds the admitted exact work envelope",
        )
    if not n:
        return

    # A diagonal presentation needs no elimination: its basis change is I.
    # This admits high-height diagonal forms without charging them for
    # determinant bounds that are irrelevant to the actual kernel.
    if not form.cross_terms:
        return

    coefficient_digits = max(
        (len(str(abs(value.num))) for value in form.diagonal_coefficients),
        default=1,
    )
    coefficient_digits = max(
        coefficient_digits,
        *(len(str(abs(term.coefficient.num))) for term in form.cross_terms),
    )

    # A_ii has denominator d.den; A_ij=c/2 has a reduced denominator that
    # occurs twice, once in each symmetric entry. Their product is a common
    # denominator for the coefficient matrix.
    denominator_exponent = sum(
        _ceil_log10_positive(value.den) for value in form.diagonal_coefficients
    )
    for term in form.cross_terms:
        twice_denominator = 2 * term.coefficient.den
        denominator = twice_denominator // gcd(
            abs(term.coefficient.num), twice_denominator
        )
        denominator_exponent += 2 * _ceil_log10_positive(denominator)
    clearing_digits = denominator_exponent + 1
    integer_entry_digits = coefficient_digits + clearing_digits
    transformed_entry_digits = integer_entry_digits + 2
    # Hadamard: |det(M_k)| <= k! * max(|M_ij|)^k <=
    # 10^(n*entry_digits + n*ceil(log10(n))).
    minor_digits = n * transformed_entry_digits + n * _ceil_log10_positive(n) + 1
    diagonal_digits = minor_digits + clearing_digits + 1
    total_output_digits = n * n * 2 * minor_digits + n * 2 * diagonal_digits
    # Fraction multiplication and subtraction in each Schur update use at
    # most a fixed number of products of these bounded minor ratios.
    intermediate_digits = 8 * (minor_digits + clearing_digits) + 64
    if (
        minor_digits > MAX_MATRIX_SCALAR_DIGITS
        or minor_digits > MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS
        or diagonal_digits > MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_DIGITS
        or total_output_digits > MAX_QUADRATIC_DIAGONALIZATION_OUTPUT_TOTAL_DIGITS
        or intermediate_digits > MAX_QUADRATIC_DIAGONALIZATION_INTERMEDIATE_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.diagonalization_coefficient_growth",
            message=(
                "diagonalization's exact minor bound exceeds the admitted "
                "intermediate or result envelope"
            ),
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
        for r in range(n):
            left, right = p[r][index], p[r][index + 1]
            p[r][index] = left + right
            p[r][index + 1] = left - right
        # T differs from I only in this 2x2 block. Apply the right
        # multiplication (column combinations), then T.T on the left
        # (matching row combinations), in O(n^2) rather than O(n^4).
        for r in range(n):
            left, right = a[r][index], a[r][index + 1]
            a[r][index] = left + right
            a[r][index + 1] = left - right
        for r in range(n):
            top, bottom = a[index][r], a[index + 1][r]
            a[index][r] = top + bottom
            a[index + 1][r] = top - bottom

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


def finite_quadratic_gauss_sum(request: FiniteGaussSumRequest) -> FiniteGaussSumResult:
    """Compute sum_x zeta_m^Q(x), reducing the full histogram in QQ[zeta_m]."""

    form, modulus = request.form, request.modulus
    if any(c.den != 1 for c in form.diagonal_coefficients) or any(
        term.coefficient.den != 1 for term in form.cross_terms
    ):
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.integral_required",
            message="finite Gauss sums require integral coefficients",
        )

    total = modulus ** len(form.axis)
    if total > MAX_QUADRATIC_GAUSS_STATES:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.gauss_sum.state_bound",
            message="complete residue domain exceeds the finite Gauss state bound",
        )

    # Every enumerated residue vector evaluates each stored polynomial term
    # once, and the result retains the complete source form alongside the
    # histogram and cyclotomic coordinates. Bound the support first so both
    # the kernel traversal and the retained source stay inside the envelope,
    # then bound their product as work before any enumeration runs.
    support = len(form.axis) + len(form.cross_terms)
    if support > MAX_QUADRATIC_GAUSS_SUPPORT_TERMS:
        raise OperationResourceAdmissionError(
            location=("form",),
            code="quadratic_form.gauss_sum.support_bound",
            message=(
                "finite Gauss form support exceeds the admitted polynomial "
                "support envelope"
            ),
        )
    if total * max(support, 1) > MAX_QUADRATIC_GAUSS_WORK:
        raise OperationResourceAdmissionError(
            location=("form", "axis"),
            code="quadratic_form.gauss_sum.work_bound",
            message="complete residue enumeration exceeds the finite Gauss work bound",
        )

    # The public modulus cap bounds both cyclotomic construction and the
    # power-basis reduction. Establish coefficient growth before enumerating.
    variable = symbols("x")
    defining = tuple(
        int(value)
        for value in Poly(cyclotomic_poly(modulus, variable), variable).all_coeffs()
    )
    degree = len(defining) - 1
    if defining[0] != 1 or degree < 1 or degree > modulus:
        raise RuntimeError("cyclotomic polynomial has an unexpected canonical shape")
    coefficient_l1 = sum(abs(value) for value in defining)
    if total * max(1, coefficient_l1) ** modulus >= 10**MAX_CYCLIC_FIELD_ELEMENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("modulus",),
            code="quadratic_form.gauss_sum.coefficient_bound",
            message="cyclotomic coefficient growth exceeds the exact output bound",
        )

    histogram, enumerated = modular_histogram(form, modulus)
    if enumerated != total:
        raise RuntimeError(
            "modular profile did not account for the complete residue domain"
        )

    # Convert Phi_m to ascending order and reduce sum_r h_r*x^r modulo Phi_m.
    phi_ascending = tuple(reversed(defining))
    reduced = list(histogram)
    for power in range(modulus - 1, degree - 1, -1):
        coefficient = reduced[power]
        if coefficient:
            reduced[power] = 0
            for lower_power in range(degree):
                reduced[power - degree + lower_power] -= (
                    coefficient * phi_ascending[lower_power]
                )
    from jacobian._exact import CanonicalRational

    element = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=modulus),
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(value))
            for value in reduced[:degree]
        ),
    )
    return FiniteGaussSumResult(
        form=form,
        modulus=modulus,
        histogram=histogram,
        total=total,
        value=element,
    )


__all__ = [
    "finite_quadratic_gauss_sum",
    "modular_histogram",
    "quadratic_diagonalization",
    "quadratic_pullback",
    "quadratic_radical",
    "quadratic_signature",
    "require_diagonalization_budget",
    "require_pullback_budget",
]
