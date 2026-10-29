"""Denominator-aware exact growth bounds for the Cayley-circle kernel."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import factorial, lcm, prod

from jacobian.canonical import decimal_digit_width
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS,
    MAX_SUP_NORM_INPUT_COMPONENT_DIGITS,
    MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS,
    MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES,
)


@dataclass(frozen=True)
class SupNormBounds:
    transform_digits: int
    numerator_digits: int
    derivative_digits: int
    resultant_digits: int
    factor_digits: int
    parameter_digits: int
    certificate_scalar_digits: int
    transform_work: int
    isolation_work: int
    resultant_work: int
    factor_work: int
    value_analysis_work: int
    postprocessing_work: int


def sup_norm_bounds(coefficients: list[tuple[Fraction, Fraction]]) -> SupNormBounds:
    """Bound expansion, elimination, factorization and retained scalar growth.

    With L the LCM of scalar denominators, M=L² and
    T=2**d sum |p_j|L/q_j, the integer numerator of Q has l1 norm <= K=T².
    Before diagonal cancellation the derivative has norm <= 6dK over M.

    The diagonal a_j²,b_j² terms in |P|² contribute multiples of (1+t²)^d,
    annihilated by Q'(1+t²)-2dtQ. Every surviving scalar product uses two
    distinct denominator positions. Thus S=product q_j clears the final
    derivative, with integer l1 norm <= 12d4**d sum_(j<k)|p_jp_k|S/(q_jq_k).
    Repeated equal denominators still occupy distinct positions in S.

    Dividing by monic (1+t²) at most d times costs an additional (2d+1)**d
    l1 allowance. A Sylvester determinant has size <= 4d, and each entry's
    y-l1 norm is bounded by A or B=M2**d+K. Hence its integer coefficient
    height is <= (4d)! A**(2d) B**(2d). Rational scaling has denominator at
    most S**(2d) M**(2d). Mignotte's factor bound contributes 2**(2d)(2d+1).
    These scalar bounds are calculated before polynomial expansion.

    Phase charges use degree/limb weights, not elapsed-time predictions.
    Value analysis includes resultant-root isolation, factor/Sturm evaluation,
    rational-level gcds and all candidate comparisons, in addition to the
    separate factorization charge. Its precision weight includes the
    degree-times-height root-separation bound. Postprocessing reserves three
    refinement targets per critical root, rational interval Horner/division,
    circle images, outward grids, square roots and canonical construction.
    """
    degree = len(coefficients) - 1
    scalars = tuple(value for pair in coefficients for value in pair)
    denominator_lcm = lcm(*(value.denominator for value in scalars))
    denominator_product = prod(value.denominator for value in scalars)
    transformed_norm = (1 << degree) * sum(
        abs(value.numerator) * (denominator_lcm // value.denominator)
        for value in scalars
    )
    numerator_norm = transformed_norm**2
    numerator_denominator = denominator_lcm**2
    off_diagonal = sum(
        abs(left.numerator * right.numerator)
        * (denominator_product // left.denominator // right.denominator)
        for left, right in combinations(scalars, 2)
    )
    derivative_norm = 12 * degree * (4**degree) * off_diagonal
    numerator_digits = max(
        decimal_digit_width(numerator_norm),
        decimal_digit_width(numerator_denominator),
    )
    transform_digits = max(
        numerator_digits, decimal_digit_width(max(1, 6 * degree) * numerator_norm)
    )
    derivative_digits = max(
        decimal_digit_width(derivative_norm),
        decimal_digit_width(denominator_product),
    )
    derivative_degree = 2 * degree
    matrix_size = 4 * degree
    coprime_norm = max(1, derivative_norm) * (2 * degree + 1) ** degree
    level_norm = (1 << degree) * numerator_denominator + numerator_norm
    resultant_norm = (
        factorial(matrix_size)
        * coprime_norm**derivative_degree
        * max(1, level_norm) ** derivative_degree
    )
    resultant_denominator = (
        denominator_product * numerator_denominator
    ) ** derivative_degree
    resultant_digits = max(
        decimal_digit_width(resultant_norm),
        decimal_digit_width(resultant_denominator),
    )
    factor_digits = decimal_digit_width(
        resultant_norm * (1 << derivative_degree) * (derivative_degree + 1)
    )
    # The primitive squarefree part of the entire derivative has degree <= 16
    # and coefficient l1 norm < 10**647 by Mignotte. Cauchy gives
    # |root| < 10**648, and Mignotte separation gives sep > 10**-9716.
    # Oversized backend endpoints are refined and rounded outward on a
    # 10000-place grid: width <= 3e-10000 preserves the unique root and
    # endpoint components need fewer than 10650 digits.
    # Already smaller encodings are retained; this explicit owner bound does
    # not assume an undocumented VAS endpoint-encoding guarantee.
    parameter_digits = MAX_SUP_NORM_PARAMETER_COMPONENT_DIGITS
    singleton_value_digits = (
        numerator_digits + 2 * degree * derivative_digits + 2 * degree + 2
    )
    image_digits = max(
        MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS, 2 * derivative_digits + 4
    )
    # Nonzero exact value singletons arise from rational parameters; the
    # separately identified zero level needs only one digit.
    value_digits = max(MAX_SUP_NORM_ENCLOSURE_COMPONENT_DIGITS, singleton_value_digits)
    critical_count = 2 * degree + 1
    certificate_digits = (
        2 * (2 * degree + 1) * numerator_digits
        + 2 * (2 * degree + 2) * derivative_digits
        + critical_count * (4 * parameter_digits + 4 * value_digits + 8 * image_digits)
        + 8 * value_digits
        + 2 * numerator_digits
        + 4 * (degree + 1) * MAX_SUP_NORM_INPUT_COMPONENT_DIGITS
        + (2 * degree + 1) * 1_000
        + 64 * critical_count
    )
    resultant_limbs = (4 * resultant_digits + 63) // 64
    factor_limbs = (4 * factor_digits + 63) // 64
    value_isolation_digits = (derivative_degree + 1) * (
        max(resultant_digits, factor_digits) + derivative_degree + 4
    )
    value_isolation_limbs = (4 * value_isolation_digits + 63) // 64
    # An interval Horner path multiplies at most 2d parameter endpoints; its
    # rational division by (1+t²)^d doubles that parameter-height allowance.
    # Shared Q denominators divide L² throughout, so no coefficient-count
    # multiplication of the numerator denominator height is needed.
    interval_scratch_digits = (
        numerator_digits
        + 2 * derivative_degree * parameter_digits
        + 2 * derivative_degree
        + 8
    )
    interval_limbs = (4 * interval_scratch_digits + 63) // 64
    return SupNormBounds(
        transform_digits=transform_digits,
        numerator_digits=numerator_digits,
        derivative_digits=derivative_digits,
        resultant_digits=resultant_digits,
        factor_digits=factor_digits,
        parameter_digits=parameter_digits,
        certificate_scalar_digits=certificate_digits,
        transform_work=(2 * degree + 1) ** 3 * (transform_digits + 64),
        # Reserve initial isolation and all per-root precision continuations,
        # including coefficient height, root magnitude and target precision,
        # even when normal endpoints need no compaction.
        isolation_work=2
        * (2 * degree + 2) ** 3
        * (2 * derivative_digits + MAX_SUP_NORM_PARAMETER_DECIMAL_PLACES + 64),
        resultant_work=(matrix_size + 1) ** 3
        * (derivative_degree + 1) ** 2
        * resultant_limbs**2,
        factor_work=(derivative_degree + 1) ** 6 * factor_limbs**3,
        value_analysis_work=(derivative_degree + 1) ** 4 * value_isolation_limbs**2
        + (derivative_degree + 1) ** 6 * factor_limbs**3,
        postprocessing_work=64 * (derivative_degree + 1) ** 4 * interval_limbs**2,
    )
