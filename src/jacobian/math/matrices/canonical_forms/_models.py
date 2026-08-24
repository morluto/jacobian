"""Typed wire contracts for exact canonical-form operations over QQ."""

from __future__ import annotations

from collections.abc import Iterable
from math import gcd
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    RationalMatrix,
    require_matrix_scalar_digits,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_CANONICAL_FORM_DIMENSION = 16
MAX_CANONICAL_FORM_SCALAR_DIGITS = 256

# A Horner pass performs ``degree`` dense n-by-n products. Ordinary execution
# performs one producer pass and result construction performs one independent
# source-bound replay. This limit therefore admits at most 2,000,000 scalar
# product terms per pass and bounds every accepted exact call before SymPy.
MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS = 4_000_000
MATRIX_POLYNOMIAL_EVALUATION_PASSES = 2
# Couple exact operation count to the largest admitted rational component.
# Reference cases, including result replay: a dense 32x32 degree-61 request
# costs about 33 billion proxy units and 1.5 seconds; a 1x1 degree-327 request
# with a 32,701-digit denominator costs about 700 billion units and one second.
MAX_MATRIX_POLYNOMIAL_DIGIT_WORK = 1_000_000_000_000

# Admission may materialize bounded powers to prove shared numerator factors
# before digit budgets are charged. Eight bits per canonical digit sits far
# above log2(10), so the estimate never undercounts a power below the ceiling.
_MAX_ADMISSION_POWER_BITS = 8 * MAX_CANONICAL_RATIONAL_DIGITS
# Total estimated bigint word-work one admission may spend on materialized
# powers and their reductions; exhaustion falls back to the dense capped
# bound. Admits several full-width cancellations per call while bounding
# repeated adversarial reduction work to milliseconds.
_ADMISSION_REDUCTION_BIT_BUDGET = 64 * _MAX_ADMISSION_POWER_BITS
# Conservative ceiling on how many full clearing-denominator powers one
# admission peels out of a proven numerator factor. Realistic cancellations
# need a handful of peels; exhaustion merely keeps the uncancelled (larger,
# still safe) denominator bound.
_MAX_PROVEN_DENOMINATOR_PEELS = 64

_RESULT_ENTRY_OVERHEAD_BYTES: int = 32
_RESULT_ENVELOPE_RESERVE_BYTES: int = 4_096
_MAX_RESULT_COMPONENT: int = 10**MAX_CANONICAL_RATIONAL_DIGITS - 1


# Work proxies may legitimately exceed one canonical component: the coupled
# digit-work bound admits intermediate digits up to the square root of the
# work limit divided by the smallest product count. Saturating work
# arithmetic at the component cap would clip exactly those honest charges,
# so work bounds saturate at a ceiling whose digits alone force the
# digit-work rejection for any admissible product count.
_MAX_WORK_BOUND = _MAX_RESULT_COMPONENT**32


def _capped_multiply(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    if left > _MAX_RESULT_COMPONENT // right:
        return _MAX_RESULT_COMPONENT + 1
    return left * right


def _capped_add(left: int, right: int) -> int:
    if left > _MAX_RESULT_COMPONENT - right:
        return _MAX_RESULT_COMPONENT + 1
    return left + right


def _work_multiply(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    # ``left * right >= 2 ** (bits(left) + bits(right) - 2)``, so operand
    # lengths at or above the ceiling's own length force saturation without
    # any arithmetic. Otherwise the product stays small enough to
    # materialize, and comparing it against the ceiling directly replaces
    # an expensive wide division while remaining exact.
    if left.bit_length() + right.bit_length() - 2 >= _MAX_WORK_BOUND.bit_length():
        return _MAX_WORK_BOUND + 1
    product = left * right
    if product > _MAX_WORK_BOUND:
        return _MAX_WORK_BOUND + 1
    return product


def _work_add(left: int, right: int) -> int:
    if left > _MAX_WORK_BOUND - right:
        return _MAX_WORK_BOUND + 1
    return left + right


def _work_power(base: int, exponent: int) -> int:
    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _work_multiply(result, factor)
            if result > _MAX_WORK_BOUND:
                return result
        remaining >>= 1
        if remaining:
            factor = _work_multiply(factor, factor)
            if factor > _MAX_WORK_BOUND:
                return factor
    return result


def _capped_power(base: int, exponent: int) -> int:
    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _capped_multiply(result, factor)
            if result > _MAX_RESULT_COMPONENT:
                return result
        remaining >>= 1
        if remaining:
            factor = _capped_multiply(factor, factor)
            if factor > _MAX_RESULT_COMPONENT:
                return factor
    return result


def _capped_lcm(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        factor = value // gcd(result, value)
        result = _capped_multiply(result, factor)
        if result > _MAX_RESULT_COMPONENT:
            return result
    return result


def _work_lcm(values: Iterable[int]) -> int:
    """Fold denominators into one common multiple under work saturation.

    Unlike ``_capped_lcm`` this never clips at a canonical component: the
    result is a work estimate whose digit width feeds the coupled digit-work
    bound, so honest compounding past one component must stay visible.
    """

    result = 1
    for value in values:
        factor = value // gcd(result, value)
        if factor != 1:
            result = _work_multiply(result, factor)
            if result > _MAX_WORK_BOUND:
                return result
    return result


def _decimal_digit_upper_bound(value: int) -> int:
    """Return a conservative decimal digit count without converting huge ints."""

    if value == 0:
        return 1
    # 0.30103 is a strict upper bound for log10(2). Using integer arithmetic
    # keeps this bound deterministic and at most one digit above the truth.
    return (value.bit_length() * 30_103 + 99_999) // 100_000


def _polynomial_degree(polynomial: RationalPolynomial) -> int:
    return max(
        (term.exponents[0] for term in polynomial.polynomial.terms),
        default=0,
    )


def _mathematical_polynomial_degree(
    polynomial: RationalPolynomial,
) -> int | None:
    if not polynomial.polynomial.terms:
        return None
    return polynomial.polynomial.terms[0].exponents[0]


def _coefficient_ratios(
    polynomial: RationalPolynomial,
) -> dict[int, tuple[int, int]]:
    return {
        term.exponents[0]: term.coefficient.as_integer_ratio()
        for term in polynomial.polynomial.terms
    }


def _bounded_rational_sum(terms: Iterable[tuple[int, int]]) -> tuple[int, int]:
    """Add signed rationals exactly and return the fully reduced sum.

    Operand signs are preserved so additive cancellation between summands
    reduces before any component bound is enforced; callers digit-bound the
    reduced result afterwards. Summands carry canonical components and the
    callers here add at most two rationals per entry, so the exact pairwise
    accumulation stays far below any resource limit.
    """

    total_numerator = 0
    total_denominator = 1
    for numerator, denominator in terms:
        if numerator == 0:
            continue
        common = gcd(total_denominator, denominator)
        total_numerator = total_numerator * (denominator // common) + numerator * (
            total_denominator // common
        )
        total_denominator *= denominator // common
    divisor = gcd(total_numerator, total_denominator)
    return total_numerator // divisor, total_denominator // divisor


def _materialized_power(base: int, exponent: int) -> tuple[int | None, int]:
    """Return ``base ** exponent`` and its estimated bit size.

    The pair ``(None, bits)`` reports a power whose estimated size exceeds
    ``_MAX_ADMISSION_POWER_BITS``; callers fall back to capped arithmetic.
    """

    if exponent == 0 or base == 1:
        return 1, 0
    bits = exponent * base.bit_length()
    if bits > _MAX_ADMISSION_POWER_BITS:
        return None, bits
    return base**exponent, bits


def _proven_numerator_factor(
    surviving_terms: tuple[tuple[int, int, int], ...],
    common_coefficient_denominator: int,
) -> int:
    """Return a factor proven to divide every cleared result numerator.

    With ``D`` the highest surviving exponent and ``V`` the common
    coefficient denominator, entry ``(i, j)`` of ``f(A)`` equals

        ``N / (V Q^D)`` with ``N = sum_k w_k Q^(D-k) M^k[i,j]``,

    where ``w_k = |c_k| (V // den c_k)`` is the exact lifted coefficient.
    Every term of ``N`` is a multiple of its own ``w_k``, so the gcd of the
    surviving ``w_k`` values divides every contributing numerator. Only such
    proven factors may later be cancelled; matrix-side quantities such as the
    cleared height ``h`` or the dimension count ``n^(k-1)`` bound entry
    magnitudes but divide no numerator, so they are never cancelled.

    The gcd chain is charged against the reduction budget; exhaustion returns
    1, which cancels nothing and keeps every downstream bound valid.
    """

    factor = 0
    budget = _ADMISSION_REDUCTION_BIT_BUDGET
    for _exponent, numerator, denominator in surviving_terms:
        lift = abs(numerator) * (common_coefficient_denominator // denominator)
        budget -= min(factor.bit_length(), lift.bit_length()) ** 2 // 65_536
        if budget < 0:
            return 1
        factor = gcd(factor, lift)
    return factor


def _reduced_general_result_bound(
    surviving_terms: tuple[tuple[int, int, int], ...],
    cleared_matrix_height: int,
    common_matrix_denominator: int,
    common_coefficient_denominator: int,
    dimension: int,
    highest_surviving_exponent: int,
    proven_numerator_factor: int,
) -> int | None:
    """Bound reduced result numerators via the exact proven cancellation.

    Every surviving term contributes at most
    ``w_k n^(k-1) h^k Q^(D-k)`` to the cleared integer numerator, so the
    exact integer sum ``S`` of those products bounds ``|N|`` from above.
    Dividing by the proven ``(scale part, peeled powers)`` of
    ``_proven_cancellation`` -- which divides every numerator and the whole
    cleared denominator -- leaves ``floor(S / scale / Q^peeled)`` as a valid
    bound on every reduced result numerator. Each summand is computed
    exactly from powers that materialize below the admission ceiling, so
    genuine cancellation between lifted coefficients and compounded
    denominators survives without ever cancelling an upper-bound factor such
    as the height maximum.

    Returns ``None`` when there is no proven cancellation to apply, when a
    required power exceeds the materialization ceiling, or when the reduction
    budget is exhausted; callers then keep the dense capped numerator bound,
    which rejects oversized growth outright.
    """

    if proven_numerator_factor <= 1:
        return None
    total = 0
    budget = _ADMISSION_REDUCTION_BIT_BUDGET
    for exponent, numerator, denominator in surviving_terms:
        lift = abs(numerator) * (common_coefficient_denominator // denominator)
        dimension_power, dimension_bits = (
            (1, 0) if exponent <= 1 else _materialized_power(dimension, exponent - 1)
        )
        height_power, height_bits = (
            (1, 0)
            if not exponent
            else _materialized_power(cleared_matrix_height, exponent)
        )
        clearing_power, clearing_bits = (
            (1, 0)
            if exponent == highest_surviving_exponent
            else _materialized_power(
                common_matrix_denominator,
                highest_surviving_exponent - exponent,
            )
        )
        if dimension_power is None or height_power is None or clearing_power is None:
            return None
        budget -= dimension_bits + height_bits + clearing_bits
        term = lift * dimension_power * height_power * clearing_power
        # A schoolbook product of this many bits costs on the order of
        # (bits / 64)^2 word operations; charging a 256th of that square
        # keeps the budget an upper estimate of one exact term reduction.
        budget -= term.bit_length() ** 2 // 65_536
        if budget < 0:
            return None
        total += term
    scale_part, peeled = _proven_cancellation(
        proven_numerator_factor,
        common_coefficient_denominator,
        common_matrix_denominator,
        highest_surviving_exponent,
    )
    total //= scale_part
    for _ in range(peeled):
        total //= common_matrix_denominator
    return total


def _proven_cancellation(
    proven_numerator_factor: int,
    common_coefficient_denominator: int,
    common_matrix_denominator: int,
    highest_surviving_exponent: int,
) -> tuple[int, int]:
    """Return the ``(scale part, peeled powers)`` proven against both sides.

    A cleared result entry equals ``N / (V Q^D)``. Reducing that fraction can
    only cancel divisor parts of ``N`` that also divide ``V Q^D``: whole
    factors shared between the proven numerator factor and ``V``, plus whole
    powers of ``Q`` peeled while they divide the remaining factor exactly.
    Any partial remainder stays uncancelled, so every later bound derived
    from this split never undercuts the true reduced components even when
    ``V Q^D`` itself is far too large to materialize.
    """

    scale_part = gcd(proven_numerator_factor, common_coefficient_denominator)
    remaining = proven_numerator_factor // scale_part
    peeled = 0
    while (
        peeled < highest_surviving_exponent
        and peeled < _MAX_PROVEN_DENOMINATOR_PEELS
        and common_matrix_denominator > 1
        and remaining % common_matrix_denominator == 0
    ):
        remaining //= common_matrix_denominator
        peeled += 1
    return scale_part, peeled


def _proven_denominator_bound(
    common_matrix_denominator: int,
    common_coefficient_denominator: int,
    highest_surviving_exponent: int,
    proven_numerator_factor: int,
) -> int:
    """Bound the reduced result denominator using proven cancellations only.

    The returned ``(V // scale_part) Q^(D - peeled)`` keeps every unproven
    factor of ``V Q^D`` in place, so it remains a valid upper bound on the
    true reduced denominator of every result entry.
    """

    scale_part, peeled = _proven_cancellation(
        proven_numerator_factor,
        common_coefficient_denominator,
        common_matrix_denominator,
        highest_surviving_exponent,
    )
    return _capped_multiply(
        common_coefficient_denominator // scale_part,
        _capped_power(common_matrix_denominator, highest_surviving_exponent - peeled),
    )


def _acyclic_support_longest_walk(
    ratios: tuple[tuple[int, int], ...],
    dimension: int,
) -> int | None:
    """Return the longest walk length in the entry-support digraph of ``M``.

    Entry ``(i, j)`` of ``M^k`` is nonzero only when the digraph of nonzero
    matrix entries carries a walk of length ``k`` from ``i`` to ``j``, so
    every power longer than the returned bound is identically zero. A support
    cycle keeps arbitrarily long walks alive and yields ``None``.
    """

    indegree = [0] * dimension
    longest = [0] * dimension
    for row_index in range(dimension):
        row_offset = row_index * dimension
        for column_index in range(dimension):
            if ratios[row_offset + column_index][0] != 0:
                indegree[column_index] += 1
    queue = [node for node in range(dimension) if indegree[node] == 0]
    processed = 0
    while processed < len(queue):
        node = queue[processed]
        processed += 1
        row_offset = node * dimension
        for column_index in range(dimension):
            if ratios[row_offset + column_index][0] != 0:
                longest[column_index] = max(longest[column_index], longest[node] + 1)
                indegree[column_index] -= 1
                if indegree[column_index] == 0:
                    queue.append(column_index)
    if processed < dimension:
        return None
    return max(longest)


def _support_reachability(
    matrix_ratios: tuple[tuple[int, int], ...],
    dimension: int,
    longest_walk: int,
) -> list[list[int]]:
    """Return per-row bitmasks of columns joined by a walk of each length.

    ``reach[length][row]`` is the bitmask of columns reachable from the row
    by a supported walk of exactly that length; lengths beyond the longest
    walk carry no walks and need no table.
    """

    adjacency = [0] * dimension
    for row_index in range(dimension):
        row_offset = row_index * dimension
        for column_index in range(dimension):
            if matrix_ratios[row_offset + column_index][0] != 0:
                adjacency[row_index] |= 1 << column_index

    reach = [[1 << row_index for row_index in range(dimension)]]
    for _length in range(longest_walk):
        previous = reach[-1]
        current = []
        for row_index in range(dimension):
            mask = 0
            neighbors = adjacency[row_index]
            column_index = 0
            while neighbors:
                if neighbors & 1:
                    mask |= previous[column_index]
                neighbors >>= 1
                column_index += 1
            current.append(mask)
        reach.append(current)
    return reach


def _reachable_length_signatures(
    reach: list[list[int]],
    dimension: int,
    longest_walk: int,
) -> list[tuple[tuple[bool, ...], tuple[bool, ...]]]:
    """Group cells by their pairs of reachable-length signatures.

    Cells share one evaluation when their signatures agree. Signature A
    reads each live shift directly after the identity add; signature B reads
    the same window once the following multiplication shifts every length by
    one.
    """

    group_keys: list[tuple[tuple[bool, ...], tuple[bool, ...]]] = []
    group_indices: dict[tuple[tuple[bool, ...], tuple[bool, ...]], int] = {}
    for row_index in range(dimension):
        for column_index in range(dimension):
            signature_a = tuple(
                bool((reach[length][row_index] >> column_index) & 1)
                for length in range(longest_walk + 1)
            )
            signature_b = tuple(
                length + 1 <= longest_walk
                and bool((reach[length + 1][row_index] >> column_index) & 1)
                for length in range(longest_walk + 1)
            )
            key = (signature_a, signature_b)
            known = group_indices.setdefault(key, len(group_keys))
            if known == len(group_keys):
                group_keys.append(key)
    return group_keys


def _evaluate_horner_group(
    signature_pair: tuple[tuple[bool, ...], tuple[bool, ...]],
    window: tuple[tuple[int, int, int], ...],
    allow_post_multiply: bool,
    longest_walk: int,
    denominator_height: int,
    magnitude_height: int,
    dimension: int,
    budget: list[int],
) -> tuple[int, int] | None:
    """Bound one signature group's widths across both state shapes.

    Returns ``(numerator, denominator)`` work estimates for the group, or
    ``None`` when charging ``budget[0]`` exhausts the admission reduction
    budget.

    One exact common multiple covers both state shapes: the lcm over every
    contributor of this group contains the lcm over each state shape's own
    contributor set. Terms reaching no cell of the group stay excluded so
    disjoint supports never compound here.
    """

    signature_a, signature_b = signature_pair

    def charge(cost: int) -> bool:
        budget[0] -= cost
        return budget[0] >= 0

    common_denominator = 1
    for shift, _numerator, denominator in window:
        reaches_group = signature_a[shift] or (
            shift + 1 <= longest_walk and signature_b[shift]
        )
        if not reaches_group:
            continue
        factor = denominator // gcd(common_denominator, denominator)
        if factor == 1:
            continue
        if not charge(
            (common_denominator.bit_length() + factor.bit_length()) ** 2 // 65_536
        ):
            return None
        widened = common_denominator * factor
        if widened.bit_length() >= _MAX_WORK_BOUND.bit_length():
            return None
        common_denominator = widened

    group_numerator = 0
    group_denominator = 1
    for index, signature in enumerate(signature_pair):
        post_multiply = index == 1
        if post_multiply and not allow_post_multiply:
            continue
        extra = 1 if post_multiply else 0
        live = [
            (shift + extra, numerator, denominator)
            for shift, numerator, denominator in window
            if shift + extra <= longest_walk and signature[shift]
        ]
        if not live:
            continue
        widest_shift = max(shift for shift, _numerator, _denominator in live)
        variant_denominator = _work_multiply(
            common_denominator,
            _work_power(denominator_height, widest_shift),
        )
        variant_numerator = 0
        for shift, numerator, denominator in live:
            # Over the common denominator Dw = lcm{d_i} * H^E, term i
            # contributes exactly ``n_i * Dw / (d_i * w_i) * u_i``. Entry
            # numerators and denominators compound independently
            # (``|u_i| <= magnitude_height^shift * n^(shift-1)`` and
            # ``w_i <= denominator_height^shift``), so the charge splits
            # into ``(lcm/d_i) * H^(E-shift)`` times that numerator product
            # instead of squaring one height.
            magnitude = _work_multiply(
                _work_power(magnitude_height, shift),
                _work_power(dimension, shift - 1 if shift else 0),
            )
            variant_numerator = _work_add(
                variant_numerator,
                _work_multiply(
                    abs(numerator),
                    _work_multiply(
                        (common_denominator // denominator)
                        * _work_power(denominator_height, widest_shift - shift),
                        magnitude,
                    ),
                ),
            )
        if not charge(variant_denominator.bit_length() ** 2 // 65_536):
            return None
        group_numerator = max(group_numerator, variant_numerator)
        group_denominator = max(group_denominator, variant_denominator)
    return group_numerator, group_denominator


def _resolved_horner_intermediate_bounds(
    coefficient_ratios: tuple[tuple[int, int, int], ...],
    matrix_ratios: tuple[tuple[int, int], ...],
    dimension: int,
    longest_walk: int,
    denominator_height: int,
    magnitude_height: int,
) -> tuple[int, int] | None:
    """Bound intermediate Horner widths by resolved entry coexistence.

    Descending Horner stores, immediately after adding ``c_j``, entries equal
    to ``sum_{i>=j} c_i A^(i-j)[p,q]``, and every following multiplication
    presents the same expansion with all lengths shifted by one. Each stored
    entry therefore reduces exactly like a sum of coefficient-weighted
    matrix-power entries, so its reduced denominator divides
    ``lcm{d_i} * H^E`` and its working numerator is bounded term-wise, where
    ``d_i`` ranges over the coefficients whose shifted power ``A^(i-j)[p,q]``
    is nonzero -- exactly the shifts in the cell's reachable-length signature
    -- ``H`` bounds one shift's denominator growth, and ``E`` is the largest
    live shift. Two coefficients whose powers never land on one shared cell
    stay disjoint, while genuinely overlapping supports compound honestly, so
    neither a largest-single-denominator guess nor a global dead-term lcm is
    needed.

    Returns the maximal working numerator and denominator widths across all
    states, both state shapes, and all cells, or ``None`` when the admission
    reduction budget is exhausted; callers then keep their conservative
    whole-window bounds.
    """

    term_count = len(coefficient_ratios)
    if not term_count:
        return 0, 1
    highest_exponent = coefficient_ratios[0][0]

    reach = _support_reachability(matrix_ratios, dimension, longest_walk)
    group_keys = _reachable_length_signatures(reach, dimension, longest_walk)

    budget = [_ADMISSION_REDUCTION_BIT_BUDGET]
    evaluations: dict[
        tuple[int, bool, tuple[tuple[int, int, int], ...]],
        tuple[int, int],
    ] = {}
    work_numerator_bound = 0
    work_denominator_bound = 1
    low = 0
    high = 0
    previous_state_key: tuple[tuple[tuple[int, int, int], ...], bool] | None = None
    for state in range(highest_exponent, -1, -1):
        while low < term_count and coefficient_ratios[low][0] > state + longest_walk:
            low += 1
        while high < term_count and coefficient_ratios[high][0] >= state:
            high += 1
        if low >= high:
            continue
        window = tuple(
            (
                coefficient_ratios[index][0] - state,
                coefficient_ratios[index][1],
                coefficient_ratios[index][2],
            )
            for index in range(low, high)
        )
        allow_post_multiply = state > 0
        # Sliding windows repeat identically across exponent gaps; evaluating
        # one representative per distinct (window, shape) pair suffices.
        state_key = (window, allow_post_multiply)
        if state_key == previous_state_key:
            continue
        previous_state_key = state_key
        for group_index, signatures in enumerate(group_keys):
            evaluation_key = (group_index, allow_post_multiply, window)
            evaluated = evaluations.get(evaluation_key)
            if evaluated is None:
                evaluated = _evaluate_horner_group(
                    signatures,
                    window,
                    allow_post_multiply,
                    longest_walk,
                    denominator_height,
                    magnitude_height,
                    dimension,
                    budget,
                )
                if evaluated is None:
                    return None
                evaluations[evaluation_key] = evaluated
            work_numerator_bound = max(work_numerator_bound, evaluated[0])
            work_denominator_bound = max(work_denominator_bound, evaluated[1])
    return work_numerator_bound, work_denominator_bound


def _linear_result_component_bounds(
    matrix: RationalMatrix,
    coefficients: dict[int, tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Bound every entry of ``c_1 A + c_0 I`` independently.

    Each product ``c_1 a`` is formed and cancelled exactly -- its components
    are at most the product of two canonical components, far below any
    resource limit -- and the diagonal sum keeps operand signs, so additive
    cancellation between ``c_1 a`` and ``c_0`` reduces to its exact rational
    value before digits are counted. Only the reduced entry bounds face the
    digit budgets; the SymPy kernel keeps every intermediate reduced the
    same way.
    """

    linear_numerator, linear_denominator = coefficients.get(1, (0, 1))
    constant_numerator, constant_denominator = coefficients.get(0, (0, 1))
    linear_magnitude = abs(linear_numerator)
    bounds: list[tuple[int, int]] = []
    for row_index, row in enumerate(matrix.entries):
        for column_index, entry in enumerate(row):
            entry_numerator, entry_denominator = entry.as_integer_ratio()
            entry_magnitude = abs(entry_numerator)
            top = gcd(linear_magnitude, entry_denominator)
            bottom = gcd(entry_magnitude, linear_denominator)
            product_numerator = (linear_magnitude // top) * (entry_magnitude // bottom)
            if (linear_numerator >= 0) != (entry_numerator >= 0):
                product_numerator = -product_numerator
            product_denominator = (linear_denominator // bottom) * (
                entry_denominator // top
            )
            terms = [(product_numerator, product_denominator)]
            if row_index == column_index:
                terms.append((constant_numerator, constant_denominator))
            numerator, denominator = _bounded_rational_sum(terms)
            bounds.append(
                (
                    _decimal_digit_upper_bound(numerator),
                    _decimal_digit_upper_bound(denominator),
                )
            )
    return (
        tuple(bounds),
        max(max(numerator, denominator) for numerator, denominator in bounds),
    )


def _general_result_component_bounds(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Bound ``f(A)`` after clearing one exact global denominator.

    If ``Q`` clears every entry denominator and ``M = Q A``, then with ``V``
    clearing the polynomial coefficients, ``V Q^D f(A)`` is the integer matrix

    ``sum_k w_k Q^(D-k) M^k``,

    where ``D`` is any exponent at least the highest surviving one. Entry
    ``(i, j)`` of ``M^k`` is nonzero only when the support digraph of
    the cleared integer entries carries a walk of length ``k`` from ``i`` to
    ``j``, so an acyclic support digraph with longest walk ``L`` makes every
    exponent beyond ``L`` identically zero. Those structurally dead powers
    are classified before any matrix-side quantity or global coefficient
    denominator is derived: when no nonconstant power survives, ``f(A)``
    equals ``c(0) I`` exactly and neither a clearing denominator nor a
    coefficient LCM is constructed at all, so coprime denominators of dead
    powers -- on entries or on coefficients alike -- can never reject a
    request whose admitted work and exact value are small.

    When some nonconstant power survives, every nonzero entry lies on a walk
    of length one, so the global clearing denominator ``Q`` is required and
    its LCM stays within the canonical component cap or the request is
    rejected here; the common coefficient denominator ``V`` is formed over
    the surviving powers only. The output bounds then charge compounded
    denominators honestly: the numerator satisfies ``|N| <= S`` where ``S``
    is the dense capped sum over surviving terms of
    ``w_k Q^(D-k) n^(k-1) h^k``, and the reduced denominator divides
    ``V Q^D``. Cancelling more than proven factors would be unsound -- the
    cleared height ``h`` is a maximum over entries and ``n^(k-1)`` is a
    count, so neither divides the contributing numerators -- hence ``S`` is
    divided only by the exact gcd of the lifted coefficients when its power
    products materialize within the admission ceiling, and ``V Q^D`` only by
    the divisor parts of that same proven factor (see
    ``_reduced_general_result_bound`` and ``_proven_denominator_bound``).
    Whenever those reductions are unavailable, the uncancelled dense bound
    stands and oversized growth is rejected during request validation rather
    than after execution.

    The dense sum is retained as a second, never-structural work bound: its
    digit size drives the coupled digit-work estimate independently of how
    small the exact result is. Horner does not skip structurally dead
    leading powers -- each rides the accumulator until a shifted
    multiplication clears it, and that clearing step pairs its full-height
    entries against live ones once. The work sum therefore charges every
    dead term at its saturated shift ``min(exponent, longest walk)``.
    Intermediate widths are resolved per state and cell rather than guessed
    coarsely: after adding ``c_j`` an accumulator entry equals
    ``sum_{i>=j} c_i A^(i-j)[p,q]``, so its denominator divides the lcm of
    exactly those coefficient denominators whose shifted power lands on that
    cell (``_resolved_horner_intermediate_bounds``). Coefficients whose
    supports never meet one shared cell during the ride stay disjoint --
    coprime denominators of such terms neither reject a request nor inflate
    its digit-work estimate -- while overlapping supports compound honestly
    through the same resolution. The resolution is budget-guarded; on
    exhaustion the conservative whole-window compounds (every surviving and
    dead coefficient denominator over the maximum live shift) stand in,
    which remain sound because they dominate every coexistence pattern.
    Work proxies saturate at the dedicated work ceiling rather than one
    canonical component: honest charges such as compounded shifted heights
    legitimately exceed a single result component before the coupled
    digit-work bound rejects them. In the constant-result branch no clearing
    denominator exists; shifted Horner intermediates there carry rational
    heights compounded over at most the longest support walk, driven by the
    largest input entry height, and the same resolution separates cells
    where two dead denominators genuinely meet from cells where they ride
    disjoint entries. The zero-matrix case is handled separately because its
    intermediates are individual input coefficients.
    """

    matrix_ratios = tuple(
        entry.as_integer_ratio() for row in matrix.entries for entry in row
    )
    dimension = len(matrix.entries)
    longest_walk = _acyclic_support_longest_walk(matrix_ratios, dimension)
    coefficient_ratios = tuple(
        (
            term.exponents[0],
            *term.coefficient.as_integer_ratio(),
        )
        for term in polynomial.polynomial.terms
    )
    surviving_terms = tuple(
        (exponent, numerator, denominator)
        for exponent, numerator, denominator in coefficient_ratios
        if longest_walk is None or exponent <= longest_walk
    )
    # Structural support is classified before any common denominator is
    # required. Coefficients of structurally dead powers never reach the
    # exact value, so coprime dead-power denominators must not be forced into
    # one global coefficient LCM ahead of the support analysis.
    common_coefficient_denominator = _capped_lcm(
        denominator for _exponent, _numerator, denominator in surviving_terms
    )
    if common_coefficient_denominator > _MAX_RESULT_COMPONENT:
        raise ValueError(
            "matrix polynomial denominator growth exceeds the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit result bound"
        )
    dead_terms = tuple(
        (exponent, numerator, denominator)
        for exponent, numerator, denominator in coefficient_ratios
        if longest_walk is not None and exponent > longest_walk
    )

    if all(exponent == 0 for exponent, _n, _d in surviving_terms):
        # Every nonconstant power vanishes structurally, so f(A) equals
        # c(0) I exactly; no Q^d clearing factor is needed and none may be
        # demanded from entries whose denominators only dead powers touch.
        numerator_bound, denominator_bound = next(
            (
                (abs(numerator), denominator)
                for exponent, numerator, denominator in coefficient_ratios
                if exponent == 0
            ),
            (0, 1),
        )
        # Work bound without a global clearing denominator: Horner
        # intermediates are sub-sums over powers shifted down from each
        # coefficient by at most the longest walk, so per-walk products
        # compound rational entry heights over at most that shift while
        # coefficient components enter additively through the sum. These
        # are work estimates, not result components, so they use the work
        # arithmetic whose ceiling sits above one canonical component:
        # clipping them at a single component cap would admit requests
        # such as a nilpotent chain whose shifted intermediates compound
        # several full input heights before the coupled digit-work bound
        # sees their true size.
        entry_height = max(
            max(abs(numerator), denominator) for numerator, denominator in matrix_ratios
        )
        shift_ceiling = max(longest_walk or 0, 0)
        resolved_bounds = (
            _resolved_horner_intermediate_bounds(
                coefficient_ratios,
                matrix_ratios,
                dimension,
                longest_walk,
                denominator_height=entry_height,
                magnitude_height=entry_height,
            )
            if longest_walk is not None
            else None
        )
        if resolved_bounds is not None:
            work_numerator_bound, work_denominator_bound = resolved_bounds
        else:
            # Conservative fallback: additive per-term shifted heights with
            # one global coefficient lcm compounding the widest shift. That
            # lcm dominates every per-cell coexistence pattern, including
            # two dead denominators meeting on a shared entry mid-ride.
            work_numerator_bound = 0
            for exponent, numerator, denominator in coefficient_ratios:
                shift = min(exponent, shift_ceiling)
                work_numerator_bound = _work_add(
                    work_numerator_bound,
                    _work_multiply(
                        max(abs(numerator), denominator),
                        _work_multiply(
                            _work_power(entry_height, shift),
                            _work_power(dimension, shift - 1 if shift else 0),
                        ),
                    ),
                )
            work_denominator_bound = _work_multiply(
                _work_lcm(
                    denominator
                    for _exponent, _numerator, denominator in coefficient_ratios
                ),
                _work_power(entry_height, shift_ceiling),
            )
    else:
        common_matrix_denominator = _capped_lcm(
            denominator for _numerator, denominator in matrix_ratios
        )
        if common_matrix_denominator > _MAX_RESULT_COMPONENT:
            raise ValueError(
                "matrix polynomial denominator growth exceeds the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit result bound"
            )
        highest_surviving_exponent = max(
            exponent for exponent, _numerator, _denominator in surviving_terms
        )
        # |numerator| * (Q // denominator) is at most the square of two canonical
        # components, so the exact cleared height is always safe to materialize.
        cleared_matrix_height = max(
            abs(numerator) * (common_matrix_denominator // denominator)
            for numerator, denominator in matrix_ratios
        )
        # Dead leading powers ride every Horner multiplication until a
        # shifted step clears them, and surviving terms ride to their own
        # exponents: the resolved bounds charge each state's live window per
        # shared cell, so coprime denominators whose powers never land on one
        # common entry stay disjoint while overlapping supports compound
        # honestly. Only budget exhaustion falls back to the whole-window
        # compounds below, which dominate every coexistence pattern.
        resolved_bounds = (
            _resolved_horner_intermediate_bounds(
                coefficient_ratios,
                matrix_ratios,
                dimension,
                longest_walk,
                denominator_height=common_matrix_denominator,
                magnitude_height=cleared_matrix_height,
            )
            if longest_walk is not None
            else None
        )
        numerator_bound = 0
        for exponent, numerator, denominator in surviving_terms:
            coefficient_lift = abs(numerator) * (
                common_coefficient_denominator // denominator
            )
            matrix_power_height = 1
            if exponent:
                matrix_power_height = _work_multiply(
                    _work_power(cleared_matrix_height, exponent),
                    _work_power(dimension, exponent - 1),
                )
            output_term = _capped_multiply(
                coefficient_lift,
                _capped_power(
                    common_matrix_denominator,
                    highest_surviving_exponent - exponent,
                ),
            )
            output_term = _capped_multiply(output_term, matrix_power_height)
            numerator_bound = _capped_add(numerator_bound, output_term)

        if resolved_bounds is not None:
            work_numerator_bound, work_denominator_bound = resolved_bounds
        else:
            # Conservative fallback. Intermediate denominators compound only
            # across live shifts -- a surviving term rides the accumulator up
            # to its own exponent while a structurally dead leading term dies
            # at its clearing multiplication -- so no accumulator ever
            # carries Q above the maximum live shift
            # ``max(min(exponent, longest walk))``. Charging the raw degree
            # of dead leading terms would reject requests whose every
            # Horner intermediate stays within one compounded denominator.
            maximum_live_shift = max(
                (
                    exponent if longest_walk is None else min(exponent, longest_walk)
                    for exponent, _numerator, _denominator in coefficient_ratios
                ),
                default=0,
            )
            dead_common_denominator = _work_lcm(
                denominator for _exponent, _numerator, denominator in dead_terms
            )
            work_denominator_bound = _work_multiply(
                _work_multiply(
                    common_coefficient_denominator,
                    dead_common_denominator,
                ),
                _work_power(common_matrix_denominator, maximum_live_shift),
            )
            work_numerator_bound = 0
            for exponent, numerator, denominator in surviving_terms:
                coefficient_lift = abs(numerator) * (
                    common_coefficient_denominator // denominator
                )
                matrix_power_height = 1
                if exponent:
                    matrix_power_height = _work_multiply(
                        _work_power(cleared_matrix_height, exponent),
                        _work_power(dimension, exponent - 1),
                    )
                term_height = _work_multiply(
                    coefficient_lift,
                    _work_power(
                        common_matrix_denominator,
                        maximum_live_shift - exponent,
                    ),
                )
                term_height = _work_multiply(term_height, matrix_power_height)
                work_numerator_bound = _work_add(work_numerator_bound, term_height)

            # Shifted dead terms keep their own Horner shifts alive up to the
            # longest support walk: the killing multiplication pairs every
            # surviving dead entry with a zero operand, so the saturated shift
            # below is the last step whose products materialize full-height
            # operands. The lift uses the combined coefficient scale so it
            # never undercounts a dead denominator against the common work
            # envelope.
            support_walk = longest_walk or 0
            for exponent, numerator, denominator in dead_terms:
                shift = min(exponent, support_walk)
                dead_lift = _work_multiply(
                    abs(numerator),
                    _work_multiply(
                        common_coefficient_denominator,
                        dead_common_denominator // denominator,
                    ),
                )
                matrix_power_height = _work_multiply(
                    _work_power(cleared_matrix_height, shift),
                    _work_power(dimension, shift - 1),
                )
                work_numerator_bound = _work_add(
                    work_numerator_bound,
                    _work_multiply(
                        dead_lift,
                        _work_multiply(
                            _work_power(
                                common_matrix_denominator,
                                maximum_live_shift - shift,
                            ),
                            matrix_power_height,
                        ),
                    ),
                )

        proven_numerator_factor = _proven_numerator_factor(
            surviving_terms, common_coefficient_denominator
        )
        reduced_numerator_bound = _reduced_general_result_bound(
            surviving_terms,
            cleared_matrix_height,
            common_matrix_denominator,
            common_coefficient_denominator,
            dimension,
            highest_surviving_exponent,
            proven_numerator_factor,
        )
        if reduced_numerator_bound is not None:
            numerator_bound = min(numerator_bound, reduced_numerator_bound)
        denominator_bound = _proven_denominator_bound(
            common_matrix_denominator,
            common_coefficient_denominator,
            highest_surviving_exponent,
            proven_numerator_factor,
        )

    maximum_work_digits = max(
        _decimal_digit_upper_bound(work_numerator_bound),
        _decimal_digit_upper_bound(work_denominator_bound),
    )
    if numerator_bound == 0:
        component_bound = (1, 1)
    else:
        if (
            numerator_bound > _MAX_RESULT_COMPONENT
            or denominator_bound > _MAX_RESULT_COMPONENT
        ):
            raise ValueError(
                "matrix polynomial coefficient growth exceeds the canonical "
                f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit result bound"
            )
        component_bound = (
            _decimal_digit_upper_bound(numerator_bound),
            _decimal_digit_upper_bound(denominator_bound),
        )
    return (
        (component_bound,) * (dimension * dimension),
        maximum_work_digits,
    )


def _require_matrix_polynomial_output_budget(
    matrix: RationalMatrix,
    polynomial: RationalPolynomial,
    degree: int,
) -> int:
    coefficients = _coefficient_ratios(polynomial)
    matrix_is_zero = all(entry.num == "0" for row in matrix.entries for entry in row)
    if degree <= 1 or matrix_is_zero:
        component_bounds, maximum_arithmetic_digits = _linear_result_component_bounds(
            matrix, coefficients
        )
    else:
        component_bounds, maximum_arithmetic_digits = _general_result_component_bounds(
            matrix, polynomial
        )
    if any(
        numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        for numerator_digits, denominator_digits in component_bounds
    ):
        raise ValueError(
            "matrix polynomial evaluation can exceed the canonical "
            f"{MAX_CANONICAL_RATIONAL_DIGITS:,}-digit rational result bound"
        )

    source_bytes = len(encode_strict_json(matrix.model_dump(mode="json")))
    polynomial_bytes = len(encode_strict_json(polynomial.model_dump(mode="json")))
    value_bytes = sum(
        numerator_digits + denominator_digits + _RESULT_ENTRY_OVERHEAD_BYTES
        for numerator_digits, denominator_digits in component_bounds
    )
    estimated_result_bytes = (
        source_bytes + polynomial_bytes + value_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise ValueError(
            "the retained matrix, polynomial, and exact value can exceed the "
            f"{output_limit}-byte canonical output limit"
        )
    arithmetic_component_digits = [
        len(component.lstrip("-"))
        for row in matrix.entries
        for entry in row
        for component in (entry.num, entry.den)
    ]
    arithmetic_component_digits.extend(
        len(component.lstrip("-"))
        for term in polynomial.polynomial.terms
        for component in (term.coefficient.num, term.coefficient.den)
    )
    arithmetic_component_digits.append(maximum_arithmetic_digits)
    return max(arithmetic_component_digits)


class MatrixPolynomialEvaluationRequest(StrictModel):
    """Evaluate one exact univariate rational polynomial at a square matrix."""

    matrix: RationalMatrix = Field(
        description=(
            "Nonempty square matrix over QQ through order "
            f"{MAX_MATRIX_DIMENSION}. Matrix and polynomial coefficients "
            "share the exact rational field."
        )
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "Sparse polynomial over QQ in exactly one declared variable; terms "
            "use the canonical descending exponent order of RationalPolynomial. "
            "Exact admission couples the ordinary degree to the matrix order: "
            f"{MATRIX_POLYNOMIAL_EVALUATION_PASSES} * degree * order^3 must "
            f"stay within {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,} scalar "
            "products across both Horner passes, so the largest admitted "
            "ordinary degree at matrix order 32 is "
            f"{MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // (MATRIX_POLYNOMIAL_EVALUATION_PASSES * MAX_MATRIX_DIMENSION**3)}. "
            "Exact admission additionally multiplies the total "
            "(2 * degree * order^3) scalar products by the square of the "
            "largest decimal-digit component among the matrix entries, the "
            "polynomial coefficients, the predicted exact-result "
            "components, and the predicted shifted Horner intermediate "
            "components, whose denominators include coprime coefficient "
            "denominators whenever their matrix-power supports overlap in "
            "one shared entry during evaluation; that digit-work product "
            "must stay within "
            f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,} units."
        )
    )

    @model_validator(mode="after")
    def require_square_univariate_bounded_evaluation(self) -> Self:
        dimension = len(self.matrix.entries)
        if len(self.matrix.entries[0]) != dimension:
            raise ValueError("matrix polynomial evaluation requires a square matrix")
        if len(self.polynomial.variables) != 1:
            raise ValueError(
                "matrix polynomial evaluation requires exactly one polynomial variable"
            )
        require_polynomial_budget(
            self.polynomial,
            maximum_terms=MAX_POLYNOMIAL_TERMS,
            maximum_exponent=MAX_POLYNOMIAL_EXPONENT,
            maximum_coefficient_digits=MAX_CANONICAL_RATIONAL_DIGITS,
            label="matrix polynomial",
        )
        degree = _polynomial_degree(self.polynomial)
        scalar_products_per_pass = degree * dimension**3
        total_scalar_products = (
            MATRIX_POLYNOMIAL_EVALUATION_PASSES * scalar_products_per_pass
        )
        if total_scalar_products > MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:
            raise ValueError(
                "matrix polynomial Horner evaluation and source-bound replay "
                f"exceed the {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,}-scalar-product "
                "work bound"
            )
        maximum_arithmetic_digits = _require_matrix_polynomial_output_budget(
            self.matrix,
            self.polynomial,
            degree,
        )
        digit_work = total_scalar_products * maximum_arithmetic_digits**2
        if digit_work > MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:
            raise ValueError(
                "matrix polynomial exact-arithmetic work exceeds the coupled "
                f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,}-unit digit-work bound"
            )
        return self


class MatrixPolynomialEvaluationResult(StrictModel):
    """Source-bound exact value of one polynomial at one rational matrix."""

    source_matrix: RationalMatrix
    polynomial: RationalPolynomial
    value: RationalMatrix
    polynomial_degree: int | None = Field(
        default=None,
        ge=0,
        le=MAX_POLYNOMIAL_EXPONENT,
        description="The ordinary degree for a nonzero polynomial; null for zero.",
    )
    matrix_multiplications: int = Field(ge=0, le=MAX_POLYNOMIAL_EXPONENT)
    scalar_product_terms: int = Field(
        ge=0,
        le=MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    )
    method: Literal["HORNER_OVER_QQ"] = "HORNER_OVER_QQ"

    @model_validator(mode="after")
    def bind_exact_evaluation(self) -> Self:
        request = MatrixPolynomialEvaluationRequest(
            matrix=self.source_matrix,
            polynomial=self.polynomial,
        )
        expected_degree = _mathematical_polynomial_degree(request.polynomial)
        if self.polynomial_degree != expected_degree:
            raise ValueError(
                "matrix polynomial result degree does not match its source"
            )
        expected_multiplications = expected_degree or 0
        if self.matrix_multiplications != expected_multiplications:
            raise ValueError(
                "Horner matrix multiplication count must equal the polynomial degree"
            )
        dimension = len(request.matrix.entries)
        if self.scalar_product_terms != expected_multiplications * dimension**3:
            raise ValueError(
                "Horner scalar-product count must equal degree times matrix order cubed"
            )
        from jacobian.math.matrices.canonical_forms._operations import (
            evaluate_matrix_polynomial_value,
        )

        expected = evaluate_matrix_polynomial_value(request)
        if self.value != expected:
            raise ValueError(
                "matrix polynomial value does not equal the retained exact evaluation"
            )
        return self


class SquareMatrixRequest(StrictModel):
    """One square rational matrix bounded for canonical-form computation."""

    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_bounded_square(self) -> Self:
        rows = len(self.matrix.entries)
        columns = len(self.matrix.entries[0])
        if rows != columns:
            raise ValueError("canonical-form operations require a square matrix")
        if rows > MAX_CANONICAL_FORM_DIMENSION:
            raise ValueError(
                "canonical-form operations are bounded to 16 x 16 matrices"
            )
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=MAX_CANONICAL_FORM_SCALAR_DIGITS,
            label="canonical-form matrix",
        )
        return self


class MonicPolynomial(StrictModel):
    """One monic univariate polynomial over QQ, as increasing-degree coefficients."""

    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_CANONICAL_FORM_DIMENSION + 1,
    )

    @model_validator(mode="after")
    def require_monic(self) -> Self:
        if self.coefficients[-1].as_fraction() != 1:
            raise ValueError("polynomial must be monic (leading coefficient = 1)")
        return self


class MinimalPolynomialResult(StrictModel):
    """Exact minimal polynomial of a square rational matrix."""

    minimal_polynomial: MonicPolynomial
    characteristic_polynomial: MonicPolynomial
    degree: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["KRYLOV_NULLSPACE"] = "KRYLOV_NULLSPACE"


class InvariantFactorEntry(StrictModel):
    """One monic invariant factor from the rational canonical form."""

    factor: MonicPolynomial
    block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)


class RationalCanonicalFormResult(StrictModel):
    """Exact rational (Frobenius) canonical form of a square rational matrix."""

    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    total_block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)
    method: Literal["SMITH_NORMAL_FORM"] = "SMITH_NORMAL_FORM"


class PrimaryDecompositionResult(StrictModel):
    """Primary decomposition of the minimal polynomial into irreducible-power components."""

    components: tuple[MonicPolynomial, ...] = Field(min_length=1)
    minimal_polynomial: MonicPolynomial
    method: Literal["FACTOR_LCM"] = "FACTOR_LCM"


__all__ = [
    "MATRIX_POLYNOMIAL_EVALUATION_PASSES",
    "MAX_CANONICAL_FORM_DIMENSION",
    "MAX_CANONICAL_FORM_SCALAR_DIGITS",
    "MAX_MATRIX_POLYNOMIAL_DIGIT_WORK",
    "MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS",
    "InvariantFactorEntry",
    "MatrixPolynomialEvaluationRequest",
    "MatrixPolynomialEvaluationResult",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SquareMatrixRequest",
]
