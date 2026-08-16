"""Boolean analysis operations backed by exact integer arithmetic."""

from __future__ import annotations

from itertools import combinations, product

from jacobian.contracts.boolean_analysis import (
    BooleanErasureNoiseRequest,
    BooleanErasureNoiseResult,
    BooleanFourierRequest,
    BooleanFourierResult,
    BooleanInfluenceRequest,
    BooleanInfluenceResult,
    BooleanMultilinearExtensionRequest,
    BooleanMultilinearExtensionResult,
    BooleanTruthTable,
    FourierCoefficient,
)


def _vertex_index(assignment: tuple[int, ...]) -> int:
    """Convert a {-1,+1}^n assignment to a canonical index."""
    index = 0
    for i, v in enumerate(assignment):
        if v == 1:
            index |= (1 << i)
    return index


def _vertex_from_index(index: int, n: int) -> tuple[int, ...]:
    """Convert a canonical index to a {-1,+1}^n assignment."""
    return tuple(1 if (index >> i) & 1 else -1 for i in range(n))


def compute_boolean_fourier(
    request: BooleanFourierRequest,
) -> BooleanFourierResult:
    """Compute all Walsh-Fourier coefficients of a Boolean function.

    For f: {-1,+1}^n -> {-1,+1}, the Walsh-Fourier coefficient is:
    f_hat(S) = (1/2^n) * sum_{x} f(x) * chi_S(x)
    where chi_S(x) = prod_{i in S} x_i.

    We return the coefficients as integers (numerator only, denominator = 2^n).
    """
    n = len(request.truth_table.variable_names)
    values = request.truth_table.values
    total = 2 ** n
    coefficients = []

    for mask in range(total):
        coeff = 0
        for idx in range(total):
            assignment = _vertex_from_index(idx, n)
            # chi_S(x) = product of x_i for i in S
            chi = 1
            for i in range(n):
                if mask & (1 << i):
                    chi *= assignment[i]
            coeff += values[idx] * chi
        coefficients.append(
            FourierCoefficient(subset_mask=mask, coefficient=coeff)
        )

    return BooleanFourierResult(
        coefficients=tuple(coefficients),
        variable_count=n,
    )


def compute_multilinear_extension(
    request: BooleanMultilinearExtensionRequest,
) -> BooleanMultilinearExtensionResult:
    """Evaluate the multilinear extension of f at a point in Z^n.

    The multilinear extension is:
    f(x_1,...,x_n) = sum_S f_hat(S) * prod_{i in S} (x_i + 1) / 2 * prod_{i not in S} (1 - x_i) / 2

    But since the truth table gives us f directly, we use:
    f(z) = sum_{x in {-1,+1}^n} f(x) * prod_i (1 + z_i * x_i) / 2
    """
    n = len(request.truth_table.variable_names)
    values = request.truth_table.values
    point = request.point
    total = 2 ** n

    result_num = 0
    result_den = 1

    for idx in range(total):
        assignment = _vertex_from_index(idx, n)
        # prod_i (1 + z_i * x_i) / 2
        num = 1
        den = 1
        for i in range(n):
            # (1 + z_i * x_i) / 2
            num *= (1 + point[i] * assignment[i])
            den *= 2
        result_num = result_num * den + values[idx] * num * result_den
        result_den = result_den * den

    # Simplify
    from math import gcd
    g = gcd(abs(result_num), abs(result_den))
    if g > 0:
        result_num //= g
        result_den //= g

    return BooleanMultilinearExtensionResult(
        value=result_num // result_den if result_den != 0 else 0,
        detail=f"Multilinear extension evaluated at {point}: numerator={result_num}, denominator={result_den}.",
    )


def compute_boolean_influence(
    request: BooleanInfluenceRequest,
) -> BooleanInfluenceResult:
    """Compute the influence of each variable.

    The influence of variable i is the number of edges (x, x^i) where f(x) != f(x^i),
    divided by 2^(n-1). We return the numerator (number of flippable edges).
    """
    n = len(request.truth_table.variable_names)
    values = request.truth_table.values
    total = 2 ** n

    influences = []
    for i in range(n):
        count = 0
        for idx in range(total):
            neighbor = idx ^ (1 << i)
            if values[idx] != values[neighbor]:
                count += 1
        influences.append(count)

    return BooleanInfluenceResult(
        influences=tuple(influences),
        total_influence=sum(influences),
    )


def compute_erasure_noise(
    request: BooleanErasureNoiseRequest,
) -> BooleanErasureNoiseResult:
    """Compute the exact expected |f(z)| where z is f with some variables erased.

    When k variables are erased, each of the C(n,k) subsets is equally likely.
    For each erasure set S of size k, we evaluate f at the 2^(n-k) remaining
    assignments and compute E|f(z)| over all erasures and completions.
    """
    n = len(request.truth_table.variable_names)
    values = request.truth_table.values
    k = request.erasure_count

    if k == 0:
        # No erasure: E|f(z)| = sum |f(x)| / 2^n = 1 since f is always +-1
        return BooleanErasureNoiseResult(
            expected_absolute_value_numerator=1,
            expected_absolute_value_denominator=1,
            detail="No erasure: E|f(z)| = 1.",
        )

    total_sum = 0
    total_count = 0

    # For each subset S of size k (erased variables)
    for erased in combinations(range(n), k):
        remaining = [i for i in range(n) if i not in erased]
        # For each assignment of the remaining variables
        for remaining_vals in product([-1, 1], repeat=len(remaining)):
            # Average over all assignments of the erased variables
            for erased_vals in product([-1, 1], repeat=k):
                # Construct full assignment
                assignment = [0] * n
                for j, idx in enumerate(remaining):
                    assignment[idx] = remaining_vals[j]
                for j, idx in enumerate(erased):
                    assignment[idx] = erased_vals[j]
                # Evaluate f
                idx = 0
                for i in range(n):
                    if assignment[i] == 1:
                        idx |= (1 << i)
                total_sum += abs(values[idx])
                total_count += 1

    from math import gcd
    g = gcd(total_sum, total_count)
    return BooleanErasureNoiseResult(
        expected_absolute_value_numerator=total_sum // g,
        expected_absolute_value_denominator=total_count // g,
        detail=f"E|f(z)| = {total_sum}/{total_count} with {k} erasures.",
    )
