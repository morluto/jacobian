"""Supported native APIs for exact classical combinatorial numbers."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._progression_hypergraph_models import (
    MAX_GROUP_ORDER,
    ProgressionHypergraphResult,
)
from jacobian.math.combinatorics._recurrence_admission import (
    _admit_linear_recurrence,
    _admit_p_recursive_recurrence,
    _admit_series,
)
from jacobian.math.combinatorics._recurrence_models import (
    IndexedRationalValue,
    LinearRecurrenceEvaluationResult,
    PolynomialCoefficientRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def _nonnegative(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OperationDomainValidationError(
            location=(name,),
            code="combinatorics.nonnegative_integer_required",
            message=f"{name} must be a nonnegative integer",
        )
    return value


def _pair(n: int, k: int) -> tuple[int, int]:
    return _nonnegative(n, name="n"), _nonnegative(k, name="k")


MAX_COUNTING_INDEX = 10_000
MAX_MULTINOMIAL_PARTS = 256
MAX_MULTINOMIAL_TOTAL = MAX_COUNTING_INDEX


def _bounded_counting_index(value: int, *, name: str) -> int:
    value = _nonnegative(value, name=name)
    if value > MAX_COUNTING_INDEX:
        raise OperationDomainValidationError(
            location=(name,),
            code="combinatorics.counting_index_out_of_range",
            message=f"{name} exceeds the {MAX_COUNTING_INDEX}-element counting bound",
        )
    return value


def factorial(n: int) -> int:
    """Return the factorial of a bounded nonnegative integer."""
    import math

    return math.factorial(_bounded_counting_index(n, name="n"))


def binomial(n: int, k: int) -> int:
    """Return the exact binomial coefficient, with zero for ``k > n``."""
    import math

    first = _bounded_counting_index(n, name="n")
    second = _bounded_counting_index(k, name="k")
    return 0 if second > first else math.comb(first, second)


def multinomial(values: tuple[int, ...]) -> int:
    """Return the exact multinomial coefficient for nonnegative part sizes."""
    import math

    if not isinstance(values, tuple) or not values:
        raise OperationDomainValidationError(
            location=("values",),
            code="combinatorics.multinomial_values_required",
            message="values must be a nonempty tuple of nonnegative integers",
        )
    parts = tuple(_nonnegative(value, name="values") for value in values)
    if len(parts) > MAX_MULTINOMIAL_PARTS:
        raise OperationDomainValidationError(
            location=("values",),
            code="combinatorics.multinomial_part_count_out_of_range",
            message=f"values exceeds the {MAX_MULTINOMIAL_PARTS}-part counting bound",
        )
    if len(parts) == 1:
        return 1
    total = sum(parts)
    if total > MAX_MULTINOMIAL_TOTAL:
        raise OperationDomainValidationError(
            location=("values",),
            code="combinatorics.multinomial_total_out_of_range",
            message=(
                "the sum of values exceeds the "
                f"{MAX_MULTINOMIAL_TOTAL}-element counting bound"
            ),
        )
    return math.factorial(total) // math.prod(math.factorial(part) for part in parts)


def permutations(n: int, k: int) -> int:
    """Return the exact number of ordered ``k``-selections from ``n``."""
    import math

    first = _bounded_counting_index(n, name="n")
    second = _bounded_counting_index(k, name="k")
    return 0 if second > first else math.perm(first, second)


def central_binomial(n: int) -> int:
    """Return the exact central binomial coefficient ``binomial(2n, n)``."""
    import math

    value = _bounded_counting_index(n, name="n")
    return math.comb(2 * value, value)


def compositions(n: int, k: int) -> int:
    """Count ordered compositions of ``n`` into ``k`` positive parts."""
    import math

    total = _bounded_counting_index(n, name="n")
    parts = _bounded_counting_index(k, name="k")
    if total == parts == 0:
        return 1
    return math.comb(total - 1, parts - 1) if 0 < parts <= total else 0


def bell_number(n: int) -> int:
    """Return the nth Bell number."""

    import sympy

    return int(sympy.bell(_nonnegative(n, name="n")))


def bernoulli_number(n: int) -> Fraction:
    """Return the nth Bernoulli number exactly."""

    import sympy

    value = sympy.bernoulli(_nonnegative(n, name="n"))
    return Fraction(int(value.p), int(value.q))


def catalan_number(n: int) -> int:
    """Return the nth Catalan number."""

    import sympy

    return int(sympy.catalan(_nonnegative(n, name="n")))


def derangement_number(n: int) -> int:
    """Return the number of derangements of n objects."""

    import sympy

    return int(sympy.subfactorial(_nonnegative(n, name="n")))


def double_factorial(n: int) -> int:
    """Return the nonnegative integer double factorial."""

    import sympy

    return int(sympy.factorial2(_nonnegative(n, name="n")))


def fibonacci_number(n: int) -> int:
    """Return the nth Fibonacci number."""

    import sympy

    return int(sympy.fibonacci(_nonnegative(n, name="n")))


def integer_partitions(
    n: int,
    *,
    max_parts: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate integer partitions in deterministic reverse-part order."""

    from sympy.utilities.iterables import partitions

    value = _nonnegative(n, name="n")
    if max_parts is not None:
        max_parts = _nonnegative(max_parts, name="max_parts")
    return tuple(
        tuple(
            part
            for part in sorted(multiplicities, reverse=True)
            for _ in range(int(multiplicities[part]))
        )
        for multiplicities in partitions(value, m=max_parts)
    )


def lucas_number(n: int) -> int:
    """Return the nth Lucas number."""

    import sympy

    return int(sympy.lucas(_nonnegative(n, name="n")))


def motzkin_number(n: int) -> int:
    """Return the nth Motzkin number."""

    import sympy

    return int(sympy.motzkin(_nonnegative(n, name="n")))


def partition_number(n: int) -> int:
    """Return the number of integer partitions of n."""

    import sympy

    return int(sympy.partition(_nonnegative(n, name="n")))


def stirling_first(n: int, k: int) -> int:
    """Return the unsigned Stirling number of the first kind."""

    from sympy.functions.combinatorial.numbers import stirling

    first, second = _pair(n, k)
    return int(stirling(first, second, kind=1))


def stirling_second(n: int, k: int) -> int:
    """Return the Stirling number of the second kind."""

    from sympy.functions.combinatorial.numbers import stirling

    first, second = _pair(n, k)
    return int(stirling(first, second, kind=2))


def progression_hypergraph(group_order: int) -> ProgressionHypergraphResult:
    """Return the 3-term progression hypergraph of ``Z/group_order Z``."""

    if not 2 <= group_order <= MAX_GROUP_ORDER:
        raise OperationDomainValidationError(
            location=("group_order",),
            code="combinatorics.progression_hypergraph.group_order",
            message=f"group order must be between 2 and {MAX_GROUP_ORDER}",
        )
    vertices = tuple(str(index) for index in range(group_order))
    edge_sets = {
        frozenset(
            (start, (start + step) % group_order, (start + 2 * step) % group_order)
        )
        for step in range(1, group_order)
        for start in range(group_order)
    }
    nondegenerate_edges = sorted(
        (edge for edge in edge_sets if len(edge) == 3),
        key=lambda edge: tuple(sorted(edge)),
    )
    edges = tuple(
        (f"e{index}", tuple(sorted(str(vertex) for vertex in edge)))
        for index, edge in enumerate(nondegenerate_edges)
    )
    return ProgressionHypergraphResult(
        group_order=group_order,
        hypergraph=FiniteHypergraph(vertices=vertices, edges=edges),
    )


def _wire(value: Fraction) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def _require_rational_tuple(values: object, *, name: str) -> None:
    if not isinstance(values, tuple) or not all(
        isinstance(value, CanonicalRational) for value in values
    ):
        raise OperationDomainValidationError(
            location=(name,),
            code="combinatorics.canonical_rational_tuple_required",
            message=f"{name} must be a tuple of CanonicalRational values",
        )


def _requested_indices(
    *,
    scope: str,
    term_count: int | None,
    indices: tuple[int, ...],
) -> tuple[int, ...]:
    if scope == "PREFIX":
        if type(term_count) is not int or not 1 <= term_count <= 513 or indices:
            raise OperationDomainValidationError(
                location=("scope",),
                code="combinatorics.recurrence_scope",
                message="PREFIX scope requires term_count and forbids indices",
            )
        return tuple(range(term_count))
    if scope != "INDICES":
        raise OperationDomainValidationError(
            location=("scope",),
            code="combinatorics.recurrence_scope",
            message="scope must be PREFIX or INDICES",
        )
    if term_count is not None or not indices:
        raise OperationDomainValidationError(
            location=("scope",),
            code="combinatorics.recurrence_scope",
            message="INDICES scope requires indices and forbids term_count",
        )
    if len(indices) > 256 or any(
        type(index) is not int or not 0 <= index <= 512 for index in indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="combinatorics.result_bound",
            message="indices are outside the recurrence bound",
        )
    if any(left >= right for left, right in pairwise(indices)):
        raise OperationDomainValidationError(
            location=("indices",),
            code="combinatorics.recurrence_scope",
            message="indices must be strictly increasing",
        )
    return indices


def evaluate_linear_recurrence(
    coefficients: tuple[CanonicalRational, ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: Literal[
        "A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"
    ],
    scope: Literal["PREFIX", "INDICES"],
    term_count: int | None = None,
    indices: tuple[int, ...] = (),
) -> LinearRecurrenceEvaluationResult:
    """Evaluate requested terms of one bounded constant-coefficient recurrence."""

    if coefficient_convention != ("A_N_EQUALS_SUM_C_J_TIMES_A_N_MINUS_J_FOR_J_FROM_1"):
        raise OperationDomainValidationError(
            location=("coefficient_convention",),
            code="combinatorics.recurrence_convention",
            message="unsupported linear recurrence coefficient convention",
        )
    _require_rational_tuple(coefficients, name="coefficients")
    _require_rational_tuple(initial_values, name="initial_values")
    if not 1 <= len(coefficients) <= 16 or not 1 <= len(initial_values) <= 16:
        raise OperationDomainValidationError(
            location=("coefficients",),
            code="combinatorics.recurrence_invariant",
            message="recurrence order is outside the bound",
        )
    if len(initial_values) != len(coefficients):
        raise OperationDomainValidationError(
            location=("initial_values",),
            code="combinatorics.recurrence_invariant",
            message="initial_values length must equal the recurrence order",
        )
    if not isinstance(indices, tuple) or not all(
        type(index) is int for index in indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="combinatorics.recurrence_indices_required",
            message="indices must be a tuple of integers",
        )
    requested_indices = _requested_indices(
        scope=scope, term_count=term_count, indices=indices
    )
    prefix = _admit_linear_recurrence(
        coefficients=coefficients,
        initial_values=initial_values,
        coefficient_convention=coefficient_convention,
        scope=scope,
        requested_indices=requested_indices,
    )
    return LinearRecurrenceEvaluationResult._from_kernel(
        coefficient_convention=coefficient_convention,
        scope=scope,
        values=tuple(
            IndexedRationalValue(index=index, value=_wire(prefix[index]))
            for index in requested_indices
        ),
    )


def evaluate_polynomial_coefficient_recurrence(
    coefficient_polynomials: tuple[tuple[CanonicalRational, ...], ...],
    initial_values: tuple[CanonicalRational, ...],
    coefficient_convention: Literal[
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ],
    polynomial_convention: Literal["ASCENDING_POWERS_OF_N"],
    scope: Literal["PREFIX", "INDICES"],
    term_count: int | None = None,
    indices: tuple[int, ...] = (),
) -> PolynomialCoefficientRecurrenceEvaluationResult:
    """Evaluate requested terms of one bounded polynomial-coefficient recurrence."""

    if coefficient_convention != (
        "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
    ):
        raise OperationDomainValidationError(
            location=("coefficient_convention",),
            code="combinatorics.recurrence_convention",
            message="unsupported polynomial recurrence coefficient convention",
        )
    if polynomial_convention != "ASCENDING_POWERS_OF_N":
        raise OperationDomainValidationError(
            location=("polynomial_convention",),
            code="combinatorics.recurrence_convention",
            message="unsupported polynomial recurrence convention",
        )
    if not isinstance(coefficient_polynomials, tuple) or not all(
        isinstance(polynomial, tuple)
        and all(isinstance(value, CanonicalRational) for value in polynomial)
        for polynomial in coefficient_polynomials
    ):
        raise OperationDomainValidationError(
            location=("coefficient_polynomials",),
            code="combinatorics.canonical_rational_tuple_required",
            message=(
                "coefficient_polynomials must be tuples of CanonicalRational values"
            ),
        )
    _require_rational_tuple(initial_values, name="initial_values")
    if not 2 <= len(coefficient_polynomials) <= 17:
        raise OperationDomainValidationError(
            location=("coefficient_polynomials",),
            code="combinatorics.recurrence_invariant",
            message="recurrence order is outside the bound",
        )
    if len(initial_values) != len(coefficient_polynomials) - 1:
        raise OperationDomainValidationError(
            location=("initial_values",),
            code="combinatorics.recurrence_invariant",
            message="initial_values length must equal the recurrence order",
        )
    if not isinstance(indices, tuple) or not all(
        type(index) is int for index in indices
    ):
        raise OperationDomainValidationError(
            location=("indices",),
            code="combinatorics.recurrence_indices_required",
            message="indices must be a tuple of integers",
        )
    requested_indices = _requested_indices(
        scope=scope, term_count=term_count, indices=indices
    )
    prefix = _admit_p_recursive_recurrence(
        coefficient_polynomials=coefficient_polynomials,
        initial_values=initial_values,
        coefficient_convention=coefficient_convention,
        polynomial_convention=polynomial_convention,
        scope=scope,
        requested_indices=requested_indices,
    )
    return PolynomialCoefficientRecurrenceEvaluationResult._from_kernel(
        coefficient_convention=coefficient_convention,
        polynomial_convention=polynomial_convention,
        scope=scope,
        recurrence_order=len(coefficient_polynomials) - 1,
        values=tuple(
            IndexedRationalValue(index=index, value=_wire(prefix[index]))
            for index in requested_indices
        ),
    )


def rational_generating_function_coefficients(
    numerator: tuple[CanonicalRational, ...],
    denominator: tuple[CanonicalRational, ...],
    coefficient_convention: Literal["ASCENDING_POWERS_OF_X"],
    expansion_point: Literal["0"],
    truncation_order: int,
) -> RationalGeneratingFunctionCoefficientsResult:
    """Expand ``N(x)/D(x)`` through one bounded exact truncation order."""

    if coefficient_convention != "ASCENDING_POWERS_OF_X":
        raise OperationDomainValidationError(
            location=("coefficient_convention",),
            code="combinatorics.generating_function_convention",
            message="unsupported generating-function coefficient convention",
        )
    if expansion_point != "0":
        raise OperationDomainValidationError(
            location=("expansion_point",),
            code="combinatorics.generating_function_expansion_point",
            message="only expansion at zero is supported",
        )
    _require_rational_tuple(numerator, name="numerator")
    _require_rational_tuple(denominator, name="denominator")
    if not 1 <= len(numerator) <= 33 or not 1 <= len(denominator) <= 33:
        raise OperationDomainValidationError(
            location=("numerator",),
            code="combinatorics.polynomial_invariant",
            message="polynomial degree is outside the bound",
        )
    if type(truncation_order) is not int or not 1 <= truncation_order <= 512:
        raise OperationDomainValidationError(
            location=("truncation_order",),
            code="combinatorics.result_bound",
            message="truncation order is outside the bound",
        )
    coefficients = _admit_series(
        numerator=numerator,
        denominator=denominator,
        coefficient_convention=coefficient_convention,
        expansion_point=expansion_point,
        truncation_order=truncation_order,
    )
    return RationalGeneratingFunctionCoefficientsResult._from_kernel(
        coefficient_convention=coefficient_convention,
        expansion_point=expansion_point,
        truncation_order=truncation_order,
        coefficients=tuple(_wire(item) for item in coefficients),
        residual_congruence=(
            "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
        ),
        residual_coefficients=tuple(
            CanonicalRational(num="0", den="1") for _ in coefficients
        ),
    )


__all__ = [
    "bell_number",
    "bernoulli_number",
    "binomial",
    "catalan_number",
    "central_binomial",
    "compositions",
    "derangement_number",
    "double_factorial",
    "evaluate_linear_recurrence",
    "evaluate_polynomial_coefficient_recurrence",
    "factorial",
    "fibonacci_number",
    "integer_partitions",
    "lucas_number",
    "motzkin_number",
    "multinomial",
    "partition_number",
    "permutations",
    "progression_hypergraph",
    "rational_generating_function_coefficients",
    "stirling_first",
    "stirling_second",
]
