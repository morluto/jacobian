"""Canonical exact number-theory operations."""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from itertools import product
from time import monotonic
from typing import Literal, SupportsIndex, cast

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._execution import current_request_execution
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.number_theory._binomial_valuation_models import (
    _MAX_BINOMIAL_ROWS_FROM_OUTPUT,
    _MAX_SAFE_JSON_INTEGER,
    MAX_BINOMIAL_DIGIT_WORK,
    MAX_BINOMIAL_PROFILE_RESULT_BYTES,
    BinomialValuationProfileResult,
    BinomialValuationProfileRow,
    _base_digit_count,
    _binomial_result_upper_bound_bytes,
)
from jacobian.math.number_theory._contiguous_sum_admission import (
    require_contiguous_sum_profile_admission,
)
from jacobian.math.number_theory._contiguous_sum_kernel import (
    run_contiguous_sum_profile,
)
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileResult,
)
from jacobian.math.number_theory._derived_models import (
    MAX_LEGENDRE_PRIME,
    BinomialPrimeValuationResult,
    FactorialValuationResult,
    FloorSquareRootResult,
    LegendreSymbolResult,
    _BinomialValuationInput,
    _FactorialValuationInput,
    admit_binomial_prime_valuation,
    admit_factorial_valuation,
)
from jacobian.math.number_theory._divisibility_graph_models import (
    MAX_FAMILY_SIZE as MAX_GRAPH_FAMILY_SIZE,
)
from jacobian.math.number_theory._divisibility_graph_models import (
    MAX_GRAPH_EDGES,
    MAX_TOTAL_FAMILY_SIZE,
    DivisibilityIncidenceGraphResult,
)
from jacobian.math.number_theory._divisibility_profile_models import (
    MAX_FAMILY_SIZE as MAX_PROFILE_FAMILY_SIZE,
)
from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileResult,
    ProductDivisibilityProfileResult,
)
from jacobian.math.number_theory._integer_models import BooleanResult
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS
from jacobian.math.number_theory._modular_basic_models import (
    MAX_CRT_COMBINED_MODULUS,
    MAX_CRT_SIZE,
    MAX_MODULUS,
    ChineseRemainderResult,
    JacobiSymbolResult,
    QuadraticResiduesResult,
)
from jacobian.math.number_theory._modular_models import (
    _MAX_RESIDUE_ASSIGNMENTS,
    _MAX_RESIDUE_EXPONENT,
    ModularPolynomialResidueCount,
    ModularPolynomialResidueImageResult,
    ModularPolynomialResidueTableRow,
    ModularPolynomialResidueWitness,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory._periodic_kernel import (
    _ExecutionPlan,
    materialize_periodic_union,
    measure_periodic_union,
    require_admitted_periodic_source,
    require_materializable_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionMeasureResult,
    PeriodicCongruenceUnionProfileResult,
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory._preimage_kernel import (
    ksigma_preimages,
)
from jacobian.math.number_theory._preimage_kernel import (
    p_adic_interval_profile as admit_p_adic_interval_profile,
)
from jacobian.math.number_theory._preimage_models import (
    KSigmaPreimageResult,
    PAdicIntervalProfileResult,
    PAdicIntervalProfileRow,
)
from jacobian.math.number_theory._prime_coverage_models import (
    MAX_COVERAGE_RESULT_BYTES,
    MAX_COVERAGE_UPPER,
    MAX_COVERAGE_WORK,
    PrimeCoverageProfileResult,
    PrimeCoverageProfileRow,
    _coverage_result_upper_bound_bytes,
    _coverage_work_upper_bound,
)
from jacobian.math.number_theory._prime_models import PrimorialResult
from jacobian.math.number_theory._prime_shift_models import (
    PrimeShiftProfileResult,
    _PrimeShiftProfileExecutionPlan,
    require_prime_shift_profile_admission,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue
from jacobian.math.number_theory.modular_polynomials import (
    ModularPolynomialTerm,
    NormalizedModularPolynomialTerm,
)

__all__ = [
    "binomial_prime_valuation",
    "binomial_valuation_profile",
    "chinese_remainder",
    "contiguous_sum_profile",
    "divisibility_incidence_graph",
    "euler_totient",
    "factorial_valuation",
    "floor_square_root",
    "gcd_quotient_profile",
    "is_prime",
    "jacobi_symbol",
    "ksigma_preimage",
    "legendre_symbol",
    "mobius",
    "modular_inverse",
    "modular_polynomial_residue_assignments",
    "modular_polynomial_residue_image",
    "multiplicative_order",
    "next_prime",
    "nth_prime",
    "p_adic_interval_profile",
    "periodic_congruence_union_measure",
    "periodic_congruence_union_profile",
    "previous_prime",
    "prime_count",
    "prime_coverage_profile",
    "prime_shift_profile",
    "primorial",
    "product_divisibility_profile",
    "quadratic_residues",
]


def _integer(value: SupportsIndex | CanonicalInteger | IntegerValue) -> int:
    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    if isinstance(value, str):
        return parse_canonical_integer(value)
    return operator.index(value)


def contiguous_sum_profile(
    lower_bound: SupportsIndex | CanonicalInteger | IntegerValue,
    upper_bound: SupportsIndex | CanonicalInteger | IntegerValue,
) -> ContiguousSumProfileResult:
    """Count contiguous-sum representations on a closed positive interval."""

    execution = current_request_execution()
    started_at = execution.started_at if execution is not None else monotonic()
    admission = require_contiguous_sum_profile_admission(
        _integer(lower_bound),
        _integer(upper_bound),
        started_at=started_at,
    )
    return run_contiguous_sum_profile(admission, profile_started=started_at)


def is_prime(value: SupportsIndex | CanonicalInteger | IntegerValue) -> BooleanResult:
    """Return whether an integer is prime."""

    from sympy import isprime

    return BooleanResult(holds=bool(isprime(_integer(value))))


def next_prime(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the least prime strictly greater than an integer."""

    from sympy import nextprime

    return IntegerValue(value=format_canonical_integer(int(nextprime(_integer(value)))))


def previous_prime(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return the greatest prime strictly below an integer."""

    from sympy import prevprime

    return IntegerValue(value=format_canonical_integer(int(prevprime(_integer(value)))))


def prime_count(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the number of primes not exceeding a nonnegative integer."""

    from sympy import primepi

    return IntegerValue(value=format_canonical_integer(int(primepi(_integer(value)))))


def nth_prime(index: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the prime at one-based positive index."""

    from sympy import prime

    return IntegerValue(value=format_canonical_integer(int(prime(_integer(index)))))


def primorial(
    index: SupportsIndex | CanonicalInteger | IntegerValue,
) -> PrimorialResult:
    """Return the product of the first ``index`` primes."""

    from sympy import primorial as sympy_primorial

    return PrimorialResult(
        value=format_canonical_integer(int(sympy_primorial(_integer(index))))
    )


def euler_totient(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
) -> IntegerValue:
    """Return Euler's totient of a positive integer."""

    from sympy import totient

    return IntegerValue(value=format_canonical_integer(int(totient(_integer(value)))))


def mobius(value: SupportsIndex | CanonicalInteger | IntegerValue) -> IntegerValue:
    """Return the Mobius function of a positive integer."""

    from sympy import mobius as sympy_mobius

    return IntegerValue(
        value=format_canonical_integer(int(sympy_mobius(_integer(value))))
    )


def _simple_sieve(limit: int) -> tuple[int, ...]:
    if limit < 2:
        return ()
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0] = flags[1] = 0
    for candidate in range(2, math.isqrt(limit) + 1):
        if flags[candidate]:
            for composite in range(candidate * candidate, limit + 1, candidate):
                flags[composite] = 0
    return tuple(candidate for candidate in range(2, limit + 1) if flags[candidate])


def _segmented_sieve(
    lower_bound: int, upper_bound: int, base_primes: tuple[int, ...]
) -> bytearray:
    flags = bytearray(b"\x01") * (upper_bound - lower_bound + 1)
    for prime in base_primes:
        if prime * prime > upper_bound:
            break
        first = max(
            prime * prime,
            ((lower_bound + prime - 1) // prime) * prime,
        )
        if first <= upper_bound:
            flags[first - lower_bound :: prime] = b"\x00" * (
                (upper_bound - first) // prime + 1
            )
    return flags


def _compute_prime_shift_profile(
    plan: _PrimeShiftProfileExecutionPlan,
) -> PrimeShiftProfileResult:
    counts = [0] * (plan.upper_bound - plan.lower_bound + 1)
    base_primes = _simple_sieve(plan.base_limit)
    for power, candidate_lower, candidate_upper in plan.candidate_intervals:
        flags = _segmented_sieve(candidate_lower, candidate_upper, base_primes)
        for offset, candidate_is_prime in enumerate(flags):
            if candidate_is_prime:
                counts[candidate_lower + offset + power - plan.lower_bound] += 1
    return PrimeShiftProfileResult._from_kernel(
        lower_bound=plan.lower_bound,
        upper_bound=plan.upper_bound,
        counts=tuple(counts),
        plan=plan,
    )


def prime_shift_profile(
    lower_bound: SupportsIndex | IntegerValue,
    upper_bound: SupportsIndex | IntegerValue,
) -> PrimeShiftProfileResult:
    """Return the exact translated-prime profile on a closed interval."""

    lower = _integer(lower_bound)
    upper = _integer(upper_bound)
    return _compute_prime_shift_profile(
        require_prime_shift_profile_admission(lower, upper)
    )


def floor_square_root(value: int) -> FloorSquareRootResult:
    """Return the exact floor of the square root of a nonnegative integer."""

    if value < 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.floor_square_root.nonnegative_required",
            message="floor square root requires a nonnegative integer",
        )
    from sympy import integer_nthroot

    root, _ = integer_nthroot(value, 2)
    return FloorSquareRootResult(root=int(root))


def legendre_symbol(a: int, prime: int) -> LegendreSymbolResult:
    """Return the Legendre symbol ``(a / prime)`` for an odd prime."""

    from sympy import isprime
    from sympy import legendre_symbol as sympy_legendre_symbol

    if not 3 <= prime <= MAX_LEGENDRE_PRIME or not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.legendre_denominator_must_be_prime",
            message="Legendre denominator must be prime",
        )
    return LegendreSymbolResult(
        a=a,
        prime=prime,
        symbol=cast(Literal[-1, 0, 1], int(sympy_legendre_symbol(a, prime))),
    )


def factorial_valuation(n: int, base: int) -> FactorialValuationResult:
    """Return the largest exponent ``e`` for which ``base**e`` divides ``n!``."""

    return _factorial_valuation(admit_factorial_valuation(n, base))


def _factorial_valuation(
    admitted: _FactorialValuationInput,
) -> FactorialValuationResult:
    from sympy.ntheory import multiplicity_in_factorial

    return FactorialValuationResult.model_construct(
        n=format_canonical_integer(admitted.n),
        base=format_canonical_integer(admitted.base),
        valuation=format_canonical_integer(
            int(multiplicity_in_factorial(admitted.base, admitted.n))
        ),
    )


def binomial_prime_valuation(
    n: int, k: int, prime: int
) -> BinomialPrimeValuationResult:
    """Return the exponent of ``prime`` in one binomial coefficient."""

    return _binomial_prime_valuation(admit_binomial_prime_valuation(n, k, prime))


def _binomial_prime_valuation(
    admitted: _BinomialValuationInput,
) -> BinomialPrimeValuationResult:
    left = admitted.k
    right = admitted.n - admitted.k
    carries = 0
    carry = 0
    while left or right or carry:
        left, left_digit = divmod(left, admitted.prime)
        right, right_digit = divmod(right, admitted.prime)
        carry = int(left_digit + right_digit + carry >= admitted.prime)
        carries += carry
    return BinomialPrimeValuationResult.model_construct(
        n=format_canonical_integer(admitted.n),
        k=format_canonical_integer(admitted.k),
        prime=format_canonical_integer(admitted.prime),
        valuation=format_canonical_integer(carries),
    )


def _modular_integer(value: SupportsIndex | CanonicalInteger | IntegerValue) -> int:
    """Convert one native integer value without accepting request envelopes."""

    return _integer(value)


def _require_modulus(modulus: int) -> None:
    if type(modulus) is not int:
        raise TypeError("modulus must be an integer")
    if not 2 <= modulus <= MAX_MODULUS:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="number_theory.modulus_out_of_range",
            message=f"modulus must be between 2 and {MAX_MODULUS:,}",
        )


def _require_bounded_integer(value: int, *, location: tuple[str, ...]) -> None:
    if len(str(abs(value))) > MAX_INTEGER_DIGITS:
        raise OperationDomainValidationError(
            location=location,
            code="number_theory.integer_exceeds_digit_bound",
            message=f"integer must have at most {MAX_INTEGER_DIGITS} digits",
        )


def jacobi_symbol(
    a: SupportsIndex | CanonicalInteger | IntegerValue,
    n: SupportsIndex,
) -> JacobiSymbolResult:
    """Return the Jacobi symbol ``(a / n)`` for an odd positive denominator."""

    a_value = _modular_integer(a)
    n_value = operator.index(n)
    _require_bounded_integer(a_value, location=("a",))
    _require_modulus(n_value)
    if n_value % 2 == 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.jacobi_symbol_denominator_must_be_odd",
            message="Jacobi symbol denominator must be odd",
        )
    from sympy import jacobi_symbol as sympy_jacobi_symbol

    return JacobiSymbolResult(
        a=format_canonical_integer(a_value),
        n=n_value,
        jacobi=cast(
            Literal[-1, 0, 1],
            int(sympy_jacobi_symbol(a_value, n_value)),
        ),
    )


def _require_unit(value: int, modulus: int) -> None:
    _require_modulus(modulus)
    _require_bounded_integer(value, location=("value",))
    if math.gcd(value, modulus) != 1:
        raise OperationDomainValidationError(
            location=("value",),
            code="number_theory.value_must_be_coprime_to_the_modulus",
            message="value must be coprime to the modulus",
        )


def modular_inverse(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
    modulus: SupportsIndex,
) -> IntegerValue:
    """Return the least nonnegative inverse of a unit modulo ``modulus``."""

    value_integer = _modular_integer(value)
    modulus_integer = operator.index(modulus)
    _require_unit(value_integer, modulus_integer)
    return IntegerValue(
        value=format_canonical_integer(pow(value_integer, -1, modulus_integer))
    )


def multiplicative_order(
    value: SupportsIndex | CanonicalInteger | IntegerValue,
    modulus: SupportsIndex,
) -> IntegerValue:
    """Return the multiplicative order of a unit modulo ``modulus``."""

    value_integer = _modular_integer(value)
    modulus_integer = operator.index(modulus)
    _require_unit(value_integer, modulus_integer)
    from sympy import n_order

    return IntegerValue(
        value=format_canonical_integer(int(n_order(value_integer, modulus_integer)))
    )


def quadratic_residues(modulus: SupportsIndex) -> QuadraticResiduesResult:
    """Return all quadratic residues modulo one bounded modulus."""

    modulus_integer = operator.index(modulus)
    _require_modulus(modulus_integer)
    from sympy.ntheory.residue_ntheory import (
        quadratic_residues as sympy_quadratic_residues,
    )

    return QuadraticResiduesResult(
        residues=tuple(
            format_canonical_integer(int(value))
            for value in sympy_quadratic_residues(modulus_integer)
        )
    )


def _require_crt_admission(residues: tuple[int, ...], moduli: tuple[int, ...]) -> None:
    if not 1 <= len(residues) <= MAX_CRT_SIZE or not 1 <= len(moduli) <= MAX_CRT_SIZE:
        raise OperationDomainValidationError(
            location=("residues",),
            code="number_theory.congruence_system_size_out_of_range",
            message=f"congruence systems must contain between 1 and {MAX_CRT_SIZE} pairs",
        )
    if len(residues) != len(moduli):
        raise OperationDomainValidationError(
            location=("residues",),
            code="number_theory.residues_and_moduli_must_have_equal_length",
            message="residues and moduli must have equal length",
        )
    combined = 1
    for index, modulus in enumerate(moduli):
        if not 2 <= modulus <= MAX_MODULUS:
            raise OperationDomainValidationError(
                location=("moduli", index),
                code="number_theory.modulus_out_of_range",
                message=f"every modulus must be between 2 and {MAX_MODULUS:,}",
            )
        combined = combined // math.gcd(combined, modulus) * modulus
        if combined > MAX_CRT_COMBINED_MODULUS:
            raise OperationDomainValidationError(
                location=("moduli", index),
                code="number_theory.combined_modulus_exceeds_bound",
                message=(
                    "the system's combined modulus must have fewer than "
                    f"{len(str(MAX_CRT_COMBINED_MODULUS))} digits; split the "
                    "congruence system into narrower subsystems"
                ),
            )
    for index, (residue, modulus) in enumerate(zip(residues, moduli, strict=True)):
        if not 0 <= residue < modulus:
            raise OperationDomainValidationError(
                location=("residues", index),
                code="number_theory.every_residue_must_be_canonical_for_its_modulus",
                message="every residue must be canonical for its modulus",
            )
        for other_index in range(index):
            if (residue - residues[other_index]) % math.gcd(
                modulus, moduli[other_index]
            ):
                raise OperationDomainValidationError(
                    location=("residues", index),
                    code="number_theory.congruence_system_is_inconsistent",
                    message="congruence system is inconsistent",
                )


def chinese_remainder(
    residues: tuple[int, ...],
    moduli: tuple[int, ...],
) -> ChineseRemainderResult:
    """Solve a finite compatible system of integer congruences."""

    if not isinstance(residues, tuple) or not all(
        type(item) is int for item in residues
    ):
        raise TypeError("residues must be a tuple of integers")
    if not isinstance(moduli, tuple) or not all(type(item) is int for item in moduli):
        raise TypeError("moduli must be a tuple of integers")
    _require_crt_admission(residues, moduli)
    from sympy.ntheory.modular import solve_congruence

    result = solve_congruence(*zip(residues, moduli, strict=True), check=True)
    if result is None or result[0] is None:
        raise AssertionError("admitted congruence system was not solved")
    residue, modulus = result
    return ChineseRemainderResult(
        residue=format_canonical_integer(int(residue)),
        modulus=format_canonical_integer(int(modulus)),
    )


def _require_residue_image_admission(
    modulus: int,
    variables: tuple[ModularPolynomialVariable, ...],
    terms: tuple[ModularPolynomialTerm, ...],
) -> None:
    _require_modulus(modulus)
    if not all(
        isinstance(variable, ModularPolynomialVariable) for variable in variables
    ):
        raise TypeError("variables must contain ModularPolynomialVariable values")
    if not all(isinstance(term, ModularPolynomialTerm) for term in terms):
        raise TypeError("terms must contain ModularPolynomialTerm values")
    if not 1 <= len(variables) <= 6:
        raise OperationDomainValidationError(
            location=("variables",),
            code="number_theory.residue_variable_count_out_of_range",
            message="variable count must be between 1 and 6",
        )
    if len(terms) > 64:
        raise OperationDomainValidationError(
            location=("terms",),
            code="number_theory.residue_term_count_exceeds_bound",
            message="residue polynomial may contain at most 64 terms",
        )
    variable_names = [variable.name for variable in variables]
    if len(variable_names) != len(set(variable_names)):
        raise OperationDomainValidationError(
            location=("variables",),
            code="number_theory.polynomial_variable_names_must_be_unique",
            message="polynomial variable names must be unique",
        )
    if any(
        variable.residues != tuple(sorted(set(variable.residues)))
        or any(residue < 0 or residue >= modulus for residue in variable.residues)
        for variable in variables
    ):
        raise OperationDomainValidationError(
            location=("variables",),
            code="number_theory.variable_residues_must_be_canonical",
            message="variable residues must be canonical and less than the modulus",
        )
    assignment_count = math.prod(len(variable.residues) for variable in variables)
    if assignment_count > _MAX_RESIDUE_ASSIGNMENTS:
        raise OperationDomainValidationError(
            location=("variables",),
            code="number_theory.residue_assignment_count_exceeds_bound",
            message=(
                "declared residue domains exceed the "
                f"{_MAX_RESIDUE_ASSIGNMENTS:,}-assignment bound"
            ),
        )
    if any(len(term.exponents) != len(variables) for term in terms):
        raise OperationDomainValidationError(
            location=("terms",),
            code="number_theory.every_term_exponent_vector_must_match_the_variable_count",
            message="every term exponent vector must match the variable count",
        )
    if any(
        len(term.coefficient) > MAX_INTEGER_DIGITS
        or any(
            exponent < 0 or exponent > _MAX_RESIDUE_EXPONENT
            for exponent in term.exponents
        )
        for term in terms
    ):
        raise OperationDomainValidationError(
            location=("terms",),
            code="number_theory.term_outside_residue_image_admission",
            message="term coefficient or exponents exceed the residue-image admission",
        )
    exponent_vectors = [term.exponents for term in terms]
    if exponent_vectors != sorted(set(exponent_vectors)):
        raise OperationDomainValidationError(
            location=("terms",),
            code="number_theory.term_exponent_vectors_must_be_unique_and_lexicographically_increasing",
            message="term exponent vectors must be unique and lexicographically increasing",
        )
    if any(int(term.coefficient) % modulus == 0 for term in terms):
        raise OperationDomainValidationError(
            location=("terms",),
            code="number_theory.sparse_polynomial_terms_must_have_nonzero_coefficient_modulo_m",
            message="sparse polynomial terms must have nonzero coefficient modulo m",
        )


def _evaluate_modular_polynomial(
    terms: tuple[NormalizedModularPolynomialTerm, ...],
    assignment: tuple[int, ...],
    modulus: int,
) -> int:
    value = 0
    for term in terms:
        monomial = term.coefficient
        for coordinate, exponent in zip(assignment, term.exponents, strict=True):
            monomial = monomial * pow(coordinate, exponent, modulus) % modulus
        value = (value + monomial) % modulus
    return value


def _residue_image(
    modulus: int,
    variables: tuple[ModularPolynomialVariable, ...],
    terms: tuple[ModularPolynomialTerm, ...],
    *,
    include_table: bool,
) -> ModularPolynomialResidueImageResult:
    _require_residue_image_admission(modulus, variables, terms)
    normalized_terms = tuple(
        NormalizedModularPolynomialTerm(
            coefficient=int(term.coefficient) % modulus,
            exponents=term.exponents,
        )
        for term in terms
    )
    counts: dict[int, int] = {}
    first_assignments: dict[int, tuple[int, ...]] = {}
    table: list[ModularPolynomialResidueTableRow] | None = [] if include_table else None
    for assignment in product(*(variable.residues for variable in variables)):
        residue = _evaluate_modular_polynomial(normalized_terms, assignment, modulus)
        if table is not None:
            table.append(
                ModularPolynomialResidueTableRow(assignment=assignment, residue=residue)
            )
        counts[residue] = counts.get(residue, 0) + 1
        first_assignments.setdefault(residue, assignment)
    image = tuple(sorted(counts))
    return ModularPolynomialResidueImageResult._from_kernel(
        modulus=modulus,
        variable_order=tuple(variable.name for variable in variables),
        domains=tuple(variable.residues for variable in variables),
        normalized_terms=normalized_terms,
        total_assignments=math.prod(len(variable.residues) for variable in variables),
        image=image,
        residue_counts=tuple(
            ModularPolynomialResidueCount(residue=residue, count=counts[residue])
            for residue in image
        ),
        witnesses=tuple(
            ModularPolynomialResidueWitness(
                residue=residue, assignment=first_assignments[residue]
            )
            for residue in image
        ),
        table=tuple(table) if table is not None else None,
    )


def modular_polynomial_residue_image(
    modulus: int,
    variables: tuple[ModularPolynomialVariable, ...],
    terms: tuple[ModularPolynomialTerm, ...],
) -> ModularPolynomialResidueImageResult:
    """Return the exact image of a sparse polynomial on finite residue domains."""

    return _residue_image(modulus, variables, terms, include_table=False)


def modular_polynomial_residue_assignments(
    modulus: int,
    variables: tuple[ModularPolynomialVariable, ...],
    terms: tuple[ModularPolynomialTerm, ...],
) -> ModularPolynomialResidueImageResult:
    """Return the exact assignment-to-residue table and image summary."""

    return _residue_image(modulus, variables, terms, include_table=True)


def binomial_valuation_profile(n: int, prime: int) -> BinomialValuationProfileResult:
    """Return ``v_prime(binomial(n, k))`` for every ``0 <= k <= n``."""
    if type(n) is not int or n < 0:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.binomial_profile_n_invalid",
            message="n must be a nonnegative integer",
        )
    if type(prime) is not int or not 2 <= prime <= _MAX_SAFE_JSON_INTEGER:
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.binomial_profile_prime_invalid",
            message="prime must be an integer between 2 and the safe JSON bound",
        )
    if n + 1 > _MAX_BINOMIAL_ROWS_FROM_OUTPUT or (
        _binomial_result_upper_bound_bytes(n, prime) > MAX_BINOMIAL_PROFILE_RESULT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.binomial_profile_output_exceeded",
            message="valuation profile exceeds the canonical output budget",
        )
    digit_work = (n + 1) * max(1, _base_digit_count(n, prime))
    if digit_work > MAX_BINOMIAL_DIGIT_WORK:
        raise OperationDomainValidationError(
            location=("n",),
            code="number_theory.binomial_profile_work_exceeded",
            message=(
                "valuation profile exceeds the digitwise work budget of "
                f"{MAX_BINOMIAL_DIGIT_WORK} steps"
            ),
        )
    from sympy import isprime

    if not isprime(prime):
        raise OperationDomainValidationError(
            location=("prime",),
            code="number_theory.binomial_profile_base_not_prime",
            message="prime must be a prime number",
        )
    rows = []
    for k in range(n + 1):
        left = k
        right = n - k
        carries = 0
        carry = 0
        while left > 0 or right > 0 or carry > 0:
            total = left % prime + right % prime + carry
            if total >= prime:
                carries += 1
                carry = 1
            else:
                carry = 0
            left //= prime
            right //= prime
        rows.append(BinomialValuationProfileRow(k=k, valuation=carries))
    return BinomialValuationProfileResult(n=n, prime=prime, rows=rows)


def _require_positive_family(
    family: tuple[int, ...], *, location: tuple[str, ...], max_size: int
) -> None:
    if not isinstance(family, tuple) or any(type(value) is not int for value in family):
        raise TypeError("families must be tuples of integers")
    if len(family) > max_size:
        raise OperationDomainValidationError(
            location=location,
            code="number_theory.family_size_exceeded",
            message=f"families may contain at most {max_size} values",
        )
    if any(value <= 0 for value in family):
        raise OperationDomainValidationError(
            location=location,
            code="number_theory.non_positive_family",
            message="family values must be positive integers",
        )


def divisibility_incidence_graph(
    left_family: tuple[int, ...], right_family: tuple[int, ...]
) -> DivisibilityIncidenceGraphResult:
    """Build the bipartite graph whose edges represent divisibility."""
    _require_positive_family(
        left_family, location=("left_family",), max_size=MAX_GRAPH_FAMILY_SIZE
    )
    _require_positive_family(
        right_family, location=("right_family",), max_size=MAX_GRAPH_FAMILY_SIZE
    )
    if len(set(left_family)) != len(left_family):
        raise OperationDomainValidationError(
            location=("left_family",),
            code="number_theory.duplicate_left_family",
            message="left_family values must be unique",
        )
    if len(set(right_family)) != len(right_family):
        raise OperationDomainValidationError(
            location=("right_family",),
            code="number_theory.duplicate_right_family",
            message="right_family values must be unique",
        )
    if len(left_family) + len(right_family) > MAX_TOTAL_FAMILY_SIZE:
        raise OperationDomainValidationError(
            location=("left_family", "right_family"),
            code="number_theory.graph_vertex_budget",
            message=f"families must contain at most {MAX_TOTAL_FAMILY_SIZE} total values",
        )
    if len(left_family) * len(right_family) > MAX_GRAPH_EDGES:
        raise OperationDomainValidationError(
            location=("left_family", "right_family"),
            code="number_theory.graph_edge_budget",
            message=f"the incidence graph may contain at most {MAX_GRAPH_EDGES} edges",
        )
    vertices = tuple(
        [f"L{index}" for index in range(len(left_family))]
        + [f"R{index}" for index in range(len(right_family))]
    )
    edges = tuple(
        (f"L{left_index}", f"R{right_index}")
        for left_index, left in enumerate(left_family)
        for right_index, right in enumerate(right_family)
        if right % left == 0
    )
    return DivisibilityIncidenceGraphResult(
        left_family=tuple(format_canonical_integer(value) for value in left_family),
        right_family=tuple(format_canonical_integer(value) for value in right_family),
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
    )


def gcd_quotient_profile(
    elements: tuple[int, ...],
) -> GcdQuotientProfileResult:
    """Return the pairwise normalized gcd quotient matrix."""
    _require_positive_family(
        elements, location=("elements",), max_size=MAX_PROFILE_FAMILY_SIZE
    )
    quotients: list[tuple[CanonicalRational, ...]] = []
    for left in elements:
        row = tuple(
            CanonicalRational.from_fraction(
                Fraction(math.gcd(left, right), max(left, right))
            )
            for right in elements
        )
        quotients.append(row)
    return GcdQuotientProfileResult(
        elements=tuple(format_canonical_integer(value) for value in elements),
        quotients=tuple(quotients),
    )


def product_divisibility_profile(
    elements: tuple[int, ...],
) -> ProductDivisibilityProfileResult:
    """Return whether each ordered pair product divides the family product."""
    _require_positive_family(
        elements, location=("elements",), max_size=MAX_PROFILE_FAMILY_SIZE
    )
    total_product = math.prod(elements)
    matrix = tuple(
        tuple(total_product % (left * right) == 0 for right in elements)
        for left in elements
    )
    return ProductDivisibilityProfileResult(
        elements=tuple(format_canonical_integer(value) for value in elements),
        divisibility_matrix=matrix,
    )


def _simple_prime_sieve(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for value in range(2, math.isqrt(limit) + 1):
        if is_prime[value]:
            for multiple in range(value * value, limit + 1, value):
                is_prime[multiple] = 0
    return [value for value in range(2, limit + 1) if is_prime[value]]


def _segmented_distinct_prime_counts(lower_bound: int, upper_bound: int) -> list[int]:
    width = upper_bound - lower_bound + 1
    residuals = list(range(lower_bound, upper_bound + 1))
    counts = bytearray(width)
    for prime in _simple_prime_sieve(math.isqrt(upper_bound)):
        first = max(prime * prime, ((lower_bound + prime - 1) // prime) * prime)
        for multiple in range(first, upper_bound + 1, prime):
            index = multiple - lower_bound
            residual = residuals[index]
            if residual % prime:
                continue
            counts[index] += 1
            while residual % prime == 0:
                residual //= prime
            residuals[index] = residual
    for index, residual in enumerate(residuals):
        if residual > 1:
            counts[index] += 1
    return list(counts)


def prime_coverage_profile(
    lower_bound: int, upper_bound: int
) -> PrimeCoverageProfileResult:
    """Return the distinct-prime-factor count for every integer in an interval."""
    if (
        type(lower_bound) is not int
        or type(upper_bound) is not int
        or lower_bound < 1
        or upper_bound < 1
        or lower_bound > MAX_COVERAGE_UPPER
        or upper_bound > MAX_COVERAGE_UPPER
    ):
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.prime_coverage_interval_invalid",
            message="interval bounds must be positive safe JSON integers",
        )
    if upper_bound < lower_bound:
        raise OperationDomainValidationError(
            location=("upper_bound",),
            code="number_theory.prime_coverage_interval_reversed",
            message="upper_bound must be >= lower_bound",
        )
    if (
        _coverage_result_upper_bound_bytes(lower_bound, upper_bound)
        > MAX_COVERAGE_RESULT_BYTES
    ):
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.prime_coverage_output_exceeded",
            message="interval result exceeds the canonical output budget of "
            f"{MAX_COVERAGE_RESULT_BYTES} bytes",
        )
    work = _coverage_work_upper_bound(lower_bound, upper_bound)
    if work > MAX_COVERAGE_WORK:
        raise OperationDomainValidationError(
            location=("lower_bound", "upper_bound"),
            code="number_theory.prime_coverage_work_exceeded",
            message="interval exceeds the segmented prime-coverage work budget of "
            f"{MAX_COVERAGE_WORK} steps",
        )
    counts = _segmented_distinct_prime_counts(lower_bound, upper_bound)
    return PrimeCoverageProfileResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        rows=[
            PrimeCoverageProfileRow(n=value, distinct_prime_count=count)
            for value, count in zip(
                range(lower_bound, upper_bound + 1), counts, strict=True
            )
        ],
    )


def _admit_periodic_source(
    source: PeriodicCongruenceUnionSource, *, materializable: bool = False
) -> _ExecutionPlan:
    try:
        plan = require_admitted_periodic_source(source)
        if materializable:
            return require_materializable_periodic_source(source, plan)
        return plan
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="number_theory.periodic.execution_bound",
            message=str(exc),
        ) from exc


def _periodic_measure_values(
    period: int,
    occupied_count: int,
) -> tuple[str, str, CanonicalRational]:
    return (
        format_canonical_integer(period),
        format_canonical_integer(occupied_count),
        CanonicalRational.from_fraction(Fraction(occupied_count, period)),
    )


def periodic_congruence_union_measure(
    source: PeriodicCongruenceUnionSource,
) -> PeriodicCongruenceUnionMeasureResult:
    """Measure one canonical finite periodic congruence union exactly."""

    plan = _admit_periodic_source(source)
    period, occupied_count, density = _periodic_measure_values(
        plan.common_period, measure_periodic_union(source, plan)
    )
    return PeriodicCongruenceUnionMeasureResult._from_kernel(
        source=source,
        common_period=period,
        occupied_count=occupied_count,
        density=density,
    )


def periodic_congruence_union_profile(
    source: PeriodicCongruenceUnionSource,
) -> PeriodicCongruenceUnionProfileResult:
    """Materialize one canonical finite periodic congruence union exactly."""

    plan = _admit_periodic_source(source, materializable=True)
    residues = materialize_periodic_union(source, plan)
    period, occupied_count, density = _periodic_measure_values(
        plan.common_period, len(residues)
    )
    return PeriodicCongruenceUnionProfileResult._from_profile_kernel(
        source=source,
        common_period=period,
        occupied_count=occupied_count,
        density=density,
        occupied_residues=tuple(
            format_canonical_integer(residue) for residue in residues
        ),
    )


def ksigma_preimage(k: int, target: int) -> KSigmaPreimageResult:
    """Return the complete positive fiber of ``n -> k * sigma(n)``."""

    preimages = ksigma_preimages(k, target)
    return KSigmaPreimageResult._from_kernel(
        k=k,
        target_value=format_canonical_integer(target),
        preimages=preimages,
    )


def p_adic_interval_profile(
    start: int,
    length: int,
    prime: int,
) -> PAdicIntervalProfileResult:
    """Return the exact valuation histogram on ``[start + 1, start + length]``."""

    plan = admit_p_adic_interval_profile(start, length, prime)
    return PAdicIntervalProfileResult._from_kernel(
        start=format_canonical_integer(start),
        length=format_canonical_integer(length),
        prime=format_canonical_integer(prime),
        rows=tuple(
            PAdicIntervalProfileRow(
                valuation=valuation,
                count=format_canonical_integer(count),
            )
            for valuation, count in plan.rows
        ),
        total_valuation=plan.total_valuation,
        maximum_valuation=plan.maximum_valuation,
    )
