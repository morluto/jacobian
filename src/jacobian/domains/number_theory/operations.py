"""Exact number-theory operations backed by maintained SymPy and stdlib APIs.

Each function takes a validated request model and returns a validated result
model.  No handrolled algorithms are used where SymPy or the Python standard
library provides a maintained implementation (``math.gcd``, ``math.lcm``,
``math.isqrt``, ``math.prod``, ``pow``, ``sympy.gcdex``, ``sympy.divisors``,
``sympy.factorint``, ``sympy.multiplicity``, ``sympy.divisor_count``,
``sympy.divisor_sigma``, ``sympy.isprime``, ``sympy.nextprime``,
``sympy.prevprime``, ``sympy.totient``, ``sympy.mobius``, ``sympy.primepi``,
``sympy.prime``, ``sympy.primorial``, ``sympy.integer_nthroot``,
``sympy.n_order``, ``sympy.ntheory.residue_ntheory.quadratic_residues``,
``sympy.ntheory.modular.solve_congruence``).
"""

from __future__ import annotations

import math
from typing import Literal, cast

from jacobian.contracts.number_theory import (
    ArithmeticFunctionRequest,
    BooleanResult,
    ChineseRemainderRequest,
    ChineseRemainderResult,
    DivisibilityRequest,
    DivisorListResult,
    ExtendedGcdResult,
    FactorialValuationRequest,
    FactorialValuationResult,
    FactorizationRequest,
    FloorSquareRootRequest,
    FloorSquareRootResult,
    IntegerPairRequest,
    IntegerValueRequest,
    IntegerValueResult,
    JacobiSymbolRequest,
    JacobiSymbolResult,
    LegendreSymbolRequest,
    LegendreSymbolResult,
    ModularPolynomialResidueCount,
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularPolynomialResidueTableRow,
    ModularPolynomialResidueWitness,
    ModularValueRequest,
    ModulusRequest,
    NonnegativeIntegerRequest,
    NormalizedModularPolynomialTerm,
    PositiveIntegerRequest,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimeFactorizationResult,
    PrimePower,
    QuadraticResiduesResult,
    ValuationRequest,
)


def compute_gcd(request: IntegerPairRequest) -> IntegerValueResult:
    """Compute gcd(a, b) using ``math.gcd``."""
    pair = request
    return IntegerValueResult(value=str(math.gcd(int(pair.left), int(pair.right))))


def compute_lcm(request: IntegerPairRequest) -> IntegerValueResult:
    """Compute lcm(a, b) using ``math.lcm``."""
    pair = request
    return IntegerValueResult(value=str(math.lcm(int(pair.left), int(pair.right))))


def compute_jacobi_symbol(request: JacobiSymbolRequest) -> JacobiSymbolResult:
    """Compute the Jacobi symbol using ``sympy.jacobi_symbol``."""
    from sympy import jacobi_symbol

    value = request
    symbol = cast(Literal[-1, 0, 1], int(jacobi_symbol(int(value.a), value.n)))
    return JacobiSymbolResult(
        a=value.a,
        n=value.n,
        jacobi=symbol,
    )


def compute_extended_gcd(request: IntegerPairRequest) -> ExtendedGcdResult:
    """Compute gcd and Bezout coefficients using ``sympy.gcdex``."""
    from sympy import gcdex

    pair = request
    x, y, divisor = gcdex(int(pair.left), int(pair.right))
    return ExtendedGcdResult(
        gcd=str(int(divisor)),
        left_coefficient=str(int(x)),
        right_coefficient=str(int(y)),
    )


def enumerate_divisors(request: FactorizationRequest) -> DivisorListResult:
    """Enumerate all positive divisors using ``sympy.divisors``."""
    from sympy import divisors as sympy_divisors

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(
        divisors=tuple(str(d) for d in sympy_divisors(abs(value))),
    )


def enumerate_proper_divisors(request: FactorizationRequest) -> DivisorListResult:
    """Enumerate all positive proper divisors using ``sympy.divisors(proper=True)``."""
    from sympy import divisors as sympy_divisors

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(
        divisors=tuple(str(d) for d in sympy_divisors(abs(value), proper=True)),
    )


def factorize_primes(request: FactorizationRequest) -> PrimeFactorizationResult:
    """Compute the complete prime-power factorization using ``sympy.factorint``."""
    from sympy import factorint

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has no finite prime factorization")
    factors = tuple(
        PrimePower(prime=str(prime), power=int(power))
        for prime, power in sorted(factorint(abs(value)).items())
    )
    return PrimeFactorizationResult(factors=factors)


def decide_powerful(request: PowerfulNumberRequest) -> PowerfulNumberResult:
    """Decide whether every prime exponent is at least two."""
    from sympy import factorint

    value = int(request.value)
    factor_items = sorted(factorint(value).items())
    factors = tuple(
        PrimePower(prime=str(prime), power=int(power)) for prime, power in factor_items
    )
    violating_primes = tuple(
        str(prime) for prime, power in factor_items if int(power) < 2
    )
    return PowerfulNumberResult(
        semantics_version="powerful-number.prime-exponents-at-least-two.v1",
        is_powerful=not violating_primes,
        factors=factors,
        violating_primes=violating_primes,
    )


def compute_valuation(request: ValuationRequest) -> IntegerValueResult:
    """Compute the p-adic valuation using ``sympy.multiplicity``."""
    from sympy import isprime, multiplicity

    req = request
    value, prime = int(req.value), int(req.prime)
    if value == 0 or abs(prime) < 2 or not isprime(abs(prime)):
        raise ValueError("valuation requires nonzero value and prime absolute base")
    return IntegerValueResult(value=str(multiplicity(abs(prime), abs(value))))


def compute_divisor_count(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Count positive divisors using ``sympy.divisor_count``."""
    from sympy import divisor_count

    n = request.n
    return IntegerValueResult(value=str(int(divisor_count(n))))


def compute_divisor_sum(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Sum positive divisors using ``sympy.divisor_sigma``."""
    from sympy import divisor_sigma

    n = request.n
    return IntegerValueResult(value=str(int(divisor_sigma(n))))


def compute_aliquot_sum(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Sum positive proper divisors using ``sympy.divisor_sigma(n) - n``."""
    from sympy import divisor_sigma

    n = request.n
    return IntegerValueResult(value=str(int(divisor_sigma(n)) - n))


def decide_coprime(request: IntegerPairRequest) -> BooleanResult:
    """Decide coprimality using ``math.gcd``."""
    pair = request
    return BooleanResult(holds=math.gcd(int(pair.left), int(pair.right)) == 1)


def decide_divides(request: DivisibilityRequest) -> BooleanResult:
    """Decide divisibility using the remainder operator."""
    req = request
    divisor, dividend = int(req.divisor), int(req.dividend)
    if divisor == 0:
        raise ValueError("divisor must be nonzero")
    return BooleanResult(holds=dividend % divisor == 0)


def decide_even(request: IntegerValueRequest) -> BooleanResult:
    """Decide evenness using the remainder operator."""
    value = int(request.value)
    return BooleanResult(holds=value % 2 == 0)


def decide_odd(request: IntegerValueRequest) -> BooleanResult:
    """Decide oddness using the remainder operator."""
    value = int(request.value)
    return BooleanResult(holds=value % 2 != 0)


def decide_square(request: NonnegativeIntegerRequest) -> BooleanResult:
    """Decide perfect square using ``math.isqrt``."""
    n = request.n
    return BooleanResult(holds=math.isqrt(n) ** 2 == n)


def decide_squarefree(request: ArithmeticFunctionRequest) -> BooleanResult:
    """Decide squarefreeness using ``sympy.factorint``."""
    from sympy import factorint

    n = request.n
    if n == 0:
        return BooleanResult(holds=False)
    return BooleanResult(holds=all(power == 1 for power in factorint(n).values()))


def decide_perfect(request: NonnegativeIntegerRequest) -> BooleanResult:
    """Decide perfect number using ``sympy.divisor_sigma``."""
    from sympy import divisor_sigma

    n = request.n
    return BooleanResult(holds=bool(n and int(divisor_sigma(n)) - n == n))


def decide_abundant(request: NonnegativeIntegerRequest) -> BooleanResult:
    """Decide abundant number using ``sympy.divisor_sigma``."""
    from sympy import divisor_sigma

    n = request.n
    return BooleanResult(holds=bool(n and int(divisor_sigma(n)) - n > n))


def decide_deficient(request: NonnegativeIntegerRequest) -> BooleanResult:
    """Decide deficient number using ``sympy.divisor_sigma``."""
    from sympy import divisor_sigma

    n = request.n
    return BooleanResult(holds=bool(n and int(divisor_sigma(n)) - n < n))


def decide_prime(request: IntegerValueRequest) -> BooleanResult:
    """Decide primality using ``sympy.isprime``."""
    from sympy import isprime

    value = int(request.value)
    return BooleanResult(holds=bool(isprime(value)))


def compute_next_prime(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    """Compute the next prime using ``sympy.nextprime``."""
    from sympy import nextprime

    n = request.n
    return IntegerValueResult(value=str(int(nextprime(n))))


def compute_previous_prime(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    """Compute the previous prime using ``sympy.prevprime``."""
    from sympy import prevprime

    n = request.n
    if n <= 2:
        raise ValueError("previous prime requires n greater than 2")
    return IntegerValueResult(value=str(int(prevprime(n))))


def compute_prime_count(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    """Count primes through n using ``sympy.primepi``."""
    from sympy import primepi

    n = request.n
    return IntegerValueResult(value=str(int(primepi(n))))


def compute_floor_square_root(request: FloorSquareRootRequest) -> FloorSquareRootResult:
    """Return the exact floor square-root."""
    from sympy import integer_nthroot

    n = request.n
    root, _ = integer_nthroot(n, 2)
    return FloorSquareRootResult(root=int(root))


def compute_legendre_symbol(request: LegendreSymbolRequest) -> LegendreSymbolResult:
    """Compute ``(a / p)`` after checking that ``p`` is prime."""
    from sympy import isprime, legendre_symbol

    value = request
    if not isprime(value.prime):
        raise ValueError("Legendre denominator must be prime")
    return LegendreSymbolResult(
        a=value.a,
        prime=value.prime,
        symbol=cast(Literal[-1, 0, 1], int(legendre_symbol(value.a, value.prime))),
    )


def compute_factorial_valuation(
    request: FactorialValuationRequest,
) -> FactorialValuationResult:
    """Compute the valuation of ``n!`` at an arbitrary composite base."""
    from sympy.ntheory import multiplicity_in_factorial

    value = request
    return FactorialValuationResult(
        n=value.n,
        base=value.base,
        valuation=int(multiplicity_in_factorial(value.base, value.n)),
    )


def compute_nth_prime(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Compute the nth prime using ``sympy.prime``."""
    from sympy import prime

    n = request.n
    return IntegerValueResult(value=str(int(prime(n))))


def compute_primorial(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    """Compute the product of the first n primes using ``sympy.primorial``."""
    from sympy import primorial

    n = request.n
    return IntegerValueResult(value=str(int(primorial(n))))


def compute_euler_totient(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Count coprime residues using ``sympy.totient``."""
    from sympy import totient

    n = request.n
    return IntegerValueResult(value=str(int(totient(n))))


def compute_mobius(request: PositiveIntegerRequest) -> IntegerValueResult:
    """Compute the Mobius function using ``sympy.mobius``."""
    from sympy import mobius

    n = request.n
    return IntegerValueResult(value=str(int(mobius(n))))


def compute_radical(request: ArithmeticFunctionRequest) -> IntegerValueResult:
    """Compute the product of distinct prime divisors using ``sympy.factorint``."""
    from sympy import factorint

    n = request.n
    return IntegerValueResult(value=str(math.prod(factorint(n))))


def compute_modular_inverse(request: ModularValueRequest) -> IntegerValueResult:
    """Compute the modular inverse using ``pow(value, -1, modulus)``."""
    req = request
    value, modulus = int(req.value), req.modulus
    return IntegerValueResult(value=str(pow(value, -1, modulus)))


def compute_multiplicative_order(request: ModularValueRequest) -> IntegerValueResult:
    """Compute the multiplicative order using ``sympy.n_order``."""
    from sympy import n_order

    req = request
    value, modulus = int(req.value), req.modulus
    if math.gcd(value, modulus) != 1:
        raise ValueError("multiplicative order requires coprime value and modulus")
    return IntegerValueResult(value=str(int(n_order(value, modulus))))


def enumerate_quadratic_residues(request: ModulusRequest) -> QuadraticResiduesResult:
    """Enumerate all quadratic residues using ``sympy.quadratic_residues``."""
    from sympy.ntheory.residue_ntheory import quadratic_residues

    modulus = request.modulus
    return QuadraticResiduesResult(
        residues=tuple(str(int(r)) for r in quadratic_residues(modulus)),
    )


def compute_modular_polynomial_residue_image(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    """Enumerate one sparse polynomial over its declared finite residue domains."""
    return _compute_modular_polynomial_residue_image(request, include_table=False)


def materialize_modular_polynomial_residue_assignments(
    request: ModularPolynomialResidueImageRequest,
) -> ModularPolynomialResidueImageResult:
    """Retain the complete assignment-to-residue ledger for explicit evidence."""
    return _compute_modular_polynomial_residue_image(request, include_table=True)


def _compute_modular_polynomial_residue_image(
    request: ModularPolynomialResidueImageRequest,
    *,
    include_table: bool,
) -> ModularPolynomialResidueImageResult:
    """Build the shared summary, optionally retaining its bulk ledger."""
    from itertools import product

    polynomial = request
    normalized_terms = tuple(
        NormalizedModularPolynomialTerm(
            coefficient=int(term.coefficient) % polynomial.modulus,
            exponents=term.exponents,
        )
        for term in polynomial.terms
    )
    table: list[ModularPolynomialResidueTableRow] = []
    counts: dict[int, int] = {}
    first_assignments: dict[int, tuple[int, ...]] = {}
    for assignment in product(
        *(variable.residues for variable in polynomial.variables)
    ):
        residue = _evaluate_modular_polynomial(
            normalized_terms,
            assignment,
            polynomial.modulus,
        )
        table.append(
            ModularPolynomialResidueTableRow(
                assignment=assignment,
                residue=residue,
            )
        )
        counts[residue] = counts.get(residue, 0) + 1
        first_assignments.setdefault(residue, assignment)
    image = tuple(sorted(counts))
    return ModularPolynomialResidueImageResult(
        semantics_version="modular-polynomial-residue-image.v1",
        modulus=polynomial.modulus,
        variable_order=tuple(variable.name for variable in polynomial.variables),
        domains=tuple(variable.residues for variable in polynomial.variables),
        normalized_terms=normalized_terms,
        enumeration_scope="COMPLETE_DECLARED_CARTESIAN_PRODUCT",
        total_assignments=len(table),
        image=image,
        residue_counts=tuple(
            ModularPolynomialResidueCount(
                residue=residue,
                count=counts[residue],
            )
            for residue in image
        ),
        witnesses=tuple(
            ModularPolynomialResidueWitness(
                residue=residue,
                assignment=first_assignments[residue],
            )
            for residue in image
        ),
        table=tuple(table) if include_table else None,
    )


def _evaluate_modular_polynomial(
    terms: tuple[NormalizedModularPolynomialTerm, ...],
    assignment: tuple[int, ...],
    modulus: int,
) -> int:
    value = 0
    for term in terms:
        monomial = term.coefficient
        for coordinate, exponent in zip(
            assignment,
            term.exponents,
            strict=True,
        ):
            monomial = monomial * pow(coordinate, exponent, modulus) % modulus
        value = (value + monomial) % modulus
    return value


def solve_chinese_remainder(request: ChineseRemainderRequest) -> ChineseRemainderResult:
    """Solve a congruence system using ``sympy.solve_congruence``."""
    from sympy.ntheory.modular import solve_congruence

    system = request
    result = solve_congruence(
        *zip(system.residues, system.moduli, strict=True),
        check=True,
    )
    if result is None or result[0] is None:
        raise ValueError("congruence system is inconsistent")
    residue, modulus = result
    return ChineseRemainderResult(
        residue=str(int(residue)),
        modulus=str(int(modulus)),
    )
