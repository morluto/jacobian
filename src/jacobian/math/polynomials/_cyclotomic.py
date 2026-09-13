"""Bounded exact cyclotomic polynomials through SymPy's ZZ backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import NoReturn

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    request_checkpoint,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._models import IntegerPolynomial
from jacobian.math.polynomials.values import MAX_POLYNOMIAL_TERMS

MAX_CYCLOTOMIC_INDEX = 100_000
MAX_CYCLOTOMIC_DEGREE = MAX_POLYNOMIAL_TERMS - 1
MAX_CYCLOTOMIC_DIVISORS = 512
MAX_CYCLOTOMIC_FACTOR_WORK = 2_000_000
MAX_CYCLOTOMIC_CONSTRUCTION_WORK = 16_000_000
MAX_CYCLOTOMIC_INTERMEDIATE_BITS = 16_384
MAX_CYCLOTOMIC_INTERMEDIATE_WORK = MAX_CYCLOTOMIC_CONSTRUCTION_WORK
MAX_CYCLOTOMIC_COEFFICIENT_DIGITS = 4_096
MAX_CYCLOTOMIC_OUTPUT_DIGITS = 8_000_000


@dataclass(frozen=True, slots=True)
class _CyclotomicAdmission:
    """One immutable preflight plan shared by backend and result construction."""

    degree: int
    factorization_work: int
    divisor_count: int
    intermediate_work: int
    coefficient_digits: int
    output_digits: int


class CyclotomicRequest(StrictModel):
    index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)


class CyclotomicResult(StrictModel):
    source_index: StrictInt = Field(ge=1, le=MAX_CYCLOTOMIC_INDEX)
    totient: StrictInt = Field(ge=1, le=MAX_POLYNOMIAL_TERMS - 1)
    polynomial: IntegerPolynomial

    @model_validator(mode="after")
    def require_degree_and_monic_shape(self) -> CyclotomicResult:
        """Check only the producer's structural shape on deserialization."""

        if self.totient != len(self.polynomial.coefficients) - 1:
            raise PydanticCustomError(
                "polynomial.cyclotomic.degree_shape",
                "totient must equal the polynomial degree",
            )
        if self.polynomial.coefficients[0] != 1:
            raise PydanticCustomError(
                "polynomial.cyclotomic.monic_shape",
                "cyclotomic polynomial must be monic",
            )
        expected_constant = -1 if self.source_index == 1 else 1
        if self.polynomial.coefficients[-1] != expected_constant:
            raise PydanticCustomError(
                "polynomial.cyclotomic.constant_shape",
                "cyclotomic polynomial has an incompatible constant term",
            )
        return self


def _backend_error(reason: BackendFailureReason, exc: BaseException) -> NoReturn:
    raise OperationBackendError(reason) from exc


def _require_typed_request(request: object) -> CyclotomicRequest:
    if not isinstance(request, CyclotomicRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="polynomial.cyclotomic.request_type",
            message="cyclotomic computation requires a CyclotomicRequest value",
        )
    return request


def _factorization_work_bound(index: int, factorization: dict[int, int] | None) -> int:
    """Charge a source-side factorization envelope before any backend expansion.

    A prime index realizes the worst-case metric ``bit_length(n) * n``. Using
    that bound on the caller integer refuses over-budget work without running
    ``factorint``. After factorization the same metric is recomputed on the
    exact prime-power support.
    """

    if factorization is None:
        return max(1, index.bit_length()) * index
    return max(1, index.bit_length()) * max(
        1, sum(prime * exponent for prime, exponent in factorization.items())
    )


def _require_factorization_work(
    index: int, factorization: dict[int, int] | None = None
) -> None:
    if _factorization_work_bound(index, factorization) > MAX_CYCLOTOMIC_FACTOR_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.factorization_work_bound",
            message="cyclotomic index factorization exceeds the admitted work bound",
        )


def _semiprime_primes(factorization: dict[int, int]) -> tuple[int, int] | None:
    """Return ``(p, q)`` when the index is a product of two distinct primes."""

    if len(factorization) != 2 or any(
        exponent != 1 for exponent in factorization.values()
    ):
        return None
    low, high = sorted(factorization)
    if not (_is_prime(low) and _is_prime(high)):
        return None
    return low, high


def _lift_prime(index: int, factorization: dict[int, int]) -> tuple[int, int] | None:
    """Return ``(p, m)`` with ``index = p * m`` and prime ``p`` not dividing ``m``.

    The prime-lifting identity ``Phi_{pm}(x) = Phi_m(x**p) / Phi_m(x)`` then
    builds the index from the smaller ``m`` through one exact division. The
    largest eligible prime is peeled so the charged division is the smallest.
    """

    candidates = [
        prime
        for prime, exponent in factorization.items()
        if exponent == 1 and index % (prime * prime) != 0
    ]
    if not candidates:
        return None
    prime = max(candidates)
    return prime, index // prime


def _totient_from_factorization(index: int, factorization: dict[int, int]) -> int:
    """Return Euler's totient from an exact prime-exponent map."""

    totient = index
    for prime in factorization:
        totient = totient // prime * (prime - 1)
    return totient


def _construction_regime(index: int, factorization: dict[int, int]) -> tuple[int, int]:
    """Return the ``(work, intermediate bits)`` for one index's construction.

    The identity ``Phi_n(x) = Phi_rad(n)(x^(n/rad(n)))`` reduces every repeated
    prime factor, so the charged regime is the reduced index's own regime. A
    dense backend construction is only charged when the index is already
    radical.
    """

    if _is_prime_index(index, factorization):
        return max(1, index.bit_length()) * index, 2
    if _is_twice_odd_index(index, factorization):
        # ``Phi_{2m}(x) = Phi_m(-x)`` for odd ``m``: the construction only has
        # to build the odd half, then negate alternate coefficients. Charging the
        # odd half's own regime keeps every reduction it admits (for example the
        # bounded semiprime quotient) available through this path.
        odd_half = index // 2
        odd_factorization = {
            prime: exponent for prime, exponent in factorization.items() if prime != 2
        }
        odd_work, odd_bits = _construction_regime(odd_half, odd_factorization)
        return odd_work + odd_half + 1, odd_bits + 1
    semiprime = _semiprime_primes(factorization)
    if semiprime is not None:
        # ``Phi_{pq}(x) = Phi_p(x**q) / Phi_p(x)`` for distinct primes ``p < q``:
        # one p-term geometric series, one sparse substitution, and one exact
        # division whose quotient has ``(p-1)*(q-1)+1`` coefficients in
        # ``{-1, 0, 1}``. Charging the universal radical-square estimate here
        # would reject cheaply executable indices such as ``447 = 3*149``.
        low, high = semiprime
        return 10 * (low + 1) * (high + 1), 4 * low + 4
    radical = prod(factorization) if factorization else 1
    if radical != index:
        return _construction_regime(radical, dict.fromkeys(factorization, 1))
    lifted = _lift_prime(index, factorization)
    if lifted is not None:
        # ``Phi_{pm}(x) = Phi_m(x**p) / Phi_m(x)`` for prime ``p`` not dividing
        # ``m``: one exact division of ``(phi(n) + 1) * (phi(m) + 1)`` products
        # on top of the smaller index's own regime. Charging the universal
        # radical-square estimate here would reject cheaply executable families
        # such as the three-prime ``455 = 5*7*13``.
        prime, other = lifted
        other_factorization = {
            base: exponent
            for base, exponent in factorization.items()
            if base != prime
        }
        other_work, other_bits = _construction_regime(other, other_factorization)
        phi_other = _totient_from_factorization(other, other_factorization)
        return (
            other_work + 10 * prime * (phi_other + 1) * (phi_other + 1),
            other_bits + 8 * (phi_other + 1),
        )
    # SymPy's dense cyclotomic construction is charged from the radical of the
    # index, matching the exact kernel bound used by spectral character sums:
    # 10 * bit_length(rad) * (rad + 1)^2, plus the intermediate bit envelope
    # 2*rad + bit_length(rad+1) + 1.
    return (
        10 * max(1, radical.bit_length()) * (radical + 1) ** 2,
        2 * radical + (radical + 1).bit_length() + 1,
    )


def _admit(index: int, factorization: dict[int, int]) -> _CyclotomicAdmission:
    """Preflight factorization, divisor, intermediate, and output envelopes."""

    _require_factorization_work(index, factorization)
    factorization_work = _factorization_work_bound(index, factorization)

    divisor_count = prod(exponent + 1 for exponent in factorization.values())
    if divisor_count > MAX_CYCLOTOMIC_DIVISORS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.divisor_work_bound",
            message=(
                "cyclotomic divisor enumeration exceeds the admitted bound of "
                f"{MAX_CYCLOTOMIC_DIVISORS} divisors"
            ),
        )

    degree = index
    for prime in factorization:
        degree = degree // prime * (prime - 1)
    if degree > MAX_CYCLOTOMIC_DEGREE:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.output_degree_bound",
            message=(
                f"cyclotomic degree {degree} exceeds the output degree bound "
                f"{MAX_CYCLOTOMIC_DEGREE}"
            ),
        )

    construction_work, intermediate_bits = _construction_regime(index, factorization)
    if construction_work > MAX_CYCLOTOMIC_CONSTRUCTION_WORK:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.construction_work_bound",
            message="cyclotomic backend construction exceeds the admitted work bound",
        )
    if intermediate_bits > MAX_CYCLOTOMIC_INTERMEDIATE_BITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.intermediate_work_bound",
            message="cyclotomic intermediate products exceed the admitted work bound",
        )

    coefficient_bits = degree + 1
    coefficient_digits = (coefficient_bits * 30_103) // 100_000 + 1
    if coefficient_digits > MAX_CYCLOTOMIC_COEFFICIENT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.coefficient_height_bound",
            message="cyclotomic coefficient height exceeds the exact-output bound",
        )
    if coefficient_digits > MAX_CANONICAL_INTEGER_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.coefficient_encoding_bound",
            message="cyclotomic coefficients exceed the canonical integer bound",
        )

    # This is the semantic exact coefficient payload, not a wire-size or
    # encoded-byte estimate.  Degree and coefficient height are separate
    # limits because either can dominate a dense exact result.
    output_digits = (degree + 1) * coefficient_digits
    if output_digits > MAX_CYCLOTOMIC_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("index",),
            code="polynomial.cyclotomic.output_digit_bound",
            message="cyclotomic exact coefficient payload exceeds the output bound",
        )
    return _CyclotomicAdmission(
        degree=degree,
        factorization_work=factorization_work,
        divisor_count=divisor_count,
        intermediate_work=construction_work,
        coefficient_digits=coefficient_digits,
        output_digits=output_digits,
    )


def _factor_index(index: int) -> dict[int, int]:
    try:
        from sympy import factorint

        factors = factorint(index)
    except Exception as exc:
        _backend_error(BackendFailureReason.INITIALIZATION, exc)
    if not isinstance(factors, dict) or any(
        type(prime) is not int or type(exponent) is not int or prime < 2 or exponent < 1
        for prime, exponent in factors.items()
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    # Every valid base is at most ``index`` and the map holds at most
    # ``bit_length(index)`` distinct primes. Reject larger bases or wider maps
    # before exponentiation so malformed backend output cannot trigger
    # unadmitted power growth while reconstructing the index.
    if len(factors) > max(1, index.bit_length()) or any(
        prime > index for prime in factors
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    # Any valid exponent satisfies ``prime**exponent <= index``, so it is bounded
    # by ``index.bit_length()``. Reject larger exponents before exponentiation so
    # malformed backend output cannot trigger unadmitted CPU or memory growth.
    exponent_limit = index.bit_length()
    if any(exponent > exponent_limit for exponent in factors.values()):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    reconstructed = prod(prime**exponent for prime, exponent in factors.items())
    if reconstructed != index:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    if any(not _is_prime(prime) for prime in factors):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return factors


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value < 4:
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    factor = 5
    while factor * factor <= value:
        if value % factor == 0 or value % (factor + 2) == 0:
            return False
        factor += 6
    return True


def _is_prime_index(index: int, factorization: dict[int, int]) -> bool:
    return factorization == {index: 1}


def _is_twice_odd_index(index: int, factorization: dict[int, int]) -> bool:
    """True when ``index = 2 * m`` for an odd ``m > 1``.

    ``Phi_{2m}(x) = Phi_m(-x)`` holds for every odd ``m``, prime or not.
    """

    return index % 2 == 0 and index >= 6 and (index // 2) % 2 == 1


def _prime_cyclotomic(prime: int) -> IntegerPolynomial:
    request_checkpoint("during prime cyclotomic geometric sum")
    return IntegerPolynomial(coefficients=(1,) * prime)


def _twice_odd_cyclotomic(
    index: int,
    factorization: dict[int, int],
    admission: _CyclotomicAdmission,
) -> IntegerPolynomial:
    """Return ``Phi_{2m}(x) = Phi_m(-x)`` for ``index = 2m`` with odd ``m``."""

    odd = index // 2
    odd_factorization = {
        prime: exponent for prime, exponent in factorization.items() if prime != 2
    }
    request_checkpoint("during twice-odd cyclotomic construction")
    if _is_prime_index(odd, odd_factorization):
        base_coefficients: tuple[int, ...] = (1,) * odd
    else:
        odd_primes = _semiprime_primes(odd_factorization)
        if odd_primes is not None:
            # Reuse the odd half's own bounded quotient instead of paying for a
            # dense backend construction of the halved index.
            base_coefficients = _semiprime_quotient_coefficients(
                odd, odd_primes, admission
            )
        else:
            # The odd half may itself need a bounded reduction (for example a
            # three-prime lift); dispatch through the admitted construction so
            # every shape it accepts is available through this path.
            _, odd_value = _compute(odd)
            base_coefficients = odd_value.coefficients
        # The odd half is ``Phi_odd`` with the same degree as ``Phi_{2*odd}``
        # and, for odd > 1, the exact constant term 1. Checking only digit
        # widths would let a malformed monic tuple ending in -1 through.
        _require_admitted_coefficients(
            base_coefficients,
            admission,
            expected_degree=admission.degree,
            expected_constant=1,
        )
    degree = len(base_coefficients) - 1
    return IntegerPolynomial(
        coefficients=tuple(
            coefficient if (degree - offset) % 2 == 0 else -coefficient
            for offset, coefficient in enumerate(base_coefficients)
        )
    )


def _exact_divide(
    dividend: IntegerPolynomial, divisor: IntegerPolynomial
) -> IntegerPolynomial:
    """Return ``dividend / divisor`` for a monic dense divisor over ``ZZ``."""

    remainder = list(dividend.coefficients)
    divisor_coefficients = list(divisor.coefficients)
    divisor_degree = len(divisor_coefficients) - 1
    leading = divisor_coefficients[0]
    if divisor_degree < 0 or leading == 0:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    quotient_length = len(remainder) - divisor_degree
    quotient = [0] * quotient_length
    for position in range(quotient_length):
        if position % 64 == 0:
            request_checkpoint("during semiprime cyclotomic division")
        factor, residual = divmod(remainder[position], leading)
        if residual:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        quotient[position] = factor
        if factor:
            for offset, coefficient in enumerate(divisor_coefficients):
                remainder[position + offset] -= factor * coefficient
    if any(remainder[quotient_length:]):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return IntegerPolynomial(coefficients=tuple(quotient))


def _semiprime_quotient_coefficients(
    index: int,
    primes: tuple[int, int],
    admission: _CyclotomicAdmission,
) -> tuple[int, ...]:
    """Return the coefficients of ``Phi_{pq}(x) = Phi_p(x**q) / Phi_p(x)``."""

    low, high = primes
    request_checkpoint("during semiprime cyclotomic construction")
    quotient = _exact_divide(
        _substitute_power(_prime_cyclotomic(low), high),
        _prime_cyclotomic(low),
    )
    if len(quotient.coefficients) != admission.degree + 1:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return quotient.coefficients


def _semiprime_quotient_cyclotomic(
    index: int,
    primes: tuple[int, int],
    admission: _CyclotomicAdmission,
) -> IntegerPolynomial:
    """Return ``Phi_{pq}(x) = Phi_p(x**q) / Phi_p(x)`` for distinct primes."""

    coefficients = _semiprime_quotient_coefficients(index, primes, admission)
    _require_admitted_coefficients(coefficients, admission, expected_constant=1)
    return IntegerPolynomial(coefficients=coefficients)


def _prime_lift_quotient_coefficients(
    index: int,
    prime: int,
    other: int,
    admission: _CyclotomicAdmission,
) -> tuple[int, ...]:
    """Return ``Phi_{pm}(x) = Phi_m(x**p) / Phi_m(x)`` for prime ``p`` not dividing ``m``."""

    request_checkpoint("during prime-lift cyclotomic construction")
    try:
        _, base = _compute(other)
        quotient = _exact_divide(_substitute_power(base, prime), base)
    except (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
        OperationBackendError,
    ):
        raise
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    if len(quotient.coefficients) != admission.degree + 1:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
    return quotient.coefficients


def _exceeds_coefficient_digits(
    coefficients: tuple[int, ...], admission: _CyclotomicAdmission
) -> bool:
    """Report whether any coefficient leaves the admitted exact-output envelope.

    A malformed backend value can exceed Python's active integer-to-string digit
    limit, where ``str`` raises ``ValueError`` before the tuple could be
    classified as a backend failure. Binary magnitude decides that case without
    any decimal conversion, and the exact digit totals are computed only for
    values that can be formatted.
    """

    total_digits = 0
    for value in coefficients:
        magnitude = abs(value)
        # ``bit_length`` bounds the decimal width from above and below:
        # digits <= bit_length, and bit_length <= 4 * digits for digits >= 1.
        bits = magnitude.bit_length()
        if bits > 4 * admission.coefficient_digits:
            return True
        try:
            digits = len(str(magnitude))
        except ValueError:
            return True
        if digits > admission.coefficient_digits:
            return True
        total_digits += digits
        if total_digits > admission.output_digits:
            return True
    return False


def _require_admitted_coefficients(
    coefficients: tuple[int, ...],
    admission: _CyclotomicAdmission,
    *,
    expected_degree: int | None = None,
    expected_constant: int | None = None,
) -> None:
    """Check a backend coefficient tuple against the admitted output envelope.

    ``expected_degree`` and ``expected_constant`` pin the carrier's shape: a
    cyclotomic returned by a reduced construction path must still have the
    admitted degree and the exact constant term, not merely bounded digit
    widths.
    """

    if (
        not coefficients
        or coefficients[0] != 1
        or (expected_degree is not None and len(coefficients) != expected_degree + 1)
        or (expected_constant is not None and coefficients[-1] != expected_constant)
        or (expected_constant is None and coefficients[-1] not in (1, -1))
        or _exceeds_coefficient_digits(coefficients, admission)
    ):
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)


def _substitute_power(
    polynomial: IntegerPolynomial, multiplier: int
) -> IntegerPolynomial:
    """Return ``polynomial(x**multiplier)`` as a sparse dense carrier."""

    coefficients = polynomial.coefficients
    degree = len(coefficients) - 1
    # ``coefficients[i]`` is the degree ``degree - i`` term; assign it to the
    # substituted degree ``(degree - i) * multiplier`` in an ascending list.
    ascending = [0] * (degree * multiplier + 1)
    for offset, coefficient in enumerate(coefficients):
        ascending[(degree - offset) * multiplier] = coefficient
    return IntegerPolynomial(coefficients=tuple(reversed(ascending)))


def _backend_cyclotomic_coefficients(index: int) -> tuple[int, ...]:
    try:
        from sympy import Symbol, cyclotomic_poly
    except Exception as exc:
        _backend_error(BackendFailureReason.INITIALIZATION, exc)
    request_checkpoint("before cyclotomic backend")
    try:
        polynomial = cyclotomic_poly(index, Symbol("x"), polys=True)
        raw_coefficients = tuple(polynomial.all_coeffs())
    except OperationBackendError:
        raise
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    request_checkpoint("after cyclotomic backend")
    coefficients: tuple[int, ...]
    normalized_coefficients: list[int] = []
    try:
        for value in raw_coefficients:
            if (
                type(value) is not int
                and getattr(value, "is_Integer", None) is not True
            ):
                raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
            normalized_coefficients.append(int(value))
    except OperationBackendError:
        raise
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    coefficients = tuple(normalized_coefficients)
    return coefficients


def _compute(index: int) -> tuple[int, IntegerPolynomial]:
    request_checkpoint("before cyclotomic admission")
    _require_factorization_work(index)
    factorization = _factor_index(index)
    admission = _admit(index, factorization)
    request_checkpoint("after cyclotomic admission")
    if _is_prime_index(index, factorization):
        polynomial_value = _prime_cyclotomic(index)
        request_checkpoint("before cyclotomic result construction")
        return admission.degree, polynomial_value
    if _is_twice_odd_index(index, factorization):
        polynomial_value = _twice_odd_cyclotomic(index, factorization, admission)
        request_checkpoint("before cyclotomic result construction")
        return admission.degree, polynomial_value
    semiprime = _semiprime_primes(factorization)
    if semiprime is not None:
        polynomial_value = _semiprime_quotient_cyclotomic(index, semiprime, admission)
        request_checkpoint("before cyclotomic result construction")
        return admission.degree, polynomial_value
    radical = prod(factorization) if factorization else 1
    if radical != index:
        # ``Phi_n(x) = Phi_rad(n)(x^(n/rad(n)))``: build the reduced cyclotomic
        # and substitute the sparse power.
        _, reduced_polynomial = _compute(radical)
        request_checkpoint("before cyclotomic result construction")
        substituted = _substitute_power(reduced_polynomial, index // radical)
        _require_admitted_coefficients(
            substituted.coefficients,
            admission,
            expected_degree=admission.degree,
            expected_constant=1,
        )
        return admission.degree, substituted
    lifted = _lift_prime(index, factorization)
    if lifted is not None:
        prime, other = lifted
        coefficients = _prime_lift_quotient_coefficients(
            index, prime, other, admission
        )
        _require_admitted_coefficients(
            coefficients,
            admission,
            expected_degree=admission.degree,
            expected_constant=1,
        )
        request_checkpoint("before cyclotomic result construction")
        try:
            return admission.degree, IntegerPolynomial(coefficients=coefficients)
        except Exception as exc:
            _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    coefficients = _backend_cyclotomic_coefficients(index)
    _require_admitted_coefficients(
        coefficients,
        admission,
        expected_degree=admission.degree,
        expected_constant=-1 if index == 1 else 1,
    )
    request_checkpoint("before cyclotomic result construction")
    try:
        polynomial_value = IntegerPolynomial(
            coefficients=coefficients,
        )
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
    return admission.degree, polynomial_value


def _require_native_index(index: int) -> int:
    if type(index) is not int:
        raise OperationDomainValidationError(
            location=("index",),
            code="polynomial.cyclotomic.index_type",
            message="cyclotomic native computation requires an integer index",
        )
    if not 1 <= index <= MAX_CYCLOTOMIC_INDEX:
        raise OperationDomainValidationError(
            location=("index",),
            code="polynomial.cyclotomic.index_bound",
            message=(
                f"cyclotomic native index must be between 1 and {MAX_CYCLOTOMIC_INDEX}"
            ),
        )
    return index


def cyclotomic(index: int) -> IntegerPolynomial:
    """Return the admitted exact cyclotomic polynomial in ``ZZ[x]``."""

    _, polynomial = _compute(_require_native_index(index))
    return polynomial


def _run(request: CyclotomicRequest) -> CyclotomicResult:
    typed_request = _require_typed_request(request)
    degree, polynomial = _compute(typed_request.index)
    try:
        return CyclotomicResult(
            source_index=typed_request.index,
            totient=degree,
            polynomial=polynomial,
        )
    except Exception as exc:
        _backend_error(BackendFailureReason.INVALID_OUTPUT, exc)
