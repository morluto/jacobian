"""Exact cyclotomic polynomial operation."""

import pytest
from flint import fmpz_poly
from pydantic import ValidationError

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math import polynomials
from jacobian.math.polynomials._cyclotomic import (
    CyclotomicRequest,
    CyclotomicResult,
    _construction_regime,
    _run,
    cyclotomic,
)


def ascending(result: CyclotomicResult) -> list[int]:
    return list(reversed(result.polynomial.coefficients))


def test_known_twelfth_cyclotomic() -> None:
    result = _run(CyclotomicRequest(index=12))
    assert result.source_index == 12
    assert result.totient == 4
    assert result.polynomial.coefficients == (1, 0, -1, 0, 1)


@pytest.mark.parametrize(
    ("index", "coefficients"),
    ((7, (1, 1, 1, 1, 1, 1, 1)), (8, (1, 0, 0, 0, 1))),
)
def test_prime_and_prime_power_cyclotomics(
    index: int, coefficients: tuple[int, ...]
) -> None:
    result = _run(CyclotomicRequest(index=index))
    assert result.polynomial.coefficients == coefficients


@pytest.mark.parametrize(
    ("index", "coefficients"),
    ((1, (1, -1)), (2, (1, 1)), (6, (1, -1, 1))),
)
def test_boundary_and_small_cyclotomic_values(
    index: int, coefficients: tuple[int, ...]
) -> None:
    result = _run(CyclotomicRequest(index=index))
    assert result.totient == len(coefficients) - 1
    assert result.polynomial.coefficients == coefficients


def test_native_api_exports_typed_cyclotomic_operation() -> None:
    assert polynomials.cyclotomic(3).coefficients == (1, 1, 1)
    assert polynomials.cyclotomic(12).coefficients == (1, 0, -1, 0, 1)


def test_direct_native_boundary_rejects_untyped_payload() -> None:
    with pytest.raises(OperationDomainValidationError, match="integer index"):
        cyclotomic({"index": 3})  # type: ignore[arg-type]


def test_result_validation_checks_shape_without_backend_recomputation() -> None:
    result = _run(CyclotomicRequest(index=12))
    payload = result.model_dump_json()
    import sympy

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("result validation must not replay the backend")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sympy, "cyclotomic_poly", fail)
        patch.setattr(sympy, "factorint", fail)
        assert CyclotomicResult.model_validate_json(payload) == result
    with pytest.raises(ValidationError, match="totient"):
        CyclotomicResult(
            source_index=12,
            totient=3,
            polynomial=result.polynomial,
        )
    with pytest.raises(ValidationError, match="constant"):
        CyclotomicResult(
            source_index=12,
            totient=4,
            polynomial=result.polynomial.model_copy(
                update={"coefficients": (1, 0, -1, 0, 0)}
            ),
        )


def test_backend_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    import sympy

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(sympy, "factorint", fail)
    with pytest.raises(OperationBackendError) as exc_info:
        _run(CyclotomicRequest(index=105))
    assert exc_info.value.reason is BackendFailureReason.INITIALIZATION


def test_inexact_division_is_a_typed_backend_failure() -> None:
    """The exact quotient seam never truncates: a remainder is typed."""
    from jacobian.math.polynomials._cyclotomic import _exact_divide
    from jacobian.math.polynomials._models import IntegerPolynomial

    with pytest.raises(OperationBackendError) as exc_info:
        _exact_divide(
            IntegerPolynomial(coefficients=(1, 0, 0)),
            IntegerPolynomial(coefficients=(1, 1, 1)),
        )
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_backend_wrong_constant_is_rejected_on_the_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _cyclotomic as module
    from jacobian.math.polynomials._models import IntegerPolynomial

    monkeypatch.setattr(
        module,
        "_exact_divide",
        lambda dividend, divisor: IntegerPolynomial(coefficients=(1,) * 48 + (-1,)),
    )
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(105)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_factor_map_must_reconstruct_the_requested_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    def fail_poly(*args: object, **kwargs: object) -> object:
        raise AssertionError("cyclotomic_poly must not run on an unbound factorization")

    monkeypatch.setattr(sympy, "factorint", lambda index: {2: 1, 3: 1, 5: 1, 7: 1})
    monkeypatch.setattr(sympy, "cyclotomic_poly", fail_poly)
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(16_530)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_factorization_work_is_admitted_before_backend_factorint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _cyclotomic as module

    monkeypatch.setattr(module, "MAX_CYCLOTOMIC_FACTOR_WORK", 1)

    def fail_factorint(index: int) -> dict[int, int]:
        raise AssertionError("factorint must not run before factorization admission")

    import sympy

    monkeypatch.setattr(sympy, "factorint", fail_factorint)
    with pytest.raises(OperationResourceAdmissionError, match="factorization"):
        cyclotomic(30)


def test_expired_request_is_rejected_before_factorization() -> None:
    from time import monotonic

    with (
        request_execution(monotonic(), outer_deadline=monotonic() - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        _run(CyclotomicRequest(index=30))


def test_prime_index_uses_the_geometric_sum_fast_path() -> None:
    prime = 4093
    result = _run(CyclotomicRequest(index=prime))
    assert result.totient == prime - 1
    assert result.polynomial.coefficients == (1,) * prime
    product = fmpz_poly([-1, 1]) * fmpz_poly(ascending(result))
    expected = fmpz_poly([-1] + [0] * (prime - 1) + [1])
    assert product == expected


def test_twice_odd_prime_index_uses_phi_m_of_minus_x() -> None:
    odd_prime = 4093
    index = 2 * odd_prime
    result = _run(CyclotomicRequest(index=index))
    assert result.totient == odd_prime - 1
    assert result.polynomial.coefficients == tuple((-1) ** k for k in range(odd_prime))
    product = (
        fmpz_poly([-1, 1])
        * fmpz_poly([1, 1])
        * fmpz_poly([1] * odd_prime)
        * fmpz_poly(ascending(result))
    )
    expected = fmpz_poly([-1] + [0] * (index - 1) + [1])
    assert product == expected


def test_twice_odd_composite_index_uses_phi_m_of_minus_x() -> None:
    """``Phi_{2m}(x) = Phi_m(-x)`` holds for composite odd ``m`` too.

    ``426 = 2 * 213`` has the same radical-free identity as a twice-prime
    index, so it must be admitted by charging the odd half rather than the
    full index radical.
    """
    import sympy

    index = 426
    result = _run(CyclotomicRequest(index=index))
    assert result.totient == 140
    expected = tuple(
        int(coefficient)
        for coefficient in sympy.cyclotomic_poly(
            index, sympy.Symbol("x"), polys=True
        ).all_coeffs()
    )
    assert result.polynomial.coefficients == expected
    # The odd half is the construction source; alternating signs reconstruct it.
    odd_result = _run(CyclotomicRequest(index=index // 2))
    assert result.polynomial.coefficients == tuple(
        coefficient
        if (len(odd_result.polynomial.coefficients) - 1 - offset) % 2 == 0
        else -coefficient
        for offset, coefficient in enumerate(odd_result.polynomial.coefficients)
    )


def test_factor_map_exponents_are_bounded_before_exponentiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed backend output cannot request ``prime**exponent`` growth."""
    import sympy

    monkeypatch.setattr(sympy, "factorint", lambda index: {2: 1_000_000_000})
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(30)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_factor_map_bases_are_bounded_before_exponentiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed factor base cannot request multi-million-digit powers."""
    import sympy

    # The exponent passes the bit-length screen, so only the base bound stops
    # the reconstruction from materializing a ~50-million-digit power.
    monkeypatch.setattr(sympy, "factorint", lambda index: {10**10_000_000 + 7: 5})
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(30)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_composite_reported_as_a_prime_base_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    def fail_poly(*args: object, **kwargs: object) -> object:
        raise AssertionError("cyclotomic_poly must not run on a composite factor base")

    monkeypatch.setattr(sympy, "factorint", lambda index: {6: 1, 5: 1})
    monkeypatch.setattr(sympy, "cyclotomic_poly", fail_poly)
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(30)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_construction_work_is_admitted_before_backend_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.polynomials import _cyclotomic as module

    monkeypatch.setattr(module, "MAX_CYCLOTOMIC_CONSTRUCTION_WORK", 1)

    def fail_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("construction must not start before work admission")

    monkeypatch.setattr(module, "_twice_odd_cyclotomic", fail_construction)
    with pytest.raises(OperationResourceAdmissionError, match="construction"):
        cyclotomic(30)


def test_reductions_reuse_the_admitted_factor_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reduction must not re-enter the factorint backend it already paid for.

    ``3988 = 2**2 * 997`` reduces to the squarefree ``997``; ``426 = 2*3*71``
    reduces through the twice-odd path. Both must construct from the factor map
    the entry point already validated, so the backend-failure point stays
    single and the recorded plan is complete.
    """
    import sympy

    original = sympy.factorint
    calls: list[int] = []

    def counting(index: int) -> dict[int, int]:
        calls.append(index)
        result: dict[int, int] = original(index)
        return result

    monkeypatch.setattr(sympy, "factorint", counting)
    assert len(cyclotomic(3988).coefficients) == 1992 + 1
    assert len(cyclotomic(426).coefficients) == 140 + 1
    assert calls == [3988, 426]


def test_backend_cardinality_is_bounded_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend polynomial wider than the admitted degree is refused early.

    The fallthrough dense path passes the admitted cardinality into the
    adapter, so a malformed response is rejected as a bounded INVALID_OUTPUT
    before it is copied or converted coefficient by coefficient.
    """
    import sympy

    from jacobian.math.polynomials import _cyclotomic as module

    class _Untouchable:
        converted = False

        def __int__(self) -> int:
            _Untouchable.converted = True
            return 0

    class _WideBackendPoly:
        def all_coeffs(self) -> list[object]:
            return [1, 2, 3, _Untouchable()]

    monkeypatch.setattr(
        sympy, "cyclotomic_poly", lambda index, symbol, polys=False: _WideBackendPoly()
    )
    with pytest.raises(OperationBackendError) as exc_info:
        module._backend_cyclotomic_coefficients(7, 3)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT
    assert _Untouchable.converted is False


def test_three_prime_lifting_charge_stays_inside_the_construction_envelope() -> None:
    """The prime-lifting regime admits cheap multi-prime squarefree indices."""
    from jacobian.math.polynomials._cyclotomic import (
        MAX_CYCLOTOMIC_CONSTRUCTION_WORK,
        _factor_index,
    )

    assert (
        _construction_regime(455, _factor_index(455))[0]
        < MAX_CYCLOTOMIC_CONSTRUCTION_WORK
    )
    assert (
        _construction_regime(16_530, _factor_index(16_530))[0]
        < MAX_CYCLOTOMIC_CONSTRUCTION_WORK
    )


def test_three_prime_cyclotomic_matches_the_backend_oracle() -> None:
    """``Phi_455 = Phi_91(x**5)/Phi_91(x)`` is admitted and exact."""
    import sympy

    result = _run(CyclotomicRequest(index=455))
    assert result.totient == 288
    assert len(result.polynomial.coefficients) == 289
    reference = sympy.Poly(
        sympy.cyclotomic_poly(455, sympy.Symbol("x")), sympy.Symbol("x")
    )
    assembled = sympy.Poly(
        sum(
            coefficient * sympy.Symbol("x") ** (288 - offset)
            for offset, coefficient in enumerate(result.polynomial.coefficients)
        ),
        sympy.Symbol("x"),
    )
    assert assembled == reference


def test_large_degree_is_admitted_before_backend_expansion() -> None:
    from jacobian.math.polynomials import _cyclotomic as module

    def fail_construction(*args: object, **kwargs: object) -> object:
        raise AssertionError("polynomial construction must not start before admission")

    constructor_names = (
        "_prime_cyclotomic",
        "_twice_odd_cyclotomic",
        "_semiprime_quotient_cyclotomic",
        "_prime_lift_quotient_coefficients",
        "_substitute_power",
        "_backend_cyclotomic_coefficients",
    )
    with pytest.MonkeyPatch.context() as patch:
        for name in constructor_names:
            patch.setattr(module, name, fail_construction)
        with pytest.raises(OperationResourceAdmissionError):
            _run(CyclotomicRequest(index=100_000))


def test_power_of_two_multiple_uses_the_reduced_path() -> None:
    """``3988 = 2**2 * 997`` reduces to ``Phi_1994(x**2)`` and is admitted."""
    import sympy

    index = 3_988
    result = _run(CyclotomicRequest(index=index))
    assert result.totient == 1_992
    expected = tuple(
        int(coefficient)
        for coefficient in sympy.cyclotomic_poly(
            index, sympy.Symbol("x"), polys=True
        ).all_coeffs()
    )
    assert result.polynomial.coefficients == expected


def test_reduced_backend_coefficients_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A twice-odd odd-half quotient tuple is checked against the envelope."""
    from jacobian.math.polynomials import _cyclotomic as module
    from jacobian.math.polynomials._models import IntegerPolynomial

    malformed = (1, 10**200, *(0 for _ in range(46)), 1)
    assert len(malformed) == 49
    monkeypatch.setattr(
        module,
        "_exact_divide",
        lambda dividend, divisor: IntegerPolynomial(coefficients=malformed),
    )
    # 210 = 2 * 105 builds its odd half through the exact prime-lifting
    # quotient (105 = 3*5*7), and a malformed quotient tuple must be checked
    # before the result is returned.
    with pytest.raises(OperationBackendError) as exc_info:
        module.cyclotomic(210)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_odd_semiprime_quotient_stays_inside_the_construction_envelope() -> None:
    """Phi_447 = Phi_3(x**149)/Phi_3(x) is cheap and must remain admissible."""
    import sympy

    result = _run(CyclotomicRequest(index=447))
    assert result.totient == 296
    assert len(result.polynomial.coefficients) == 297
    assert set(result.polynomial.coefficients) <= {-1, 0, 1}
    # Independent oracle: the maintained backend's own construction.
    reference = sympy.Poly(
        sympy.cyclotomic_poly(447, sympy.Symbol("x")), sympy.Symbol("x")
    )
    assembled = sympy.Poly(
        sum(
            coefficient * sympy.Symbol("x") ** (296 - offset)
            for offset, coefficient in enumerate(result.polynomial.coefficients)
        ),
        sympy.Symbol("x"),
    )
    assert assembled == reference
    # The quotient regime is charged far below the universal radical-square
    # estimate that used to refuse this index.
    assert _construction_regime(447, {3: 1, 149: 1})[0] < 16_000_000


@pytest.mark.parametrize("index", (15, 21, 35, 77))
def test_semiprime_quotient_matches_exact_backend_coefficients(index: int) -> None:
    """Distinct-prime semiprime quotient coefficients match an exact oracle."""
    import sympy

    result = _run(CyclotomicRequest(index=index))
    x = sympy.Symbol("x")
    expected = tuple(
        int(coefficient)
        for coefficient in sympy.cyclotomic_poly(index, x, polys=True).all_coeffs()
    )
    assert result.polynomial.coefficients == expected


def test_twice_odd_base_shape_is_required_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed odd half must not reach the native return."""
    from jacobian.math.polynomials import _cyclotomic as module
    from jacobian.math.polynomials._cyclotomic import _admit, _factor_index

    admission = _admit(210, _factor_index(210))

    class WrongConstant:
        coefficients: tuple[int, ...] = (1,) + (0,) * (admission.degree - 1) + (-1,)

    class Short:
        coefficients: tuple[int, ...] = (1,) + (0,) * (admission.degree - 2)

    for fake in (WrongConstant(), Short()):
        monkeypatch.setattr(
            module,
            "_prime_lift_quotient_coefficients",
            lambda *args, carrier=fake, **kwargs: carrier.coefficients,
        )
        with pytest.raises(OperationBackendError) as exc_info:
            cyclotomic(210)
        assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_twice_odd_halves_agree_with_the_backend_for_every_reduction() -> None:
    """The reused odd-half kernel matches the backend across reduction shapes."""
    import sympy

    for index in (6, 12, 30, 42, 894):
        result = _run(CyclotomicRequest(index=index))
        reference = sympy.Poly(
            sympy.cyclotomic_poly(index, sympy.Symbol("x")), sympy.Symbol("x")
        )
        degree = result.totient
        assembled = sympy.Poly(
            sum(
                coefficient * sympy.Symbol("x") ** (degree - offset)
                for offset, coefficient in enumerate(result.polynomial.coefficients)
            ),
            sympy.Symbol("x"),
        )
        assert assembled == reference


def test_oversized_backend_integer_is_a_typed_backend_failure() -> None:
    """A float-digit backend coefficient must not leak a raw ValueError."""
    from jacobian._execution import OperationBackendError
    from jacobian.math.polynomials._cyclotomic import (
        _admit,
        _factor_index,
        _require_admitted_coefficients,
    )

    admission = _admit(30, _factor_index(30))
    oversized = 10**5000
    for coefficients in ((1, oversized, 1), (1, oversized)):
        with pytest.raises(OperationBackendError) as exc_info:
            _require_admitted_coefficients(coefficients, admission)
        assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT
    _require_admitted_coefficients((1, 0, 1), admission)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _convolve(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Descending-coefficient convolution, matching the result carrier."""
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return tuple(result)


def _cyclotomic_coefficients(index: int) -> tuple[int, ...]:
    return _run(CyclotomicRequest(index=index)).polynomial.coefficients


def test_divisor_product_reconstructs_xn_minus_one() -> None:
    """x^n - 1 == prod_{d|n} Phi_d, by independent integer convolution."""
    for n in range(1, 21):
        product: tuple[int, ...] = (1,)
        for divisor in _divisors(n):
            product = _convolve(product, _cyclotomic_coefficients(divisor))
        assert product == (1, *((0,) * (n - 1)), -1)


def test_cyclotomic_at_one_is_prime_power_detector() -> None:
    """Phi_n(1) is p for n = p^k and 1 otherwise, evaluated directly."""
    assert sum(_cyclotomic_coefficients(8)) == 2
    assert sum(_cyclotomic_coefficients(9)) == 3
    assert sum(_cyclotomic_coefficients(12)) == 1
    assert sum(_cyclotomic_coefficients(7)) == 7
    assert sum(_cyclotomic_coefficients(1)) == 0


def test_cyclotomics_are_irreducible_over_qq() -> None:
    from sympy import Poly, Symbol

    x = Symbol("x")
    for index in (3, 4, 5, 6, 7, 8, 9, 12):
        assert Poly(
            list(_cyclotomic_coefficients(index)), x, domain="QQ"
        ).is_irreducible
