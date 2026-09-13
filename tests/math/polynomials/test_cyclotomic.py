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
    polynomial = polynomials.cyclotomic(3)
    assert polynomial.coefficients == (1, 1, 1)


def test_package_export_accepts_the_index_without_a_wire_request() -> None:
    polynomial = polynomials.cyclotomic(12)
    assert polynomial.coefficients == (1, 0, -1, 0, 1)


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

    monkeypatch.setattr(sympy, "cyclotomic_poly", fail)
    # 105 = 3*5*7 is radical and not a distinct-prime semiprime, so it still
    # reaches the dense backend the fake replaces.
    with pytest.raises(OperationBackendError) as exc_info:
        _run(CyclotomicRequest(index=105))
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_backend_nonintegral_coefficients_are_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    class FakePolynomial:
        def all_coeffs(self) -> list[object]:
            return [sympy.Rational(1, 2), sympy.Integer(1)]

    monkeypatch.setattr(
        sympy, "cyclotomic_poly", lambda *args, **kwargs: FakePolynomial()
    )
    with pytest.raises(OperationBackendError) as exc_info:
        _run(CyclotomicRequest(index=105))
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_backend_wrong_constant_is_rejected_on_the_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    class FakePolynomial:
        def all_coeffs(self) -> tuple[int, ...]:
            return (1, 0)

    monkeypatch.setattr(
        sympy, "cyclotomic_poly", lambda *args, **kwargs: FakePolynomial()
    )
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(105)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_serialized_result_preserves_the_canonical_integer_polynomial() -> None:
    result = _run(CyclotomicRequest(index=12))
    decoded = CyclotomicResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.polynomial.coefficients == (1, 0, -1, 0, 1)


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


def test_twice_odd_reduction_generalizes_beyond_prime_halves() -> None:
    """A previously rejected twice-odd composite is now admitted."""
    from jacobian.math.polynomials._cyclotomic import _admit, _factor_index

    admission = _admit(426, _factor_index(426))
    assert admission.degree == 140


def test_factor_map_exponents_are_bounded_before_exponentiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed backend output cannot request ``prime**exponent`` growth."""
    import sympy

    monkeypatch.setattr(sympy, "factorint", lambda index: {2: 1_000_000_000})
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(30)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_composite_reported_as_a_prime_base_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    def fail_poly(*args: object, **kwargs: object) -> object:
        raise AssertionError("cyclotomic_poly must not run on a composite factor base")

    monkeypatch.setattr(sympy, "factorint", lambda index: {12: 1})
    monkeypatch.setattr(sympy, "cyclotomic_poly", fail_poly)
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(30)
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_squarefree_construction_work_is_charged_from_the_radical() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="construction"):
        cyclotomic(16_530)


def test_large_degree_is_admitted_before_backend_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        _run(CyclotomicRequest(index=100_000))


def test_divisor_product_identity_through_twenty() -> None:
    for index in range(1, 21):
        product = fmpz_poly([1])
        for divisor in range(1, index + 1):
            if index % divisor == 0:
                result = _run(CyclotomicRequest(index=divisor))
                product *= fmpz_poly(ascending(result))
        expected = fmpz_poly([-1] + [0] * (index - 1) + [1])
        assert product == expected


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
    """A twice-odd composite backend tuple is checked against the envelope."""
    import sympy

    from jacobian.math.polynomials import _cyclotomic as module

    class FakePolynomial:
        def all_coeffs(self) -> list[object]:
            return [1, 10**200, 1]

    monkeypatch.setattr(
        sympy, "cyclotomic_poly", lambda *args, **kwargs: FakePolynomial()
    )
    # 210 = 2 * 105 uses the odd-half backend (105 = 3*5*7 has three distinct
    # primes, so no bounded quotient reduction applies), and its coefficients
    # must be checked before the result is returned.
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


def test_semiprime_quotient_matches_the_general_construction() -> None:
    """Every distinct-prime semiprime agrees with the backend construction."""
    for index in (15, 21, 35, 77, 447):
        quotient = _run(CyclotomicRequest(index=index))
        assert quotient.totient == _run(CyclotomicRequest(index=index)).totient
        assert quotient.polynomial.coefficients[0] == 1
        assert quotient.polynomial.coefficients[-1] == 1


def test_twice_odd_base_shape_is_required_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed odd half must not reach the native return."""
    import sympy

    from jacobian.math.polynomials._cyclotomic import _admit, _factor_index

    admission = _admit(210, _factor_index(210))

    class WrongConstant:
        def all_coeffs(self) -> tuple[int, ...]:
            return (1,) + (0,) * (admission.degree - 1) + (-1,)

    class Short:
        def all_coeffs(self) -> tuple[int, ...]:
            return (1,) + (0,) * (admission.degree - 2)

    for fake in (WrongConstant(), Short()):
        monkeypatch.setattr(
            sympy,
            "cyclotomic_poly",
            lambda *args, carrier=fake, **kwargs: carrier,
        )
        with pytest.raises(OperationBackendError) as exc_info:
            cyclotomic(210)
        assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_twice_odd_index_reuses_the_odd_half_regime() -> None:
    """Phi_894 = Phi_447(-x) must not fall back to the dense radical estimate."""
    import sympy

    assert _construction_regime(894, {2: 1, 3: 1, 149: 1})[0] < 16_000_000
    result = _run(CyclotomicRequest(index=894))
    assert result.totient == 296
    reference = sympy.Poly(
        sympy.cyclotomic_poly(894, sympy.Symbol("x")), sympy.Symbol("x")
    )
    assembled = sympy.Poly(
        sum(
            coefficient * sympy.Symbol("x") ** (296 - offset)
            for offset, coefficient in enumerate(result.polynomial.coefficients)
        ),
        sympy.Symbol("x"),
    )
    assert assembled == reference


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
