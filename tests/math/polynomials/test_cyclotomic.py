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
    with pytest.raises(OperationBackendError) as exc_info:
        _run(CyclotomicRequest(index=12))
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
        _run(CyclotomicRequest(index=12))
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
        cyclotomic(12)
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
        cyclotomic(12)


def test_expired_request_is_rejected_before_factorization() -> None:
    from time import monotonic

    with (
        request_execution(monotonic(), outer_deadline=monotonic() - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        _run(CyclotomicRequest(index=12))


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
    assert result.polynomial.coefficients == tuple(
        (-1) ** k for k in range(odd_prime)
    )
    product = (
        fmpz_poly([-1, 1])
        * fmpz_poly([1, 1])
        * fmpz_poly([1] * odd_prime)
        * fmpz_poly(ascending(result))
    )
    expected = fmpz_poly([-1] + [0] * (index - 1) + [1])
    assert product == expected


def test_composite_reported_as_a_prime_base_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    def fail_poly(*args: object, **kwargs: object) -> object:
        raise AssertionError("cyclotomic_poly must not run on a composite factor base")

    monkeypatch.setattr(sympy, "factorint", lambda index: {12: 1})
    monkeypatch.setattr(sympy, "cyclotomic_poly", fail_poly)
    with pytest.raises(OperationBackendError) as exc_info:
        cyclotomic(12)
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
