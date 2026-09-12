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
    cyclotomic,
)


def ascending(result: CyclotomicResult) -> list[int]:
    return list(reversed(result.polynomial.coefficients))


def test_known_twelfth_cyclotomic() -> None:
    result = cyclotomic(CyclotomicRequest(index=12))
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
    result = cyclotomic(CyclotomicRequest(index=index))
    assert result.polynomial.coefficients == coefficients


@pytest.mark.parametrize(
    ("index", "coefficients"),
    ((1, (1, -1)), (2, (1, 1)), (6, (1, -1, 1))),
)
def test_boundary_and_small_cyclotomic_values(
    index: int, coefficients: tuple[int, ...]
) -> None:
    result = cyclotomic(CyclotomicRequest(index=index))
    assert result.totient == len(coefficients) - 1
    assert result.polynomial.coefficients == coefficients


def test_native_api_exports_typed_cyclotomic_operation() -> None:
    result = polynomials.cyclotomic(CyclotomicRequest(index=3))
    assert isinstance(result, CyclotomicResult)
    assert result.polynomial.coefficients == (1, 1, 1)


def test_direct_native_boundary_rejects_untyped_payload() -> None:
    with pytest.raises(OperationDomainValidationError, match="CyclotomicRequest"):
        cyclotomic({"index": 3})  # type: ignore[arg-type]


def test_result_validation_checks_shape_without_backend_recomputation() -> None:
    result = cyclotomic(CyclotomicRequest(index=12))
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
        cyclotomic(CyclotomicRequest(index=3))
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
        cyclotomic(CyclotomicRequest(index=2))
    assert exc_info.value.reason is BackendFailureReason.INVALID_OUTPUT


def test_serialized_result_preserves_the_canonical_integer_polynomial() -> None:
    result = cyclotomic(CyclotomicRequest(index=12))
    decoded = CyclotomicResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.polynomial.coefficients == (1, 0, -1, 0, 1)


def test_expired_request_is_rejected_before_factorization() -> None:
    from time import monotonic

    with (
        request_execution(monotonic(), outer_deadline=monotonic() - 1),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        cyclotomic(CyclotomicRequest(index=12))


def test_large_degree_is_admitted_before_backend_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        cyclotomic(CyclotomicRequest(index=100_000))


def test_divisor_product_identity_through_twenty() -> None:
    for index in range(1, 21):
        product = fmpz_poly([1])
        for divisor in range(1, index + 1):
            if index % divisor == 0:
                result = cyclotomic(CyclotomicRequest(index=divisor))
                product *= fmpz_poly(ascending(result))
        expected = fmpz_poly([-1] + [0] * (index - 1) + [1])
        assert product == expected
