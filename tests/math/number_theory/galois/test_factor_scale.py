"""Motivating packets, independent reconstruction and bounded execution."""

from threading import Event

import pytest
from flint import fmpz_mod_poly_ctx, fmpz_poly, nmod_poly

from jacobian._execution import OperationExecutionCancelledError, request_cancellation
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.galois._models import (
    GaloisFactorRequest,
    GaloisFactorResult,
)
from jacobian.math.number_theory.galois._tools import _galois_factor
from jacobian.math.number_theory.galois.operations import galois_factor


def _packets() -> list[tuple[fmpz_poly, int]]:
    y = fmpz_poly([0, 1])
    b = y * (y - 1) * (y - 2)
    a1 = 2 * y**2 * (y - 1) * (y - 2) ** 2
    a2 = 2 * y**2 * (y - 1) ** 2 * (y - 2)
    r = 1 + b * (y + 10)
    z1, z2 = a1**2 * a2**3 * r**5, a1 * a2**2 * r**3
    return [
        (1 + z1, 163),
        (z1**2 - z1 + 1, 5),
        (1 + z2, 79),
        (z2**4 - z2**3 + z2**2 - z2 + 1, 3),
    ]


@pytest.mark.parametrize("polynomial,prime", _packets())
def test_motivating_irreducible_packets(polynomial: fmpz_poly, prime: int) -> None:
    coefficients = tuple(int(c) % prime for c in polynomial.coeffs())
    request = GaloisFactorRequest(field_order=prime, coefficients=coefficients)
    result = _galois_factor(request)
    assert result.is_irreducible
    assert fmpz_mod_poly_ctx(prime)(list(coefficients)).is_irreducible()
    assert result == galois_factor(prime, coefficients)
    assert GaloisFactorResult.model_validate_json(result.model_dump_json()) == result


def test_repeated_degree_108_reconstructs_with_an_independent_backend() -> None:
    coefficients = (1,) + (0,) * 107 + (1,)
    result = galois_factor(3, coefficients)
    reconstructed = nmod_poly([result.unit], 3)
    for factor in result.factors:
        polynomial = nmod_poly(list(factor.coefficients), 3)
        assert fmpz_mod_poly_ctx(3)(list(factor.coefficients)).is_irreducible()
        reconstructed *= polynomial**factor.multiplicity
    assert reconstructed == nmod_poly(list(coefficients), 3)
    assert not result.is_irreducible
    assert any(factor.multiplicity > 1 for factor in result.factors)


def test_dense_work_rejection_and_cancellation_are_operational() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        galois_factor(251, (1,) * 129)
    event = Event()
    event.set()
    with request_cancellation(event), pytest.raises(OperationExecutionCancelledError):
        galois_factor(3, (1,) + (0,) * 107 + (1,))


def test_native_modulus_is_bounded_before_primality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sympy

    def unexpected(value: object) -> None:
        pytest.fail("out-of-envelope modulus must not reach primality testing")

    monkeypatch.setattr(sympy, "isprime", unexpected)
    with pytest.raises(OperationDomainValidationError, match=r"2\.\.251"):
        galois_factor(10**10000 + 1, (1, 1))
