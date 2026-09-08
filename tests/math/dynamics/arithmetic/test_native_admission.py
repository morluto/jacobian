"""Native arithmetic-dynamics admission and claim failure boundaries."""

import json
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
import sympy

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.dynamics.arithmetic import _tools as wire
from jacobian.math.dynamics.arithmetic import operations as native
from jacobian.math.dynamics.arithmetic._models import MapIterateRequest


class OversizedCoefficients(list[int]):
    def __iter__(self) -> Any:
        pytest.fail("oversized input was converted before length admission")


def test_coefficients_length_is_admitted_before_conversion() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        native.polynomial_from_coefficients(OversizedCoefficients([1] * 32))


def test_dynatomic_index_is_admitted_before_coefficient_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polynomial = native.polynomial_from_coefficients((0, 0, 1))
    monkeypatch.setattr(
        native,
        "polynomial_coefficients",
        lambda *_: pytest.fail("expanded before index admission"),
    )
    with pytest.raises(ValueError, match="index"):
        native.dynatomic_polynomial(polynomial, 10**100)


def test_cycle_multiplier_admits_and_converts_its_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polynomial = native.polynomial_from_coefficients((-1, 0, 1))
    original = native._require_input_polynomial
    calls = 0

    def count(source: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(native, "_require_input_polynomial", count)
    assert (
        native.cycle_multiplier(
            polynomial,
            (
                CanonicalRational.from_fraction(Fraction(0)),
                CanonicalRational.from_fraction(Fraction(-1)),
            ),
        ).as_fraction()
        == 0
    )
    assert calls == 1


@pytest.mark.parametrize("operation", wire.TOOLS)
def test_claim_verifiers_propagate_backend_value_errors(
    operation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = operation.request_type.model_validate_json(
        json.dumps(operation.examples[0].input)
    )
    # Every declaration has its native verifier in this owner.
    names: dict[str, tuple[Callable[..., Any], Callable[..., bool], str]] = {
        "arithmetic_dynamics.map.iterate.compute": (
            wire.compute_map_iterate,
            wire.verify_map_iterate,
            "compose",
        ),
        "arithmetic_dynamics.point.orbit.compute": (
            wire.compute_orbit_prefix,
            wire.verify_orbit_prefix,
            "eval",
        ),
        "arithmetic_dynamics.dynatomic_polynomial.compute": (
            wire.compute_dynatomic_polynomial,
            wire.verify_dynatomic_polynomial,
            "compose",
        ),
        "arithmetic_dynamics.cycle.multiplier.compute": (
            wire.compute_cycle_multiplier,
            wire.verify_cycle_multiplier,
            "eval",
        ),
        "arithmetic_dynamics.finite_field.functional_graph.compute": (
            wire.compute_finite_field_map,
            wire.verify_finite_field_map,
            "isprime",
        ),
    }
    compute, verify, method = names[operation.operation_id]
    claim = compute(request)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("unexpected backend failure")

    monkeypatch.setattr(sympy if method == "isprime" else sympy.Poly, method, fail)
    with pytest.raises(ValueError, match="unexpected backend failure"):
        verify(claim)


def test_unadmitted_iterate_claim_is_not_refuted() -> None:
    request = MapIterateRequest(
        polynomial=native.polynomial_from_coefficients((0, 0, 1)), n=1
    )
    claim = wire.compute_map_iterate(request)
    authored = claim.model_copy(update={"n": 11})
    restored = type(claim).model_validate_json(authored.model_dump_json())
    with pytest.raises(OperationResourceAdmissionError, match="degree"):
        wire.verify_map_iterate(restored)


def test_finite_field_coefficients_length_is_admitted_before_conversion() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        native.finite_field_functional_graph(OversizedCoefficients([1] * 32), 5)


def test_source_degree_is_admitted_before_backend_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = native.iterate_polynomial(
        native.polynomial_from_coefficients((0, 0, 1)), 5
    )
    monkeypatch.setattr(
        native,
        "rational_polynomial_to_sympy",
        lambda *_: pytest.fail("converted before input degree admission"),
    )
    with pytest.raises(ValueError, match="degree"):
        native.orbit_prefix(source, CanonicalRational.from_fraction(Fraction(0)), 1)
