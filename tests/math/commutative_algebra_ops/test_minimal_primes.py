"""Tests for exact source-bound rational ideal minimal primes."""

from __future__ import annotations

import shutil

import pytest
from pydantic import ValidationError

from jacobian.math.commutative_algebra_ops import _singular
from jacobian.math.commutative_algebra_ops._models import (
    IdealMinimalPrimesRequest,
    IdealMinimalPrimesResult,
)
from jacobian.math.commutative_algebra_ops._operations import (
    compute_ideal_minimal_primes,
)
from jacobian.math.commutative_algebra_ops._singular import SingularMinimalPrimesResult
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
)


def _poly(
    variables: tuple[str, ...],
    *terms: tuple[int, int, tuple[int, ...]],
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "polynomial_schema_version": "1",
            "domain": "QQ",
            "variables": list(variables),
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": str(numerator), "den": str(denominator)},
                        "exponents": list(exponents),
                    }
                    for numerator, denominator, exponents in terms
                ]
            },
        }
    )


def _ideal(
    variables: tuple[str, ...], *generators: RationalPolynomial
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(variables=variables, generators=generators)


def _axes_request() -> IdealMinimalPrimesRequest:
    variables = ("x", "y")
    return IdealMinimalPrimesRequest(
        ideal=_ideal(variables, _poly(variables, (1, 1, (1, 1))))
    )


def _axes_components() -> tuple[RationalPolynomialIdeal, ...]:
    variables = ("x", "y")
    return tuple(
        sorted(
            (
                _ideal(variables, _poly(variables, (1, 1, (1, 0)))),
                _ideal(variables, _poly(variables, (1, 1, (0, 1)))),
            ),
            key=lambda ideal: ideal.model_dump_json(),
        )
    )


def test_computed_family_replays_the_complete_source_bound_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    calls = 0

    def replay(
        source: RationalPolynomialIdeal, budget: object
    ) -> SingularMinimalPrimesResult:
        nonlocal calls
        calls += 1
        assert source == request.ideal
        return SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        )

    monkeypatch.setattr(_singular, "run_singular_minimal_primes", replay)
    result = IdealMinimalPrimesResult(
        request=request,
        outcome="COMPUTED",
        components=components,
        backend_version="4.4.0",
    )

    assert result.components == components
    assert calls == 1


def test_computed_family_rejects_a_missing_minimal_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()

    monkeypatch.setattr(
        _singular,
        "run_singular_minimal_primes",
        lambda source, budget: SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        ),
    )

    with pytest.raises(ValidationError, match="complete minimal-prime family"):
        IdealMinimalPrimesResult(
            request=request,
            outcome="COMPUTED",
            components=components[:1],
            backend_version="4.4.0",
        )


def test_missing_backend_is_typed_and_makes_no_component_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._operations.run_singular_minimal_primes",
        lambda source, budget: SingularMinimalPrimesResult(
            outcome="UNAVAILABLE", detail="backend is unavailable"
        ),
    )

    result = compute_ideal_minimal_primes(_axes_request())

    assert result.outcome == "UNAVAILABLE"
    assert result.components is None


def test_producer_replays_once_without_a_third_backend_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    calls = 0

    def backend(
        source: RationalPolynomialIdeal, budget: object
    ) -> SingularMinimalPrimesResult:
        nonlocal calls
        calls += 1
        assert source == request.ideal
        return SingularMinimalPrimesResult(
            outcome="COMPUTED", components=components, backend_version="4.4.0"
        )

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._operations.run_singular_minimal_primes",
        backend,
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "COMPUTED"
    assert result.components == components
    assert calls == 2


def test_nonreproducible_producer_output_is_a_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _axes_request()
    components = _axes_components()
    calls = 0

    def backend(
        source: RationalPolynomialIdeal, budget: object
    ) -> SingularMinimalPrimesResult:
        nonlocal calls
        calls += 1
        return SingularMinimalPrimesResult(
            outcome="COMPUTED",
            components=components if calls == 1 else components[:1],
            backend_version="4.4.0",
        )

    monkeypatch.setattr(
        "jacobian.math.commutative_algebra_ops._operations.run_singular_minimal_primes",
        backend,
    )

    result = compute_ideal_minimal_primes(request)

    assert result.outcome == "ERROR"
    assert result.components is None


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_coordinate_axes_are_the_two_qq_minimal_primes() -> None:
    result = compute_ideal_minimal_primes(_axes_request())

    assert result.outcome == "COMPUTED"
    assert result.components == _axes_components()


@pytest.mark.skipif(
    shutil.which("Singular") is None,
    reason="Singular 4.4 backend is not installed",
)
def test_unit_zero_and_embedded_sources_have_their_exact_family_shapes() -> None:
    variables = ("x", "y")
    zero = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(ideal=_ideal(variables, _poly(variables)))
    )
    unit = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(
            ideal=_ideal(variables, _poly(variables, (1, 1, (0, 0))))
        )
    )
    embedded = compute_ideal_minimal_primes(
        IdealMinimalPrimesRequest(
            ideal=_ideal(
                variables,
                _poly(variables, (1, 1, (2, 0))),
                _poly(variables, (1, 1, (1, 1))),
            )
        )
    )

    assert zero.outcome == unit.outcome == embedded.outcome == "COMPUTED"
    assert zero.components == (_ideal(variables, _poly(variables)),)
    assert unit.components == ()
    assert embedded.components == (_ideal(variables, _poly(variables, (1, 1, (1, 0)))),)
