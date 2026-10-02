"""Resultant support admission prices active coefficients, not ambient axes."""

from math import comb

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.multivariate import operations
from jacobian.math.polynomials.multivariate._resultant import (
    MultivariateResultantResult,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

_VARIABLES = ("x", "y", "z", "w", "u", "v", "r", "t")


def _poly(
    variables: tuple[str, ...], terms: tuple[tuple[int, dict[str, int]], ...]
) -> RationalPolynomial:
    coefficients = {
        tuple(powers.get(variable, 0) for variable in variables): coefficient
        for coefficient, powers in terms
    }
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_integer_ratio(coefficient, 1),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(coefficients.items(), reverse=True)
            )
        ),
    )


@pytest.mark.parametrize("power", [5, 6, 63])
@pytest.mark.parametrize(
    "variables",
    [_VARIABLES, ("t", "y", "v", "x", "w", "u", "z", "r"), _VARIABLES[::-1]],
)
def test_inactive_axes_do_not_inflate_resultant_support(
    variables: tuple[str, ...], power: int
) -> None:
    left = _poly(variables, ((1, {"x": 1}), (1, {"y": 1})))
    right = _poly(variables, ((1, {"x": power}),))
    result = operations.multivariate_resultant(left, right, "x")
    remaining = tuple(variable for variable in variables if variable != "x")
    assert result.resultant.kind == "POLYNOMIAL"
    # Res_x(x+y, x^n) = (-y)^n, including the odd-degree swap sign.
    assert result.resultant.value == _poly(remaining, (((-1) ** power, {"y": power}),))
    decoded = MultivariateResultantResult.model_validate_json(result.model_dump_json())
    assert operations.verify_multivariate_resultant(decoded)
    swapped = operations.multivariate_resultant(right, left, "x")
    assert swapped.resultant.kind == "POLYNOMIAL"
    assert swapped.resultant.value == _poly(remaining, ((1, {"y": power}),))


def test_coefficients_of_both_operands_contribute_to_active_degrees() -> None:
    left = _poly(_VARIABLES, ((1, {"x": 2}), (1, {"y": 1})))
    right = _poly(_VARIABLES, ((1, {"x": 3}), (1, {"z": 1})))
    result = operations.multivariate_resultant(left, right, "x")
    assert result.resultant.kind == "POLYNOMIAL"
    # Modulo x^2+y, x^3+z = z-y*x; its norm is z^2+y^3.
    assert result.resultant.value == _poly(
        _VARIABLES[1:], ((1, {"y": 3}), (1, {"z": 2}))
    )


def test_axis_degree_box_admits_exact_output_term_boundary() -> None:
    left = _poly(
        _VARIABLES,
        (
            (1, {"x": 1}),
            (-1, {"y": 1, "z": 1}),
            (-1, {"y": 1}),
            (-1, {"z": 1}),
            (-1, {}),
        ),
    )
    right = _poly(_VARIABLES, ((1, {"x": 31}),))
    result = operations.multivariate_resultant(left, right, "x")
    assert result.resultant.kind == "POLYNOMIAL"
    # Res_x(x-(1+y)(1+z), x^31) = (1+y)^31 (1+z)^31: exactly 1,024 terms.
    expected = _poly(
        _VARIABLES[1:],
        tuple(
            (comb(31, y) * comb(31, z), {"y": y, "z": z})
            for y in range(32)
            for z in range(32)
        ),
    )
    assert result.resultant.value == expected
    decoded = MultivariateResultantResult.model_validate_json(result.model_dump_json())
    assert operations.verify_multivariate_resultant(decoded)


def test_active_total_degree_bound_can_be_tighter_than_axis_box() -> None:
    left = _poly(_VARIABLES, ((1, {"x": 1}), (-1, {"y": 1}), (-1, {"z": 1}), (-1, {})))
    right = _poly(_VARIABLES, ((1, {"x": 40}),))
    result = operations.multivariate_resultant(left, right, "x")
    assert result.resultant.kind == "POLYNOMIAL"
    # (1+y+z)^40 has C(42,2)=861 terms, below its 41^2 axis box.
    assert result.resultant.value == _poly(
        _VARIABLES[1:],
        tuple(
            (comb(40, y) * comb(40 - y, z), {"y": y, "z": z})
            for y in range(41)
            for z in range(41 - y)
        ),
    )


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize(
    "kind", ["zero", "constant", "both_constants", "no_active_axes"]
)
def test_retained_axes_survive_degenerate_resultants(kind: str, swap: bool) -> None:
    left = _poly(_VARIABLES, ((1, {"x": 6}),))
    right = _poly(_VARIABLES, ((1, {"y": 1}),))
    expected = _poly(_VARIABLES[1:], ((1, {"y": 6}),))
    if kind == "zero":
        right = _poly(_VARIABLES, ())
        expected = _poly(_VARIABLES[1:], ())
    elif kind == "both_constants":
        left = _poly(_VARIABLES, ((1, {"z": 64}),))
        expected = _poly(_VARIABLES[1:], ((1, {}),))
    elif kind == "no_active_axes":
        right = _poly(_VARIABLES, ((1, {"x": 1}), (-2, {})))
        expected = _poly(_VARIABLES[1:], ((64, {}),))
    if swap:
        left, right = right, left
    result = operations.multivariate_resultant(left, right, "x")
    assert result.resultant.kind == "POLYNOMIAL"
    assert result.resultant.value == expected
    decoded = MultivariateResultantResult.model_validate_json(result.model_dump_json())
    assert operations.verify_multivariate_resultant(decoded)


@pytest.mark.parametrize("family", ["axis_box", "all_active"])
def test_genuine_active_growth_is_rejected_before_backend(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    terms: tuple[tuple[int, dict[str, int]], ...]
    if family == "axis_box":
        terms = (
            (1, {"x": 1}),
            (-1, {"y": 1, "z": 1}),
            (-1, {"y": 1}),
            (-1, {"z": 1}),
            (-1, {}),
        )
        power = 32  # (1+y)^32 (1+z)^32 has 1,089 terms.
    else:
        terms = tuple((1, {variable: 1}) for variable in _VARIABLES)
        power = 7  # (y+z+w+u+v+r+t)^7 has C(13,6)=1,716 terms.
    left = _poly(_VARIABLES, terms)
    right = _poly(_VARIABLES, ((1, {"x": power}),))

    def unexpected_backend(*args: object, **kwargs: object) -> None:
        pytest.fail("resultant output admission must precede the backend")

    monkeypatch.setattr(operations, "_sylvester_resultant_value", unexpected_backend)
    with pytest.raises(
        OperationDomainValidationError, match="resultant output"
    ) as error:
        operations.multivariate_resultant(left, right, "x")
    assert error.value.errors()[0]["type"] == "polynomial.multivariate_contract"
