"""Serialized gradient values compose with both public vector consumers."""

from fractions import Fraction

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.polynomials.vector_calculus._models import ScalarResult, VectorResult
from jacobian.math.polynomials.vector_calculus.operations import (
    verify_curl,
    verify_divergence,
)


def _coefficients(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        term.exponents: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _partial(
    terms: dict[tuple[int, ...], Fraction], axis: int
) -> dict[tuple[int, ...], Fraction]:
    """Independent coefficient-map differentiation, without SymPy."""

    result = {}
    for exponents, coefficient in terms.items():
        if exponents[axis]:
            powers = list(exponents)
            powers[axis] -= 1
            result[tuple(powers)] = coefficient * exponents[axis]
    return result


@pytest.mark.parametrize(
    "coefficient", [10**126, 10**128 - 1], ids=["128-digits", "130-digits"]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_public_gradient_composes_at_coefficient_expansion_boundary(
    coefficient: int, mixed: bool
) -> None:
    terms: dict[tuple[int, ...], Fraction] = {(64, 0, 0): Fraction(coefficient)}
    if mixed:
        terms.update(
            {(31, 32, 1): Fraction(-coefficient, 7), (0, 2, 1): Fraction(5, 11)}
        )
    source = {
        "variables": ["z", "x", "y"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {
                        "num": str(value.numerator),
                        "den": str(value.denominator),
                    },
                    "exponents": list(powers),
                }
                for powers, value in sorted(terms.items(), reverse=True)
            ]
        },
    }
    catalog = Catalog.open()
    produced = invoke_operation(
        "polynomial_field.scalar.gradient.compute", {"polynomial": source}, catalog
    )
    gradient = VectorResult.model_validate_json(encode_strict_json(produced.output))
    expected_gradient = tuple(_partial(terms, axis) for axis in range(3))
    assert tuple(map(_coefficients, gradient.components)) == expected_gradient
    assert len(str(gradient.components[0].polynomial.terms[0].coefficient.num)) == (
        128 if coefficient == 10**126 else 130
    )
    # Feed the actual public payload unchanged: no coefficient rescaling or
    # manual reconstruction of the producer's components.
    payload = {"components": produced.output["components"]}
    divergence_wire = invoke_operation(
        "polynomial_field.vector.divergence.compute", payload, catalog
    )
    curl_wire = invoke_operation(
        "polynomial_field.vector.curl.compute", payload, catalog
    )
    direct_wire = invoke_operation(
        "polynomial_field.scalar.laplacian.compute", {"polynomial": source}, catalog
    )
    divergence = ScalarResult.model_validate_json(
        encode_strict_json(divergence_wire.output)
    )
    curl = VectorResult.model_validate_json(encode_strict_json(curl_wire.output))
    direct = ScalarResult.model_validate_json(encode_strict_json(direct_wire.output))
    expected_laplacian: dict[tuple[int, ...], Fraction] = {}
    for axis, component in enumerate(expected_gradient):
        for powers, value in _partial(component, axis).items():
            expected_laplacian[powers] = (
                expected_laplacian.get(powers, Fraction()) + value
            )
    assert _coefficients(divergence.result) == expected_laplacian
    assert divergence.result == direct.result
    assert all(not component.polynomial.terms for component in curl.components)
    assert divergence.source_components == curl.source_components == gradient.components
    assert divergence.result.variables == ("z", "x", "y")
    assert all(component.variables == ("z", "x", "y") for component in curl.components)
    assert verify_divergence(divergence)
    assert verify_curl(curl)


def test_public_zero_gradient_composes_on_retained_axes() -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "polynomial_field.scalar.gradient.compute",
        {"polynomial": {"variables": ["z", "x", "y"], "polynomial": {"terms": []}}},
        catalog,
    )
    gradient = VectorResult.model_validate_json(encode_strict_json(produced.output))
    payload = {"components": produced.output["components"]}
    for operation in ("divergence", "curl"):
        result = invoke_operation(
            f"polynomial_field.vector.{operation}.compute", payload, catalog
        )
        if operation == "divergence":
            scalar = ScalarResult.model_validate_json(encode_strict_json(result.output))
            assert scalar.result == gradient.components[0]
            assert scalar.source_components == gradient.components
        else:
            vector = VectorResult.model_validate_json(encode_strict_json(result.output))
            assert vector.components == gradient.components
            assert vector.source_components == gradient.components
