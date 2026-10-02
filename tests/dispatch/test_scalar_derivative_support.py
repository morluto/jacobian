"""Scalar derivative sparse admission through the public operation boundary."""

from __future__ import annotations

import json
from itertools import islice, product
from math import comb
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.values import RationalPolynomial
from jacobian.math.polynomials.vector_calculus._models import ScalarResult
from jacobian.math.polynomials.vector_calculus.operations import (
    directional_derivative,
    laplacian,
    verify_directional_derivative,
    verify_laplacian,
)


def _source(
    variables: tuple[str, ...], terms: dict[tuple[int, ...], int]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate_json(
        json.dumps(
            {
                "domain": "QQ",
                "variables": variables,
                "polynomial": {
                    "terms": [
                        {
                            "coefficient": {"num": str(value), "den": "1"},
                            "exponents": key,
                        }
                        for key, value in sorted(terms.items(), reverse=True)
                    ]
                },
            }
        )
    )


def _public_scalar(
    source: RationalPolynomial, weights: tuple[int, ...] | None
) -> ScalarResult:
    payload: dict[str, Any] = {"polynomial": source.model_dump(mode="json")}
    if weights is None:
        operation = "laplacian"
        native = laplacian(source)
        verify = verify_laplacian
    else:
        operation = "directional_derivative"
        direction = tuple(CanonicalRational(num=weight, den=1) for weight in weights)
        payload["direction"] = [
            coordinate.model_dump(mode="json") for coordinate in direction
        ]
        native = directional_derivative(source, direction)
        verify = verify_directional_derivative
    public = invoke_operation(
        f"polynomial_field.scalar.{operation}.compute", payload, Catalog.open()
    ).output
    decoded = ScalarResult.model_validate_json(json.dumps(public), strict=True)
    assert decoded == native
    assert decoded.source_polynomial == source
    assert decoded.result.variables == source.variables
    assert verify(decoded)
    return decoded


@pytest.mark.parametrize(
    "variables", [("x", "y", "z"), ("x", "y", "z", "w"), ("w", "z", "y", "x")]
)
def test_public_sparse_scalar_derivatives_retain_axes(
    variables: tuple[str, ...],
) -> None:
    source = _source(
        variables,
        {
            tuple(degree if name == "x" else 0 for name in variables): 1
            for degree in range(65)
        },
    )
    laplace = _public_scalar(source, None)
    directional = _public_scalar(source, tuple(int(name == "x") for name in variables))
    for result, order in [(laplace, 2), (directional, 1)]:
        assert result.result == _source(
            variables,
            {
                tuple(
                    degree - order if name == "x" else 0 for name in variables
                ): degree if order == 1 else degree * (degree - 1)
                for degree in range(order, 65)
            },
        )


def test_public_harmonic_scalar_laplacian_is_zero() -> None:
    source = _source(
        ("x", "y", "z", "w"),
        {
            (64 - degree, degree, 0, 0): (-1) ** (degree // 2) * comb(64, degree)
            for degree in range(65)
        },
    )
    assert not _public_scalar(source, None).result.polynomial.terms


def test_public_scalar_output_boundary_and_selected_directions() -> None:
    variables = ("x", "y", "z", "w")
    terms = {
        tuple(3 * value + 2 for value in key): 1
        for key in islice(product(range(4), repeat=4), 65)
    }
    source = _source(variables, terms)
    selected = _public_scalar(source, (1, 0, 0, 0))
    assert selected.result == _source(
        variables, {(key[0] - 1, *key[1:]): key[0] for key in terms}
    )
    assert not _public_scalar(source, (0, 0, 0, 0)).result.polynomial.terms
    for weights in [None, (1, 1, 1, 1)]:
        boundary = _source(variables, dict(islice(terms.items(), 64)))
        assert len(_public_scalar(boundary, weights).result.polynomial.terms) == 256
        payload: dict[str, Any] = {"polynomial": source.model_dump(mode="json")}
        operation = "laplacian"
        if weights is not None:
            operation = "directional_derivative"
            payload["direction"] = [{"num": "1", "den": "1"}] * 4
        with pytest.raises(OperationDomainValidationError) as caught:
            invoke_operation(
                f"polynomial_field.scalar.{operation}.compute", payload, Catalog.open()
            )
        assert (
            caught.value.errors()[0]["type"]
            == "polynomial_vector_calc.derivative_term_budget"
        )
