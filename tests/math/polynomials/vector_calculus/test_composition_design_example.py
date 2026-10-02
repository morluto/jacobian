"""A no-gap design outcome: the exact div(grad(f)) composition already exists."""

import json

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)
from jacobian.math.polynomials.vector_calculus._models import VectorFieldRequest
from jacobian.math.polynomials.vector_calculus.operations import (
    divergence,
    gradient,
    laplacian,
)


def _polynomial(
    axes: tuple[str, ...], terms: dict[tuple[int, ...], int]
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=axes,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    exponents=e, coefficient=CanonicalRational.from_integer_ratio(c, 1)
                )
                for e, c in sorted(terms.items(), reverse=True)
                if c
            )
        ),
    )


@pytest.mark.parametrize("axes", [("x", "y"), ("y", "x")])
@pytest.mark.parametrize(
    "terms,expected",
    [
        ({(2, 0): 1, (0, 2): 1}, {(0, 0): 4}),
        ({(3, 0): 1, (1, 2): 1}, {(1, 0): 8}),
        ({(0, 0): 5}, {}),
    ],
)
def test_existing_gradient_divergence_composition(
    axes: tuple[str, ...],
    terms: dict[tuple[int, ...], int],
    expected: dict[tuple[int, ...], int],
) -> None:
    source = _polynomial(axes, terms)
    produced = gradient(source).model_dump(mode="json")
    consumer = VectorFieldRequest.model_validate_json(
        json.dumps({"components": produced["components"]})
    )
    composed = divergence(consumer.components)
    oracle = _polynomial(axes, expected)
    assert composed.result == oracle
    assert composed.result.variables == axes
    assert laplacian(source).result == oracle
