"""Canonical public derivative outputs remain usable by the same operation."""

import json

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.values import RationalPolynomial


def test_repeated_public_differentiation_preserves_expanded_coefficients() -> None:
    catalog = Catalog.open()
    coefficient = 10**256 - 1
    polynomial = {
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {
                    "coefficient": {"num": str(coefficient), "den": "1"},
                    "exponents": [127],
                }
            ]
        },
    }
    expected_coefficient = coefficient
    for exponent in range(127, 123, -1):
        result = invoke_operation(
            "polynomial.rational.compute.derivative",
            {"polynomial": polynomial},
            catalog,
        )
        # Pass the serialized producer value without caller-side reconstruction.
        polynomial = json.loads(json.dumps(result.output["derivative"]))
        decoded = RationalPolynomial.model_validate_json(json.dumps(polynomial))
        expected_coefficient *= exponent
        assert decoded.variables == ("x",)
        assert len(decoded.polynomial.terms) == 1
        term = decoded.polynomial.terms[0]
        assert term.exponents == (exponent - 1,)
        assert term.coefficient.as_integer_ratio() == (expected_coefficient, 1)
