"""Wider exact infinity residues compose through their canonical scalar carrier."""

import json
from fractions import Fraction

import pytest
from sympy import Symbol

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.structured_operations import (
    residue_at_infinity,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT,
)


@pytest.mark.parametrize(
    "kind", ("quartic_zero", "quartic_pole", "height", "full_degree", "improper")
)
def test_residue_native_and_public_acceptance_has_exact_known_answers(
    kind: str,
) -> None:
    x = Symbol("t")
    bound = MAX_RATIONAL_FUNCTION_REPRESENTATION_EXPONENT
    expressions = {
        "quartic_zero": (1 / (x**4 + 1), Fraction(0)),
        "quartic_pole": (x**3 / (x**4 + 1), Fraction(-1)),
        "height": (1 / (100 * (x - 1)), Fraction(-1, 100)),
        "full_degree": (x ** (bound - 1) / (x**bound + 1), Fraction(-1)),
        "improper": (x**8 / (x - 2), Fraction(-256)),
    }
    expression, expected = expressions[kind]
    source = rational_function_from_sympy(expression, ("t",))
    catalog = Catalog.open()
    public = invoke_operation(
        "rational_function.residue_at_infinity.compute",
        {"function": source.model_dump(mode="json")},
        catalog,
    )
    decoded = CanonicalRational.model_validate_json(json.dumps(public.output))
    assert decoded == residue_at_infinity(source)
    assert decoded.as_fraction() == expected
    scaled = invoke_operation(
        "formal_series.rational.scalar_multiply.compute",
        {
            "series": {
                "variable": "t",
                "truncation_order": 2,
                "coefficients": [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
            },
            "scalar": public.output,
        },
        catalog,
    )
    values = scaled.output["result"]["coefficients"]
    assert [
        CanonicalRational.model_validate_json(json.dumps(value)).as_fraction()
        for value in values
    ] == [expected, 2 * expected]
