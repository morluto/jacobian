"""Value-specific publication checks for polynomial group-action examples.

These assert that each published example executes and produces the exact
admitted result, which the catalog-wide example lane does not check. They
live here rather than under tests/math because they boot the product catalog,
dispatch, and canonicalization boundaries.
"""

import json
from fractions import Fraction

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationMatchRequest,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.derivations._models import GaActionRequest
from jacobian.math.polynomials.derivations._tools import TOOLS
from jacobian.math.polynomials.derivations._weight_models import (
    PolynomialWeightSubrepresentationRequest,
)
from jacobian.math.polynomials.derivations._weight_operations import (
    diagonal_weight_action,
    gm_generated_subrepresentation,
)
from jacobian.math.polynomials.values import RationalPolynomial


def test_ga_action_catalog_example_and_request_are_publishable() -> None:
    tool = next(
        item
        for item in TOOLS
        if item.operation_id == "algebraic_group.ga.action_from_derivation.compute"
    )
    request = tool.request_type.model_validate_json(
        json.dumps(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.generator_images[0].variables == ("x", "y", "t")
    assert GaActionRequest.model_validate(request.model_dump()) == request

    assert result.parameter == "t"
    assert [
        (term.coefficient.num, term.coefficient.den, term.exponents)
        for term in result.generator_images[0].polynomial.terms
    ] == [(1, 1, (1, 0, 0)), (1, 1, (0, 1, 1))]
    catalog = Catalog(TOOLS)
    operation = catalog.operation(tool.operation_id)
    assert operation is not None
    invocation = invoke_operation(
        operation.operation_id, operation.examples[0].input, catalog
    )
    assert invocation.output["parameter"] == "t"
    assert invocation.output["generator_images"][0]["polynomial"]["terms"] == [
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1, 0, 0]},
        {"coefficient": {"num": "1", "den": "1"}, "exponents": [0, 1, 1]},
    ]


def _poly(
    variables: tuple[str, ...], *terms: tuple[int, tuple[int, ...]]
) -> RationalPolynomial:
    return RationalPolynomial.model_validate(
        {
            "variables": variables,
            "polynomial": {
                "terms": [
                    {
                        "coefficient": {"num": coefficient, "den": 1},
                        "exponents": exponents,
                    }
                    for coefficient, exponents in sorted(
                        terms, key=lambda item: item[1], reverse=True
                    )
                ]
            },
        }
    )


def _coefficient_map(polynomial: RationalPolynomial) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def test_support_is_admitted_before_projection_and_catalog_publishes_operation() -> (
    None
):
    terms = tuple((1, (degree,)) for degree in range(64, 0, -1))
    with pytest.raises(OperationResourceAdmissionError):
        gm_generated_subrepresentation(
            {
                "action": {"variables": ["x"], "weights": [1]},
                "generators": [_poly(("x",), *terms) for _ in range(5)],
            }
        )

    operation = Catalog.open().operation(
        "algebraic_group.gm.finite_subrepresentation.compute"
    )
    assert operation is not None
    assert operation.request_type.__name__ == "PolynomialWeightSubrepresentationRequest"
    assert operation.result_type.__name__ == "PolynomialWeightSubrepresentationResult"
    parsed = PolynomialWeightSubrepresentationRequest.model_validate(
        {
            "action": {"variables": ["x"], "weights": [3]},
            "generators": [_poly(("x",), (1, (1,)))],
        }
    )
    assert operation.run(parsed).weights == (3,)


def test_weight_action_is_catalogued_with_valid_example() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.math.polynomials.derivations._tools import TOOLS

    operation_id = "algebraic_group.gm.diagonal_weight_action.compute"
    tool = next(item for item in TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    assert (
        tool.run(request).weight_zero
        == diagonal_weight_action(
            request.action, request.polynomial, request.parameter
        ).weight_zero
    )
    assert operation_id in {item.operation_id for item in TOOLS}
    found = Catalog.open().match(OperationMatchRequest(need="diagonal integer weights"))
    assert any(item.operation_id == operation_id for item in found.matches)


def test_operation_is_catalogued_with_usable_example() -> None:
    from jacobian.catalog.catalog import Catalog
    from jacobian.math.polynomials.derivations._tools import TOOLS

    operation_id = "algebraic_group.gm.invariants_through_degree.compute"
    tool = next(item for item in TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(
        encode_strict_json(tool.examples[0].input), strict=True
    )
    result = tool.run(request)
    assert result.dimension == sum(row.dimension for row in result.hilbert_prefix)
    assert operation_id in {item.operation_id for item in TOOLS}
    found = Catalog.open().match(
        OperationMatchRequest(need="weight zero polynomial basis by degree")
    )
    assert any(item.operation_id == operation_id for item in found.matches)
