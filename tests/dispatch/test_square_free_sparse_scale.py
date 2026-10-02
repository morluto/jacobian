"""Native and public square-free calls share the compact-source envelope."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.polynomials._models import PolynomialSquareFreeDecompositionResult
from jacobian.math.polynomials.operations import (
    polynomial_square_free_decomposition,
    verify_polynomial_square_free_decomposition,
)
from jacobian.math.polynomials.values import RationalPolynomial

_OPERATION = "polynomial.compute.square_free_decomposition"


def _source(degree: int, *, constant: bool = True) -> dict[str, Any]:
    return {
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [exponent]}
                for exponent in ((degree, 0) if constant else (degree,))
            ]
        },
    }


@pytest.mark.parametrize("degree", (64, 65, 256, 500, 32768))
def test_public_compact_source_matches_native_and_strict_decoding(degree: int) -> None:
    source = _source(degree)
    output = invoke_operation(_OPERATION, {"polynomial": source}, Catalog.open()).output
    claim = PolynomialSquareFreeDecompositionResult.model_validate_json(
        json.dumps(output), strict=True
    )
    assert claim.model_dump(mode="json") == output
    assert claim == polynomial_square_free_decomposition(claim.polynomial)
    assert output["polynomial"] == output["reconstructed"] == source
    assert output["coefficient"] == {"num": "1", "den": "1"}
    assert output["factors"] == [{"factor": source, "multiplicity": 1}]
    assert verify_polynomial_square_free_decomposition(claim)


def test_actual_factorization_output_is_an_unchanged_square_free_input() -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "polynomial.factor.compute", {"polynomial": _source(256)}, catalog
    ).output
    assert len(produced["factors"]) == 1
    factor = produced["factors"][0]["factor"]
    assert factor == _source(256)
    output = invoke_operation(_OPERATION, {"polynomial": factor}, catalog).output
    assert output["polynomial"] == output["reconstructed"] == factor
    assert output["factors"] == [{"factor": factor, "multiplicity": 1}]
    claim = PolynomialSquareFreeDecompositionResult.model_validate_json(
        json.dumps(output), strict=True
    )
    assert verify_polynomial_square_free_decomposition(claim)


def test_public_monomial_multiplicity_boundary_matches_native() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        _OPERATION, {"polynomial": _source(64, constant=False)}, catalog
    ).output
    assert result["factors"][0]["multiplicity"] == 64
    source = _source(65, constant=False)
    with pytest.raises(OperationResourceAdmissionError) as public:
        invoke_operation(_OPERATION, {"polynomial": source}, catalog)
    with pytest.raises(OperationResourceAdmissionError) as native:
        polynomial_square_free_decomposition(
            RationalPolynomial.model_validate_json(json.dumps(source), strict=True)
        )
    assert public.value.errors() == native.value.errors()
    assert (
        public.value.errors()[0]["type"] == "polynomial.square_free_multiplicity_budget"
    )


def test_public_unrecognized_high_degree_shape_keeps_typed_refusal() -> None:
    source = _source(65)
    source["polynomial"]["terms"][1]["exponents"] = [1]
    with pytest.raises(OperationDomainValidationError) as public:
        invoke_operation(_OPERATION, {"polynomial": source}, Catalog.open())
    with pytest.raises(OperationDomainValidationError) as native:
        polynomial_square_free_decomposition(
            RationalPolynomial.model_validate_json(json.dumps(source), strict=True)
        )
    assert public.value.errors() == native.value.errors()


def test_binomial_lift_does_not_widen_the_canonical_exponent_carrier() -> None:
    with pytest.raises(OperationRequestValidationError) as error:
        invoke_operation(_OPERATION, {"polynomial": _source(32769)}, Catalog.open())
    assert error.value.errors()[0]["type"] == "polynomial.exponent_bound"
