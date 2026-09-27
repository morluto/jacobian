"""Exact graded projections of sparse noncommutative polynomials."""

import json
from collections import Counter
from itertools import islice, product
from typing import Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial
from jacobian.math.free_algebras.homogeneous_component._models import (
    FreeAlgebraHomogeneousComponent,
    FreeAlgebraHomogeneousComponentRequest,
)
from jacobian.math.free_algebras.homogeneous_component.operations import (
    homogeneous_component,
)


def _term(word: list[str], coefficient: int = 1) -> dict[str, object]:
    return {
        "coefficient": {"num": coefficient, "den": 1},
        "word": word,
    }


def _polynomial() -> FreeAlgebraPolynomial:
    return FreeAlgebraPolynomial.model_validate(
        {
            "alphabet": ["x", "y"],
            "terms": [
                _term(["x", "y"], 3),
                _term(["y"], 2),
                _term(["x"], 1),
                _term([], 5),
            ],
        }
    )


def test_projection_preserves_exact_terms_parent_and_degree_after_round_trip() -> None:
    result = homogeneous_component(_polynomial(), 1)
    restored = FreeAlgebraHomogeneousComponent.model_validate(result.model_dump())
    assert restored.degree == 1
    assert restored.polynomial.alphabet == ("x", "y")
    assert [term.word for term in restored.polynomial.terms] == [("y",), ("x",)]
    assert [term.coefficient.num for term in restored.polynomial.terms] == [2, 1]


def test_degree_family_reconstructs_source_and_preserves_zero_parent() -> None:
    source = _polynomial()
    components = [homogeneous_component(source, degree) for degree in range(4)]
    projected = Counter(
        (term.word, term.coefficient.num, term.coefficient.den)
        for component in components
        for term in component.polynomial.terms
    )
    original = Counter(
        (term.word, term.coefficient.num, term.coefficient.den) for term in source.terms
    )
    assert projected == original

    missing = homogeneous_component(source, 3)
    assert missing.degree == 3
    assert missing.polynomial.is_zero
    assert missing.polynomial.alphabet == source.alphabet


def test_native_projection_composes_with_polynomial_multiplication() -> None:
    component = FreeAlgebraHomogeneousComponent.model_validate_json(
        json.dumps(homogeneous_component(_polynomial(), 1).model_dump(mode="json"))
    )
    from jacobian.math.free_algebras.operations import multiply

    product_result = multiply(
        component.polynomial,
        FreeAlgebraPolynomial.model_validate(
            {"alphabet": ["x", "y"], "terms": [_term(["x"])]}
        ),
    )
    words = [term.word for term in product_result.product.terms]
    assert words == [("y", "x"), ("x", "x")]


def test_large_selected_support_is_admitted_by_mathematical_bounds() -> None:
    alphabet = tuple(chr(ord("A") + index) * 64 for index in range(26))
    words = list(islice(product(alphabet, repeat=8), 4096))
    polynomial = FreeAlgebraPolynomial.model_validate(
        {
            "alphabet": alphabet,
            "terms": [_term(list(word)) for word in reversed(words)],
        }
    )
    result = homogeneous_component(polynomial, 8)
    assert len(result.polynomial.terms) == len(words)


def test_native_operation_rejects_invalid_or_over_bound_degree() -> None:
    polynomial = _polynomial()
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        homogeneous_component(polynomial, -1)
    assert homogeneous_component(polynomial, 65).polynomial.is_zero
    with pytest.raises(OperationDomainValidationError, match="nonnegative integer"):
        homogeneous_component(polynomial, True)
    forged = FreeAlgebraPolynomial.model_construct(alphabet=("x", "x"), terms=())
    with pytest.raises(OperationDomainValidationError, match="not canonical"):
        homogeneous_component(forged, 0)
    oversized = FreeAlgebraPolynomial.model_construct(
        alphabet=(), terms=(None,) * 600001
    )
    with pytest.raises(OperationResourceAdmissionError, match="scan bound"):
        homogeneous_component(oversized, 0)


def _component_wire(degree: str) -> dict[str, object]:
    return {
        "degree": degree,
        "polynomial": _polynomial().model_dump(mode="json"),
    }


def test_degree_wire_encoding_preserves_integers_beyond_json_numbers() -> None:
    degree = (1 << 53) + 1
    result = homogeneous_component(_polynomial(), degree)
    wire = json.loads(result.model_dump_json())
    assert wire["degree"] == str(degree)
    restored = FreeAlgebraHomogeneousComponent.model_validate_json(
        json.dumps(wire), strict=True
    )
    assert restored.degree == degree
    assert restored.polynomial.is_zero


def test_request_degree_wire_encoding_round_trips_exactly() -> None:
    degree = (1 << 53) + 1
    payload = {
        "polynomial": _polynomial().model_dump(mode="json"),
        "degree": str(degree),
    }
    request = FreeAlgebraHomogeneousComponentRequest.model_validate_json(
        json.dumps(payload), strict=True
    )
    assert request.degree == degree
    assert request.model_dump()["degree"] == degree


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_degree_schema_publishes_canonical_nonnegative_decimal_strings(
    mode: Literal["validation", "serialization"],
) -> None:
    schema = FreeAlgebraHomogeneousComponent.model_json_schema(mode=mode)
    degree_schema = schema["properties"]["degree"]
    assert degree_schema["type"] == "string"
    assert degree_schema["maxLength"] == MAX_CANONICAL_INTEGER_DIGITS
    validator = Draft202012Validator(degree_schema)
    for value in ("0", "42", "9" * MAX_CANONICAL_INTEGER_DIGITS):
        assert validator.is_valid(value)
    for invalid in (
        "9" * (MAX_CANONICAL_INTEGER_DIGITS + 1),
        "-1",
        "01",
        "-0",
        "+1",
        "1\n",
        42,
    ):
        assert not validator.is_valid(invalid)


def test_degree_wire_validation_matches_the_published_schema() -> None:
    degree_schema = FreeAlgebraHomogeneousComponent.model_json_schema(
        mode="validation"
    )["properties"]["degree"]
    validator = Draft202012Validator(degree_schema)
    for value in ("0", "1", str((1 << 53) + 1)):
        assert validator.is_valid(value)
        restored = FreeAlgebraHomogeneousComponent.model_validate_json(
            json.dumps(_component_wire(value)), strict=True
        )
        assert restored.degree == int(value)
    for invalid in ("-1", "01", "1\n"):
        assert not validator.is_valid(invalid)
        with pytest.raises(ValidationError):
            FreeAlgebraHomogeneousComponent.model_validate_json(
                json.dumps(_component_wire(invalid)), strict=True
            )


def test_native_degree_envelope_matches_the_exact_integer_wire_bound() -> None:
    polynomial = _polynomial()
    boundary = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    assert homogeneous_component(polynomial, boundary).degree == boundary
    with pytest.raises(OperationDomainValidationError, match="decimal digits"):
        homogeneous_component(polynomial, 10**MAX_CANONICAL_INTEGER_DIGITS)
