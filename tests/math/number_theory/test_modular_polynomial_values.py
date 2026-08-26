"""Composition checks for sparse modular-polynomial values."""

import copy
import re

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.math.number_theory._validation import expect_validation

from jacobian.math.modular_polynomials import (
    _INTEGER,
    ModularPolynomialIdentityRequest,
    ModularPolynomialTerm,
    NormalizedModularPolynomialTerm,
    modular_polynomial_identity,
)
from jacobian.math.number_theory._models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory._modular_operations import (
    compute_modular_polynomial_residue_assignments,
    compute_modular_polynomial_residue_image,
)


def _request(*, exponent: int = 1) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient="3", exponents=(exponent,)),),
    )


def _request_with_coefficient(
    coefficient: str,
) -> ModularPolynomialResidueImageRequest:
    return ModularPolynomialResidueImageRequest(
        modulus=5,
        variables=(ModularPolynomialVariable(name="x", residues=(0, 1)),),
        terms=(ModularPolynomialTerm(coefficient=coefficient, exponents=(1,)),),
    )


def _request_payload(coefficient: str) -> dict[str, object]:
    return {
        "modulus": 5,
        "variables": [{"name": "x", "residues": [0, 1]}],
        "terms": [{"coefficient": coefficient, "exponents": [1]}],
    }


def test_residue_image_consumes_and_produces_the_canonical_term_types() -> None:
    request = _request()

    assert type(request.terms[0]) is ModularPolynomialTerm
    assert request.model_dump(mode="json")["terms"] == [
        {"coefficient": "3", "exponents": [1]}
    ]

    identity = modular_polynomial_identity(
        ModularPolynomialIdentityRequest(
            modulus=5,
            variables=("x",),
            left=request.terms,
        )
    )
    result = compute_modular_polynomial_residue_image(request)

    assert type(result.normalized_terms[0]) is NormalizedModularPolynomialTerm
    assert identity.normalized_left == result.normalized_terms
    assert result.model_dump(mode="json")["normalized_terms"] == [
        {"coefficient": 3, "exponents": [1]}
    ]


def test_residue_image_keeps_its_narrower_exponent_admission() -> None:
    with expect_validation("number_theory.term_outside_residue_image_admission"):
        _request(exponent=33)


def test_published_term_schema_matches_residue_image_admission() -> None:
    term_schema = ModularPolynomialResidueImageRequest.model_json_schema()[
        "properties"
    ]["terms"]["items"]
    coefficient_schema = term_schema["properties"]["coefficient"]

    assert coefficient_schema["maxLength"] == 256
    assert coefficient_schema["pattern"] == _INTEGER.pattern
    assert term_schema["properties"]["exponents"]["maxItems"] == 6
    assert term_schema["properties"]["exponents"]["items"]["maximum"] == 32

    shared_schema = ModularPolynomialTerm.model_json_schema()
    assert shared_schema["properties"]["coefficient"]["maxLength"] == 257
    assert "pattern" not in shared_schema["properties"]["coefficient"]
    assert shared_schema["properties"]["exponents"]["maxItems"] == 20


def test_published_input_schema_rejects_non_canonical_coefficients() -> None:
    request_schema = ModularPolynomialResidueImageRequest.model_json_schema()
    validator = Draft202012Validator(request_schema)

    pattern = request_schema["properties"]["terms"]["items"]["properties"][
        "coefficient"
    ]["pattern"]
    assert re.search(pattern, "abc") is None
    assert re.fullmatch(pattern, "-" + "9" * 255)
    assert re.fullmatch(pattern, "0")

    assert list(validator.iter_errors(_request_payload("abc")))
    assert not list(validator.iter_errors(_request_payload("-" + "9" * 255)))
    assert not list(validator.iter_errors(_request_payload("3")))


def test_published_result_schema_matches_residue_image_output_scope() -> None:
    term_items = ModularPolynomialResidueImageResult.model_json_schema()["properties"][
        "normalized_terms"
    ]["items"]
    exponents_schema = term_items["properties"]["exponents"]

    assert exponents_schema["maxItems"] == 6
    assert exponents_schema["items"]["minimum"] == 0
    assert exponents_schema["items"]["maximum"] == 32

    shared_schema = NormalizedModularPolynomialTerm.model_json_schema()
    assert shared_schema["properties"]["exponents"]["maxItems"] == 20
    assert "minimum" not in shared_schema["properties"]["exponents"]["items"]
    assert "maximum" not in shared_schema["properties"]["exponents"]["items"]


def test_both_residue_image_operations_advertise_the_restored_bounds() -> None:
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    residue_image_tools = [
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id.startswith("modular.polynomial_residue_image.")
    ]
    assert len(residue_image_tools) == 2

    for tool in residue_image_tools:
        exponents_schema = tool.result_type.model_json_schema()["properties"][
            "normalized_terms"
        ]["items"]["properties"]["exponents"]
        assert exponents_schema["maxItems"] == 6
        assert exponents_schema["items"]["minimum"] == 0
        assert exponents_schema["items"]["maximum"] == 32


def test_emitted_results_parse_under_their_advertised_output_schema() -> None:
    request = _request()
    validator = Draft202012Validator(
        ModularPolynomialResidueImageResult.model_json_schema()
    )

    documents = [
        operation.model_dump(mode="json")
        for operation in (
            compute_modular_polynomial_residue_image(request),
            compute_modular_polynomial_residue_assignments(request),
        )
    ]
    for document in documents:
        assert not list(validator.iter_errors(document))

    widened = copy.deepcopy(documents[0])
    widened["normalized_terms"][0]["exponents"] = [1] * 7
    assert list(validator.iter_errors(widened))
    with pytest.raises(ValidationError):
        ModularPolynomialResidueImageResult.model_validate(widened)

    inflated = copy.deepcopy(documents[0])
    inflated["normalized_terms"][0]["exponents"] = [33]
    assert list(validator.iter_errors(inflated))


def test_shared_normalized_term_type_retains_its_wider_envelope_elsewhere() -> None:
    wide_vector = (0, 0, 0, 0, 0, 0, 1)
    identity = modular_polynomial_identity(
        ModularPolynomialIdentityRequest(
            modulus=5,
            variables=("a", "b", "c", "d", "e", "f", "g"),
            left=(ModularPolynomialTerm(coefficient="1", exponents=wide_vector),),
        )
    )

    assert identity.normalized_left[0].exponents == wide_vector


def test_coefficient_boundary_follows_the_advertised_envelope() -> None:
    admitted = _request_with_coefficient("-" + "9" * 255)
    assert int(admitted.terms[0].coefficient) < 0

    with expect_validation("number_theory.term_outside_residue_image_admission"):
        _request_with_coefficient("-" + "9" * 256)


def test_shared_term_type_retains_its_wider_envelope_elsewhere() -> None:
    widest = "-" + "9" * 256
    identity = modular_polynomial_identity(
        ModularPolynomialIdentityRequest(
            modulus=5,
            variables=("x",),
            left=(ModularPolynomialTerm(coefficient=widest, exponents=(1,)),),
        )
    )

    assert identity.normalized_left[0].coefficient == int(widest) % 5
