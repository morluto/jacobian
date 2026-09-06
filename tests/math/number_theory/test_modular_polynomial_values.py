"""Composition checks for sparse modular-polynomial values."""

import copy
import re

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory import modular_polynomials
from jacobian.math.number_theory._modular_models import (
    ModularPolynomialResidueImageRequest,
    ModularPolynomialResidueImageResult,
    ModularPolynomialVariable,
)
from jacobian.math.number_theory.modular_polynomials import (
    _INTEGER,
    ModularPolynomialTerm,
    NormalizedModularPolynomialTerm,
    modular_polynomial_identity,
    verify_modular_polynomial_identity,
)
from jacobian.math.number_theory.operations import (
    modular_polynomial_residue_assignments,
    modular_polynomial_residue_image,
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
        5,
        ("x",),
        request.terms,
    )
    result = modular_polynomial_residue_image(
        request.modulus, request.variables, request.terms
    )

    assert type(result.normalized_terms[0]) is NormalizedModularPolynomialTerm
    assert identity.normalized_left == result.normalized_terms
    assert result.model_dump(mode="json")["normalized_terms"] == [
        {"coefficient": 3, "exponents": [1]}
    ]


def test_identity_native_api_exposes_values_and_semantic_scalars() -> None:
    term = ModularPolynomialTerm(coefficient="6", exponents=(1,))

    result = modular_polynomial_identity(5, ("x",), (term,))

    assert "ModularPolynomialIdentityRequest" not in modular_polynomials.__all__
    assert result.normalized_left == (
        NormalizedModularPolynomialTerm(coefficient=1, exponents=(1,)),
    )


def test_identity_result_verifier_rejects_a_forged_residual() -> None:
    identity = modular_polynomial_identity(
        5,
        ("x",),
        (ModularPolynomialTerm(coefficient="1", exponents=(1,)),),
    )
    payload = identity.model_dump(mode="json")
    payload["residual"] = []
    payload["identical"] = True

    decoded = type(identity).model_validate(payload)
    assert not list(
        Draft202012Validator(type(identity).model_json_schema()).iter_errors(payload)
    )
    assert not verify_modular_polynomial_identity(decoded)


def test_identity_result_schema_matches_structural_parser_bounds() -> None:
    schema = modular_polynomials.ModularPolynomialIdentityValue.model_json_schema()
    variable_order = schema["properties"]["variable_order"]
    assert variable_order["items"]["pattern"] == r"^[a-z][a-z0-9_]{0,31}$"
    assert variable_order["uniqueItems"] is True

    for name in ("normalized_left", "normalized_right", "residual"):
        terms = schema["properties"][name]
        expected_max_items = (
            modular_polynomials._MAX_TERMS
            if name != "residual"
            else modular_polynomials._MAX_TERMS * 2
        )
        assert terms["maxItems"] == expected_max_items
        assert terms["items"]["properties"]["exponents"]["items"]["minimum"] == 0
        assert (
            terms["items"]["properties"]["exponents"]["items"]["maximum"]
            == modular_polynomials._MAX_EXPONENT
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("variable_order", ["X"]),
        ("variable_order", ["x", "x"]),
        ("normalized_left", [{"coefficient": 1, "exponents": [257]}]),
    ),
)
def test_identity_schema_and_parser_reject_the_same_structural_forgery(
    field: str, value: object
) -> None:
    identity = modular_polynomial_identity(5, ("x",), (), ())
    payload = identity.model_dump(mode="json")
    payload[field] = value

    schema_errors = list(
        Draft202012Validator(type(identity).model_json_schema()).iter_errors(payload)
    )
    assert schema_errors
    with pytest.raises(ValidationError):
        type(identity).model_validate(payload)


def test_identity_verifier_preflights_malicious_tuple_subclass_without_hashing() -> (
    None
):
    class ExplodingTuple(tuple[object, ...]):
        def __hash__(self) -> int:
            raise RuntimeError("must not hash untrusted tuple subclass")

    term = NormalizedModularPolynomialTerm.model_construct(
        coefficient=1,
        exponents=ExplodingTuple((1,)),
    )
    claim = modular_polynomials.ModularPolynomialIdentityValue.model_construct(
        modulus=5,
        variable_order=("x",),
        normalized_left=(term,),
        normalized_right=(),
        residual=(term,),
        identical=False,
        comparison_scope="FORMAL_COEFFICIENTWISE_IDENTITY",
    )

    assert verify_modular_polynomial_identity(claim) is False


def test_identity_verifier_rejects_oversized_claim_before_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    term = NormalizedModularPolynomialTerm(coefficient=1, exponents=(1,))
    claim = modular_polynomials.ModularPolynomialIdentityValue.model_construct(
        modulus=5,
        variable_order=("x",),
        normalized_left=(term,) * (modular_polynomials._MAX_TERMS + 1),
        normalized_right=(),
        residual=(term,),
        identical=False,
        comparison_scope="FORMAL_COEFFICIENTWISE_IDENTITY",
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("oversized claims must be rejected before subtraction")

    monkeypatch.setattr(modular_polynomials, "_subtract_normalized", fail)
    assert verify_modular_polynomial_identity(claim) is False


def test_identity_result_round_trips_through_canonical_json() -> None:
    identity = modular_polynomial_identity(
        5,
        ("x",),
        (ModularPolynomialTerm(coefficient="1", exponents=(1,)),),
        (ModularPolynomialTerm(coefficient="1", exponents=(1,)),),
    )

    decoded = type(identity).model_validate_json(identity.model_dump_json())

    assert decoded == identity
    assert verify_modular_polynomial_identity(decoded)


def test_identity_result_decoding_does_not_replay_subtraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = modular_polynomial_identity(
        5,
        ("x",),
        (ModularPolynomialTerm(coefficient="1", exponents=(1,)),),
    )
    payload = identity.model_dump(mode="json")
    payload["residual"] = []
    payload["identical"] = True

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("deserialization must not replay subtraction")

    monkeypatch.setattr(modular_polynomials, "_subtract_normalized", fail)
    decoded = type(identity).model_validate(payload)

    assert decoded.residual == ()
    assert decoded.identical is True


def test_identity_verifier_accepts_the_bounded_maximum_sparse_claim() -> None:
    terms_left = tuple(
        ModularPolynomialTerm(coefficient="1", exponents=(index // 257, index % 257))
        for index in range(512)
    )
    terms_right = tuple(
        ModularPolynomialTerm(
            coefficient="1",
            exponents=(index // 257 + 2, index % 257),
        )
        for index in range(512)
    )

    identity = modular_polynomial_identity(7, ("x", "y"), terms_left, terms_right)

    assert len(identity.normalized_left) == 512
    assert len(identity.normalized_right) == 512
    assert len(identity.residual) == 1024
    assert verify_modular_polynomial_identity(identity)


def test_residue_image_keeps_its_narrower_exponent_admission() -> None:
    request = _request(exponent=33)
    with pytest.raises(OperationDomainValidationError, match="exponents"):
        modular_polynomial_residue_image(
            request.modulus,
            request.variables,
            request.terms,
        )


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
            modular_polynomial_residue_image(
                request.modulus, request.variables, request.terms
            ),
            modular_polynomial_residue_assignments(
                request.modulus, request.variables, request.terms
            ),
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
        5,
        ("a", "b", "c", "d", "e", "f", "g"),
        (ModularPolynomialTerm(coefficient="1", exponents=wide_vector),),
    )

    assert identity.normalized_left[0].exponents == wide_vector


def test_coefficient_boundary_follows_the_advertised_envelope() -> None:
    admitted = _request_with_coefficient("-" + "9" * 255)
    assert int(admitted.terms[0].coefficient) < 0

    with pytest.raises(OperationDomainValidationError, match="coefficient"):
        request = _request_with_coefficient("-" + "9" * 256)
        modular_polynomial_residue_image(
            request.modulus, request.variables, request.terms
        )


def test_shared_term_type_retains_its_wider_envelope_elsewhere() -> None:
    widest = "-" + "9" * 256
    identity = modular_polynomial_identity(
        5,
        ("x",),
        (ModularPolynomialTerm(coefficient=widest, exponents=(1,)),),
    )

    assert identity.normalized_left[0].coefficient == int(widest) % 5
