"""Inventory and schemas distinguish analysis operands from canonical evidence."""

import json
from typing import Any, get_args

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tests.dispatch._analysis_rational_cases import (
    ANALYSIS_RATIONAL_CASES,
    AnalysisRationalCase,
)
from tests.dispatch._rational_request_cases import ratio

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.catalog import Catalog
from jacobian.math.matrices.analysis._models import (
    FarkasCertificateRequest,
    RationalSpectrumClaimRequest,
)
from jacobian.math.matrices.completion._models import PartialSymmetricRationalMatrix
from jacobian.math.matrices.values import RationalMatrix


def contains(annotation: Any, target: type[Any]) -> bool:
    return annotation is target or any(
        contains(arg, target) for arg in get_args(annotation)
    )


def test_inventory_covers_every_ordinary_rational_position_in_the_four_owners() -> None:
    owners = {"analysis", "collatz_wielandt", "chordal_psd", "completion"}
    found = set()
    strict_claims = set()
    for operation in BUILTIN_TOOLS:
        request = operation.request_type
        if request.__module__ not in {
            f"jacobian.math.matrices.{owner}._models" for owner in owners
        }:
            continue
        if request in (RationalSpectrumClaimRequest, FarkasCertificateRequest):
            strict_claims.add(request)
            continue
        for name, field in request.model_fields.items():
            if contains(field.annotation, RationalMatrix) or contains(
                field.annotation, CanonicalRational
            ):
                found.add((operation.operation_id, name))
            elif contains(field.annotation, PartialSymmetricRationalMatrix):
                found.add((operation.operation_id, "specified_entries"))
    assert strict_claims == {RationalSpectrumClaimRequest, FarkasCertificateRequest}
    assert found == {
        (case.operation, case.position) for case in ANALYSIS_RATIONAL_CASES
    }
    assert len(found) == 5


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize(
    "value",
    [
        ratio("2", "4"),
        ratio("-2", "-4"),
        ratio("0", "-5"),
        ratio("-" + "9" * 32768, "9" * 32768),
    ],
)
def test_schema_and_decoder_accept_full_carrier_request_presentations(
    case: AnalysisRationalCase, value: dict[str, Any]
) -> None:
    operation = Catalog.open().operation(case.operation)
    assert operation is not None
    payload = case.payload(value)
    schema = operation.request_type.model_json_schema(mode="validation")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    parsed = operation.request_type.model_validate_json(
        json.dumps(payload), strict=True
    )
    canonical = parsed.model_dump(mode="json")
    Draft202012Validator(
        operation.request_type.model_json_schema(mode="serialization")
    ).validate(canonical)
    if value["den"].startswith("-"):
        assert not Draft202012Validator(
            operation.request_type.model_json_schema(mode="serialization")
        ).is_valid(payload)


@pytest.mark.parametrize("case", ANALYSIS_RATIONAL_CASES)
@pytest.mark.parametrize(
    "value",
    [
        ratio("1", "0"),
        ratio("1", "-0"),
        ratio("9" * 32769, "9" * 32769),
        ratio("1", "9" * 32769),
    ],
)
def test_schema_and_decoder_reject_raw_bounds_and_zero_denominator(
    case: AnalysisRationalCase, value: dict[str, Any]
) -> None:
    operation = Catalog.open().operation(case.operation)
    assert operation is not None
    payload = case.payload(value)
    assert not Draft202012Validator(
        operation.request_type.model_json_schema()
    ).is_valid(payload)
    with pytest.raises(ValidationError):
        operation.request_type.model_validate_json(json.dumps(payload), strict=True)
