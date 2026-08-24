"""Catalog projection checks for finite periodic congruence unions."""

import json
from fractions import Fraction

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
)


def test_periodic_congruence_examples_execute_through_catalog() -> None:
    catalog = Catalog.open()
    for operation_id in (
        "congruence.periodic_union.measure.compute",
        "congruence.periodic_union.profile.compute",
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert operation.examples
        for invocation_example in operation.examples:
            result = invoke_operation(operation_id, invocation_example.input, catalog)
            validated = operation.result_type.model_validate(result.output)
            assert validated.model_dump(mode="json") == result.output


def test_published_profile_example_executes_exactly() -> None:
    operation = Catalog.open().operation("congruence.periodic_union.profile.compute")
    assert operation is not None
    assert operation.request_type is PeriodicCongruenceUnionProfileRequest

    result = invoke_operation(
        operation.operation_id,
        operation.examples[0].input,
        Catalog.open(),
    )
    validated = PeriodicCongruenceUnionProfileResult.model_validate(result.output)

    assert validated.source.model_dump(mode="json") == {
        "subsets": [
            {"modulus": "3", "residues": ["1"]},
            {"modulus": "4", "residues": ["0", "2"]},
        ],
        "complement": True,
    }
    assert validated.common_period == "12"
    assert validated.occupied_count == "4"
    assert validated.density.as_fraction() == Fraction(1, 3)
    assert validated.occupied_residues == ("3", "5", "9", "11")


def test_advertised_profile_example_parses_through_strict_json() -> None:
    operation = Catalog.open().operation("congruence.periodic_union.profile.compute")
    assert operation is not None

    request = PeriodicCongruenceUnionProfileRequest.model_validate_json(
        json.dumps(operation.examples[0].input),
        strict=True,
    )

    assert request.subsets[0].model_dump(mode="json") == {
        "modulus": "4",
        "residues": ["0", "2"],
    }


def test_published_profile_consumes_serialized_measure_source_unchanged() -> None:
    catalog = Catalog.open()
    measure_output = invoke_operation(
        "congruence.periodic_union.measure.compute",
        {"subsets": [{"modulus": "5", "residues": ["0"]}], "complement": False},
        catalog,
    ).output

    operation = catalog.operation("congruence.periodic_union.profile.compute")
    assert operation is not None
    assert operation.request_type is PeriodicCongruenceUnionProfileRequest
    result = invoke_operation(
        operation.operation_id,
        measure_output["source"],
        catalog,
    )
    validated = PeriodicCongruenceUnionProfileResult.model_validate(result.output)

    assert validated.source.model_dump(mode="json") == measure_output["source"]
    assert validated.common_period == "5"
    assert validated.occupied_count == "1"
    assert validated.density.as_fraction() == Fraction(1, 5)
    assert validated.occupied_residues == ("0",)
