"""Catalog projection checks for interval arithmetic-function profiles."""

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.number_theory._interval_profile_models import (
    MAX_INTERVAL_WIDTH,
    MAX_PROFILE_RESULT_BYTES,
)

_INTERVAL_PROFILE_OPERATION_IDS = (
    "number_theory.integer_interval.squarefree_profile.compute",
    "number_theory.integer_interval.divisor_count_profile.compute",
    "number_theory.integer_interval.greatest_prime_factor_profile.compute",
    "number_theory.prime_gap_profile.compute",
    "number_theory.integer_interval.least_prime_factor_profile.compute",
    "number_theory.integer_interval.euler_totient_profile.compute",
    "number_theory.integer_interval.divisor_sum_profile.compute",
)


def test_interval_profile_examples_execute_through_catalog() -> None:
    catalog = Catalog.open()
    for operation_id in _INTERVAL_PROFILE_OPERATION_IDS:
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert operation.examples
        for invocation_example in operation.examples:
            result = invoke_operation(operation_id, invocation_example.input, catalog)
            validated = operation.result_type.model_validate(result.output)
            assert validated.model_dump(mode="json") == result.output


def test_squarefree_example_values() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "number_theory.integer_interval.squarefree_profile.compute",
        {"lower_bound": 1, "upper_bound": 12},
        catalog,
    )
    output = result.output
    assert output["squarefree_values"] == [1, 2, 3, 5, 6, 7, 10, 11]
    assert output["nonsquarefree_values"] == [4, 8, 9, 12]
    assert output["squarefree_count"] == 8
    assert output["nonsquarefree_count"] == 4


def test_divisor_count_example_values() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "number_theory.integer_interval.divisor_count_profile.compute",
        {"lower_bound": 1, "upper_bound": 6},
        catalog,
    )
    output = result.output
    assert [(r["n"], r["divisor_count"]) for r in output["rows"]] == [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 2),
        (6, 4),
    ]


def test_greatest_prime_factor_example_values() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "number_theory.integer_interval.greatest_prime_factor_profile.compute",
        {"lower_bound": 1, "upper_bound": 10},
        catalog,
    )
    output = result.output
    assert [r["greatest_prime_factor"] for r in output["rows"]] == [
        1,
        2,
        3,
        2,
        5,
        3,
        7,
        2,
        3,
        5,
    ]


def test_prime_gap_example_values() -> None:
    catalog = Catalog.open()
    result = invoke_operation(
        "number_theory.prime_gap_profile.compute",
        {"lower_bound": 3, "upper_bound": 5},
        catalog,
    )
    output = result.output
    assert len(output["rows"]) == 2
    assert output["rows"][0] == {"lower_prime": 3, "upper_prime": 5, "gap": 2}
    assert output["rows"][1] == {"lower_prime": 5, "upper_prime": 7, "gap": 2}


def test_interval_profile_schemas_publish_coupled_admission_limits() -> None:
    catalog = Catalog.open()
    for operation_id in _INTERVAL_PROFILE_OPERATION_IDS:
        operation = catalog.operation(operation_id)
        assert operation is not None
        description = operation.request_type.model_json_schema()["description"]
        assert f"{MAX_PROFILE_RESULT_BYTES:,}" in description

    for operation_id in (
        "number_theory.integer_interval.divisor_count_profile.compute",
        "number_theory.integer_interval.greatest_prime_factor_profile.compute",
    ):
        operation = catalog.operation(operation_id)
        assert operation is not None
        assert (
            f"{MAX_INTERVAL_WIDTH:,}"
            in operation.request_type.model_json_schema()["description"]
        )


def test_row_profile_rejection_is_request_validation_at_the_catalog_boundary() -> None:
    with pytest.raises(
        OperationRequestValidationError, match="payload failed validation"
    ):
        invoke_operation(
            "number_theory.integer_interval.divisor_count_profile.compute",
            {"lower_bound": 1, "upper_bound": MAX_INTERVAL_WIDTH},
            Catalog.open(),
        )
