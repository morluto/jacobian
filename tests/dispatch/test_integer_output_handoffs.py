"""Exact integer output composes unchanged with bounded scalar consumers."""

import math
from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.number_theory._integer_models import MAX_SAFE_INTEGER


@pytest.mark.parametrize(
    ("operation", "expected"),
    (
        ("euler_totient", {"value": "2"}),
        ("divisor_count", {"value": "4"}),
        ("divisor_sum", {"value": "12"}),
        ("mobius", {"value": "1"}),
        ("next_prime", {"value": "7"}),
        ("nth_prime", {"value": "13"}),
        ("prime_count", {"value": "3"}),
        ("previous_prime", {"value": "5"}),
        ("primorial", {"value": "30030"}),
        ("floor_square_root", {"root": 2}),
    ),
)
def test_gcd_scalar_composes_unchanged_with_bounded_consumers(
    operation: str, expected: dict[str, Any]
) -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "integer.compute.gcd", {"left": "12", "right": "18"}, catalog
    ).output
    assert produced == {"value": "6"}
    operation_id = "integer.compute." + operation
    assert (
        invoke_operation(operation_id, {"n": produced["value"]}, catalog).output
        == expected
    )
    assert invoke_operation(operation_id, {"n": 6}, catalog).output == expected


@pytest.mark.parametrize("left", (12, "12", -12, "-12"))
@pytest.mark.parametrize("right", (18, "18"))
def test_gcd_request_spellings_preserve_exact_bezout_relations(
    left: int | str, right: int | str
) -> None:
    catalog = Catalog.open()
    payload = {"left": left, "right": right}
    expected = math.gcd(int(left), int(right))
    assert invoke_operation("integer.compute.gcd", payload, catalog).output == {
        "value": str(expected)
    }
    extended = invoke_operation("integer.compute.extended_gcd", payload, catalog).output
    assert int(extended["gcd"]) == expected
    assert (
        int(left) * int(extended["left_coefficient"])
        + int(right) * int(extended["right_coefficient"])
        == expected
    )


def test_large_gcd_output_retains_the_safe_integer_consumer_boundary() -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "integer.compute.gcd", {"left": str(MAX_SAFE_INTEGER), "right": "0"}, catalog
    ).output
    result = invoke_operation(
        "integer.compute.floor_square_root", {"n": produced["value"]}, catalog
    ).output
    root = result["root"]
    assert root**2 <= MAX_SAFE_INTEGER < (root + 1) ** 2
    for too_large in (10_001, MAX_SAFE_INTEGER + 1, 10**255):
        output = invoke_operation(
            "integer.compute.gcd", {"left": str(too_large), "right": "0"}, catalog
        ).output
        with pytest.raises(OperationRequestValidationError):
            invoke_operation(
                "integer.compute.euler_totient", {"n": output["value"]}, catalog
            )
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            "integer.compute.floor_square_root",
            {"n": str(MAX_SAFE_INTEGER + 1)},
            catalog,
        )


def test_zero_gcd_output_preserves_each_consumers_mathematical_domain() -> None:
    catalog = Catalog.open()
    produced = invoke_operation(
        "integer.compute.gcd", {"left": 0, "right": "0"}, catalog
    ).output
    assert produced == {"value": "0"}
    for operation, expected in (
        ("floor_square_root", {"root": 0}),
        ("prime_count", {"value": "0"}),
        ("next_prime", {"value": "2"}),
    ):
        assert (
            invoke_operation(
                "integer.compute." + operation, {"n": produced["value"]}, catalog
            ).output
            == expected
        )
    with pytest.raises(OperationRequestValidationError):
        invoke_operation(
            "integer.compute.euler_totient", {"n": produced["value"]}, catalog
        )


@pytest.mark.parametrize(
    "invalid",
    (True, 6.0, 6.5, "+6", "06", "-0", " 6", "6\n", "1" * 257, MAX_SAFE_INTEGER + 1),
)
def test_public_integer_ingress_refuses_coercion_and_unbounded_numeric_tokens(
    invalid: Any,
) -> None:
    catalog = Catalog.open()
    for operation_id, payload in (
        ("integer.compute.gcd", {"left": invalid, "right": "18"}),
        ("integer.compute.euler_totient", {"n": invalid}),
    ):
        with pytest.raises(OperationRequestValidationError):
            invoke_operation(operation_id, payload, catalog)
