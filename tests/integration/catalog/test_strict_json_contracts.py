"""Strict JSON contracts that require the public dispatch boundary."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.quadratic_forms._models import (
    MAX_ENTRY_DIGITS,
    MAX_VECTOR_DIGITS,
)


def test_large_quadratic_integer_result_survives_public_dispatch() -> None:
    entry = "1" + "0" * (MAX_ENTRY_DIGITS - 1)
    vector = "1" + "0" * (MAX_VECTOR_DIGITS - 1)
    result = invoke_operation(
        "quadratic_form.evaluate.compute",
        {"form": {"matrix": [[entry]]}, "vector": [vector]},
        Catalog.open(),
    )

    assert result.output["value"] == str(int(entry) * int(vector) ** 2)
