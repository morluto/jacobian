"""Strict JSON contracts that require the public dispatch boundary."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.petri_nets.values import MAX_PETRI_ARC_WEIGHT, MAX_PETRI_MARKING
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


def test_petri_firing_reports_successor_outside_marking_envelope() -> None:
    result = invoke_operation(
        "petri_net.fire_transition.compute",
        {
            "net": {
                "place_count": 1,
                "transition_count": 1,
                "pre": [[0]],
                "post": [[MAX_PETRI_ARC_WEIGHT]],
            },
            "marking": {"tokens": [MAX_PETRI_MARKING]},
            "transition": 0,
        },
        Catalog.open(),
    )

    assert result.output == {
        "status": "ESCAPES_DECLARED_ENVELOPE",
        "new_marking": None,
        "envelope_escape": [MAX_PETRI_MARKING + MAX_PETRI_ARC_WEIGHT],
    }
