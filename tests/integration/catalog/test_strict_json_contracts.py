"""Strict JSON contracts that require the public dispatch boundary."""

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.petri_nets.values import MAX_PETRI_ARC_WEIGHT, MAX_PETRI_MARKING
from jacobian.math.quadratic_forms.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS,
)


def test_large_quadratic_rational_result_survives_public_dispatch() -> None:
    coefficient = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    coordinate = "1" + "0" * (MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS - 1)
    result = invoke_operation(
        "quadratic_form.evaluate.compute",
        {
            "form": {
                "axis": ["x"],
                "diagonal_coefficients": [{"num": coefficient, "den": "1"}],
            },
            "vector": {
                "axis": ["x"],
                "coordinates": [{"num": coordinate, "den": "1"}],
            },
        },
        Catalog.open(),
    )

    assert result.output["value"] == {
        "num": str(int(coefficient) * int(coordinate) ** 2),
        "den": "1",
    }


def test_unsafe_quadratic_global_enumeration_leaves_are_not_published() -> None:
    catalog = Catalog.open()

    assert catalog.operation("quadratic_form.representation_numbers.compute") is None
    assert catalog.operation("quadratic_form.theta_series_prefix.compute") is None


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
