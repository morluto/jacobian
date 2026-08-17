"""Public JSON Schema guidance for canonical input values."""

from __future__ import annotations

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.finite_fields import FiniteFieldPresentation
from jacobian.math.logic._operations import CanonicalCnf


@pytest.mark.parametrize(
    ("model", "fields"),
    (
        (CanonicalRational, ("num", "den")),
        (CanonicalCnf, ("variables", "clauses")),
        (
            FiniteFieldPresentation,
            (
                "characteristic",
                "modulus_coefficients",
                "generator",
                "element_encoding_version",
            ),
        ),
    ),
)
def test_canonical_value_schema_exposes_examples_and_field_guidance(
    model: type[CanonicalRational | CanonicalCnf | FiniteFieldPresentation],
    fields: tuple[str, ...],
) -> None:
    schema = model.model_json_schema()

    assert schema["examples"]
    for field in fields:
        assert schema["properties"][field]["description"]
        assert schema["properties"][field]["examples"]
