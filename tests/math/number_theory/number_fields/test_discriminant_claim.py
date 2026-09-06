"""Canonical field-discriminant claims and explicit consumer checking."""

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
    discriminant,
    verify_discriminant,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
)


@pytest.mark.parametrize("value", ["nope", "08", "+8", "-0"])
def test_discriminant_rejects_noncanonical_integer(value: str) -> None:
    with pytest.raises(ValidationError):
        NumberFieldDiscriminantResult.model_validate(
            {
                "field": {"coefficients_descending": ["1", "0", "-2"]},
                "discriminant": value,
            }
        )


def test_field_discriminant_claim_round_trip() -> None:
    field = SimpleNumberFieldPresentation(coefficients_descending=("1", "0", "-5"))
    # Q(sqrt(5)) has field discriminant 5, not polynomial discriminant 20.
    assert discriminant(field) == 5
    claim = NumberFieldDiscriminantResult(field=field, discriminant="5")
    assert claim.discriminant == 5
    assert claim.model_dump(mode="json")["discriminant"] == "5"
    assert verify_discriminant(
        NumberFieldDiscriminantResult.model_validate_json(claim.model_dump_json())
    )
    forged = NumberFieldDiscriminantResult(field=field, discriminant="20")
    assert not verify_discriminant(forged)


def test_reducible_transport_is_rejected_by_verifier() -> None:
    claim = NumberFieldDiscriminantResult(
        field=SimpleNumberFieldPresentation(coefficients_descending=("1", "0", "-1")),
        discriminant=1,
    )

    assert not verify_discriminant(claim)
