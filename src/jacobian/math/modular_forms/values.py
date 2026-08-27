"""Canonical exact values for the reviewed level-one modular forms."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.formal_power_series._models import TruncatedSeries
from jacobian.math.modular_forms.kernel import (
    NamedLevelOneModularForm,
    expected_coefficients,
    metadata,
    require_level_one_replay,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"modular_forms.{reason}", message)


class LevelOneModularQExpansion(StrictModel):
    """One normalized named form in QQ[[q]] through a declared precision.

    ``q_expansion`` contains every coefficient from q^0 through q^(P-1).
    Coefficients beyond that finite prefix are intentionally not represented.
    """

    form: NamedLevelOneModularForm
    congruence_subgroup: Literal["SL2Z"] = "SL2Z"
    level: Literal[1] = 1
    weight: Literal[4, 6, 12]
    space_kind: Literal["HOLOMORPHIC", "CUSP"]
    coefficient_domain: Literal["QQ"] = "QQ"
    normalization: str = Field(min_length=1, max_length=96)
    q_expansion: TruncatedSeries
    exactness: Literal["EXACT_RATIONAL"] = "EXACT_RATIONAL"

    @model_validator(mode="after")
    def require_structural_named_form(self) -> Self:
        weight, space_kind, normalization = metadata(self.form)
        if (
            self.weight != weight
            or self.space_kind != space_kind
            or self.normalization != normalization
        ):
            raise _validation_error(
                "metadata_mismatch",
                "level-one modular metadata does not match the named form",
            )
        if self.q_expansion.variable != "q":
            raise _validation_error(
                "variable_mismatch",
                "a modular q-expansion must use the canonical variable q",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        form: NamedLevelOneModularForm,
        weight: Literal[4, 6, 12],
        space_kind: Literal["HOLOMORPHIC", "CUSP"],
        normalization: str,
        q_expansion: TruncatedSeries,
    ) -> Self:
        """Construct a value after the owner kernel established its coefficients."""

        return cls.model_construct(
            form=form,
            weight=weight,
            space_kind=space_kind,
            normalization=normalization,
            q_expansion=q_expansion,
        )


def verify_level_one_q_expansion(value: LevelOneModularQExpansion) -> bool:
    """Boundedly verify an independently supplied named q-expansion."""

    try:
        require_level_one_replay(value.form, value.q_expansion.truncation_order)
    except ValueError:
        return False
    expected = expected_coefficients(value.form, value.q_expansion.truncation_order)
    actual = tuple(
        coefficient.as_fraction() for coefficient in value.q_expansion.coefficients
    )
    return actual == expected


__all__ = ["LevelOneModularQExpansion", "verify_level_one_q_expansion"]
