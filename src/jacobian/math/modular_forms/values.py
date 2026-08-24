"""Canonical exact values for the reviewed level-one modular forms."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.formal_power_series._models import TruncatedSeries
from jacobian.math.modular_forms.kernel import (
    NamedLevelOneModularForm,
    expected_coefficients,
    metadata,
    require_level_one_replay,
)


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
    def require_normalized_replayable_form(self) -> Self:
        require_level_one_replay(self.form, self.q_expansion.truncation_order)
        weight, space_kind, normalization = metadata(self.form)
        if (
            self.weight != weight
            or self.space_kind != space_kind
            or self.normalization != normalization
        ):
            raise ValueError("level-one modular metadata does not match the named form")
        if self.q_expansion.variable != "q":
            raise ValueError("a modular q-expansion must use the canonical variable q")
        expected = expected_coefficients(self.form, self.q_expansion.truncation_order)
        actual = tuple(
            coefficient.as_fraction() for coefficient in self.q_expansion.coefficients
        )
        if actual != expected:
            raise ValueError("q-expansion does not match the normalized named form")
        return self


__all__ = ["LevelOneModularQExpansion"]
