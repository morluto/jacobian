"""Canonical exact values for the reviewed level-one modular forms."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.kernel import (
    NamedLevelOneModularForm,
    metadata,
)
from jacobian.math.polynomials.series._models import TruncatedSeries

MAX_MODULAR_FORM_LEVEL = 100_000
MAX_MODULAR_FORM_WEIGHT = 1_000_000


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


class ModularFormSpace(StrictModel):
    """One supported exact holomorphic or cuspidal modular-form space.

    Version 1 binds the trivial character over QQ on Gamma0(N); only level
    one admits dimension computation so far, and higher levels are rejected
    by the owning operation before any backend work.
    """

    group: Literal["GAMMA0"] = Field(
        default="GAMMA0", description="Congruence subgroup family."
    )
    level: StrictInt = Field(
        ge=1,
        le=MAX_MODULAR_FORM_LEVEL,
        description="Level N of Gamma0(N).",
    )
    weight: StrictInt = Field(
        ge=0,
        le=MAX_MODULAR_FORM_WEIGHT,
        description="Integer modular weight k.",
    )
    kind: Literal["M", "S"] = Field(
        description="Full holomorphic space M_k or cuspidal subspace S_k."
    )
    character: Literal["TRIVIAL"] = Field(
        default="TRIVIAL", description="Dirichlet character (trivial only)."
    )
    coefficient_domain: Literal["QQ"] = Field(
        default="QQ", description="Exact coefficient domain."
    )


__all__ = [
    "MAX_MODULAR_FORM_LEVEL",
    "MAX_MODULAR_FORM_WEIGHT",
    "LevelOneModularQExpansion",
    "ModularFormSpace",
]
