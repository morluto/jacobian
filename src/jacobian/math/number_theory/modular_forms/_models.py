"""Wire contracts for exact named level-one q-expansion construction."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.kernel import NamedLevelOneModularForm
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"modular_forms.{reason}", message)


class LevelOneNamedQExpansionRequest(StrictModel):
    """Construct one reviewed normalized level-one modular-form q-prefix."""

    form: NamedLevelOneModularForm = Field(
        description="Closed normalized family: E4, E6, or Ramanujan DELTA."
    )
    truncation_order: StrictInt = Field(
        ge=1,
        description=(
            "Return coefficients q^0 through q^(P-1); exact work and retained "
            "coefficient cardinality bound P before finite-series arithmetic."
        ),
    )


class SpaceDimensionRequest(StrictModel):
    """Compute the exact dimension of one supported modular-form space."""

    space: ModularFormSpace = Field(
        description=(
            "Holomorphic M_k or cuspidal S_k space on Gamma0(N) with trivial "
            "character over QQ; only level one is admitted so far."
        ),
    )


class SpaceDimensionResult(StrictModel):
    """Exact dimension of a source space with its level-one formula data.

    For the declared space, ``dimension`` is the exact complex dimension;
    ``eisenstein_dimension`` and ``cusp_dimension`` are the explicit
    projections of the same space family (the Eisenstein subspace of the
    ambient ``M_k`` and the cuspidal subspace). For ``kind == "M"`` the
    dimension is their sum; for ``kind == "S"`` it equals the cusp
    projection. No q-expansion coefficients are carried.
    """

    space: ModularFormSpace
    dimension: StrictInt = Field(ge=0)
    floor_term: StrictInt = Field(
        ge=0, description="weight // 12, the level-one dimension driver."
    )
    congruence_class: StrictInt = Field(
        ge=0, le=11, description="weight mod 12 selecting the correction."
    )
    eisenstein_dimension: StrictInt = Field(ge=0)
    cusp_dimension: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_dimension_decomposition(self) -> Self:
        if self.space.kind == "M":
            expected = self.eisenstein_dimension + self.cusp_dimension
        else:
            expected = self.cusp_dimension
        if self.dimension != expected:
            raise _validation_error(
                "dimension_decomposition",
                "dimension must decompose into the reported Eisenstein/cusp data",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        space: ModularFormSpace,
        *,
        dimension: int,
        eisenstein_dimension: int,
        cusp_dimension: int,
    ) -> Self:
        """Build one result after the admitted kernel established its count."""

        return cls.model_construct(
            space=space,
            dimension=dimension,
            floor_term=space.weight // 12,
            congruence_class=space.weight % 12,
            eisenstein_dimension=eisenstein_dimension,
            cusp_dimension=cusp_dimension,
        )


__all__ = [
    "LevelOneNamedQExpansionRequest",
    "SpaceDimensionRequest",
    "SpaceDimensionResult",
]
