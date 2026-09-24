"""Requests/results for finite modular q-series transforms."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.number_theory.modular_forms.values import (
    MAX_GAMMA0_OPERATION_LEVEL,
    MAX_Q_TRANSFORM_OUTPUT_PRECISION,
    ModularFormSpace,
)
from jacobian.math.polynomials.series._models import TruncatedSeries


class NamedQExpansionRequest(StrictModel):
    space: ModularFormSpace
    form: str
    precision: StrictInt = Field(ge=1)


class SturmBoundRequest(StrictModel):
    space: ModularFormSpace


class SturmBoundResult(StrictModel):
    space: ModularFormSpace
    index: StrictInt = Field(ge=1)
    bound: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_rational_space(self) -> SturmBoundResult:
        if self.space.coefficient_domain != "QQ":
            raise PydanticCustomError(
                "modular_forms.rational_sturm_result_parent",
                "this Sturm result carrier currently represents QQ spaces only",
            )
        return self


class FormalQSeriesOperatorRequest(StrictModel):
    """A finite formal q-prefix and a requested index-transform prefix."""

    series: TruncatedSeries
    prime: StrictInt = Field(ge=2, le=MAX_GAMMA0_OPERATION_LEVEL)
    output_precision: StrictInt = Field(ge=1, le=MAX_Q_TRANSFORM_OUTPUT_PRECISION)


class FormalQSeriesURequest(FormalQSeriesOperatorRequest):
    """Request a formal coefficient-selection U_p prefix."""


class FormalQSeriesVRequest(FormalQSeriesOperatorRequest):
    """Request a formal q-index dilation V_p prefix."""


__all__ = [
    "FormalQSeriesOperatorRequest",
    "FormalQSeriesURequest",
    "FormalQSeriesVRequest",
    "SturmBoundRequest",
    "SturmBoundResult",
]
