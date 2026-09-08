"""Requests for a single rational exposed-face reduction."""

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.semidefinite.values import (
    MAX_SEMIDEFINITE_CELLS,
    RationalSemidefiniteSystem,
)


class SemidefiniteFaceReductionRequest(StrictModel):
    system: RationalSemidefiniteSystem
    multipliers: tuple[CanonicalRational, ...] = Field(
        max_length=8192,
        description="Supplied y with nonzero PSD sum y_i A_i and sum y_i b_i = 0.",
    )

    @model_validator(mode="before")
    @classmethod
    def require_aggregate_cells(cls, data: object) -> object:
        """Reject over-budget dense systems before nested matrix parsing."""

        data = canonicalize_json_containers(data)
        if not isinstance(data, dict):
            return data
        system = data.get("system")
        if not isinstance(system, dict):
            return data
        order = system.get("order")
        matrices = system.get("matrices")
        if type(order) is not int or not isinstance(matrices, (list, tuple)):
            return data
        if order < 0 or len(matrices) * max(order, 0) ** 2 > MAX_SEMIDEFINITE_CELLS:
            raise ValueError(
                "source and reduced matrices exceed the dense cell envelope"
            )
        return data
