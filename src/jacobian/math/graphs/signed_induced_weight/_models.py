"""Typed contracts for signed induced-weight extrema."""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._models import (
    RationalWeightedGraph,
)
from jacobian.math.graphs.signed_induced_weight._bounds import (
    MAX_SIGNED_WEIGHT_EDGES,
    MAX_SIGNED_WEIGHT_VERTICES,
)


class _SignedWeightGraphSchema:
    """Project this operation's envelope onto its shared graph carrier."""

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        schema = handler.resolve_ref_schema(handler(core_schema)).copy()
        schema["description"] = (
            "A canonical rational-weighted simple graph admitted for "
            "component-bounded signed induced-weight optimization: at most "
            f"{MAX_SIGNED_WEIGHT_VERTICES} vertices and "
            f"{MAX_SIGNED_WEIGHT_EDGES} edges, with every nonzero-weight "
            "support component inside the published exhaustive-search "
            "envelope, subject to the published arithmetic work and "
            "exact-result height budgets."
        )
        properties = schema["properties"]
        properties["vertices"] = {
            **properties["vertices"],
            "maxItems": MAX_SIGNED_WEIGHT_VERTICES,
        }
        properties["edges"] = {
            **properties["edges"],
            "maxItems": MAX_SIGNED_WEIGHT_EDGES,
        }
        return schema


SignedInducedWeightGraph = Annotated[
    RationalWeightedGraph,
    _SignedWeightGraphSchema,
]


class SignedInducedWeightRequest(StrictModel):
    """Request for exact signed induced-edge weight extrema."""

    graph: SignedInducedWeightGraph


class WeightExtremum(StrictModel):
    """One extremum (min or max) with a witness vertex subset."""

    value: CanonicalRational
    witness_vertices: tuple[str, ...]


class SignedInducedWeightResult(StrictModel):
    """Exact min and max of the signed induced-edge weight over all subsets."""

    graph: RationalWeightedGraph
    minimum: WeightExtremum
    maximum: WeightExtremum

    @model_validator(mode="after")
    def require_canonical_witness_axes(self) -> Self:
        for name, extremum in (("minimum", self.minimum), ("maximum", self.maximum)):
            witness = set(extremum.witness_vertices)
            if extremum.witness_vertices != tuple(
                vertex for vertex in self.graph.vertices if vertex in witness
            ):
                raise PydanticCustomError(
                    "graph.signed_induced_weight.witness_axis",
                    f"{name} witness vertices must be a subset in source-vertex order",
                )
        return self


__all__ = [
    "MAX_SIGNED_WEIGHT_EDGES",
    "MAX_SIGNED_WEIGHT_VERTICES",
    "SignedInducedWeightRequest",
    "SignedInducedWeightResult",
    "WeightExtremum",
]
