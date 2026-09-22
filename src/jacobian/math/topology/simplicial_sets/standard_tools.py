"""Public finite standard-simplex operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.standard import (
    simplex_boundary,
    simplex_horn,
    standard_simplex,
)


class StandardSimplexRequest(StrictModel):
    dimension: int = Field(ge=0, le=4)
    max_degree: int = Field(
        ge=0,
        le=4,
        description=(
            "Finite prefix degree; coupled admission requires every generated "
            "degree to contain at most 32 simplices and the total prefix at most 96."
        ),
    )


class SimplexBoundaryRequest(StandardSimplexRequest):
    pass


class SimplexHornRequest(StandardSimplexRequest):
    missing_face: int = Field(ge=0, le=4)


def _simplex(request: StandardSimplexRequest) -> FiniteTruncatedSimplicialSet:
    return standard_simplex(request.dimension, request.max_degree)


def _boundary(request: SimplexBoundaryRequest) -> FiniteTruncatedSimplicialSet:
    return simplex_boundary(request.dimension, request.max_degree)


def _horn(request: SimplexHornRequest) -> FiniteTruncatedSimplicialSet:
    return simplex_horn(request.dimension, request.missing_face, request.max_degree)


TOOLS = (
    MathTool(
        operation_id="topology.simplicial_set.standard_simplex.compute",
        title="Construct a finite standard simplicial simplex",
        description="Construct the complete finite degree prefix of Delta[n] using monotone maps and all face and degeneracy tables; the degree prefix is finite and bounded.",
        request_type=StandardSimplexRequest,
        result_type=FiniteTruncatedSimplicialSet,
        run=_simplex,
        tags=("topology", "simplicial-set", "standard-simplex", "exact"),
        discovery_terms=("standard simplex", "Delta n", "simplicial simplex"),
        examples=(
            OperationExample(
                name="delta_one",
                description="Construct the degree-0..1 prefix of Delta[1]; the prefix degree is finite and all maps are retained.",
                input={"dimension": 1, "max_degree": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_set.boundary.compute",
        title="Construct a finite simplicial boundary",
        description="Construct the finite degree prefix of the boundary of Delta[n], retaining every in-range face and degeneracy map; n must be positive because the empty boundary is outside this nonempty carrier.",
        request_type=SimplexBoundaryRequest,
        result_type=FiniteTruncatedSimplicialSet,
        run=_boundary,
        tags=("topology", "simplicial-set", "boundary", "exact"),
        discovery_terms=("simplicial boundary", "boundary of simplex"),
        examples=(
            OperationExample(
                name="triangle_boundary",
                description="Construct the degree-0..1 prefix of the boundary of Delta[2]; the simplex dimension is positive.",
                input={"dimension": 2, "max_degree": 1},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.simplicial_set.horn.compute",
        title="Construct a finite simplicial horn",
        description="Construct the finite degree prefix of the horn Lambda[n,k], consisting of faces opposite every vertex except k, with complete in-range maps.",
        request_type=SimplexHornRequest,
        result_type=FiniteTruncatedSimplicialSet,
        run=_horn,
        tags=("topology", "simplicial-set", "horn", "exact"),
        discovery_terms=("simplicial horn", "Kan horn"),
        examples=(
            OperationExample(
                name="horn_2_1",
                description="Construct the degree-0..1 prefix of Lambda[2,1]; missing_face selects the omitted face.",
                input={"dimension": 2, "missing_face": 1, "max_degree": 1},
            ),
        ),
    ),
)
__all__ = [
    "TOOLS",
    "SimplexBoundaryRequest",
    "SimplexHornRequest",
    "StandardSimplexRequest",
]
