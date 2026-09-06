"""Typed wire contracts for exact bounded graphical-model operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.probability.graphical_models.values import (
    MAX_MODEL_VARS,
    Factor,
    Variable,
)


class FactorMultiplyRequest(StrictModel):
    left: Factor
    right: Factor


class FactorMultiplyResult(FactorMultiplyRequest):
    factor: Factor

    @classmethod
    def _from_kernel(cls, left: Factor, right: Factor, factor: Factor) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(left=left, right=right, factor=factor)


class FactorMarginalizeRequest(StrictModel):
    factor: Factor
    variable: Variable


class FactorMarginalizeResult(StrictModel):
    source_factor: Factor
    variable: Variable
    factor: Factor

    @classmethod
    def _from_kernel(
        cls, source_factor: Factor, variable: Variable, factor: Factor
    ) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            source_factor=source_factor, variable=variable, factor=factor
        )


class BayesianDAG(StrictModel):
    """A structural directed graph on one explicit ordered variable axis."""

    variables: tuple[Variable, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    edges: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=MAX_MODEL_VARS * (MAX_MODEL_VARS - 1) // 2
    )

    @model_validator(mode="after")
    def require_structural_graph(self) -> Self:
        if self.variables != tuple(range(len(self.variables))):
            raise ValueError("variables must be the canonical ordered axis")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("directed edges must be distinct")
        if any(
            parent == child
            or parent not in self.variables
            or child not in self.variables
            for parent, child in self.edges
        ):
            raise ValueError("edge endpoints must belong to the variable axis")
        return self

    @property
    def variable_count(self) -> int:
        return len(self.variables)


class DSeparationQuery(StrictModel):
    """A d-separation query bound to one Bayesian DAG."""

    dag: BayesianDAG
    set_a: tuple[Variable, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    set_b: tuple[Variable, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    set_c: tuple[Variable, ...] = Field(default=(), max_length=MAX_MODEL_VARS)

    @model_validator(mode="after")
    def require_structural_query(self) -> Self:
        node_sets = (self.set_a, self.set_b, self.set_c)
        if any(len(values) != len(set(values)) for values in node_sets):
            raise ValueError("d-separation node sets cannot contain duplicates")
        if any(
            node not in self.dag.variables for values in node_sets for node in values
        ):
            raise ValueError("d-separation node is outside the graph")
        if set(self.set_a) & set(self.set_b) or set(self.set_a) & set(self.set_c) or set(
            self.set_b
        ) & set(self.set_c):
            raise ValueError("d-separation node sets must be pairwise disjoint")
        return self


class DSeparationRequest(StrictModel):
    query: DSeparationQuery


class DSeparationResult(StrictModel):
    query: DSeparationQuery
    d_separated: bool

    @classmethod
    def _from_kernel(cls, request: DSeparationRequest, d_separated: bool) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            query=request.query,
            d_separated=d_separated,
        )


__all__ = [
    "BayesianDAG",
    "DSeparationQuery",
    "DSeparationRequest",
    "DSeparationResult",
    "FactorMarginalizeRequest",
    "FactorMarginalizeResult",
    "FactorMultiplyRequest",
    "FactorMultiplyResult",
]
