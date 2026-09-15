"""Typed wire contracts for exact bounded graphical-model operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.probability.graphical_models.values import (
    MAX_MODEL_VARS,
    BayesianNetwork,
    ConditionalProbabilityTable,
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
        if (
            set(self.set_a) & set(self.set_b)
            or set(self.set_a) & set(self.set_c)
            or set(self.set_b) & set(self.set_c)
        ):
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


class BayesNetConstructRequest(StrictModel):
    variable_count: int = Field(ge=1, le=MAX_MODEL_VARS)
    edges: tuple[tuple[int, int], ...] = Field(default=())
    domain_sizes: tuple[int, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    tables: tuple[ConditionalProbabilityTable, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )


class BayesNetConstructResult(StrictModel):
    network: BayesianNetwork

    @classmethod
    def _from_kernel(cls, network: BayesianNetwork) -> Self:
        return cls.model_construct(network=network)


class BayesNetJointRequest(StrictModel):
    network: BayesianNetwork


class BayesNetJointResult(StrictModel):
    network: BayesianNetwork
    joint: Factor

    @classmethod
    def _from_kernel(cls, network: BayesianNetwork, joint: Factor) -> Self:
        return cls.model_construct(network=network, joint=joint)


class EliminationStep(StrictModel):
    eliminated: Variable
    input_scopes: tuple[tuple[Variable, ...], ...]
    product_scope: tuple[Variable, ...]
    product: Factor
    output_scope: tuple[Variable, ...]
    output: Factor
    fill_edges: tuple[tuple[Variable, ...], ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        eliminated: int,
        input_scopes: tuple[tuple[int, ...], ...],
        product_scope: tuple[int, ...],
        product: Factor,
        output_scope: tuple[int, ...],
        output: Factor,
        fill_edges: tuple[tuple[int, int], ...],
    ) -> Self:
        return cls.model_construct(
            eliminated=eliminated,
            input_scopes=input_scopes,
            product_scope=product_scope,
            product=product,
            output_scope=output_scope,
            output=output,
            fill_edges=fill_edges,
        )


class VariableEliminationTraceRequest(StrictModel):
    factors: tuple[Factor, ...] = Field(min_length=1, max_length=64)
    domain_sizes: tuple[int, ...] = Field(min_length=1, max_length=MAX_MODEL_VARS)
    elimination_order: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)
    query_variables: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)


class VariableEliminationTraceResult(StrictModel):
    factors: tuple[Factor, ...]
    domain_sizes: tuple[int, ...]
    elimination_order: tuple[Variable, ...]
    query_variables: tuple[Variable, ...]
    steps: tuple[EliminationStep, ...]
    final: Factor

    @classmethod
    def _from_kernel(
        cls,
        *,
        factors: tuple[Factor, ...],
        domain_sizes: tuple[int, ...],
        elimination_order: tuple[Variable, ...],
        query_variables: tuple[Variable, ...],
        steps: tuple[EliminationStep, ...],
        final: Factor,
    ) -> Self:
        return cls.model_construct(
            factors=factors,
            domain_sizes=domain_sizes,
            elimination_order=elimination_order,
            query_variables=query_variables,
            steps=steps,
            final=final,
        )


class JunctionTreeCalibrateRequest(StrictModel):
    network: BayesianNetwork
    elimination_order: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)


class JunctionTreeCalibrateResult(StrictModel):
    network: BayesianNetwork
    elimination_order: tuple[Variable, ...]
    cliques: tuple[tuple[Variable, ...], ...]
    separators: tuple[tuple[Variable, ...], ...]
    clique_marginals: tuple[Factor, ...]
    separator_marginals: tuple[Factor, ...]
    partition: Factor

    @classmethod
    def _from_kernel(
        cls,
        *,
        network: BayesianNetwork,
        elimination_order: tuple[Variable, ...],
        cliques: tuple[tuple[int, ...], ...],
        separators: tuple[tuple[int, ...], ...],
        clique_marginals: tuple[Factor, ...],
        separator_marginals: tuple[Factor, ...],
        partition: Factor,
    ) -> Self:
        return cls.model_construct(
            network=network,
            elimination_order=elimination_order,
            cliques=cliques,
            separators=separators,
            clique_marginals=clique_marginals,
            separator_marginals=separator_marginals,
            partition=partition,
        )


__all__ = [
    "BayesNetConstructRequest",
    "BayesNetConstructResult",
    "BayesNetJointRequest",
    "BayesNetJointResult",
    "BayesianDAG",
    "DSeparationQuery",
    "DSeparationRequest",
    "DSeparationResult",
    "EliminationStep",
    "FactorMarginalizeRequest",
    "FactorMarginalizeResult",
    "FactorMultiplyRequest",
    "FactorMultiplyResult",
    "JunctionTreeCalibrateRequest",
    "JunctionTreeCalibrateResult",
    "VariableEliminationTraceRequest",
    "VariableEliminationTraceResult",
]
