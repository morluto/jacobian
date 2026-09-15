"""Provider-independent values for exact bounded finite graphical models."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

# Ambient model metadata is intentionally bounded separately from active factor
# work.  Local products/marginals may touch only a few axes of a larger model;
# ``scope_size`` continues to admit the materialized table before expansion.
MAX_MODEL_VARS = 64
MAX_VAR_DOMAIN = 32
MAX_FACTOR_TABLE_SIZE = 4_096
MAX_FACTOR_COUNT = 64
MAX_RATIONAL_DIGITS = 256

DomainSize = Annotated[int, Field(ge=1, le=MAX_VAR_DOMAIN)]
Variable = Annotated[int, Field(ge=0, lt=MAX_MODEL_VARS)]


def scope_size(variables: tuple[int, ...], domain_sizes: tuple[int, ...]) -> int:
    """Return a bounded factor-table size for a validated variable scope."""

    size = 1
    for variable in variables:
        if variable >= len(domain_sizes):
            raise ValueError("variable index exceeds domain_sizes")
        size *= domain_sizes[variable]
        if size > MAX_FACTOR_TABLE_SIZE:
            raise ValueError("factor table exceeds the supported size bound")
    return size


class Factor(StrictModel):
    """An exact factor indexed lexicographically by its ordered variable scope.

    The empty scope represents a scalar and therefore has exactly one table entry.
    ``domain_sizes`` describes the complete shared model domain.
    Every table entry must be a nonnegative canonical rational.
    """

    variables: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)
    domain_sizes: tuple[DomainSize, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )
    table: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_FACTOR_TABLE_SIZE,
        description=(
            "Nonnegative CanonicalRational potentials in lexicographic scope "
            "order; numerators must be nonnegative."
        ),
    )

    @model_validator(mode="after")
    def require_valid_factor(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise PydanticCustomError(
                "graphical_model.factor_variables_not_unique",
                "factor variables must be distinct",
            )
        try:
            expected_size = scope_size(self.variables, self.domain_sizes)
        except ValueError as error:
            raise PydanticCustomError(
                "graphical_model.factor_scope_invalid",
                str(error),
            ) from error
        if len(self.table) != expected_size:
            raise PydanticCustomError(
                "graphical_model.factor_table_size",
                f"table size {len(self.table)} does not match expected {expected_size}",
            )
        for value in self.table:
            try:
                require_bounded_rational(
                    value,
                    max_digits=MAX_RATIONAL_DIGITS,
                    label="factor entry",
                )
            except ValueError as error:
                raise PydanticCustomError(
                    "graphical_model.factor_entry_invalid",
                    str(error),
                ) from error
            if value.num < 0:
                raise PydanticCustomError(
                    "graphical_model.factor_entry_negative",
                    "factor entries must be nonnegative",
                )
        return self


class ConditionalProbabilityTable(StrictModel):
    """An exact row-normalized CPT for one variable given its parents.

    ``variables`` is ``(variable, *sorted_parents)`` and the table is
    lexicographic in that order. Every parent-assignment row sums exactly to
    one; serialization preserves exact variable/outcome semantics.
    """

    variable: Variable
    parents: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)
    domain_sizes: tuple[DomainSize, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )
    table: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_FACTOR_TABLE_SIZE
    )

    @model_validator(mode="after")
    def require_valid_cpt(self) -> Self:
        if self.variable >= len(self.domain_sizes):
            raise PydanticCustomError(
                "graphical_model.cpt_variable_out_of_range",
                "cpt variable is outside the model domain",
            )
        if len({self.variable, *self.parents}) != 1 + len(self.parents):
            raise PydanticCustomError(
                "graphical_model.cpt_scope_not_distinct",
                "cpt variable and parents must be distinct",
            )
        if self.parents != tuple(sorted(self.parents)):
            raise PydanticCustomError(
                "graphical_model.cpt_parents_not_sorted",
                "cpt parents must be sorted",
            )
        if any(parent >= len(self.domain_sizes) for parent in self.parents):
            raise PydanticCustomError(
                "graphical_model.cpt_parent_out_of_range",
                "cpt parent is outside the model domain",
            )
        variables = (self.variable, *self.parents)
        try:
            expected = scope_size(variables, self.domain_sizes)
        except ValueError as error:
            raise PydanticCustomError(
                "graphical_model.cpt_scope_invalid", str(error)
            ) from error
        if len(self.table) != expected:
            raise PydanticCustomError(
                "graphical_model.cpt_table_size",
                "cpt table size does not match its scope",
            )
        for value in self.table:
            try:
                require_bounded_rational(
                    value, max_digits=MAX_RATIONAL_DIGITS, label="cpt entry"
                )
            except ValueError as error:
                raise PydanticCustomError(
                    "graphical_model.cpt_entry_invalid", str(error)
                ) from error
            if value.num < 0:
                raise PydanticCustomError(
                    "graphical_model.cpt_entry_negative",
                    "cpt entries must be nonnegative",
                )
        return self

    @property
    def variables(self) -> tuple[int, ...]:
        return (self.variable, *self.parents)

    def as_factor(self) -> Factor:
        return Factor.model_construct(
            variables=self.variables,
            domain_sizes=self.domain_sizes,
            table=self.table,
        )


class BayesianNetwork(StrictModel):
    """A DAG with one CPT per variable bound exactly to its parent set."""

    variable_count: int = Field(ge=1, le=MAX_MODEL_VARS)
    edges: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=MAX_MODEL_VARS * (MAX_MODEL_VARS - 1) // 2
    )
    domain_sizes: tuple[DomainSize, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )
    tables: tuple[ConditionalProbabilityTable, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )

    @model_validator(mode="after")
    def require_bound_network(self) -> Self:
        if len(self.domain_sizes) != self.variable_count:
            raise PydanticCustomError(
                "graphical_model.network_domain_count",
                "domain_sizes must describe every network variable",
            )
        if len(self.tables) != self.variable_count:
            raise PydanticCustomError(
                "graphical_model.network_table_count",
                "network must carry exactly one table per variable",
            )
        if self.edges != tuple(sorted(set(self.edges))):
            raise PydanticCustomError(
                "graphical_model.network_edges_canonical",
                "network edges must be distinct and sorted",
            )
        for parent, child in self.edges:
            if (
                not 0 <= parent < self.variable_count
                or not 0 <= child < self.variable_count
                or parent == child
            ):
                raise PydanticCustomError(
                    "graphical_model.network_edge_endpoints",
                    "network edges must join distinct model variables",
                )
        # Acyclicity is established by the constructing operation; structural
        # decoding checks only the cheap parent-set binding below.
        parents: dict[int, tuple[int, ...]] = {
            variable: tuple(
                sorted(parent for parent, child in self.edges if child == variable)
            )
            for variable in range(self.variable_count)
        }
        seen: set[int] = set()
        for table in self.tables:
            if table.variable in seen:
                raise PydanticCustomError(
                    "graphical_model.network_duplicate_table",
                    "network must carry exactly one table per variable",
                )
            seen.add(table.variable)
            if table.domain_sizes != self.domain_sizes:
                raise PydanticCustomError(
                    "graphical_model.network_domain_mismatch",
                    "every table must share the network domain_sizes",
                )
            if table.parents != parents[table.variable]:
                raise PydanticCustomError(
                    "graphical_model.network_parent_binding",
                    "every table must bind exactly its DAG parent set",
                )
        if seen != set(range(self.variable_count)):
            raise PydanticCustomError(
                "graphical_model.network_table_coverage",
                "network tables must cover every variable once",
            )
        return self


__all__ = [
    "MAX_FACTOR_COUNT",
    "MAX_FACTOR_TABLE_SIZE",
    "MAX_MODEL_VARS",
    "MAX_RATIONAL_DIGITS",
    "MAX_VAR_DOMAIN",
    "BayesianNetwork",
    "ConditionalProbabilityTable",
    "Factor",
    "scope_size",
]
