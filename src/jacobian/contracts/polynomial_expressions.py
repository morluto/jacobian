"""Typed exact rational-polynomial expression contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import comb
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri, Sha256Digest
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.operations import (
    ProviderAvailability,
    ProviderObservation,
)
from jacobian.contracts.polynomials import (
    PolynomialVariable,
    SparseRationalPolynomial,
)
from jacobian.contracts.results import ContractModel

MAX_EXPRESSION_VARIABLES = 4
MAX_EXPRESSION_NODES = 128
MAX_EXPRESSION_DEPTH = 16
MAX_EXPRESSION_OPERANDS = 16
MAX_EXPRESSION_POWER = 32
MAX_EXPRESSION_TERMS = 1024
MAX_EXPRESSION_EXPONENT = 127
MAX_EXPRESSION_INTEGER_DIGITS = 256
MAX_EXPRESSION_COEFFICIENT_DIGIT_BUDGET = 4096
POLYNOMIAL_EXPRESSION_PUBLIC_VALIDATION_MESSAGES = frozenset(
    {
        "polynomial expression variables must be unique",
        "polynomial expressions are limited to 128 AST nodes",
        "polynomial expressions are limited to depth 16",
        "polynomial expression exponent exceeds 127 for a declared variable",
        "polynomial expression exceeds the exact coefficient digit budget",
        "polynomial rational literals are limited to 256 decimal digits",
        "expression variable is not declared",
    }
)
SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION = {
    "distribution": "sympy",
    "domain": "QQ",
    "operation": "Poly(expression, *variables, domain=QQ).terms()",
    "expression_schema_version": "1",
    "maximum_variables": MAX_EXPRESSION_VARIABLES,
    "maximum_nodes": MAX_EXPRESSION_NODES,
    "maximum_depth": MAX_EXPRESSION_DEPTH,
    "maximum_expanded_terms": MAX_EXPRESSION_TERMS,
    "maximum_exponent_per_variable": MAX_EXPRESSION_EXPONENT,
    "maximum_coefficient_digit_budget": MAX_EXPRESSION_COEFFICIENT_DIGIT_BUDGET,
}


class PolynomialRationalExpression(ContractModel):
    kind: Literal["rational"] = "rational"
    value: CanonicalRational


class PolynomialVariableExpression(ContractModel):
    kind: Literal["variable"] = "variable"
    name: PolynomialVariable


class PolynomialAddExpression(ContractModel):
    kind: Literal["add"] = "add"
    operands: tuple[PolynomialExpressionNode, ...] = Field(
        min_length=2,
        max_length=MAX_EXPRESSION_OPERANDS,
    )


class PolynomialMultiplyExpression(ContractModel):
    kind: Literal["multiply"] = "multiply"
    operands: tuple[PolynomialExpressionNode, ...] = Field(
        min_length=2,
        max_length=MAX_EXPRESSION_OPERANDS,
    )


class PolynomialNegateExpression(ContractModel):
    kind: Literal["negate"] = "negate"
    operand: PolynomialExpressionNode


class PolynomialPowerExpression(ContractModel):
    kind: Literal["power"] = "power"
    base: PolynomialExpressionNode
    exponent: StrictInt = Field(ge=0, le=MAX_EXPRESSION_POWER)


PolynomialExpressionNode = Annotated[
    PolynomialRationalExpression
    | PolynomialVariableExpression
    | PolynomialAddExpression
    | PolynomialMultiplyExpression
    | PolynomialNegateExpression
    | PolynomialPowerExpression,
    Field(discriminator="kind"),
]

for _recursive_model in (
    PolynomialAddExpression,
    PolynomialMultiplyExpression,
    PolynomialNegateExpression,
    PolynomialPowerExpression,
):
    _recursive_model.model_rebuild(
        _types_namespace={"PolynomialExpressionNode": PolynomialExpressionNode}
    )


@dataclass(frozen=True, slots=True)
class PolynomialExpressionAnalysis:
    node_count: int
    depth: int
    expanded_term_upper_bound: int
    maximum_exponents: tuple[int, ...]
    coefficient_digit_budget: int


class PolynomialExpansionTermBudgetError(ValueError):
    """A conservative static expansion bound exceeds the hard term cap."""

    def __init__(
        self,
        *,
        estimated_expanded_terms_upper_bound: int,
        requested_exponent: int | None,
    ) -> None:
        self.limit = MAX_EXPRESSION_TERMS
        self.estimated_expanded_terms_upper_bound = estimated_expanded_terms_upper_bound
        self.requested_exponent = requested_exponent
        super().__init__(
            "the conservative polynomial expansion bound exceeds the "
            f"{self.limit}-term budget"
        )


@dataclass(frozen=True, slots=True)
class _NodeAnalysis:
    node_count: int
    depth: int
    term_upper_bound: int
    maximum_exponents: tuple[int, ...]
    coefficient_digit_budget: int


def analyze_polynomial_expression(
    expression: PolynomialExpressionNode,
    variables: tuple[PolynomialVariable, ...],
) -> PolynomialExpressionAnalysis:
    """Validate and conservatively bound one typed polynomial expression."""

    variable_indices = {variable: index for index, variable in enumerate(variables)}
    if len(variable_indices) != len(variables):
        raise ValueError("polynomial expression variables must be unique")
    analysis = _analyze_node(expression, variable_indices)
    if analysis.node_count > MAX_EXPRESSION_NODES:
        raise ValueError("polynomial expressions are limited to 128 AST nodes")
    if analysis.depth > MAX_EXPRESSION_DEPTH:
        raise ValueError("polynomial expressions are limited to depth 16")
    if analysis.term_upper_bound > MAX_EXPRESSION_TERMS:
        raise PolynomialExpansionTermBudgetError(
            estimated_expanded_terms_upper_bound=analysis.term_upper_bound,
            requested_exponent=_maximum_requested_power(expression),
        )
    if any(
        exponent > MAX_EXPRESSION_EXPONENT for exponent in analysis.maximum_exponents
    ):
        raise ValueError(
            "polynomial expression exponent exceeds 127 for a declared variable"
        )
    if analysis.coefficient_digit_budget > MAX_EXPRESSION_COEFFICIENT_DIGIT_BUDGET:
        raise ValueError(
            "polynomial expression exceeds the exact coefficient digit budget"
        )
    return PolynomialExpressionAnalysis(
        node_count=analysis.node_count,
        depth=analysis.depth,
        expanded_term_upper_bound=analysis.term_upper_bound,
        maximum_exponents=analysis.maximum_exponents,
        coefficient_digit_budget=analysis.coefficient_digit_budget,
    )


def _maximum_requested_power(expression: PolynomialExpressionNode) -> int | None:
    if isinstance(expression, PolynomialPowerExpression):
        nested = _maximum_requested_power(expression.base)
        return (
            expression.exponent if nested is None else max(expression.exponent, nested)
        )
    if isinstance(expression, (PolynomialAddExpression, PolynomialMultiplyExpression)):
        powers = tuple(
            power
            for operand in expression.operands
            if (power := _maximum_requested_power(operand)) is not None
        )
        return max(powers) if powers else None
    if isinstance(expression, PolynomialNegateExpression):
        return _maximum_requested_power(expression.operand)
    return None


def _analyze_rational_node(
    expression: PolynomialRationalExpression,
    zero_degrees: tuple[int, ...],
) -> _NodeAnalysis:
    if (
        len(expression.value.num.lstrip("-")) > MAX_EXPRESSION_INTEGER_DIGITS
        or len(expression.value.den) > MAX_EXPRESSION_INTEGER_DIGITS
    ):
        raise ValueError(
            "polynomial rational literals are limited to 256 decimal digits"
        )
    return _NodeAnalysis(
        node_count=1,
        depth=1,
        term_upper_bound=int(expression.value.as_fraction() != 0),
        maximum_exponents=zero_degrees,
        coefficient_digit_budget=(
            len(expression.value.num.lstrip("-")) + len(expression.value.den)
        ),
    )


def _analyze_variable_node(
    expression: PolynomialVariableExpression,
    variable_indices: dict[str, int],
    dimension: int,
) -> _NodeAnalysis:
    index = variable_indices.get(expression.name)
    if index is None:
        raise ValueError("expression variable is not declared")
    variable_degrees = [0] * dimension
    variable_degrees[index] = 1
    return _NodeAnalysis(1, 1, 1, tuple(variable_degrees), 1)


def _analyze_negate_node(
    expression: PolynomialNegateExpression,
    variable_indices: dict[str, int],
) -> _NodeAnalysis:
    operand = _analyze_node(expression.operand, variable_indices)
    return _NodeAnalysis(
        node_count=1 + operand.node_count,
        depth=1 + operand.depth,
        term_upper_bound=operand.term_upper_bound,
        maximum_exponents=operand.maximum_exponents,
        coefficient_digit_budget=operand.coefficient_digit_budget,
    )


def _analyze_power_node(
    expression: PolynomialPowerExpression,
    variable_indices: dict[str, int],
    zero_degrees: tuple[int, ...],
) -> _NodeAnalysis:
    base = _analyze_node(expression.base, variable_indices)
    exponent = expression.exponent
    if exponent == 0:
        terms = 1
        power_degrees = zero_degrees
    elif base.term_upper_bound == 0:
        terms = 0
        power_degrees = zero_degrees
    else:
        terms = _bounded_combination_count(base.term_upper_bound, exponent)
        power_degrees = tuple(value * exponent for value in base.maximum_exponents)
    return _NodeAnalysis(
        node_count=1 + base.node_count,
        depth=1 + base.depth,
        term_upper_bound=terms,
        maximum_exponents=power_degrees,
        coefficient_digit_budget=(
            1 if exponent == 0 else base.coefficient_digit_budget * exponent
        ),
    )


def _analyze_add_or_multiply_node(
    expression: PolynomialAddExpression | PolynomialMultiplyExpression,
    variable_indices: dict[str, int],
    dimension: int,
    zero_degrees: tuple[int, ...],
) -> _NodeAnalysis:
    children = tuple(
        _analyze_node(operand, variable_indices) for operand in expression.operands
    )
    node_count = 1 + sum(child.node_count for child in children)
    depth = 1 + max(child.depth for child in children)
    if isinstance(expression, PolynomialAddExpression):
        terms = _bounded_sum(child.term_upper_bound for child in children)
        combined_degrees = tuple(
            max(child.maximum_exponents[index] for child in children)
            for index in range(dimension)
        )
        coefficient_digit_budget = sum(
            child.coefficient_digit_budget for child in children
        ) + len(children)
    else:
        if any(child.term_upper_bound == 0 for child in children):
            terms = 0
            combined_degrees = zero_degrees
        else:
            terms = _bounded_product(child.term_upper_bound for child in children)
            combined_degrees = tuple(
                sum(child.maximum_exponents[index] for child in children)
                for index in range(dimension)
            )
        coefficient_digit_budget = sum(
            child.coefficient_digit_budget for child in children
        )
    return _NodeAnalysis(
        node_count,
        depth,
        terms,
        combined_degrees,
        coefficient_digit_budget,
    )


def _analyze_node(
    expression: PolynomialExpressionNode,
    variable_indices: dict[str, int],
) -> _NodeAnalysis:
    dimension = len(variable_indices)
    zero_degrees = (0,) * dimension
    if isinstance(expression, PolynomialRationalExpression):
        return _analyze_rational_node(expression, zero_degrees)
    if isinstance(expression, PolynomialVariableExpression):
        return _analyze_variable_node(expression, variable_indices, dimension)
    if isinstance(expression, PolynomialNegateExpression):
        return _analyze_negate_node(expression, variable_indices)
    if isinstance(expression, PolynomialPowerExpression):
        return _analyze_power_node(expression, variable_indices, zero_degrees)
    if isinstance(expression, (PolynomialAddExpression, PolynomialMultiplyExpression)):
        return _analyze_add_or_multiply_node(
            expression, variable_indices, dimension, zero_degrees
        )
    raise TypeError("unsupported polynomial expression node")


def _bounded_sum(values: Iterable[int]) -> int:
    total = 0
    for value in values:
        total += value
        if total > MAX_EXPRESSION_TERMS:
            return MAX_EXPRESSION_TERMS + 1
    return total


def _bounded_product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= value
        if total > MAX_EXPRESSION_TERMS:
            return MAX_EXPRESSION_TERMS + 1
    return total


def _bounded_combination_count(terms: int, exponent: int) -> int:
    if terms == 1:
        return 1
    result = comb(terms + exponent - 1, exponent)
    return min(result, MAX_EXPRESSION_TERMS + 1)


class PolynomialExpressionArtifact(ContractModel):
    expression_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_VARIABLES,
    )
    expression: PolynomialExpressionNode

    @model_validator(mode="after")
    def require_bounded_declared_polynomial(self) -> Self:
        analyze_polynomial_expression(self.expression, self.variables)
        return self


class PolynomialExpressionResourceBudget(ContractModel):
    budget_version: Literal["1"] = "1"
    wall_seconds: StrictInt = Field(default=10, ge=1, le=60)


class PolynomialExpressionBinding(ContractModel):
    binding_version: Literal["1"] = "1"
    expression_artifact_uri: ArtifactUri
    expression_object_digest: Sha256Digest
    expression_payload_digest: Sha256Digest
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_VARIABLES,
    )
    node_count: StrictInt = Field(ge=1, le=MAX_EXPRESSION_NODES)
    depth: StrictInt = Field(ge=1, le=MAX_EXPRESSION_DEPTH)
    expanded_term_upper_bound: StrictInt = Field(
        ge=0,
        le=MAX_EXPRESSION_TERMS,
    )
    coefficient_digit_budget: StrictInt = Field(
        ge=1,
        le=MAX_EXPRESSION_COEFFICIENT_DIGIT_BUDGET,
    )


class PolynomialExpressionNormalizationArtifact(ContractModel):
    normalization_schema_version: Literal["1"] = "1"
    source: PolynomialExpressionBinding
    declared_scope: Literal["FULL_EXPRESSION"] = "FULL_EXPRESSION"
    normalized: SparseRationalPolynomial
    producer: ProviderObservation
    resource_budget: PolynomialExpressionResourceBudget
    method: Literal["SYMPY_POLY_QQ_CANONICAL_TERMS"] = "SYMPY_POLY_QQ_CANONICAL_TERMS"

    @model_validator(mode="after")
    def require_matching_ring_and_pinned_producer(self) -> Self:
        dimension = len(self.source.variables)
        if any(len(term.exponents) != dimension for term in self.normalized.terms):
            raise ValueError(
                "normalized monomials must match the declared variable order"
            )
        if (
            self.producer.provider != "jacobian.sympy"
            or self.producer.availability is not ProviderAvailability.AVAILABLE
            or self.producer.version != "1.14.0"
            or self.producer.configuration
            != SYMPY_POLYNOMIAL_NORMALIZATION_CONFIGURATION
        ):
            raise ValueError(
                "normalization producer must be the pinned SymPy polynomial profile"
            )
        return self


class PolynomialExpressionNormalizeRequest(ContractModel):
    expression: PolynomialExpressionArtifact
    resource_budget: PolynomialExpressionResourceBudget = Field(
        default_factory=PolynomialExpressionResourceBudget
    )


class PolynomialExpressionNormalizeOutput(ContractModel):
    status: Literal["NORMALIZATION_PRODUCED", "NO_NORMALIZATION_PRODUCED"]
    expression_uri: ArtifactUri
    normalization_uri: ArtifactUri | None = None
    normalized: SparseRationalPolynomial | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_candidate_projection(self) -> Self:
        produced = self.status == "NORMALIZATION_PRODUCED"
        if produced != (
            self.normalization_uri is not None and self.normalized is not None
        ):
            raise ValueError(
                "produced output requires one durable normalization candidate"
            )
        if not produced and (
            self.normalization_uri is not None or self.normalized is not None
        ):
            raise ValueError("failed output cannot carry normalization evidence")
        return self


class PolynomialExpressionNormalizationVerificationRequest(ContractModel):
    normalization_uri: ArtifactUri


class PolynomialExpressionNormalizationVerificationOutput(ContractModel):
    status: Literal[
        "VERIFIED_NORMALIZATION",
        "REJECTED",
        "TIMEOUT",
        "CANCELLED",
        "ERROR",
    ]
    conclusion: Literal["TRUE", "UNKNOWN"]
    expression_uri: ArtifactUri
    normalization_uri: ArtifactUri
    witness_uri: ArtifactUri
    checker_id: CheckerUri
    verification_record_uri: ArtifactUri | None = None
    detail: str = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def bind_verified_projection(self) -> Self:
        if self.status == "VERIFIED_NORMALIZATION":
            if self.conclusion != "TRUE" or self.verification_record_uri is None:
                raise ValueError(
                    "verified normalization requires TRUE and a verification record"
                )
        elif self.conclusion != "UNKNOWN" or self.verification_record_uri is not None:
            raise ValueError(
                "non-verified normalization cannot carry a conclusion or record"
            )
        return self
