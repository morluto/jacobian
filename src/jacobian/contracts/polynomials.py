"""Exact sparse rational-polynomial map contracts."""

from __future__ import annotations

from enum import StrEnum
from itertools import permutations
from math import prod
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian.contracts.common import ArtifactUri, CheckerUri
from jacobian.contracts.exact import (
    RATIONAL_SEARCH_GRID_LIMIT,
    CanonicalRational,
    bounded_rational_grid_size,
)
from jacobian.contracts.results import Conclusion, ContractModel, InputValidation

PolynomialVariable = Annotated[
    str,
    StringConstraints(
        pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$",
        strict=True,
    ),
]
_MAX_SOURCE_EXPONENT = 32
_MAX_DERIVED_EXPONENT = 4 * _MAX_SOURCE_EXPONENT - 1
_MAX_JACOBIAN_PRODUCT_TERM_ESTIMATE = 1024


class RationalPolynomialTerm(ContractModel):
    coefficient: CanonicalRational
    exponents: tuple[int, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_nonzero_coefficient_and_bounded_exponents(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero polynomial terms must be omitted")
        if any(
            exponent < 0 or exponent > _MAX_DERIVED_EXPONENT
            for exponent in self.exponents
        ):
            raise ValueError(
                "polynomial exponents exceed the bounded derived-polynomial limit"
            )
        return self


class SparseRationalPolynomial(ContractModel):
    terms: tuple[RationalPolynomialTerm, ...] = Field(
        default=(),
        max_length=1024,
    )

    @model_validator(mode="after")
    def require_unique_canonical_term_order(self) -> Self:
        exponents = tuple(term.exponents for term in self.terms)
        if len(set(exponents)) != len(exponents):
            raise ValueError("polynomial exponent tuples must be unique")
        if exponents != tuple(sorted(exponents, reverse=True)):
            raise ValueError("polynomial terms must use descending lexicographic order")
        return self


class RationalPolynomial(ContractModel):
    """One sparse polynomial together with its exact coefficient ring."""

    polynomial_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    polynomial: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial variables must be unique")
        if any(
            len(term.exponents) != len(self.variables) for term in self.polynomial.terms
        ):
            raise ValueError("every monomial must match the declared variable order")
        return self


class RationalPolynomialMap(ContractModel):
    map_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    coordinates: tuple[SparseRationalPolynomial, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def require_square_map_and_matching_monomials(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial variables must be unique")
        if len(self.coordinates) != len(self.variables):
            raise ValueError("the first polynomial-map contract supports square maps")
        if any(
            len(term.exponents) != len(self.variables)
            for polynomial in self.coordinates
            for term in polynomial.terms
        ):
            raise ValueError("every monomial must match the declared variable order")
        if any(
            exponent > _MAX_SOURCE_EXPONENT
            for polynomial in self.coordinates
            for term in polynomial.terms
            for exponent in term.exponents
        ):
            raise ValueError("source polynomial exponents must be between zero and 32")
        return self


class RationalPolynomialPoint(ContractModel):
    values: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)


class PolynomialEvaluationRequest(ContractModel):
    map: RationalPolynomialMap
    point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_point_dimension(self) -> Self:
        if len(self.point) != len(self.map.variables):
            raise ValueError("evaluation point dimension must match the polynomial map")
        return self


class PolynomialJacobianRequest(ContractModel):
    map: RationalPolynomialMap

    @model_validator(mode="after")
    def require_bounded_determinant_expansion(self) -> Self:
        dimension = len(self.map.variables)
        derivative_term_counts = tuple(
            tuple(
                sum(term.exponents[column] > 0 for term in polynomial.terms)
                for column in range(dimension)
            )
            for polynomial in self.map.coordinates
        )
        estimate = sum(
            prod(
                derivative_term_counts[row][permutation[row]]
                for row in range(dimension)
            )
            for permutation in permutations(range(dimension))
        )
        if estimate > _MAX_JACOBIAN_PRODUCT_TERM_ESTIMATE:
            raise ValueError(
                "Jacobian determinant expansion exceeds the exact operation budget"
            )
        return self


class PolynomialKellerConditionVerifyRequest(ContractModel):
    """Verify that an exact polynomial map has a nonzero constant Jacobian."""

    map: RationalPolynomialMap


class PolynomialFactorRequest(ContractModel):
    variable: PolynomialVariable
    polynomial: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_univariate_terms(self) -> Self:
        if any(len(term.exponents) != 1 for term in self.polynomial.terms):
            raise ValueError("factorization currently supports univariate polynomials")
        return self


class PolynomialFactorRecord(ContractModel):
    factor: SparseRationalPolynomial
    multiplicity: int = Field(ge=1, le=127)


class PolynomialFactorizationArtifact(ContractModel):
    factorization_schema_version: Literal["1"] = "1"
    variable: PolynomialVariable
    source_polynomial_uri: ArtifactUri
    coefficient: CanonicalRational
    factors: tuple[PolynomialFactorRecord, ...] = Field(max_length=1024)
    reconstructed: SparseRationalPolynomial
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)


class PolynomialFactorOutput(ContractModel):
    source_polynomial_uri: ArtifactUri
    factorization_uri: ArtifactUri
    variable: PolynomialVariable
    coefficient: CanonicalRational
    factors: tuple[PolynomialFactorRecord, ...]
    reconstructed: SparseRationalPolynomial
    exactness: Literal["EXACT"] = "EXACT"
    product_reconstruction: Literal["EXACT"] = "EXACT"
    irreducibility_verification: Literal["UNVERIFIED"] = "UNVERIFIED"
    backend: Literal["sympy"] = "sympy"
    backend_version: str


class PolynomialCollisionRequest(ContractModel):
    first_evaluation_uri: ArtifactUri
    second_evaluation_uri: ArtifactUri

    @model_validator(mode="after")
    def require_distinct_evaluation_artifacts(self) -> Self:
        if self.first_evaluation_uri == self.second_evaluation_uri:
            raise ValueError(
                "collision comparison requires distinct evaluation artifacts"
            )
        return self


class PolynomialIdentityRequest(ContractModel):
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left: SparseRationalPolynomial
    right: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_matching_ring(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("polynomial variables must be unique")
        dimension = len(self.variables)
        if any(
            len(term.exponents) != dimension
            for polynomial in (self.left, self.right)
            for term in polynomial.terms
        ):
            raise ValueError("every monomial must match the declared variable order")
        return self


class SparseRationalFunction(ContractModel):
    """One bounded fraction of sparse polynomials over a shared QQ ring."""

    numerator: SparseRationalPolynomial
    denominator: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_nonzero_denominator(self) -> Self:
        if not self.denominator.terms:
            raise ValueError("rational-function denominator must be nonzero")
        return self


class RationalFunctionIdentityRequest(ContractModel):
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left: SparseRationalFunction
    right: SparseRationalFunction

    @model_validator(mode="after")
    def require_matching_bounded_fraction_field(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("rational-function variables must be unique")
        dimension = len(self.variables)
        polynomials = (
            self.left.numerator,
            self.left.denominator,
            self.right.numerator,
            self.right.denominator,
        )
        if any(
            len(term.exponents) != dimension
            for polynomial in polynomials
            for term in polynomial.terms
        ):
            raise ValueError("every monomial must match the declared variable order")
        if (
            max(
                len(self.left.numerator.terms) * len(self.right.denominator.terms),
                len(self.right.numerator.terms) * len(self.left.denominator.terms),
            )
            > 4096
        ):
            raise ValueError("rational-function cross product exceeds 4096 term pairs")
        return self


class RationalFunctionArtifact(ContractModel):
    rational_function_schema_version: Literal["1"] = "1"
    domain: Literal["QQ_FRACTION_FIELD"] = "QQ_FRACTION_FIELD"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    numerator: SparseRationalPolynomial
    denominator: SparseRationalPolynomial

    @model_validator(mode="after")
    def require_matching_fraction_field(self) -> Self:
        # Validate the artifact directly rather than reusing the
        # two-function cross-product validator: the identity-request bound
        # applies the pair limit to a single fraction's self-product
        # (numerator.terms * denominator.terms), which incorrectly rejects a
        # valid 65x65 fraction.  The artifact only needs variable uniqueness,
        # exponent dimension matching, and a nonzero denominator.
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("rational-function variables must be unique")
        dimension = len(self.variables)
        for polynomial in (self.numerator, self.denominator):
            if any(len(term.exponents) != dimension for term in polynomial.terms):
                raise ValueError(
                    "every monomial must match the declared variable order"
                )
        if not self.denominator.terms:
            raise ValueError("rational-function denominator must be nonzero")
        return self


def _validate_ring_variable_alignment(
    forward_map: RationalPolynomialMap,
    inverse_map: RationalPolynomialMap,
    source_variables: tuple[PolynomialVariable, ...],
    target_variables: tuple[PolynomialVariable, ...],
) -> None:
    if forward_map.variables != source_variables:
        raise ValueError("forward map variables must equal source_variables")
    if inverse_map.variables != target_variables:
        raise ValueError("inverse map variables must equal target_variables")
    if len(source_variables) != len(target_variables):
        raise ValueError("source and target dimensions must agree")
    if len(set(source_variables)) != len(source_variables):
        raise ValueError("source variables must be unique")
    if len(set(target_variables)) != len(target_variables):
        raise ValueError("target variables must be unique")


def _validate_composition_residual_bounds(
    outer: RationalPolynomialMap,
    inner: RationalPolynomialMap,
) -> None:
    inner_term_counts = tuple(len(coordinate.terms) for coordinate in inner.coordinates)
    for coordinate in outer.coordinates:
        term_bound = 0
        for term in coordinate.terms:
            term_bound += prod(
                count**exponent
                for count, exponent in zip(
                    inner_term_counts,
                    term.exponents,
                    strict=True,
                )
            )
            if term_bound > 1024:
                raise ValueError("composition residual term bound exceeds 1024")
        outer_degree = max(
            (sum(term.exponents) for term in coordinate.terms),
            default=0,
        )
        inner_degree = max(
            (
                sum(term.exponents)
                for inner_coordinate in inner.coordinates
                for term in inner_coordinate.terms
            ),
            default=0,
        )
        if outer_degree * inner_degree > _MAX_DERIVED_EXPONENT:
            raise ValueError("composition residual degree bound exceeds 127")


class PolynomialMapInverseVerifyRequest(ContractModel):
    forward_map: RationalPolynomialMap
    inverse_map: RationalPolynomialMap
    source_variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    target_variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_compatible_ordered_rings(self) -> Self:
        _validate_ring_variable_alignment(
            self.forward_map,
            self.inverse_map,
            self.source_variables,
            self.target_variables,
        )
        for outer, inner in (
            (self.inverse_map, self.forward_map),
            (self.forward_map, self.inverse_map),
        ):
            _validate_composition_residual_bounds(outer, inner)
        return self


class PolynomialInverseSynthesisStatus(StrEnum):
    FOUND = "FOUND"
    NO_CANDIDATE_WITHIN_ANSATZ = "NO_CANDIDATE_WITHIN_ANSATZ"
    UNDERDETERMINED = "UNDERDETERMINED"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNSUPPORTED = "UNSUPPORTED"


class PolynomialInverseSupportMode(StrEnum):
    EXPLICIT = "EXPLICIT"
    FULL_TOTAL_DEGREE = "FULL_TOTAL_DEGREE"


class PolynomialInverseSynthesisLimits(ContractModel):
    timeout_ms: int = Field(ge=0, le=120_000)
    max_inverse_degree: int = Field(ge=0, le=8)
    max_composition_degree: int = Field(ge=1, le=_MAX_DERIVED_EXPONENT)
    max_unknown_coefficients: int = Field(ge=1, le=512)
    max_coefficient_equations: int = Field(ge=1, le=4096)
    max_residual_terms: int = Field(ge=1, le=8192)


def _validate_ansatz_ring_alignment(
    forward_map: RationalPolynomialMap,
    source_variables: tuple[PolynomialVariable, ...],
    target_variables: tuple[PolynomialVariable, ...],
) -> None:
    if forward_map.variables != source_variables:
        raise ValueError("forward map variables must equal source_variables")
    if len(source_variables) != len(target_variables):
        raise ValueError("source and target dimensions must agree")
    if len(set(source_variables)) != len(source_variables):
        raise ValueError("source variables must be unique")
    if len(set(target_variables)) != len(target_variables):
        raise ValueError("target variables must be unique")


def _validate_explicit_coordinate_support(
    coordinate: tuple[tuple[int, ...], ...],
    target_dimension: int,
    inverse_degree_bound: int,
) -> None:
    if not coordinate:
        raise ValueError("each explicit coordinate support must be nonempty")
    if coordinate != tuple(sorted(coordinate, reverse=True)):
        raise ValueError("explicit support must use descending lexicographic order")
    for exponents in coordinate:
        if len(exponents) != target_dimension:
            raise ValueError("support monomials must match target variable order")
        if any(exponent < 0 for exponent in exponents):
            raise ValueError("support exponents must be nonnegative")
        if sum(exponents) > inverse_degree_bound:
            raise ValueError("explicit support exceeds inverse_degree_bound")


class PolynomialMapInverseSynthesisRequest(ContractModel):
    forward_map: RationalPolynomialMap
    source_variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    target_variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    inverse_degree_bound: int = Field(ge=0, le=8)
    support_mode: PolynomialInverseSupportMode
    explicit_support: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    solver: str = Field(min_length=1, max_length=64)
    limits: PolynomialInverseSynthesisLimits

    @model_validator(mode="after")
    def require_bounded_square_qq_ansatz(self) -> Self:
        _validate_ansatz_ring_alignment(
            self.forward_map,
            self.source_variables,
            self.target_variables,
        )
        if self.inverse_degree_bound > self.limits.max_inverse_degree:
            raise ValueError("inverse_degree_bound exceeds the declared degree limit")
        if self.support_mode is PolynomialInverseSupportMode.EXPLICIT:
            if self.explicit_support is None:
                raise ValueError("EXPLICIT support_mode requires explicit_support")
            if len(self.explicit_support) != len(self.target_variables):
                raise ValueError(
                    "explicit_support must contain one support per coordinate"
                )
            for coordinate in self.explicit_support:
                _validate_explicit_coordinate_support(
                    coordinate,
                    len(self.target_variables),
                    self.inverse_degree_bound,
                )
        elif self.explicit_support is not None:
            raise ValueError(
                "FULL_TOTAL_DEGREE support_mode must not carry explicit_support"
            )
        return self


class PolynomialInverseAnsatzSpecification(ContractModel):
    support_mode: PolynomialInverseSupportMode
    inverse_degree_bound: int
    source_variables: tuple[PolynomialVariable, ...]
    target_variables: tuple[PolynomialVariable, ...]
    coordinate_supports: tuple[tuple[tuple[int, ...], ...], ...]
    coefficient_symbols: tuple[tuple[str, ...], ...]


class PolynomialInverseCoefficientEquation(ContractModel):
    direction: Literal["INVERSE_AFTER_FORWARD", "FORWARD_AFTER_INVERSE"]
    coordinate: int = Field(ge=0, le=3)
    monomial_exponents: tuple[int, ...]
    expression: str = Field(min_length=1, max_length=100_000)


class PolynomialInverseSolverProvenance(ContractModel):
    solver: str
    backend: Literal["sympy"] = "sympy"
    backend_version: str
    exact_domain: Literal["QQ"] = "QQ"
    timeout_ms: int
    unknown_count: int = Field(ge=0)
    equation_count: int = Field(ge=0)
    residual_term_count: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)


class PolynomialMapInverseSynthesisArtifact(ContractModel):
    synthesis_schema_version: Literal["1"] = "1"
    status: PolynomialInverseSynthesisStatus
    forward_map: RationalPolynomialMap
    ansatz: PolynomialInverseAnsatzSpecification
    coefficient_equations: tuple[PolynomialInverseCoefficientEquation, ...]
    solver_provenance: PolynomialInverseSolverProvenance
    candidate_inverse_map: RationalPolynomialMap | None = None
    inverse_after_forward: tuple[SparseRationalPolynomial, ...] = ()
    forward_after_inverse: tuple[SparseRationalPolynomial, ...] = ()
    verification_output: dict[str, Any] | None = None
    verification_artifact_uri: ArtifactUri | None = None
    verification_failure: str | None = None
    noninvertibility_proved: Literal[False] = False

    @model_validator(mode="after")
    def require_status_consistent_candidate_bundle(self) -> Self:
        found = self.status is PolynomialInverseSynthesisStatus.FOUND
        if found != (self.candidate_inverse_map is not None):
            raise ValueError("FOUND status must agree with candidate_inverse_map")
        if found and (not self.inverse_after_forward or not self.forward_after_inverse):
            raise ValueError("FOUND requires both composition residual families")
        if not found and (self.inverse_after_forward or self.forward_after_inverse):
            raise ValueError("only FOUND may carry composition residual families")
        if self.verification_artifact_uri is not None and not found:
            raise ValueError("only FOUND may carry a verification artifact")
        return self


class PolynomialMapInverseSynthesisOutput(PolynomialMapInverseSynthesisArtifact):
    synthesis_uri: ArtifactUri
    forward_map_uri: ArtifactUri


class PolynomialCollisionSearchRequest(ContractModel):
    map: RationalPolynomialMap
    max_abs_numerator: int = Field(ge=0, le=8)
    max_denominator: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def require_bounded_grid(self) -> Self:
        if (
            bounded_rational_grid_size(
                self.max_abs_numerator,
                self.max_denominator,
                len(self.map.variables),
            )
            > RATIONAL_SEARCH_GRID_LIMIT
        ):
            raise ValueError("declared rational grid exceeds the 10,000-point limit")
        return self


class PolynomialCollisionSearchStopReason(StrEnum):
    FIRST_COLLISION = "FIRST_COLLISION"
    GRID_EXHAUSTED = "GRID_EXHAUSTED"


class PolynomialCollisionVerifyRequest(ContractModel):
    map: RationalPolynomialMap
    first_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    second_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    claimed_image: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_collision_dimensions_and_distinct_points(self) -> Self:
        dimension = len(self.map.variables)
        if not (
            len(self.first_point)
            == len(self.second_point)
            == len(self.claimed_image)
            == dimension
        ):
            raise ValueError("collision points and image must match the map dimension")
        if self.first_point == self.second_point:
            raise ValueError("collision verification requires distinct points")
        return self


class PolynomialMapInverseCollisionVerifyRequest(ContractModel):
    """Use one exact collision to refute a two-sided polynomial inverse."""

    map: RationalPolynomialMap
    first_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    second_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    claimed_image: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def require_collision_dimensions_and_distinct_points(self) -> Self:
        dimension = len(self.map.variables)
        if not (
            len(self.first_point)
            == len(self.second_point)
            == len(self.claimed_image)
            == dimension
        ):
            raise ValueError("collision points and image must match the map dimension")
        if self.first_point == self.second_point:
            raise ValueError("collision verification requires distinct points")
        return self


class PolynomialMapEvaluation(ContractModel):
    evaluation_schema_version: Literal["1"] = "1"
    map_uri: ArtifactUri
    point: RationalPolynomialPoint
    image: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_equal_point_and_image_dimensions(self) -> Self:
        if len(self.point.values) != len(self.image):
            raise ValueError("evaluation point and image dimensions must agree")
        return self


class PolynomialJacobian(ContractModel):
    jacobian_schema_version: Literal["1"] = "1"
    map_uri: ArtifactUri
    variable_order: tuple[PolynomialVariable, ...] = Field(
        min_length=1,
        max_length=4,
    )
    matrix: tuple[tuple[SparseRationalPolynomial, ...], ...] = Field(
        min_length=1,
        max_length=4,
    )
    determinant: SparseRationalPolynomial
    backend: Literal["sympy"] = "sympy"
    backend_version: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_square_jacobian(self) -> Self:
        dimension = len(self.variable_order)
        if len(self.matrix) != dimension or any(
            len(row) != dimension for row in self.matrix
        ):
            raise ValueError("Jacobian matrix must match the variable order")
        if any(
            len(term.exponents) != dimension
            for row in self.matrix
            for polynomial in row
            for term in polynomial.terms
        ) or any(len(term.exponents) != dimension for term in self.determinant.terms):
            raise ValueError("Jacobian monomials must match the variable order")
        return self


class PolynomialInjectivityClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_MAP_INJECTIVE"] = "POLYNOMIAL_MAP_INJECTIVE"
    domain: Literal["QQ"] = "QQ"
    map_uri: ArtifactUri


class PolynomialKellerConditionClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_MAP_KELLER_CONDITION"] = (
        "POLYNOMIAL_MAP_KELLER_CONDITION"
    )
    domain: Literal["QQ"] = "QQ"
    map_uri: ArtifactUri
    jacobian_uri: ArtifactUri


class PolynomialNoTwoSidedInverseClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_MAP_NO_TWO_SIDED_INVERSE"] = (
        "POLYNOMIAL_MAP_NO_TWO_SIDED_INVERSE"
    )
    domain: Literal["QQ"] = "QQ"
    map_uri: ArtifactUri


class PolynomialJacobianClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["EXACT_POLYNOMIAL_JACOBIAN"] = "EXACT_POLYNOMIAL_JACOBIAN"
    source_map_uri: ArtifactUri


class PolynomialIdentityClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_IDENTITY"] = "POLYNOMIAL_IDENTITY"
    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left_uri: ArtifactUri
    right_uri: ArtifactUri


class PolynomialIdentityReplayPayload(ContractModel):
    method: Literal["DIRECT_SPARSE_REPLAY"] = "DIRECT_SPARSE_REPLAY"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left_uri: ArtifactUri
    right_uri: ArtifactUri


class RationalFunctionIdentityClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["RATIONAL_FUNCTION_IDENTITY"] = "RATIONAL_FUNCTION_IDENTITY"
    domain: Literal["QQ_FRACTION_FIELD"] = "QQ_FRACTION_FIELD"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left_uri: ArtifactUri
    right_uri: ArtifactUri


class RationalFunctionIdentityReplayPayload(ContractModel):
    method: Literal["CROSS_MULTIPLY_SPARSE_POLYNOMIALS"] = (
        "CROSS_MULTIPLY_SPARSE_POLYNOMIALS"
    )
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=4)
    left_uri: ArtifactUri
    right_uri: ArtifactUri


class PolynomialMapInverseClaim(ContractModel):
    claim_schema_version: Literal["1"] = "1"
    predicate: Literal["POLYNOMIAL_MAP_TWO_SIDED_INVERSE"] = (
        "POLYNOMIAL_MAP_TWO_SIDED_INVERSE"
    )
    domain: Literal["QQ"] = "QQ"
    forward_map_uri: ArtifactUri
    inverse_map_uri: ArtifactUri
    source_variables: tuple[PolynomialVariable, ...]
    target_variables: tuple[PolynomialVariable, ...]


class PolynomialMapCompositionResiduals(ContractModel):
    residual_schema_version: Literal["1"] = "1"
    domain: Literal["QQ"] = "QQ"
    forward_map_uri: ArtifactUri
    inverse_map_uri: ArtifactUri
    source_variables: tuple[PolynomialVariable, ...]
    target_variables: tuple[PolynomialVariable, ...]
    inverse_after_forward: tuple[SparseRationalPolynomial, ...]
    forward_after_inverse: tuple[SparseRationalPolynomial, ...]
    inverse_after_forward_checker_records: tuple[ArtifactUri, ...]
    forward_after_inverse_checker_records: tuple[ArtifactUri, ...]

    @model_validator(mode="after")
    def require_complete_two_sided_bundle(self) -> Self:
        dimension = len(self.source_variables)
        if not (
            dimension
            == len(self.target_variables)
            == len(self.inverse_after_forward)
            == len(self.forward_after_inverse)
            == len(self.inverse_after_forward_checker_records)
            == len(self.forward_after_inverse_checker_records)
        ):
            raise ValueError(
                "both residual and checker-record families must be complete"
            )
        return self


class PolynomialMapInverseReplayPayload(ContractModel):
    method: Literal["DIRECT_TWO_SIDED_SPARSE_REPLAY"] = "DIRECT_TWO_SIDED_SPARSE_REPLAY"
    forward_map_uri: ArtifactUri
    inverse_map_uri: ArtifactUri
    residuals_uri: ArtifactUri
    source_variables: tuple[PolynomialVariable, ...]
    target_variables: tuple[PolynomialVariable, ...]
    inverse_after_forward_checker_records: tuple[ArtifactUri, ...]
    forward_after_inverse_checker_records: tuple[ArtifactUri, ...]


class PolynomialJacobianReplayPayload(ContractModel):
    method: Literal["DIRECT_SPARSE_REPLAY"] = "DIRECT_SPARSE_REPLAY"
    source_map_uri: ArtifactUri
    jacobian_uri: ArtifactUri


class PolynomialKellerConditionReplayPayload(ContractModel):
    method: Literal["DIRECT_SPARSE_KELLER_REPLAY"] = "DIRECT_SPARSE_KELLER_REPLAY"
    map_uri: ArtifactUri
    jacobian_uri: ArtifactUri


class PolynomialCollisionPayload(ContractModel):
    first_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    second_point: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)
    image: tuple[CanonicalRational, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if not (len(self.first_point) == len(self.second_point) == len(self.image)):
            raise ValueError("collision points and image dimensions must agree")
        return self


class PolynomialExactness(StrEnum):
    EXACT = "EXACT"


class PolynomialDeterminism(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"


class PolynomialVerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"


class PolynomialEvaluationOutput(ContractModel):
    map_uri: ArtifactUri
    evaluation_uri: ArtifactUri
    point: tuple[CanonicalRational, ...]
    image: tuple[CanonicalRational, ...]
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    determinism: PolynomialDeterminism = PolynomialDeterminism.DETERMINISTIC
    verification: PolynomialVerificationStatus = PolynomialVerificationStatus.UNVERIFIED
    certificate_available: Literal[False] = False
    checker_id: None = None
    backend: Literal["sympy"] = "sympy"
    backend_version: str

    @model_validator(mode="after")
    def require_equal_point_and_image_dimensions(self) -> Self:
        if len(self.point) != len(self.image):
            raise ValueError("evaluation output dimensions must agree")
        return self


class PolynomialJacobianOutput(ContractModel):
    map_uri: ArtifactUri
    jacobian_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    checker_id: CheckerUri | None = None
    matrix: tuple[tuple[SparseRationalPolynomial, ...], ...]
    determinant: SparseRationalPolynomial
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    determinism: PolynomialDeterminism = PolynomialDeterminism.DETERMINISTIC
    verification: PolynomialVerificationStatus = PolynomialVerificationStatus.UNVERIFIED
    certificate_available: Literal[True] = True
    backend: Literal["sympy"] = "sympy"
    backend_version: str


class PolynomialCollisionOutput(ContractModel):
    claim_uri: ArtifactUri
    candidate_uri: ArtifactUri
    first_evaluation_uri: ArtifactUri
    second_evaluation_uri: ArtifactUri
    first_point: tuple[CanonicalRational, ...]
    second_point: tuple[CanonicalRational, ...]
    first_image: tuple[CanonicalRational, ...]
    second_image: tuple[CanonicalRational, ...]
    candidate_collision: bool
    witness_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    determinism: PolynomialDeterminism = PolynomialDeterminism.DETERMINISTIC
    verification: PolynomialVerificationStatus = PolynomialVerificationStatus.UNVERIFIED
    certificate_available: bool
    comparison_method: Literal["EXACT_EVALUATION_ARTIFACT_COMPARISON"] = (
        "EXACT_EVALUATION_ARTIFACT_COMPARISON"
    )

    @model_validator(mode="after")
    def witness_matches_collision(self) -> Self:
        if self.first_evaluation_uri == self.second_evaluation_uri:
            raise ValueError("collision output requires distinct evaluation artifacts")
        if not (
            len(self.first_point)
            == len(self.second_point)
            == len(self.first_image)
            == len(self.second_image)
        ):
            raise ValueError("collision output point and image dimensions must agree")
        expected_collision = (
            self.first_point != self.second_point
            and self.first_image == self.second_image
        )
        if self.candidate_collision != expected_collision:
            raise ValueError(
                "candidate collision status must match distinct points with equal images"
            )
        if self.candidate_collision != (self.witness_uri is not None):
            raise ValueError("only candidate collisions may carry a witness")
        if self.certificate_available != (
            self.witness_uri is not None and self.checker_id is not None
        ):
            raise ValueError("certificate availability requires witness and checker")
        return self


class PolynomialIdentityOutput(ContractModel):
    identical: bool | None
    conclusion: Conclusion
    left_uri: ArtifactUri
    right_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    determinism: PolynomialDeterminism = PolynomialDeterminism.DETERMINISTIC

    @model_validator(mode="after")
    def identity_matches_conclusion(self) -> Self:
        expected = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }
        if self.conclusion not in expected:
            raise ValueError(
                "polynomial identity conclusion must be TRUE, FALSE, or UNKNOWN"
            )
        if self.identical is not expected[self.conclusion]:
            raise ValueError("identical must preserve an unknown checker conclusion")
        return self


class RationalFunctionIdentityOutput(ContractModel):
    identical: bool | None
    conclusion: Conclusion
    left_uri: ArtifactUri
    right_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    determinism: PolynomialDeterminism = PolynomialDeterminism.DETERMINISTIC
    equality_semantics: Literal["QQ_FRACTION_FIELD_CROSS_MULTIPLICATION"] = (
        "QQ_FRACTION_FIELD_CROSS_MULTIPLICATION"
    )

    @model_validator(mode="after")
    def identity_matches_conclusion(self) -> Self:
        expected = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }
        if self.conclusion not in expected:
            raise ValueError(
                "rational-function identity conclusion must be TRUE, FALSE, or UNKNOWN"
            )
        if self.identical is not expected[self.conclusion]:
            raise ValueError("identical must preserve an unknown checker conclusion")
        return self


class PolynomialMapInverseVerifyOutput(ContractModel):
    inverse_verified: bool | None
    conclusion: Conclusion
    forward_map_uri: ArtifactUri
    inverse_map_uri: ArtifactUri
    residuals_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    inverse_after_forward_checker_records: tuple[ArtifactUri, ...]
    forward_after_inverse_checker_records: tuple[ArtifactUri, ...]
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    domain: Literal["QQ"] = "QQ"
    source_variables: tuple[PolynomialVariable, ...]
    target_variables: tuple[PolynomialVariable, ...]
    exactness: PolynomialExactness = PolynomialExactness.EXACT

    @model_validator(mode="after")
    def inverse_matches_conclusion(self) -> Self:
        expected = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }
        if self.conclusion not in expected:
            raise ValueError("inverse conclusion must be TRUE, FALSE, or UNKNOWN")
        if self.inverse_verified is not expected[self.conclusion]:
            raise ValueError("inverse_verified must preserve checker conclusion")
        return self


class PolynomialCollisionSearchOutput(ContractModel):
    found: bool
    map_uri: ArtifactUri
    examined_point_count: int = Field(ge=0, le=RATIONAL_SEARCH_GRID_LIMIT)
    grid_point_count: int = Field(ge=1, le=RATIONAL_SEARCH_GRID_LIMIT)
    first_point: tuple[CanonicalRational, ...] | None = None
    second_point: tuple[CanonicalRational, ...] | None = None
    common_image: tuple[CanonicalRational, ...] | None = None
    first_evaluation_uri: ArtifactUri | None = None
    second_evaluation_uri: ArtifactUri | None = None
    claim_uri: ArtifactUri | None = None
    witness_uri: ArtifactUri | None = None
    checker_id: CheckerUri | None = None
    stop_reason: PolynomialCollisionSearchStopReason
    verification: PolynomialVerificationStatus = PolynomialVerificationStatus.UNVERIFIED

    @model_validator(mode="after")
    def require_complete_candidate_bundle(self) -> Self:
        candidate_fields = (
            self.first_point,
            self.second_point,
            self.common_image,
            self.first_evaluation_uri,
            self.second_evaluation_uri,
            self.claim_uri,
            self.witness_uri,
        )
        if self.found and not all(value is not None for value in candidate_fields):
            raise ValueError("found status must match the complete collision bundle")
        if not self.found and any(value is not None for value in candidate_fields):
            raise ValueError("not-found results cannot carry a collision bundle")
        if self.examined_point_count > self.grid_point_count:
            raise ValueError("examined point count cannot exceed grid size")
        if self.found and (
            self.stop_reason is not PolynomialCollisionSearchStopReason.FIRST_COLLISION
        ):
            raise ValueError("found results must stop at the first collision")
        if not self.found and (
            self.stop_reason is not PolynomialCollisionSearchStopReason.GRID_EXHAUSTED
            or self.examined_point_count != self.grid_point_count
        ):
            raise ValueError("not-found results require an exhausted grid")
        return self


class PolynomialCollisionVerifyOutput(ContractModel):
    collision_verified: bool
    conclusion: Literal["FALSE", "UNKNOWN"]
    verification_input: InputValidation
    map_uri: ArtifactUri
    claim_uri: ArtifactUri
    witness_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    first_point: tuple[CanonicalRational, ...]
    second_point: tuple[CanonicalRational, ...]
    claimed_image: tuple[CanonicalRational, ...]
    exactness: PolynomialExactness = PolynomialExactness.EXACT
    coverage: Literal["NOT_APPLICABLE"] = "NOT_APPLICABLE"


class PolynomialKellerConditionVerifyOutput(ContractModel):
    keller_condition_verified: bool | None
    conclusion: Conclusion
    map_uri: ArtifactUri
    jacobian_uri: ArtifactUri
    claim_uri: ArtifactUri
    certificate_uri: ArtifactUri
    determinant: SparseRationalPolynomial
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    domain: Literal["QQ"] = "QQ"
    exactness: PolynomialExactness = PolynomialExactness.EXACT

    @model_validator(mode="after")
    def condition_matches_conclusion(self) -> Self:
        expected = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }
        if self.conclusion not in expected:
            raise ValueError(
                "Keller-condition conclusion must be TRUE, FALSE, or UNKNOWN"
            )
        if self.keller_condition_verified is not expected[self.conclusion]:
            raise ValueError(
                "keller_condition_verified must preserve the checker conclusion"
            )
        return self


class PolynomialMapInverseCollisionVerifyOutput(ContractModel):
    noninvertibility_verified: bool | None
    conclusion: Conclusion
    verification_input: InputValidation
    map_uri: ArtifactUri
    claim_uri: ArtifactUri
    witness_uri: ArtifactUri
    verification_record_uri: ArtifactUri | None = None
    checker_id: CheckerUri
    first_point: tuple[CanonicalRational, ...]
    second_point: tuple[CanonicalRational, ...]
    claimed_image: tuple[CanonicalRational, ...]
    domain: Literal["QQ"] = "QQ"
    exactness: PolynomialExactness = PolynomialExactness.EXACT

    @model_validator(mode="after")
    def noninvertibility_matches_conclusion(self) -> Self:
        expected = {
            Conclusion.TRUE: True,
            Conclusion.FALSE: False,
            Conclusion.UNKNOWN: None,
        }
        if self.conclusion not in expected:
            raise ValueError(
                "non-invertibility conclusion must be TRUE, FALSE, or UNKNOWN"
            )
        if self.noninvertibility_verified is not expected[self.conclusion]:
            raise ValueError(
                "noninvertibility_verified must preserve the checker conclusion"
            )
        return self
