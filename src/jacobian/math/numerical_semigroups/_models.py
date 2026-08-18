"""Typed wire contracts for numerical semigroup operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

MAX_GENERATORS = 20
MAX_GENERATOR = 500
MAX_ELEMENT = 10_000
MAX_FACTOR_SEARCH = 100


def _require_positive_bounded_generators(generators: tuple[str, ...]) -> None:
    values: list[int] = []
    for generator in generators:
        value = parse_canonical_integer(generator)
        if value <= 0:
            raise ValueError("generators must be positive integers")
        if value > MAX_GENERATOR:
            raise ValueError(f"generators must be at most {MAX_GENERATOR}")
        values.append(value)
    gcd = values[0]
    for value in values[1:]:
        while value:
            gcd, value = value, gcd % value
    if gcd != 1:
        raise ValueError(f"generators must have gcd 1, got gcd {gcd}")


class NumericalSemigroupRequest(StrictModel):
    """A numerical semigroup defined by a finite set of positive generators."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryRequest(StrictModel):
    """Compute the full summary of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class NumericalSemigroupSummaryResult(StrictModel):
    """Summary of a numerical semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    multiplicity: CanonicalInteger
    embedding_dimension: int = Field(ge=1)
    frobenius_number: str
    conductor: str
    genus: int = Field(ge=0)
    gaps: tuple[CanonicalInteger, ...]


class SemigroupMembershipRequest(StrictModel):
    """Check membership of an integer in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"membership value must be at most {MAX_ELEMENT}")
        return self


class SemigroupMembershipResult(StrictModel):
    """Whether the value is in the semigroup."""

    value: CanonicalInteger
    in_semigroup: bool

# ---------------------------------------------------------------------------
# Extended operations: factorization, elasticity, catenary degree, etc.
# ---------------------------------------------------------------------------


class FactorizationComputeRequest(StrictModel):
    """Compute the complete factorization family Z(s) for one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class FactorizationComputeResult(StrictModel):
    """Complete factorization family Z(s) for one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    factorizations: tuple[tuple[int, ...], ...]


class FactorizationLengthsComputeRequest(StrictModel):
    """Compute the complete sorted length set of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class FactorizationLengthsComputeResult(StrictModel):
    """Sorted length set of one element."""

    value: CanonicalInteger
    lengths: tuple[int, ...]


class FactorizationDistanceRequest(StrictModel):
    """Distance between two factorizations of the same element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger
    first: tuple[int, ...] = Field(min_length=1)
    second: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        if len(self.first) != len(self.second):
            raise ValueError("factorizations must have equal coordinate length")
        if any(c < 0 for c in self.first) or any(c < 0 for c in self.second):
            raise ValueError("factorization coordinates must be non-negative")
        return self


class FactorizationDistanceResult(StrictModel):
    """Distance between two factorizations."""

    value: CanonicalInteger
    distance: int = Field(ge=0)
    first_length: int = Field(ge=0)
    second_length: int = Field(ge=0)


class FactorizationGraphComputeRequest(StrictModel):
    """Compute the standard factorization graph of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class FactorizationGraphComputeResult(StrictModel):
    """Standard factorization graph with connected components."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...]
    factorizations: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    connected_components: tuple[tuple[int, ...], ...]
    is_connected: bool


class ElementDeltaSetRequest(StrictModel):
    """Delta set of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class ElementDeltaSetResult(StrictModel):
    """Delta set of one element."""

    value: CanonicalInteger
    delta_set: tuple[int, ...]


class ElementElasticityRequest(StrictModel):
    """Elasticity of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class ElementElasticityResult(StrictModel):
    """Elasticity of one element."""

    value: CanonicalInteger
    elasticity: str


class ElementCatenaryDegreeRequest(StrictModel):
    """Catenary degree of one element in a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        if parse_canonical_integer(self.value) > MAX_ELEMENT:
            raise ValueError(f"value must be at most {MAX_ELEMENT}")
        return self


class ElementCatenaryDegreeResult(StrictModel):
    """Catenary degree of one element."""

    value: CanonicalInteger
    catenary_degree: int = Field(ge=0)


class BettiElementsRequest(StrictModel):
    """Betti elements of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class BettiElementsResult(StrictModel):
    """Betti elements of a semigroup."""

    betti_elements: tuple[CanonicalInteger, ...]


class MinimalPresentationRequest(StrictModel):
    """One minimal presentation of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class MinimalPresentationRelation(StrictModel):
    """One relation (pair of distinct factorizations) in a presentation."""

    first: tuple[int, ...]
    second: tuple[int, ...]


class MinimalPresentationResult(StrictModel):
    """One minimal presentation of the semigroup."""

    minimal_generators: tuple[CanonicalInteger, ...]
    betti_elements: tuple[CanonicalInteger, ...]
    relations: tuple[MinimalPresentationRelation, ...]


class PresentationBinomialsRequest(StrictModel):
    """Convert a minimal presentation to sparse binomial form."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    relations: tuple[MinimalPresentationRelation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class PresentationBinomial(StrictModel):
    """One sparse binomial (aX - bX) arising from a presentation relation."""

    left_coefficient: str
    left_exponents: tuple[int, ...]
    right_coefficient: str
    right_exponents: tuple[int, ...]


class PresentationBinomialsResult(StrictModel):
    """Presentation converted to sparse binomials."""

    minimal_generators: tuple[CanonicalInteger, ...]
    binomials: tuple[PresentationBinomial, ...]


class DeltaSetRequest(StrictModel):
    """Global delta set of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class DeltaSetResult(StrictModel):
    """Global delta set of the semigroup."""

    delta_set: tuple[int, ...]


class ElasticityRequest(StrictModel):
    """Global elasticity of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class ElasticityResult(StrictModel):
    """Global elasticity of the semigroup."""

    elasticity: str


class CatenaryDegreeRequest(StrictModel):
    """Global catenary degree of a numerical semigroup."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )

    @model_validator(mode="after")
    def require_positive_generators(self) -> Self:
        _require_positive_bounded_generators(self.generators)
        return self


class CatenaryDegreeResult(StrictModel):
    """Global catenary degree of the semigroup."""

    catenary_degree: int = Field(ge=0)


__all__ = [
    "NumericalSemigroupSummaryRequest",
    "NumericalSemigroupSummaryResult",
    "SemigroupMembershipRequest",
    "SemigroupMembershipResult",
    # Factorization family
    "FactorizationComputeRequest",
    "FactorizationComputeResult",
    # Factorization lengths
    "FactorizationLengthsComputeRequest",
    "FactorizationLengthsComputeResult",
    # Distance
    "FactorizationDistanceRequest",
    "FactorizationDistanceResult",
    # Factorization graph
    "FactorizationGraphComputeRequest",
    "FactorizationGraphComputeResult",
    # Element-level
    "ElementDeltaSetRequest",
    "ElementDeltaSetResult",
    "ElementElasticityRequest",
    "ElementElasticityResult",
    "ElementCatenaryDegreeRequest",
    "ElementCatenaryDegreeResult",
    # Betti
    "BettiElementsRequest",
    "BettiElementsResult",
    # Minimal presentation
    "MinimalPresentationRequest",
    "MinimalPresentationRelation",
    "MinimalPresentationResult",
    "PresentationBinomialsRequest",
    "PresentationBinomial",
    "PresentationBinomialsResult",
    # Global
    "DeltaSetRequest",
    "DeltaSetResult",
    "ElasticityRequest",
    "ElasticityResult",
    "CatenaryDegreeRequest",
    "CatenaryDegreeResult",
]

