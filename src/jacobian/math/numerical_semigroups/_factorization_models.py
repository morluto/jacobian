"""Contracts owned by numerical-semigroup factorization kernels."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.numerical_semigroups._models import (
    _GENERAL_GENERATOR_ENVELOPE,
    MAX_GENERATORS,
    MAX_GRAPH_FACTORIZATIONS,
    MAX_MATERIALIZED_FACTORIZATIONS,
    _require_bounded_value,
    _require_canonical_minimal_axis,
    _require_materializable_factorizations,
    _require_minimal_generators,
    _validation_error,
)


class FactorizationComputeRequest(StrictModel):
    """Compute the complete factorization family Z(s) for one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; returned coordinates use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_materializable_factorizations(
            generators,
            _require_bounded_value(generators, self.value),
            MAX_MATERIALIZED_FACTORIZATIONS,
        )
        return self


class FactorizationComputeResult(StrictModel):
    """Complete factorization family Z(s) for one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        in_semigroup: bool,
        factorizations: tuple[tuple[int, ...], ...],
    ) -> Self:
        """Construct a complete family materialized by the admitted kernel."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            in_semigroup=in_semigroup,
            factorizations=factorizations,
        )

    @model_validator(mode="after")
    def require_exact_family(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error(
                "membership must agree with the factorization family"
            )
        return self


class FactorizationLengthsComputeRequest(StrictModel):
    """Compute the complete sorted length set of one element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; factorization lengths use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_positive_generators_and_bounded_value(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_bounded_value(generators, self.value)
        return self


class FactorizationLengthsComputeResult(StrictModel):
    """Sorted length set of one element."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    lengths: tuple[int, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        in_semigroup: bool,
        lengths: tuple[int, ...],
    ) -> Self:
        """Construct a length set derived by the admitted kernel."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            in_semigroup=in_semigroup,
            lengths=lengths,
        )

    @model_validator(mode="after")
    def require_length_set(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        if self.in_semigroup != bool(self.lengths):
            raise _validation_error(
                "membership must agree with the factorization lengths"
            )
        if self.lengths != tuple(sorted(set(self.lengths))):
            raise _validation_error(
                "lengths must be strictly increasing and duplicate-free"
            )
        if any(length < 0 for length in self.lengths):
            raise _validation_error("factorization lengths must be non-negative")
        return self


class FactorizationDistanceRequest(StrictModel):
    """Distance between two factorizations of the same element."""

    generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant, but both factorization coordinate tuples must use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger
    first: tuple[int, ...] = Field(min_length=1)
    second: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        value = _require_bounded_value(generators, self.value)
        if any(c < 0 for c in self.first) or any(c < 0 for c in self.second):
            raise _validation_error("factorization coordinates must be non-negative")
        if len(self.first) != len(generators) or len(self.second) != len(generators):
            raise _validation_error(
                "factorization coordinates must match the minimal generating system"
            )
        if any(
            sum(
                coefficient * generator
                for coefficient, generator in zip(
                    factorization, generators, strict=True
                )
            )
            != value
            for factorization in (self.first, self.second)
        ):
            raise _validation_error(
                "both factorizations must evaluate to the declared value"
            )
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
        min_length=1,
        max_length=MAX_GENERATORS,
        description=(
            "Positive generators with gcd 1. "
            + _GENERAL_GENERATOR_ENVELOPE
            + "The presentation may be reordered or redundant; graph vertices use its increasing minimal generator axis."
        ),
    )
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_complete_materialization(self) -> Self:
        generators = _require_minimal_generators(self.generators)
        _require_materializable_factorizations(
            generators,
            _require_bounded_value(generators, self.value),
            MAX_GRAPH_FACTORIZATIONS,
        )
        return self


class FactorizationGraphComputeResult(StrictModel):
    """Standard factorization graph with connected components."""

    value: CanonicalInteger
    minimal_generators: tuple[CanonicalInteger, ...] = Field(
        min_length=1, max_length=MAX_GENERATORS
    )
    in_semigroup: bool
    factorizations: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    connected_components: tuple[tuple[int, ...], ...]
    is_connected: bool

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CanonicalInteger,
        minimal_generators: tuple[CanonicalInteger, ...],
        in_semigroup: bool,
        factorizations: tuple[tuple[int, ...], ...],
        edges: tuple[tuple[int, int], ...],
        connected_components: tuple[tuple[int, ...], ...],
        is_connected: bool,
    ) -> Self:
        """Construct a graph derived from one admitted factorization family."""

        return cls.model_construct(
            value=value,
            minimal_generators=minimal_generators,
            in_semigroup=in_semigroup,
            factorizations=factorizations,
            edges=edges,
            connected_components=connected_components,
            is_connected=is_connected,
        )

    @model_validator(mode="after")
    def require_graph_partition(self) -> Self:
        generators = _require_canonical_minimal_axis(self.minimal_generators)
        _require_bounded_value(generators, self.value)
        vertex_count = len(self.factorizations)
        if self.in_semigroup != bool(self.factorizations):
            raise _validation_error("membership must agree with graph vertices")
        vertices = tuple(
            index for component in self.connected_components for index in component
        )
        if tuple(sorted(vertices)) != tuple(range(vertex_count)):
            raise _validation_error(
                "connected components must partition all graph vertices"
            )
        if self.is_connected != (len(self.connected_components) <= 1):
            raise _validation_error("is_connected must agree with connected components")
        if any(not 0 <= left < right < vertex_count for left, right in self.edges):
            raise _validation_error("graph edge has invalid vertex indices")
        return self


__all__ = [name for name in globals() if name.startswith("Factorization")]
