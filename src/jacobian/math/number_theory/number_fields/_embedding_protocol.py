"""Strict private protocol for the one-shot embedding worker."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, TypeAdapter, model_validator

from jacobian._models import StrictModel
from jacobian.math.number_theory.algebraic_numbers.complex import (
    RationalComplexIsolatingRectangle,
)
from jacobian.math.number_theory.algebraic_numbers.real import (
    RationalIsolatingInterval,
)
from jacobian.math.number_theory.number_fields._embedding_limits import (
    MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_EMBEDDING_DEGREE,
    NumberFieldInteger,
    SimpleNumberFieldPresentation,
)


class NumberFieldEmbeddingWorkerRequest(StrictModel):
    """Retained source plus the owner-admitted static isolation plan."""

    field: SimpleNumberFieldPresentation
    root_isolation_bits: StrictInt = Field(
        ge=5,
        le=MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS,
    )
    evidence_grid_bits: StrictInt = Field(
        ge=1,
        le=MAX_NUMBER_FIELD_ROOT_REFINEMENT_BITS - 4,
    )

    @model_validator(mode="after")
    def bind_admitted_grids(self) -> Self:
        if self.root_isolation_bits != self.evidence_grid_bits + 4:
            raise ValueError(
                "worker isolation grid must refine the public evidence grid by 4 bits"
            )
        return self


class NumberFieldEmbeddingWorkerComplete(StrictModel):
    kind: Literal["complete"]
    real_intervals: tuple[RationalIsolatingInterval, ...] = Field(
        max_length=MAX_NUMBER_FIELD_EMBEDDING_DEGREE
    )
    negative_complex_rectangles: tuple[RationalComplexIsolatingRectangle, ...] = Field(
        max_length=MAX_NUMBER_FIELD_EMBEDDING_DEGREE // 2
    )
    defining_polynomial_discriminant: NumberFieldInteger

    @model_validator(mode="after")
    def bind_degree_and_half_plane(self) -> Self:
        degree = len(self.real_intervals) + 2 * len(self.negative_complex_rectangles)
        if not 1 <= degree <= MAX_NUMBER_FIELD_EMBEDDING_DEGREE:
            raise ValueError("worker root evidence must describe a bounded degree")
        if any(
            rectangle.imaginary_upper.as_fraction() >= 0
            for rectangle in self.negative_complex_rectangles
        ):
            raise ValueError("worker complex evidence must select negative roots")
        return self


class NumberFieldEmbeddingWorkerInvalid(StrictModel):
    kind: Literal["invalid"]
    reason: Literal["not_irreducible"]


class NumberFieldEmbeddingWorkerRejected(StrictModel):
    kind: Literal["rejected"]
    reason: Literal["pair_ordering_precision_bound"]


NumberFieldEmbeddingWorkerResponse = Annotated[
    NumberFieldEmbeddingWorkerComplete
    | NumberFieldEmbeddingWorkerInvalid
    | NumberFieldEmbeddingWorkerRejected,
    Field(discriminator="kind"),
]

NUMBER_FIELD_EMBEDDING_WORKER_RESPONSE_ADAPTER: TypeAdapter[
    NumberFieldEmbeddingWorkerResponse
] = TypeAdapter(NumberFieldEmbeddingWorkerResponse)


__all__ = [
    "NUMBER_FIELD_EMBEDDING_WORKER_RESPONSE_ADAPTER",
    "NumberFieldEmbeddingWorkerComplete",
    "NumberFieldEmbeddingWorkerInvalid",
    "NumberFieldEmbeddingWorkerRejected",
    "NumberFieldEmbeddingWorkerRequest",
    "NumberFieldEmbeddingWorkerResponse",
]
