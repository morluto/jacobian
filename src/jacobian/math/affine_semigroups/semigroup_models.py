from __future__ import annotations

from typing import Annotated

from pydantic import Field, WithJsonSchema
from pydantic.json_schema import JsonSchemaValue

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.math.affine_semigroups.semigroup import (
    MAX_AFFINE_GENERATORS,
    MAX_AFFINE_GRAPH_MOVES,
    AffineConfiguration,
    AffineFiber,
    AffineFiberGraph,
    AffineHilbertBasis,
    AffineMembershipResult,
    AffineSemigroupNormalization,
    PositiveAffineSemigroup,
    PositiveGradingResult,
)


class PositiveGradingRequest(StrictModel):
    configuration: AffineConfiguration


def _hilbert_configuration_json_schema() -> JsonSchemaValue:
    schema = AffineConfiguration.model_json_schema()
    properties = schema["properties"]
    properties["row_labels"].update(minItems=2, maxItems=2)
    properties["generator_labels"].update(minItems=1, maxItems=MAX_AFFINE_GENERATORS)
    properties["entries"].update(minItems=2, maxItems=2)
    properties["entries"]["items"].update(minItems=1, maxItems=MAX_AFFINE_GENERATORS)
    schema["description"] = (
        "Affine configuration with exactly two labelled rows and 1..10 generators."
    )
    return schema


class AffineSemigroupRequest(StrictModel):
    configuration: AffineConfiguration
    grading: tuple[CanonicalRational, ...]


class AffineFiberRequest(StrictModel):
    semigroup: PositiveAffineSemigroup
    target: tuple[ExactInteger, ...] = Field(
        description=(
            "Target on the ambient row axis. After canonicalization, the "
            "product of target-derived coefficient ranges must fit the "
            "50,000-state exact fiber envelope."
        )
    )


class AffineFactorizationCountRequest(AffineFiberRequest):
    """Count the finite fiber without returning its factorization vectors."""

    target: tuple[ExactInteger, ...] = Field(
        description=(
            "Exact target on the ambient row axis. The operation admits the "
            "univariate dynamic-program state/work envelope or the general "
            "positive-grading coefficient-box envelope before counting."
        )
    )


class AffineFactorizationRequest(StrictModel):
    """Evaluate one nonnegative coefficient vector in its semigroup parent."""

    semigroup: PositiveAffineSemigroup
    coordinates: tuple[Annotated[ExactInteger, Field(ge=0, max_length=32)], ...] = (
        Field(
            max_length=MAX_AFFINE_GENERATORS,
            description=(
                "Nonnegative exact integers on the generator axis. The owner "
                "admits the axis, sign, digit, arithmetic-work, and output bounds "
                "before evaluating the matrix product."
            ),
        )
    )


class AffineMembershipRequest(AffineFiberRequest):
    pass


class AffineFiberGraphRequest(AffineFiberRequest):
    moves: tuple[
        Annotated[tuple[ExactInteger, ...], Field(max_length=MAX_AFFINE_GENERATORS)],
        ...,
    ] = Field(
        description=(
            "At most 16 nonzero integer kernel vectors on the generator axis. "
            "Each move z must satisfy A z = 0 exactly."
        ),
        max_length=MAX_AFFINE_GRAPH_MOVES,
    )


class AffineHilbertBasisRequest(StrictModel):
    configuration: Annotated[
        AffineConfiguration, WithJsonSchema(_hilbert_configuration_json_schema())
    ] = Field(
        description=(
            "Generators of a full-dimensional pointed cone in Z^2. The exact "
            "primitive-ray determinant must be at most 1,000."
        )
    )


class AffineSemigroupNormalizationRequest(StrictModel):
    semigroup: PositiveAffineSemigroup


__all__ = [
    "AffineFactorizationCountRequest",
    "AffineFactorizationRequest",
    "AffineFiber",
    "AffineFiberGraph",
    "AffineFiberGraphRequest",
    "AffineFiberRequest",
    "AffineHilbertBasis",
    "AffineHilbertBasisRequest",
    "AffineMembershipRequest",
    "AffineMembershipResult",
    "AffineSemigroupNormalization",
    "AffineSemigroupNormalizationRequest",
    "AffineSemigroupRequest",
    "PositiveAffineSemigroup",
    "PositiveGradingRequest",
    "PositiveGradingResult",
]
