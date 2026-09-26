"""Typed values for finite root-system connection indices."""

from __future__ import annotations

from itertools import pairwise
from math import prod
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.groups.root_systems._models import MAX_RANK, CartanMatrix
from jacobian.math.matrices.values import IntegerMatrix


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"root_system.connection_index.{reason}", message)


class RootSystemConnectionIndexResult(StrictModel):
    """The finite quotient of the weight lattice by the root lattice."""

    cartan_matrix: CartanMatrix
    root_to_weight: IntegerMatrix
    invariant_factors: tuple[ExactInteger, ...] = Field(max_length=MAX_RANK)
    connection_index: ExactInteger = Field(ge=1)
    quotient: Literal["WEIGHT_LATTICE_MOD_ROOT_LATTICE"] = (
        "WEIGHT_LATTICE_MOD_ROOT_LATTICE"
    )

    @model_validator(mode="after")
    def require_canonical_quotient_shape(self) -> Self:
        rank = len(self.cartan_matrix)
        if not 1 <= rank <= MAX_RANK:
            raise _validation_error(
                "rank", "connection-index Cartan rank is outside its bound"
            )
        if (
            self.root_to_weight.row_count != rank
            or self.root_to_weight.column_count != rank
            or self.root_to_weight.entries != self.cartan_matrix.entries
        ):
            raise _validation_error(
                "inclusion",
                "root-to-weight inclusion must equal the retained Cartan matrix",
            )
        factors = tuple(self.invariant_factors)
        if any(type(factor) is not int or factor <= 1 for factor in factors) or any(
            right % left != 0 for left, right in pairwise(factors)
        ):
            raise _validation_error(
                "invariant_factors",
                "nontrivial quotient invariant factors must form a divisibility chain",
            )
        if prod(factors, start=1) != self.connection_index:
            raise _validation_error(
                "index",
                "connection index must equal the product of quotient invariant factors",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        cartan_matrix: CartanMatrix,
        root_to_weight: IntegerMatrix,
        invariant_factors: tuple[int, ...],
        connection_index: int,
    ) -> Self:
        return cls.model_construct(
            cartan_matrix=cartan_matrix,
            root_to_weight=root_to_weight,
            invariant_factors=invariant_factors,
            connection_index=connection_index,
        )


__all__ = ["RootSystemConnectionIndexResult"]
