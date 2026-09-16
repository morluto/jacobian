"""Typed wire contracts for exact tropical scalar addition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.tropical.values import TropicalScalar, TropicalSemiring


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"tropical.{reason}", message)


class ScalarAddRequest(StrictModel):
    """Add two tropical scalars on one explicit semiring.

    Both operands must carry the request semiring as their parent; scalars
    from the dual convention are rejected before arithmetic. Finite-scalar
    digits are preflighted in ``tropical_scalar_add`` before comparison.
    """

    semiring: TropicalSemiring = Field(
        description="The explicit semiring owning tropical-plus on both operands."
    )
    left: TropicalScalar = Field(description="First tropical-plus operand.")
    right: TropicalScalar = Field(description="Second tropical-plus operand.")

    @model_validator(mode="after")
    def require_shared_semiring(self) -> Self:
        if self.left.semiring != self.semiring or self.right.semiring != self.semiring:
            raise _validation_error(
                "semiring_mismatch",
                "tropical-plus operands must carry the request semiring",
            )
        return self


AddBranch = Literal["LEFT", "RIGHT", "TIE"]
InfinityCase = Literal["NONE", "LEFT_INFINITE", "RIGHT_INFINITE", "BOTH_INFINITE"]


class ScalarAddResult(StrictModel):
    """A source-bound tropical sum with its finite/infinity branch recorded.

    ``branch`` names the winning operand (``TIE`` when both operands are
    equal, in which case the result equals both). ``infinity_case`` records
    which operands were infinite, so identity absorption is visible without
    re-reading the sources.
    """

    semiring: TropicalSemiring
    left: TropicalScalar
    right: TropicalScalar
    result: TropicalScalar
    branch: AddBranch
    infinity_case: InfinityCase

    @model_validator(mode="after")
    def require_branch_shape(self) -> Self:
        if self.result.semiring != self.semiring:
            raise _validation_error(
                "result_semiring_mismatch",
                "the tropical sum must carry the request semiring",
            )
        left_infinite = self.left.kind != "FINITE"
        right_infinite = self.right.kind != "FINITE"
        expected: InfinityCase = (
            "BOTH_INFINITE"
            if left_infinite and right_infinite
            else "LEFT_INFINITE"
            if left_infinite
            else "RIGHT_INFINITE"
            if right_infinite
            else "NONE"
        )
        if self.infinity_case != expected:
            raise _validation_error(
                "infinity_case_mismatch",
                "infinity_case must match the operand variants",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: ScalarAddRequest,
        *,
        result: TropicalScalar,
        branch: AddBranch,
        infinity_case: InfinityCase,
    ) -> Self:
        """Build one result after the admitted kernel established its branch."""

        return cls.model_construct(
            semiring=request.semiring,
            left=request.left,
            right=request.right,
            result=result,
            branch=branch,
            infinity_case=infinity_case,
        )


__all__ = ["AddBranch", "InfinityCase", "ScalarAddRequest", "ScalarAddResult"]
