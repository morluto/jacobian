"""Wire values for the sparse free-algebra commutator operation."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.free_algebras._models import FreeAlgebraPolynomial

MAX_COMMUTATOR_OUTPUT_WORD_CELLS = 262_144
MAX_COMMUTATOR_WORK = 20_000_000


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"free_algebra.commutator.{reason}", message)


class FreeAlgebraCommutatorRequest(StrictModel):
    """Two sparse polynomials over the same ordered generator alphabet."""

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial

    @model_validator(mode="after")
    def require_common_alphabet(self) -> Self:
        if self.left.alphabet != self.right.alphabet:
            raise _error(
                "alphabet_mismatch",
                "both polynomials must use the same ordered generator alphabet",
            )
        return self


class FreeAlgebraCommutatorResult(StrictModel):
    """The exact commutator ``left*right - right*left``."""

    left: FreeAlgebraPolynomial
    right: FreeAlgebraPolynomial
    commutator: FreeAlgebraPolynomial

    @model_validator(mode="after")
    def require_common_alphabet(self) -> Self:
        if not (self.left.alphabet == self.right.alphabet == self.commutator.alphabet):
            raise _error(
                "result_alphabet_mismatch",
                "the commutator must retain its operands' ordered alphabet",
            )
        return self


__all__ = [
    "MAX_COMMUTATOR_OUTPUT_WORD_CELLS",
    "MAX_COMMUTATOR_WORK",
    "FreeAlgebraCommutatorRequest",
    "FreeAlgebraCommutatorResult",
]
