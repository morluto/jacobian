"""Canonical ordered finite alphabets shared by automata and transducers."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_FINITE_ALPHABET_SIZE = 32
MAX_FINITE_ALPHABET_SYMBOL_LENGTH = 64


class FiniteAlphabet(StrictModel):
    """An ordered, uniquely labeled finite symbol axis."""

    symbols: tuple[
        Annotated[
            str,
            Field(min_length=1, max_length=MAX_FINITE_ALPHABET_SYMBOL_LENGTH),
        ],
        ...,
    ] = Field(min_length=1, max_length=MAX_FINITE_ALPHABET_SIZE)

    @model_validator(mode="after")
    def require_unique_symbols(self) -> Self:
        if len(set(self.symbols)) != len(self.symbols):
            raise PydanticCustomError(
                "finite_state_transducer.alphabet_symbols_not_unique",
                "alphabet symbols must be unique",
            )
        return self


__all__ = ["MAX_FINITE_ALPHABET_SIZE", "FiniteAlphabet"]
