"""Canonical ordered finite alphabets shared by automata and transducers."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BeforeValidator, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_FINITE_ALPHABET_SIZE = 32
MAX_FINITE_ALPHABET_SYMBOL_LENGTH = 64


def _reject_lone_surrogate_symbol(value: object) -> object:
    """Reject lone surrogates before Pydantic's incidental UTF-8 decoding.

    Canonical transport cannot encode non-scalar strings, so a carrier that
    accepted them would produce a mathematical value that fails its own
    serialization contract. This mirrors the word-domain ``Symbol`` check and
    makes the rejection a domain-owned validation error on native calls.
    """

    if isinstance(value, str) and any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        raise PydanticCustomError(
            "finite_state_transducer.alphabet_symbol_not_unicode_scalar",
            "alphabet symbols must contain only Unicode scalar values",
        )
    return value


class FiniteAlphabet(StrictModel):
    """An ordered, uniquely labeled finite symbol axis."""

    symbols: tuple[
        Annotated[
            str,
            BeforeValidator(_reject_lone_surrogate_symbol),
            Field(min_length=1, max_length=MAX_FINITE_ALPHABET_SYMBOL_LENGTH),
        ],
        ...,
    ] = Field(max_length=MAX_FINITE_ALPHABET_SIZE)

    @model_validator(mode="after")
    def require_unique_symbols(self) -> Self:
        if len(set(self.symbols)) != len(self.symbols):
            raise PydanticCustomError(
                "finite_state_transducer.alphabet_symbols_not_unique",
                "alphabet symbols must be unique",
            )
        return self


__all__ = ["MAX_FINITE_ALPHABET_SIZE", "FiniteAlphabet"]
