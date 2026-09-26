"""Typed contract for restricting a subsequential function's domain."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    SubsequentialTransducer,
    alphabet_parent_mismatch,
)
from jacobian.math.logic.languages.regular.values import DFA


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_state_transducer.{reason}", message)


class SubsequentialDomainRestrictionRequest(StrictModel):
    """Restrict a partial transduction to one regular input language."""

    transducer: SubsequentialTransducer
    domain_dfa: DFA

    @model_validator(mode="after")
    def require_identical_input_alphabet(self) -> Self:
        if self.transducer.input_alphabet_size != self.domain_dfa.alphabet_size:
            raise _validation_error(
                "domain_alphabet_size_mismatch",
                "domain DFA and transducer input alphabet sizes must match",
            )
        mismatch = alphabet_parent_mismatch(
            self.transducer.input_alphabet_id,
            self.transducer.input_alphabet,
            self.domain_dfa.alphabet_id,
            self.domain_dfa.alphabet,
        )
        if mismatch is not None:
            reason, message = mismatch
            suffix = reason.removeprefix("composition_")
            raise _validation_error(f"domain_{suffix}", message)
        return self
