"""Wire contracts for regular-language restriction of transducer outputs."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import DFA

MAX_RESTRICT_OUTPUT_LABEL_CELLS = 250_000
MAX_RESTRICT_OUTPUT_RESULT_EDGES = 4096
MAX_RESTRICT_OUTPUT_WORK = 20_000_000


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"rational_transducer.restrict_output.{reason}", message)


class RestrictRationalOutputRequest(StrictModel):
    """Restrict a rational relation by a total DFA on its output tape.

    ``output_alphabet`` binds the DFA's zero-based symbol axis to the output
    symbols of ``transducer``. When the relation already retains an alphabet
    context, it must equal this context exactly.
    """

    transducer: RationalTransducer
    output_language: DFA
    output_alphabet: FiniteAlphabet

    @model_validator(mode="after")
    def require_matching_output_axis(self) -> Self:
        if self.output_language.alphabet_size != self.transducer.output_alphabet_size:
            raise _error(
                "alphabet_size_mismatch",
                "the DFA alphabet size must equal the transducer output alphabet size",
            )
        if len(self.output_alphabet.symbols) != self.transducer.output_alphabet_size:
            raise _error(
                "alphabet_context_size_mismatch",
                "the output alphabet context must match the transducer output axis",
            )
        if (
            self.transducer.output_alphabet is not None
            and self.transducer.output_alphabet != self.output_alphabet
        ):
            raise _error(
                "alphabet_context_mismatch",
                "the supplied output alphabet must match the relation's retained context",
            )
        return self


class RestrictOutputProductState(StrictModel):
    """Transport from an output-restriction product state to its factors."""

    product_state: int = Field(ge=0, le=63)
    transducer_state: int = Field(ge=0, le=63)
    language_state: int = Field(ge=0, le=63)


class RestrictOutputEdgeSource(StrictModel):
    """Source edge coordinate for one edge in the restricted relation."""

    restricted_edge: int = Field(ge=0, le=MAX_RESTRICT_OUTPUT_RESULT_EDGES - 1)
    source_edge: int = Field(ge=0, le=4095)


class RestrictRationalOutputResult(StrictModel):
    """Exact product relation ``R ∩ (A* x L)`` with source transports."""

    source: RationalTransducer
    output_language: DFA
    output_alphabet: FiniteAlphabet
    restricted: RationalTransducer
    product_states: tuple[RestrictOutputProductState, ...] = Field(max_length=64)
    edge_sources: tuple[RestrictOutputEdgeSource, ...] = Field(
        max_length=MAX_RESTRICT_OUTPUT_RESULT_EDGES
    )

    @model_validator(mode="after")
    def require_canonical_transports(self) -> Self:
        if len(self.product_states) != self.restricted.state_count:
            raise _error(
                "product_state_transport_mismatch",
                "every restricted state must have one product-state transport",
            )
        if tuple(row.product_state for row in self.product_states) != tuple(
            range(self.restricted.state_count)
        ):
            raise _error(
                "product_state_transport_order",
                "product-state transports must follow restricted state order",
            )
        if len(self.edge_sources) != len(self.restricted.edges):
            raise _error(
                "edge_transport_mismatch",
                "every restricted edge must have one source-edge transport",
            )
        if tuple(row.restricted_edge for row in self.edge_sources) != tuple(
            range(len(self.restricted.edges))
        ):
            raise _error(
                "edge_transport_order",
                "edge transports must follow restricted edge order",
            )
        if self.restricted.input_alphabet_size != self.source.input_alphabet_size:
            raise _error(
                "input_axis_mismatch", "the input alphabet axis must be retained"
            )
        if self.restricted.output_alphabet_size != self.source.output_alphabet_size:
            raise _error(
                "output_axis_mismatch", "the output alphabet axis must be retained"
            )
        if self.restricted.output_alphabet != self.output_alphabet:
            raise _error(
                "output_context_mismatch",
                "the restricted relation must retain its output alphabet",
            )
        return self


__all__ = [
    "MAX_RESTRICT_OUTPUT_LABEL_CELLS",
    "MAX_RESTRICT_OUTPUT_RESULT_EDGES",
    "MAX_RESTRICT_OUTPUT_WORK",
    "RestrictOutputEdgeSource",
    "RestrictOutputProductState",
    "RestrictRationalOutputRequest",
    "RestrictRationalOutputResult",
]
