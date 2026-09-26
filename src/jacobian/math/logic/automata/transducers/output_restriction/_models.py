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


def _output_edge_advances_language(
    transitions: dict[tuple[int, int], int],
    initial: int,
    label: tuple[int, ...],
    target: int,
) -> bool:
    state = initial
    for symbol in label:
        next_state = transitions.get((state, symbol))
        if next_state is None:
            return False
        state = next_state
    return state == target


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


def _require_transport_indices(result: RestrictRationalOutputResult) -> None:
    source_edges = result.source.edges
    restricted_edges = result.restricted.edges
    product_rows = result.product_states
    if len(product_rows) != result.restricted.state_count:
        raise _error(
            "product_state_transport_mismatch",
            "every restricted state must have one product-state transport",
        )
    if tuple(row.product_state for row in product_rows) != tuple(
        range(result.restricted.state_count)
    ):
        raise _error(
            "product_state_transport_order",
            "product-state transports must follow restricted state order",
        )
    if len(result.edge_sources) != len(restricted_edges):
        raise _error(
            "edge_transport_mismatch",
            "every restricted edge must have one source-edge transport",
        )
    if tuple(row.restricted_edge for row in result.edge_sources) != tuple(
        range(len(restricted_edges))
    ):
        raise _error(
            "edge_transport_order", "edge transports must follow restricted edge order"
        )
    if any(
        row.transducer_state >= result.source.state_count
        or row.language_state >= result.output_language.state_count
        for row in product_rows
    ):
        raise _error(
            "product_state_transport_out_of_range",
            "product-state transport leaves an embedded source axis",
        )
    if any(row.source_edge >= len(source_edges) for row in result.edge_sources):
        raise _error(
            "edge_transport_out_of_range",
            "edge transport leaves the embedded source relation",
        )
    if any(
        edge.source >= len(product_rows) or edge.target >= len(product_rows)
        for edge in restricted_edges
    ):
        raise _error(
            "edge_endpoint_out_of_range",
            "restricted edge endpoint leaves the product-state axis",
        )


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
        _require_transport_indices(self)
        source_edges = self.source.edges
        restricted_edges = self.restricted.edges
        product_rows = self.product_states
        transitions = {
            (row.source, row.symbol): row.target
            for row in self.output_language.transitions
        }
        for transport in self.edge_sources:
            restricted_edge = restricted_edges[transport.restricted_edge]
            source_edge = source_edges[transport.source_edge]
            source_state = product_rows[restricted_edge.source]
            target_state = product_rows[restricted_edge.target]
            if (
                source_state.transducer_state != source_edge.source
                or target_state.transducer_state != source_edge.target
                or restricted_edge.input_label != source_edge.input_label
                or restricted_edge.output_label != source_edge.output_label
                or not _output_edge_advances_language(
                    transitions,
                    source_state.language_state,
                    source_edge.output_label,
                    target_state.language_state,
                )
            ):
                raise _error(
                    "edge_transport_mismatch",
                    "edge transport must preserve its source edge and product endpoints",
                )
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
        for row in self.product_states:
            if (
                row.transducer_state >= self.source.state_count
                or row.language_state >= self.output_language.state_count
            ):
                raise _error(
                    "product_state_transport_out_of_range",
                    "product-state transport leaves an embedded source axis",
                )
        if any(row.source_edge >= len(self.source.edges) for row in self.edge_sources):
            raise _error(
                "edge_transport_out_of_range",
                "edge transport leaves the embedded source relation",
            )
        if (
            self.output_language.alphabet_size != self.source.output_alphabet_size
            or len(self.output_alphabet.symbols) != self.source.output_alphabet_size
            or (
                self.source.output_alphabet is not None
                and self.source.output_alphabet != self.output_alphabet
            )
        ):
            raise _error(
                "source_output_context_mismatch",
                "embedded language and alphabet must match the source output axis",
            )
        if self.restricted.input_alphabet_size != self.source.input_alphabet_size:
            raise _error(
                "input_axis_mismatch", "the input alphabet axis must be retained"
            )
        if self.restricted.output_alphabet_size != self.source.output_alphabet_size:
            raise _error(
                "output_axis_mismatch", "the output alphabet axis must be retained"
            )
        if (
            self.restricted.input_alphabet != self.source.input_alphabet
            or self.restricted.input_alphabet_id != self.source.input_alphabet_id
            or self.restricted.output_alphabet_id != self.source.output_alphabet_id
        ):
            raise _error(
                "alphabet_parent_mismatch",
                "restricted relation must retain both source alphabet parents",
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
