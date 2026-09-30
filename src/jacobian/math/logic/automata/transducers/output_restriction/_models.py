"""Wire contracts for regular-language restriction of transducer outputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self, cast

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_ALPHABET,
    MAX_FST_ALPHABET_ID_LENGTH,
    MAX_FST_ALPHABET_SYMBOL_LENGTH,
    MAX_FST_EDGES,
    MAX_FST_STATES,
    MAX_FST_WORD_LENGTH,
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_DFA_ALPHABET,
    MAX_DFA_ALPHABET_ID_LENGTH,
    MAX_DFA_STATES,
    MAX_DFA_TRANSITIONS,
    DFATransition,
)

MAX_RESTRICT_OUTPUT_LABEL_CELLS = 250_000
MAX_RESTRICT_OUTPUT_RESULT_EDGES = 4096
MAX_RESTRICT_OUTPUT_WORK = 20_000_000


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"rational_transducer.restrict_output.{reason}", message)


def _require_bounded_relation_labels(relation: object) -> None:
    """Check both retained label axes before parsing, copying, or replay."""
    native = isinstance(relation, RationalTransducer)
    if native:
        fields = vars(relation)
    elif type(relation) is dict:
        fields = relation
    else:
        raise _error("edge_label_invalid", "relation label fields are malformed")
    edges = fields.get("edges")
    input_size = fields.get("input_alphabet_size")
    output_size = fields.get("output_alphabet_size")
    if (
        type(input_size) is not int
        or type(output_size) is not int
        or not 1 <= input_size <= MAX_FST_ALPHABET
        or not 1 <= output_size <= MAX_FST_ALPHABET
        or type(edges) not in ((tuple,) if native else (tuple, list))
    ):
        raise _error("edge_label_invalid", "relation label axes are malformed")
    edges = cast(tuple[object, ...] | list[object], edges)
    if len(edges) > MAX_FST_EDGES:
        raise _error("edge_label_invalid", "relation edge family exceeds its carrier")
    label_cells = 0
    for index, edge in enumerate(edges):
        if index % 32 == 0:
            request_checkpoint("during output-restriction label admission")
        if type(edge) is RationalEdge:
            edge_fields = vars(edge)
            containers: tuple[type[object], ...] = (tuple,)
        elif not native and type(edge) is dict:
            edge_fields = edge
            containers = (tuple, list)
        else:
            raise _error("edge_label_invalid", "relation label row is malformed")
        for name, axis_size in (
            ("input_label", input_size),
            ("output_label", output_size),
        ):
            label = edge_fields.get(name, ())
            if type(label) not in containers or len(label) > MAX_FST_WORD_LENGTH:
                raise _error(
                    "edge_label_invalid", "edge label exceeds its canonical carrier"
                )
            label_cells += len(label)
            if label_cells > MAX_RESTRICT_OUTPUT_LABEL_CELLS:
                raise _error(
                    "edge_label_invalid",
                    "retained labels exceed the admitted cell envelope",
                )
            if any(
                type(symbol) is not int or not 0 <= symbol < axis_size
                for symbol in label
            ):
                raise _error(
                    "edge_label_invalid",
                    "edge label must contain integer symbols on its axis",
                )

        if not edge_fields.get("input_label", ()) and not edge_fields.get(
            "output_label", ()
        ):
            raise _error("edge_label_invalid", "an edge must have a nonempty label")


def _raw_fields(
    value: object, model: type[StrictModel], *, allow_mapping: bool = True
) -> tuple[dict[str, object], bool]:
    native = type(value) is model
    if native:
        fields = vars(value)
    elif allow_mapping and type(value) is dict:
        fields = value
    else:
        raise _error("result_shape_invalid", "retained value has a noncanonical type")
    declared = model.model_fields
    if len(fields) > len(declared) or any(
        type(name) is not str or name not in declared for name in fields
    ):
        raise _error("result_shape_invalid", "retained value has unknown fields")
    if any(
        field.is_required() and name not in fields for name, field in declared.items()
    ):
        raise _error(
            "result_shape_invalid", "retained value is missing required fields"
        )
    return cast(dict[str, object], fields), native


def _raw_int(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _error("result_shape_invalid", "retained integer leaves its bounded axis")
    return value


def _raw_sequence(
    value: object, maximum: int, native: bool
) -> tuple[object, ...] | list[object]:
    if type(value) not in ((tuple,) if native else (tuple, list)):
        raise _error(
            "result_shape_invalid", "retained axis must be a canonical sequence"
        )
    sequence = cast(tuple[object, ...] | list[object], value)
    if len(sequence) > maximum:
        raise _error("result_shape_invalid", "retained axis exceeds its carrier bound")
    return sequence


def _raw_text(value: object, maximum: int) -> str:
    if type(value) is not str or len(value) > maximum:
        raise _error("result_shape_invalid", "retained text exceeds its carrier bound")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _error("result_shape_invalid", "retained text must be UTF-8") from exc
    return value


def _preflight_alphabet(value: object, *, allow_mapping: bool) -> None:
    fields, native = _raw_fields(value, FiniteAlphabet, allow_mapping=allow_mapping)
    symbols = _raw_sequence(fields["symbols"], MAX_FST_ALPHABET, native)
    if not symbols:
        raise _error("result_shape_invalid", "retained alphabet must be nonempty")
    for symbol in symbols:
        if not _raw_text(symbol, MAX_FST_ALPHABET_SYMBOL_LENGTH):
            raise _error("result_shape_invalid", "alphabet symbols must be nonempty")


def _preflight_state_axis(value: object, state_count: int, native: bool) -> None:
    for state in _raw_sequence(value, MAX_FST_STATES, native):
        _raw_int(state, 0, state_count - 1)


def _preflight_relation(value: object, *, allow_mapping: bool) -> None:
    fields, native = _raw_fields(value, RationalTransducer, allow_mapping=allow_mapping)
    _require_bounded_relation_labels(value)
    state_count = _raw_int(fields["state_count"], 1, MAX_FST_STATES)
    for name in ("initial_states", "accepting_states"):
        _preflight_state_axis(fields[name], state_count, native)
    for name in ("input_alphabet_id", "output_alphabet_id"):
        if fields.get(name) is not None:
            _raw_text(fields[name], MAX_FST_ALPHABET_ID_LENGTH)
    for name in ("input_alphabet", "output_alphabet"):
        if fields.get(name) is not None:
            _preflight_alphabet(fields[name], allow_mapping=not native)
    for edge in _raw_sequence(fields["edges"], MAX_FST_EDGES, native):
        row, _ = _raw_fields(edge, RationalEdge, allow_mapping=not native)
        _raw_int(row["source"], 0, state_count - 1)
        _raw_int(row["target"], 0, state_count - 1)


def _preflight_language(value: object, *, allow_mapping: bool) -> None:
    fields, native = _raw_fields(value, DFA, allow_mapping=allow_mapping)
    state_count = _raw_int(fields["state_count"], 1, MAX_DFA_STATES)
    alphabet_size = _raw_int(fields["alphabet_size"], 0, MAX_DFA_ALPHABET)
    _raw_int(fields["initial_state"], 0, state_count - 1)
    _preflight_state_axis(fields["accepting_states"], state_count, native)
    if fields.get("alphabet_id") is not None:
        _raw_text(fields["alphabet_id"], MAX_DFA_ALPHABET_ID_LENGTH)
    if fields.get("alphabet") is not None:
        _preflight_alphabet(fields["alphabet"], allow_mapping=not native)
    for index, transition in enumerate(
        _raw_sequence(fields["transitions"], MAX_DFA_TRANSITIONS, native)
    ):
        if index % 256 == 0:
            request_checkpoint("during output-restriction language admission")
        row, _ = _raw_fields(transition, DFATransition, allow_mapping=not native)
        _raw_int(row["source"], 0, state_count - 1)
        _raw_int(row["target"], 0, state_count - 1)
        _raw_int(row["symbol"], 0, alphabet_size - 1)


def _preflight_retained_result(data: object) -> None:
    """Bound the complete retained shape before recursive canonicalization."""
    fields, native = _raw_fields(data, RestrictRationalOutputResult)
    for name in ("source", "restricted"):
        _preflight_relation(fields[name], allow_mapping=not native)
    _preflight_language(fields["output_language"], allow_mapping=not native)
    _preflight_alphabet(fields["output_alphabet"], allow_mapping=not native)
    for name, model, maximum in (
        ("product_states", RestrictOutputProductState, MAX_FST_STATES),
        ("edge_sources", RestrictOutputEdgeSource, MAX_RESTRICT_OUTPUT_RESULT_EDGES),
    ):
        rows = _raw_sequence(fields[name], maximum, native)
        for index, item in enumerate(rows):
            if index % 256 == 0:
                request_checkpoint("during output-restriction transport admission")
            row, _ = _raw_fields(item, model, allow_mapping=not native)
            for entry in row.values():
                _raw_int(entry, 0, maximum - 1)


def _output_edge_advances_language(
    transitions: dict[tuple[int, int], int],
    initial: int,
    label: tuple[int, ...],
    target: int,
) -> bool:
    state = initial
    for index, symbol in enumerate(label):
        if index % 256 == 0:
            request_checkpoint("during output-restriction label replay")
        next_state = transitions.get((state, symbol))
        if next_state is None:
            return False
        state = next_state
    return state == target


def _admitted_edge_transports(
    edge_sources: tuple[RestrictOutputEdgeSource, ...],
    restricted_edges: tuple[RationalEdge, ...],
) -> Iterator[RestrictOutputEdgeSource]:
    """Charge and yield edge transports in bounded, cancellable steps.

    Confirming the edge transport replays the output language once per edge,
    which is admitted-scale work. Charging it against the same envelope the
    kernel uses, and yielding it in checkpointed steps, means a decode inside a
    tight or cancelled request is accounted for and interruptible instead of
    running to completion unchecked.
    """
    replay_cells = sum(
        len(restricted_edges[transport.restricted_edge].output_label)
        for transport in edge_sources
    )
    if replay_cells > MAX_RESTRICT_OUTPUT_WORK:
        raise _error(
            "edge_transport_replay_exceeded",
            "edge-transport replay exceeds the admitted work envelope",
        )
    for index, transport in enumerate(edge_sources):
        if index % 256 == 0:
            request_checkpoint("during output-restriction edge transport")
        yield transport


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

    @model_validator(mode="before")
    @classmethod
    def require_bounded_retained_result(cls, data: object) -> object:
        _preflight_retained_result(data)
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_canonical_transports(self) -> Self:
        # Existing native result instances may bypass before-validation.
        _preflight_retained_result(self)
        _require_transport_indices(self)
        source_edges = self.source.edges
        restricted_edges = self.restricted.edges
        product_rows = self.product_states
        transitions = {
            (row.source, row.symbol): row.target
            for row in self.output_language.transitions
        }
        for transport in _admitted_edge_transports(self.edge_sources, restricted_edges):
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
