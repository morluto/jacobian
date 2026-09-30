"""Bound constructed labels before decoding and checkpoint admitted replay."""

from __future__ import annotations

import time
from collections.abc import Iterator
from threading import Event

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
    request_execution,
)
from jacobian.math.logic.automata.transducers.output_restriction import _models
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    RestrictRationalOutputResult,
)
from jacobian.math.logic.automata.transducers.output_restriction.operations import (
    restrict_rational_output,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


def _result() -> RestrictRationalOutputResult:
    alphabet = FiniteAlphabet(symbols=("a",))
    source = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=1,
        initial_states=(0,),
        accepting_states=(0,),
        edges=(
            RationalEdge(
                source=0, target=0, input_label=(0,) * 512, output_label=(0,) * 512
            ),
        ),
    )
    language = DFA(
        state_count=1,
        alphabet_size=1,
        initial_state=0,
        accepting_states=(0,),
        transitions=(DFATransition(source=0, symbol=0, target=0),),
    )
    return restrict_rational_output(source, language, alphabet)


def _decode(
    result: RestrictRationalOutputResult, representation: str
) -> RestrictRationalOutputResult:
    if representation == "json":
        return RestrictRationalOutputResult.model_validate_json(
            result.model_dump_json()
        )
    if representation == "native-mapping":
        return RestrictRationalOutputResult.model_validate(dict(result))
    return RestrictRationalOutputResult.model_validate(result)


@pytest.mark.parametrize("relation_name", ("source", "restricted"))
@pytest.mark.parametrize("label_name", ("input_label", "output_label"))
@pytest.mark.parametrize("representation", ("native", "native-mapping"))
@pytest.mark.parametrize(
    "label",
    ((0,) * 513, [0], {"symbols": [0]}, None, (False,), ([0],), (1,), (-1,)),
    ids=(
        "overlong",
        "list",
        "mapping",
        "missing",
        "bool",
        "nested",
        "outside-axis",
        "negative",
    ),
)
def test_malformed_retained_label_is_rejected_before_replay(
    monkeypatch: pytest.MonkeyPatch,
    relation_name: str,
    label_name: str,
    representation: str,
    label: object,
) -> None:
    result = _result()
    relation = getattr(result, relation_name)
    edge = relation.edges[0].model_copy(update={label_name: label})
    result = result.model_copy(
        update={relation_name: relation.model_copy(update={"edges": (edge,)})}
    )
    replays: list[object] = []

    def unexpected_replay(*args: object) -> bool:
        replays.append(args)
        raise AssertionError("malformed label reached replay")

    monkeypatch.setattr(_models, "_output_edge_advances_language", unexpected_replay)
    with pytest.raises(ValidationError) as error:
        _decode(result, representation)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_label_invalid"
    )
    assert replays == []


@pytest.mark.parametrize("representation", ("native", "native-mapping", "json"))
@pytest.mark.parametrize("interruption", ("cancellation", "deadline"))
def test_valid_maximum_label_replay_can_be_interrupted_mid_edge(
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
    interruption: str,
) -> None:
    result = _result()
    signal = Event()
    clock = [0.0]
    replay_checkpoints: list[str] = []

    def interrupt_after_first_symbol_checkpoint(stage: str) -> None:
        request_checkpoint(stage)
        if stage == "during output-restriction label replay":
            replay_checkpoints.append(stage)
            signal.set()
            clock[0] = 2.0

    monkeypatch.setattr(
        _models, "request_checkpoint", interrupt_after_first_symbol_checkpoint
    )
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    error_type = (
        OperationExecutionCancelledError
        if interruption == "cancellation"
        else OperationExecutionTimeoutError
    )
    with (
        request_execution(
            0.0,
            cancellation_signal=signal if interruption == "cancellation" else None,
            outer_deadline=1.0 if interruption == "deadline" else None,
        ),
        pytest.raises(error_type),
    ):
        _decode(result, representation)
    assert replay_checkpoints == ["during output-restriction label replay"]


@pytest.mark.parametrize("representation", ("native", "native-mapping", "json"))
def test_valid_maximum_labels_preserve_the_single_state_relation(
    representation: str,
) -> None:
    result = _result()
    decoded = _decode(result, representation)
    assert decoded == result
    assert decoded.restricted.edges == result.source.edges
    assert (
        decoded.restricted.initial_states == decoded.restricted.accepting_states == (0,)
    )
    assert (
        decoded.product_states[0].transducer_state
        == decoded.product_states[0].language_state
        == 0
    )


@pytest.mark.parametrize("relation_name", ("source", "restricted"))
@pytest.mark.parametrize("label_name", ("input_label", "output_label"))
def test_raw_label_is_rejected_before_parser_iteration(
    relation_name: str, label_name: str
) -> None:
    iterations: list[str] = []

    class UncopyableLabel:
        def __iter__(self) -> Iterator[int]:
            iterations.append("iterated")
            raise AssertionError("raw label reached parser iteration")

    payload = _result().model_dump(mode="json")
    payload[relation_name]["edges"][0][label_name] = UncopyableLabel()
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_label_invalid"
    )
    assert iterations == []


@pytest.mark.parametrize("relation_name", ("source", "restricted"))
@pytest.mark.parametrize("label_name", ("input_label", "output_label"))
def test_oversized_wire_label_is_refused_by_preflight(
    relation_name: str, label_name: str
) -> None:
    payload = _result().model_dump(mode="json")
    payload[relation_name]["edges"][0][label_name] = [0] * 513
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(payload)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_label_invalid"
    )


@pytest.mark.parametrize("relation_name", ("source", "restricted"))
@pytest.mark.parametrize("label_name", ("input_label", "output_label"))
def test_native_label_subclass_is_rejected_without_iteration(
    relation_name: str,
    label_name: str,
) -> None:
    iterations: list[object] = []

    class HookedLabel(tuple[int, ...]):
        def __iter__(self) -> Iterator[int]:
            iterations.append(self)
            raise AssertionError("untrusted label iteration")

    result = _result()
    relation = getattr(result, relation_name)
    edge = relation.edges[0].model_copy(update={label_name: HookedLabel((0,))})
    result = result.model_copy(
        update={relation_name: relation.model_copy(update={"edges": (edge,)})}
    )
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(result)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_label_invalid"
    )
    assert iterations == []


def test_untransported_source_label_is_still_bounded() -> None:
    result = _result()
    extra = result.source.edges[0].model_copy(update={"output_label": (0,) * 513})
    result = result.model_copy(
        update={
            "source": result.source.model_copy(
                update={"edges": (*result.source.edges, extra)}
            )
        }
    )
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(result)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.edge_label_invalid"
    )
