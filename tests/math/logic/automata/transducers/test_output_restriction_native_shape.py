"""Native admission must establish the shape retained by the result decoder."""

from __future__ import annotations

import pytest
from pydantic import ConfigDict

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.transducers.output_restriction import operations
from jacobian.math.logic.automata.transducers.output_restriction._models import (
    RestrictRationalOutputResult,
)
from jacobian.math.logic.automata.transducers.values import (
    FiniteAlphabet,
    RationalEdge,
    RationalTransducer,
)
from jacobian.math.logic.languages.regular.values import DFA, DFATransition


class MutableRelation(RationalTransducer):
    model_config = ConfigDict(frozen=False)


class MutableDFA(DFA):
    model_config = ConfigDict(frozen=False)


class MutableAlphabet(FiniteAlphabet):
    model_config = ConfigDict(frozen=False)


class TupleSubclass(tuple[object, ...]):
    pass


class TextSubclass(str):
    pass


def _inputs() -> tuple[RationalTransducer, DFA, FiniteAlphabet]:
    alphabet = FiniteAlphabet(symbols=("a",))
    source = RationalTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        input_alphabet=alphabet,
        output_alphabet=alphabet,
        state_count=2,
        initial_states=(0,),
        accepting_states=(1,),
        edges=(RationalEdge(source=0, target=1, input_label=(0,), output_label=(0,)),),
    )
    language = DFA(
        state_count=1,
        alphabet_size=1,
        alphabet=alphabet,
        initial_state=0,
        accepting_states=(0,),
        transitions=(DFATransition(source=0, symbol=0, target=0),),
    )
    return source, language, alphabet


def _cases() -> dict[str, tuple[RationalTransducer, DFA, FiniteAlphabet, str]]:
    source, language, alphabet = _inputs()
    edge = source.edges[0]
    mutable_alphabet = MutableAlphabet.model_construct(**dict(alphabet))
    variants = {
        "source-subclass": (
            MutableRelation.model_construct(**dict(source)),
            language,
            alphabet,
            "relation_type_invalid",
        ),
        "dfa-subclass": (
            source,
            MutableDFA.model_construct(**dict(language)),
            alphabet,
            "language_type_invalid",
        ),
        "output-alphabet-subclass": (
            source.model_copy(update={"output_alphabet": None}),
            language,
            mutable_alphabet,
            "output_context_invalid",
        ),
        "input-alphabet-subclass": (
            source.model_copy(update={"input_alphabet": mutable_alphabet}),
            language,
            alphabet,
            "alphabet_context_invalid",
        ),
        "source-output-alphabet-subclass": (
            source.model_copy(update={"output_alphabet": mutable_alphabet}),
            language,
            alphabet,
            "alphabet_context_invalid",
        ),
        "dfa-alphabet-subclass": (
            source,
            language.model_copy(update={"alphabet": mutable_alphabet}),
            alphabet,
            "native_input_shape",
        ),
        "source-edges-subclass": (
            source.model_copy(update={"edges": TupleSubclass(source.edges)}),
            language,
            alphabet,
            "edge_shape_invalid",
        ),
        "source-initial-subclass": (
            source.model_copy(update={"initial_states": TupleSubclass((0,))}),
            language,
            alphabet,
            "initial_states_invalid",
        ),
        "source-accepting-subclass": (
            source.model_copy(update={"accepting_states": TupleSubclass((1,))}),
            language,
            alphabet,
            "accepting_states_invalid",
        ),
        "dfa-transitions-subclass": (
            source,
            language.model_copy(
                update={"transitions": TupleSubclass(language.transitions)}
            ),
            alphabet,
            "dfa_transitions_invalid",
        ),
        "dfa-accepting-subclass": (
            source,
            language.model_copy(update={"accepting_states": TupleSubclass((0,))}),
            alphabet,
            "dfa_accepting_states_invalid",
        ),
        "source-id-subclass": (
            source.model_copy(update={"input_alphabet_id": TextSubclass("id")}),
            language,
            alphabet,
            "alphabet_identity_too_long",
        ),
        "dfa-id-subclass": (
            source,
            language.model_copy(update={"alphabet_id": TextSubclass("id")}),
            alphabet,
            "native_input_shape",
        ),
        "source-long-id": (
            source.model_copy(update={"input_alphabet_id": "a" * 129}),
            language,
            alphabet,
            "native_input_shape",
        ),
        "source-extra": (
            source.model_copy(update={"extra": [0] * 65}),
            language,
            alphabet,
            "relation_type_invalid",
        ),
        "dfa-extra": (
            source,
            language.model_copy(update={"extra": [0] * 65}),
            alphabet,
            "language_type_invalid",
        ),
        "edge-extra": (
            source.model_copy(
                update={"edges": (edge.model_copy(update={"extra": [0] * 65}),)}
            ),
            language,
            alphabet,
            "edge_invalid",
        ),
        "input-label-subclass": (
            source.model_copy(
                update={
                    "edges": (
                        edge.model_copy(update={"input_label": TupleSubclass((0,))}),
                    )
                }
            ),
            language,
            alphabet,
            "edge_label_invalid",
        ),
        "output-label-subclass": (
            source.model_copy(
                update={
                    "edges": (
                        edge.model_copy(update={"output_label": TupleSubclass((0,))}),
                    )
                }
            ),
            language,
            alphabet,
            "edge_label_invalid",
        ),
    }
    for case, symbols, reason in (
        ("symbols-subclass", TupleSubclass(("a",)), "alphabet_context_invalid"),
        ("symbol-subclass", (TextSubclass("a"),), "alphabet_symbol_too_long"),
        ("long-symbol", ("a" * 65,), "native_input_shape"),
    ):
        variants[case] = (
            source.model_copy(
                update={
                    "input_alphabet": alphabet.model_copy(update={"symbols": symbols})
                }
            ),
            language,
            alphabet,
            reason,
        )
    return variants


@pytest.mark.parametrize("case", tuple(_cases()))
def test_noncanonical_native_shape_is_rejected_before_product_expansion(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    source, language, alphabet, reason = _cases()[case]
    starts: list[object] = []

    def unexpected_product(*args: object) -> None:
        starts.append(args)
        raise AssertionError("noncanonical native input reached the product kernel")

    monkeypatch.setattr(operations, "deque", unexpected_product)
    with pytest.raises(OperationDomainValidationError) as error:
        operations.restrict_rational_output(source, language, alphabet)
    assert (
        error.value.errors()[0]["type"]
        == f"rational_transducer.restrict_output.{reason}"
    )
    assert starts == []


def test_canonical_native_result_still_decodes_and_preserves_the_relation() -> None:
    source, language, alphabet = _inputs()
    result = operations.restrict_rational_output(source, language, alphabet)
    assert result.restricted.edges == source.edges
    assert result.restricted.initial_states == (0,)
    assert result.restricted.accepting_states == (1,)
    assert (
        RestrictRationalOutputResult.model_validate_json(result.model_dump_json())
        == result
    )
