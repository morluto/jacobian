"""Every retained result axis is bounded before the decoder copies containers."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

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
        edges=(RationalEdge(source=0, target=0, input_label=(0,), output_label=(0,)),),
    )
    language = DFA(
        state_count=1,
        alphabet_size=1,
        alphabet=alphabet,
        initial_state=0,
        accepting_states=(0,),
        transitions=(DFATransition(source=0, symbol=0, target=0),),
    )
    return restrict_rational_output(source, language, alphabet)


CASES: tuple[tuple[tuple[str | int, ...], object], ...] = (
    (("product_states",), [None] * 65),
    (("edge_sources",), [None] * 4097),
    (("output_language", "transitions"), [None] * 4097),
    (("output_language", "accepting_states"), [0] * 65),
    (("output_language", "state_count"), [0] * 65),
    (("output_language", "initial_state"), [0] * 65),
    (("output_language", "alphabet_id"), [0] * 65),
    (("output_language", "transitions", 0, "symbol"), [0] * 65),
    (("output_language", "transitions", 0, "extra"), [0] * 65),
    (("product_states", 0, "product_state"), [0] * 65),
    (("edge_sources", 0, "source_edge"), [0] * 65),
    (("output_alphabet", "symbols"), ["a"] * 33),
    (("output_alphabet", "symbols", 0), ["a"] * 65),
    (("output_alphabet", "symbols", 0), "a" * 65),
    (("extra",), [0] * 65),
    (("product_states", 0, "extra"), [0] * 65),
    (("edge_sources", 0, "extra"), [0] * 65),
    (("output_alphabet", "extra"), [0] * 65),
    (("output_language", "extra"), [0] * 65),
    *(
        ((owner, field), [0] * 65)
        for owner in ("source", "restricted")
        for field in (
            "initial_states",
            "accepting_states",
            "state_count",
            "input_alphabet_id",
            "output_alphabet_id",
            "extra",
        )
    ),
    *(
        ((owner, "edges", 0, field), [0] * 65)
        for owner in ("source", "restricted")
        for field in ("source", "target", "extra")
    ),
    *(
        ((owner, field, "symbols"), ["a"] * 33)
        for owner in ("source", "restricted")
        for field in ("input_alphabet", "output_alphabet")
    ),
    (("output_language", "alphabet", "symbols"), ["a"] * 33),
)


def _replace(value: Any, path: tuple[str | int, ...], replacement: object) -> Any:
    if not path:
        return replacement
    key, *remaining = path
    if isinstance(value, BaseModel):
        child = getattr(value, str(key), None)
        return value.model_copy(
            update={str(key): _replace(child, tuple(remaining), replacement)}
        )
    if isinstance(value, tuple):
        rows = list(value)
        rows[int(key)] = _replace(rows[int(key)], tuple(remaining), replacement)
        return tuple(rows)
    raise AssertionError("fixture path must stay within canonical models and tuples")


@pytest.mark.parametrize(
    "path,replacement", CASES, ids=["-".join(map(str, path)) for path, _ in CASES]
)
@pytest.mark.parametrize("representation", ("mapping", "native"))
def test_raw_shape_is_refused_before_recursive_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str | int, ...],
    replacement: object,
    representation: str,
) -> None:
    result = _result()
    if representation == "native":
        payload = _replace(result, path, replacement)
    else:
        payload = result.model_dump(mode="json")
        parent: Any = payload
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replacement
    copies: list[object] = []

    def unexpected_copy(data: object) -> object:
        copies.append(data)
        raise AssertionError("unbounded result reached recursive canonicalization")

    monkeypatch.setattr(_models, "canonicalize_json_containers", unexpected_copy)
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(payload)
    assert copies == []
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.result_shape_invalid"
    )


@pytest.mark.parametrize(
    "field",
    (
        "product_states",
        "edge_sources",
        "output_language",
        "output_alphabet",
        "source",
        "restricted",
    ),
)
def test_constructed_result_missing_required_field_has_a_structured_error(
    field: str,
) -> None:
    fields = dict(_result())
    del fields[field]
    forged = RestrictRationalOutputResult.model_construct(**fields)
    with pytest.raises(ValidationError) as error:
        RestrictRationalOutputResult.model_validate(forged)
    assert (
        error.value.errors()[0]["type"]
        == "rational_transducer.restrict_output.result_shape_invalid"
    )


@pytest.mark.parametrize("representation", ("native", "mapping", "json"))
def test_valid_result_keeps_its_self_loop_and_round_trips(representation: str) -> None:
    result = _result()
    if representation == "json":
        decoded = RestrictRationalOutputResult.model_validate_json(
            result.model_dump_json()
        )
    else:
        decoded = RestrictRationalOutputResult.model_validate(
            result if representation == "native" else result.model_dump()
        )
    assert decoded == result
    assert decoded.restricted.edges == (
        RationalEdge(source=0, target=0, input_label=(0,), output_label=(0,)),
    )
    assert decoded.product_states[0].product_state == 0
    assert decoded.edge_sources[0].source_edge == 0
