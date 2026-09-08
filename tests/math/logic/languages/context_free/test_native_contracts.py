"""Degenerate grammar axes and native verifier failure boundaries."""

import pytest
from pydantic import ValidationError

from jacobian.math.logic.languages.context_free import operations as native
from jacobian.math.logic.languages.context_free._models import (
    FiniteCFGO,
    FirstSetsResult,
)


def test_ruleless_grammar_retains_nonterminal_axes() -> None:
    grammar = FiniteCFGO(
        nonterminals=("S", "T"), terminals=(), rules=(), start_symbol="S"
    )
    assert native.nullable_nonterminals(grammar) == (False, False)
    assert native.first_sets(grammar) == ((), ())
    assert native.dependency_edges(grammar) == ()
    result = FirstSetsResult(grammar=grammar, first_sets=((), ()))
    assert native.verify_first_sets(
        type(result).model_validate_json(result.model_dump_json())
    )


@pytest.mark.parametrize("axis", ["nonterminals", "terminals"])
def test_grammar_symbol_axes_are_unique(axis: str) -> None:
    payload = {
        "nonterminals": ("S",),
        "terminals": ("a",),
        "rules": ({"head": "S", "body": ("a",)},),
        "start_symbol": "S",
    }
    payload[axis] = ("S", "S") if axis == "nonterminals" else ("a", "a")
    with pytest.raises(ValidationError, match="unique"):
        FiniteCFGO.model_validate(payload)


@pytest.mark.parametrize(
    "kernel,verifier,result_type",
    [
        ("nullable_nonterminals", "verify_symbol_profiles", "SymbolProfilesResult"),
        ("dependency_edges", "verify_dependency_graph", "DependencyGraphResult"),
        ("first_sets", "verify_first_sets", "FirstSetsResult"),
    ],
)
def test_verifiers_propagate_unexpected_kernel_failures(
    kernel: str, verifier: str, result_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jacobian.math.logic.languages.context_free import _models

    grammar = FiniteCFGO(nonterminals=("S",), terminals=(), rules=(), start_symbol="S")
    field = {
        "nullable_nonterminals": "nullable",
        "dependency_edges": "edges",
        "first_sets": "first_sets",
    }[kernel]
    claim = getattr(_models, result_type)(
        grammar=grammar, **{field: getattr(native, kernel)(grammar)}
    )

    def fail(*args: object) -> object:
        raise ValueError("unexpected execution failure")

    monkeypatch.setattr(native, kernel, fail)
    with pytest.raises(ValueError, match="unexpected execution failure"):
        getattr(native, verifier)(claim)
