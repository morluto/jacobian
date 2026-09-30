"""Re-admit native context frames and descendants before retaining them."""

from collections.abc import Iterator
from typing import NoReturn

import pytest
from pydantic import ValidationError

from jacobian.math.logic.automata.tree.contexts import (
    FiniteTreeContext,
    TreeContextFrame,
)
from jacobian.math.logic.automata.tree.values import RankedTree


@pytest.mark.parametrize(
    "fields",
    [
        {"symbol": -1},
        {"symbol": True},
        {"symbol": 1.0},
        {"symbol": None},
        {"hole_child": -1},
        {"hole_child": False},
        {"hole_child": 0.0},
        {"hole_child": None},
        {"hole_child": 16},
        {"siblings": []},
        {"siblings": None},
        {"siblings": ({"symbol": 0},)},
    ],
)
@pytest.mark.parametrize("native_context", [False, True])
def test_context_revalidates_constructed_native_frame_fields(
    fields: dict[str, object], native_context: bool
) -> None:
    frame = TreeContextFrame.model_construct(
        symbol=fields.get("symbol", 2),
        hole_child=fields.get("hole_child", 0),
        siblings=fields.get("siblings", (RankedTree(symbol=0),)),
    )
    payload = (
        FiniteTreeContext.model_construct(arity=(0, 1, 2), frames=(frame,))
        if native_context
        else {"arity": [0, 1, 2], "frames": [frame]}
    )
    with pytest.raises(ValidationError) as error:
        FiniteTreeContext.model_validate(payload)
    assert error.value.errors()[0]["type"].startswith("tree_context.")


@pytest.mark.parametrize(
    "fields",
    [
        {"symbol": -3},
        {"symbol": False},
        {"symbol": 0.0},
        {"symbol": "0"},
        {"symbol": None},
        {"children": []},
        {"children": None},
        {"children": (42,)},
        {"children": ({"symbol": 0},)},
    ],
)
def test_context_revalidates_every_native_descendant_without_dump(
    fields: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = RankedTree.model_construct(
        symbol=fields.get("symbol", 0), children=fields.get("children", ())
    )
    sibling = RankedTree.model_construct(symbol=1, children=(malformed,))
    frame = TreeContextFrame.model_construct(
        symbol=2, hole_child=0, siblings=(sibling,)
    )

    def dump_must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError(
            "native context was serialized before structural admission"
        )

    monkeypatch.setattr(TreeContextFrame, "model_dump", dump_must_not_run)
    monkeypatch.setattr(RankedTree, "model_dump", dump_must_not_run)
    with pytest.raises(ValidationError) as error:
        FiniteTreeContext.model_validate({"arity": [0, 1, 2], "frames": [frame]})
    assert error.value.errors()[0]["type"].startswith("tree_context.")


def test_context_rejects_nested_raw_extras_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.logic.automata.tree import contexts

    def copy_must_not_run(_value: object) -> NoReturn:
        raise AssertionError("malformed nested tree reached recursive canonicalization")

    monkeypatch.setattr(contexts, "canonicalize_json_containers", copy_must_not_run)
    with pytest.raises(ValidationError) as error:
        FiniteTreeContext.model_validate(
            {
                "arity": [0, 1, 2],
                "frames": [
                    {
                        "symbol": 2,
                        "hole_child": 0,
                        "siblings": [
                            {
                                "symbol": 1,
                                "children": [
                                    {"symbol": 0, "children": [], "extra": []}
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    assert error.value.errors()[0]["type"] == "tree_context.shape"


@pytest.mark.parametrize("target", ["siblings", "children"])
def test_context_bounds_native_container_types_before_iteration(target: str) -> None:
    class UniterableTrees(tuple[RankedTree, ...]):
        def __iter__(self) -> Iterator[RankedTree]:
            raise AssertionError("native container was iterated before type admission")

    leaf = RankedTree(symbol=0)
    siblings = (
        UniterableTrees((leaf,))
        if target == "siblings"
        else (RankedTree.model_construct(symbol=1, children=UniterableTrees((leaf,))),)
    )
    frame = TreeContextFrame.model_construct(symbol=2, hole_child=0, siblings=siblings)
    with pytest.raises(ValidationError) as error:
        FiniteTreeContext.model_validate({"arity": [0, 1, 2], "frames": [frame]})
    assert error.value.errors()[0]["type"] == "tree_context.shape"
