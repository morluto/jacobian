"""Catalog-wide conformance checks for accepted public-operation requests."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.dispatch import invoke_operation

MAX_MUTATIONS_PER_EXAMPLE = 256


def _builtin_operations() -> tuple[MathTool[Any, Any], ...]:
    catalog = Catalog.open()
    return tuple(
        operation
        for descriptor in catalog.snapshot().operations
        if (operation := catalog.operation(descriptor.operation_id)) is not None
    )


def _scalar_replacements(value: object) -> tuple[object, ...]:
    if isinstance(value, bool):
        return (not value,)
    if isinstance(value, int):
        return (-1, 0, 1, value + 1)
    if isinstance(value, float):
        return (-1.0, 0.0, 1.0)
    if isinstance(value, str):
        return (
            "",
            " ",
            "0",
            "-1",
            "0/0",
            "1/0",
            "nan",
            "oo",
            "undeclared_symbol",
        )
    if value is None:
        return (0, "", {}, [])
    return ()


def _replace_at_path(
    payload: object, path: tuple[object, ...], value: object
) -> object:
    result = deepcopy(payload)
    target = result
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return result


def _dict_mutations(
    value: dict[object, object],
    *,
    root: object,
    path: tuple[object, ...],
) -> Iterator[tuple[str, object]]:
    for key in value:
        without = deepcopy(root)
        target = without
        for component in path:
            target = target[component]  # type: ignore[index]
        del target[key]  # type: ignore[attr-defined]
        yield f"remove {(*path, key)}", without
    extra = deepcopy(root)
    target = extra
    for component in path:
        target = target[component]  # type: ignore[index]
    target["unexpected_contract_field"] = 0  # type: ignore[index]
    yield f"add field at {path}", extra
    for key, child in value.items():
        yield from _mutations(child, root=root, path=(*path, key))


def _list_mutations(
    value: list[object],
    *,
    root: object,
    path: tuple[object, ...],
) -> Iterator[tuple[str, object]]:
    yield f"empty {path}", _replace_at_path(root, path, [])
    if not value:
        return
    yield f"duplicate first in {path}", _replace_at_path(root, path, [value[0], *value])
    yield f"duplicate last in {path}", _replace_at_path(root, path, [*value, value[-1]])
    if len(value) > 1:
        yield f"reverse {path}", _replace_at_path(root, path, value[::-1])
    for index, child in enumerate(value):
        yield from _mutations(child, root=root, path=(*path, index))


def _mutations(
    value: object,
    *,
    root: object,
    path: tuple[object, ...] = (),
) -> Iterator[tuple[str, object]]:
    for replacement in _scalar_replacements(value):
        if replacement != value:
            yield f"{path}={replacement!r}", _replace_at_path(root, path, replacement)

    if isinstance(value, dict):
        yield from _dict_mutations(value, root=root, path=path)
    elif isinstance(value, list):
        yield from _list_mutations(value, root=root, path=path)


@pytest.mark.parametrize(
    "operation",
    _builtin_operations(),
    ids=lambda operation: operation.operation_id,
)
def test_every_accepted_boundary_mutation_returns_the_declared_result(
    operation: MathTool[Any, Any],
) -> None:
    for invocation_example in operation.examples:
        mutations = _mutations(invocation_example.input, root=invocation_example.input)
        for index, (mutation, payload) in enumerate(mutations):
            if index >= MAX_MUTATIONS_PER_EXAMPLE:
                break
            try:
                request = operation.request_type.model_validate(payload)
            except ValidationError:
                continue
            try:
                result = operation.run(request)
            except Exception as exc:
                pytest.fail(
                    f"{operation.operation_id} accepted {mutation} but raised "
                    f"{type(exc).__name__}: {exc}"
                )
            assert isinstance(result, operation.result_type), (
                operation.operation_id,
                mutation,
                type(result),
                operation.result_type,
            )


def test_large_periodic_profile_survives_public_result_wrapping() -> None:
    result = invoke_operation(
        "symbolic_dynamics.periodic_point_profile.compute",
        {"shift": {"matrix": [[1_000_000]], "two_sided": True}, "max_period": 3},
        Catalog.open(),
    )
    assert result.output["fixed_point_counts"][-1] == "1000000000000000000"
