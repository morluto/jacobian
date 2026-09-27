"""Exact Weyl-element order with a direct matrix-power oracle."""

from __future__ import annotations

import json

import pytest

from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    WeylElementOrderResult,
    WeylElementRequest,
)
from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import weyl_element_order


def _matmul(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(len(left)))
            for j in range(len(right))
        )
        for i in range(len(left))
    )


def _a2_reflection(index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    # Matrices on the simple-root basis, derived directly from
    # s_i(alpha_j) = alpha_j - A_ij alpha_i.
    return (((-1, 1), (0, 1)), ((1, 0), (1, -1)))[index]


def _matrix_power_order(matrix: tuple[tuple[int, ...], ...], cap: int = 12) -> int:
    identity = tuple(
        tuple(int(i == j) for j in range(len(matrix))) for i in range(len(matrix))
    )
    power = identity
    for exponent in range(1, cap + 1):
        power = _matmul(power, matrix)
        if power == identity:
            return exponent
    raise AssertionError("oracle cap did not contain the finite order")


def test_a2_word_order_agrees_with_direct_reflection_matrix_powers() -> None:
    matrix = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    word = (0, 1)
    action = _matmul(_a2_reflection(1), _a2_reflection(0))
    expected = _matrix_power_order(action)
    result = weyl_element_order(matrix, word)
    assert expected == result.order == 3
    assert result.matrix == matrix and result.word == word
    assert (
        WeylElementOrderResult.model_validate_json(result.model_dump_json()) == result
    )
    assert weyl_element_order(matrix, (1, 0)).order == 3


def test_small_a2_words_agree_with_independent_linear_action_oracle() -> None:
    from itertools import product

    matrix = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    for length in range(4):
        for word in product(range(2), repeat=length):
            action = tuple(tuple(int(i == j) for j in range(2)) for i in range(2))
            for index in word:
                action = _matmul(_a2_reflection(index), action)
            assert weyl_element_order(matrix, word).order == _matrix_power_order(action)


@pytest.mark.parametrize(
    ("word", "expected"), [((), 1), ((0,), 2), ((0, 1, 0, 1, 0, 1), 1)]
)
def test_a2_identity_and_reflection_orders(
    word: tuple[int, ...], expected: int
) -> None:
    matrix = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    # Independent linear action also catches cancellation in a non-reduced word.
    action = tuple(tuple(int(i == j) for j in range(2)) for i in range(2))
    for index in word:
        action = _matmul(_a2_reflection(index), action)
    assert (
        weyl_element_order(matrix, word).order
        == _matrix_power_order(action)
        == expected
    )


def test_order_preflight_happens_before_signed_root_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.groups.root_systems.operations as operations

    monkeypatch.setattr(operations, "MAX_WEYL_ELEMENT_ORDER_WORK", 0)
    monkeypatch.setattr(
        operations,
        "_signed_roots",
        lambda _rows: pytest.fail("signed roots were expanded before admission"),
    )
    with pytest.raises(OperationResourceAdmissionError):
        operations.weyl_element_order(
            CartanMatrix.model_validate(((2, -1), (-1, 2))), (0,)
        )


def test_order_rejects_invalid_word_index() -> None:
    matrix = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    with pytest.raises(OperationDomainValidationError, match="word"):
        weyl_element_order(matrix, (2,))


def test_manifest_example_invokes_published_order_operation() -> None:
    local = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "weyl_group.element.order.compute"
    )
    request = WeylElementRequest.model_validate_json(
        json.dumps(local.examples[0].input), strict=True
    )
    public = next(
        tool
        for tool in BUILTIN_TOOLS
        if tool.operation_id == "weyl_group.element.order.compute"
    )
    assert public.run(request).order == 3
