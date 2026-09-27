"""Canonical composable Weyl elements on simple-root coordinates."""

from collections import deque

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    WeylElement,
    WeylElementComposeRequest,
    WeylElementInverseRequest,
    WeylElementRequest,
)
from jacobian.math.groups.root_systems.operations import (
    weyl_element_compose,
    weyl_element_from_word,
    weyl_element_inverse,
)


def _multiply(a, b):
    n = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n))
        for i in range(n)
    )


def _oracle_group(cartan):
    """Enumerate actions independently by multiplying simple reflections."""
    n = len(cartan)
    generators = []
    for index in range(n):
        reflection = [[int(i == j) for j in range(n)] for i in range(n)]
        for column in range(n):
            reflection[index][column] -= cartan[index][column]
        generators.append(tuple(tuple(row) for row in reflection))
    identity = tuple(tuple(int(i == j) for j in range(n)) for i in range(n))
    seen = {identity}
    queue = deque([identity])
    while queue:
        current = queue.popleft()
        for generator in generators:
            image = _multiply(generator, current)
            if image not in seen:
                seen.add(image)
                queue.append(image)
    return seen


@pytest.mark.parametrize(
    ("cartan", "expected_order"),
    [(((2, -1), (-1, 2)), 6), (((2, -2), (-1, 2)), 8)],
)
def test_canonical_element_values_match_independent_finite_group(
    cartan, expected_order
):
    parent = CartanMatrix.model_validate(cartan)
    identity = weyl_element_from_word(parent, ())
    oracle = _oracle_group(cartan)
    generated = {identity.root_action.entries}
    frontier = [identity]
    generators = [weyl_element_from_word(parent, (i,)) for i in range(len(cartan))]
    while frontier:
        element = frontier.pop()
        for generator in generators:
            next_element = weyl_element_compose(element, generator)
            key = next_element.root_action.entries
            if key not in generated:
                generated.add(key)
                frontier.append(next_element)
    assert len(oracle) == expected_order
    assert generated == oracle


def test_equal_words_share_canonical_action_and_serialize_composably():
    parent = CartanMatrix.model_validate(((2, -1), (-1, 2)))
    first = weyl_element_from_word(parent, (0, 1, 0))
    same = weyl_element_from_word(parent, (1, 0, 1))
    assert first == same
    restored = WeylElement.model_validate_json(first.model_dump_json())
    identity = weyl_element_compose(restored, weyl_element_inverse(restored))
    assert identity.root_action.entries == ((1, 0), (0, 1))
    assert weyl_element_inverse(first).root_action.entries == first.root_action.entries


def test_composition_rejects_different_cartan_parents():
    a2 = weyl_element_from_word(((2, -1), (-1, 2)), (0,))
    b2 = weyl_element_from_word(((2, -2), (-1, 2)), (0,))
    with pytest.raises(
        OperationDomainValidationError, match="same ordered Cartan parent"
    ):
        weyl_element_compose(a2, b2)


def test_requests_round_trip_element_contracts():
    element = weyl_element_from_word(((2, -1), (-1, 2)), (1,))
    assert WeylElementRequest.model_validate(
        {"matrix": ((2, -1), (-1, 2)), "word": [1]}
    )
    assert WeylElementComposeRequest.model_validate_json(
        '{"first":'
        + element.model_dump_json()
        + ',"then":'
        + element.model_dump_json()
        + "}"
    )
    assert WeylElementInverseRequest.model_validate_json(
        '{"element":' + element.model_dump_json() + "}"
    )
