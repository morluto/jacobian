"""Tests for the finite-Abelian forbidden-prefix sequencing operation."""

from __future__ import annotations

from itertools import permutations
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import MathTool
from jacobian.math.combinatorics.additive.cyclic_prefix_sum import (
    search_forbidden_prefix_cyclic_ordering,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._models import (
    MAX_SEQUENCING_PERMUTATION_NODES,
    MAX_SEQUENCING_SOURCE_ITEMS,
    ForbiddenPrefixSequencingRequest,
    ForbiddenPrefixSequencingResult,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._sequencing_kernel import (
    search_forbidden_prefix_sequencing,
)
from jacobian.math.combinatorics.additive.cyclic_prefix_sum._tools import TOOLS


def _operation() -> MathTool[
    ForbiddenPrefixSequencingRequest, ForbiddenPrefixSequencingResult
]:
    return cast(
        MathTool[ForbiddenPrefixSequencingRequest, ForbiddenPrefixSequencingResult],
        next(
            operation
            for operation in TOOLS
            if operation.operation_id
            == "additive.cyclic_prefix_sum.forbidden_prefix_sequencing.find"
        ),
    )


def _request(
    elements: tuple[tuple[int, ...], ...],
    moduli: tuple[int, ...],
    *,
    first_element: tuple[int, ...] | None = None,
    forbidden_values: tuple[tuple[int, ...], ...] = (),
) -> ForbiddenPrefixSequencingRequest:
    return ForbiddenPrefixSequencingRequest.model_validate(
        {
            "source": {
                "group": {"moduli": list(moduli)},
                "elements": [list(e) for e in elements],
            },
            **(
                {"first_element": list(first_element)}
                if first_element is not None
                else {}
            ),
            "forbidden_values": [list(value) for value in forbidden_values],
        }
    )


def _run(
    request: ForbiddenPrefixSequencingRequest,
    *,
    node_limit: int = MAX_SEQUENCING_PERMUTATION_NODES,
) -> ForbiddenPrefixSequencingResult:
    return search_forbidden_prefix_cyclic_ordering(
        request.source,
        request.first_element,
        request.forbidden_values,
        search_node_limit=node_limit,
    )


def _replay(
    result: ForbiddenPrefixSequencingResult,
    request: ForbiddenPrefixSequencingRequest,
) -> None:
    assert result.status == "FOUND"
    ordering = result.ordering
    assert ordering is not None
    moduli = request.source.group.moduli
    running: list[int] = [0] * len(moduli)
    seen: set[tuple[int, ...]] = set()
    forbidden = set(request.forbidden_values)
    for position, row in enumerate(ordering):
        running = [
            (left + right) % modulus
            for left, right, modulus in zip(running, row.element, moduli, strict=True)
        ]
        assert tuple(running) == row.prefix_sum
        if position < len(ordering) - 1:
            proper = tuple(running)
            assert proper != tuple(0 for _ in moduli)
            assert proper not in forbidden
            assert proper not in seen
            seen.add(proper)
    assert {row.element for row in ordering} == set(request.source.elements)
    if request.first_element is not None:
        assert ordering[0].element == request.first_element


def test_positive_fixed_start_avoids_one_target() -> None:
    request = _request(
        ((1,), (2,), (4,), (5,)), (7,), first_element=(2,), forbidden_values=((1,),)
    )

    result = _run(request)

    assert result.status == "FOUND"
    assert result.first_element == (2,)
    assert result.forbidden_values == ((1,),)
    _replay(result, request)


def test_exceptional_four_element_family_is_exactly_negative() -> None:
    for modulus in (5, 7, 11):
        request = _request(
            (
                (1,),
                (2,),
                (modulus - 2,),
                (modulus - 1,),
            ),
            (modulus,),
            first_element=(2,),
            forbidden_values=((1,),),
        )

        result = _run(request)

        assert result.status == "EXHAUSTED"
        assert result.ordering is None
        assert result.source.elements == (
            (1,),
            (2,),
            (modulus - 2,),
            (modulus - 1,),
        )


def test_two_and_three_element_zero_sum_sources_use_cyclic_endpoint() -> None:
    # In Z/7Z: {1,6} and {1,2,4} each sum to zero, and the chosen witnesses
    # return to zero only at their terminal prefix sum.
    pair = _request(((1,), (6,)), (7,))
    triple = _request(((1,), (2,), (4,)), (7,), forbidden_values=((0,),))

    for request in (pair, triple):
        result = _run(request)
        assert result.status == "FOUND"
        _replay(result, request)
        assert result.ordering is not None
        assert result.ordering[-1].prefix_sum == (0,)


def test_empty_forbidden_set_and_nonzero_sum_source() -> None:
    nonzero = _request(((1,), (2,), (3,)), (5,))

    result = _run(nonzero)

    assert result.status == "FOUND"
    _replay(result, nonzero)
    assert result.ordering is not None
    assert result.ordering[-1].prefix_sum != (0,)


def test_empty_source_has_the_vacuous_found_ordering() -> None:
    result = _run(_request((), (3,)))

    assert result.status == "FOUND"
    assert result.ordering == ()
    assert result.states_explored == 1


def test_node_budget_returns_unknown_not_nonexistence() -> None:
    # Every fixed-start ordering is excluded, so a small budget can stop the
    # complete fixed-start tree before either a witness or exhaustion.
    request = _request(
        ((1,), (2,), (5,), (6,)),
        (7,),
        first_element=(2,),
        forbidden_values=((1,),),
    )

    limited = _run(request, node_limit=2)

    assert limited.status == "UNKNOWN"
    assert limited.ordering is None
    replayed = ForbiddenPrefixSequencingResult.model_validate(
        limited.model_dump(mode="json")
    )
    assert replayed == limited


def test_kernel_matches_exhaustive_permutation_oracle() -> None:
    elements = ((1,), (2,), (4,), (5,))
    moduli = (7,)
    forbidden = ((1,), (3,))

    def oracle() -> tuple[int, ...] | None:
        for indices in permutations(range(len(elements))):
            seen: set[int] = set()
            running = 0
            ok = True
            for position, index in enumerate(indices):
                running = (running + elements[index][0]) % moduli[0]
                if position < len(indices) - 1 and (
                    running == 0
                    or running in {value[0] for value in forbidden}
                    or running in seen
                ):
                    ok = False
                    break
                seen.add(running)
            if ok:
                return indices
        return None

    expected = oracle()
    actual = search_forbidden_prefix_sequencing(
        elements, moduli, forbidden, None, MAX_SEQUENCING_PERMUTATION_NODES
    )

    if expected is None:
        assert actual.status == "EXHAUSTED"
    else:
        assert actual.status == "FOUND"
        assert actual.ordering_indices == expected


def test_product_group_witness_replays_across_axes() -> None:
    request = _request(
        ((0, 1), (1, 0), (1, 2), (2, 1)),
        (3, 4),
        forbidden_values=((1, 1),),
    )

    result = _run(request)

    assert result.status == "FOUND"
    _replay(result, request)


def test_request_and_result_round_trip_canonically() -> None:
    request = _request(
        ((1,), (2,), (4,), (5,)), (7,), first_element=(2,), forbidden_values=((1,),)
    )
    result = _operation().run(request)

    assert request == ForbiddenPrefixSequencingRequest.model_validate(
        request.model_dump(mode="json")
    )
    assert result == ForbiddenPrefixSequencingResult.model_validate(
        result.model_dump(mode="json")
    )


def test_catalog_example_is_negative_and_replays_through_public_operation() -> None:
    operation = _operation()
    assert len(operation.examples) == 1
    request = operation.request_type.model_validate(operation.examples[0].input)

    result = operation.run(request)

    assert result.status == "EXHAUSTED"
    assert result.states_explored == 3


def test_request_reduces_and_validates_canonical_sources() -> None:
    request = _request(
        (
            (7,),
            (2,),
        ),
        (7,),
        first_element=(7,),
        forbidden_values=((8,), (2,)),
    )

    assert request.source.elements == ((0,), (2,))
    assert request.first_element == (0,)
    assert request.forbidden_values == ((1,), (2,))

    with pytest.raises(ValidationError, match="distinct and sorted"):
        ForbiddenPrefixSequencingRequest.model_validate(
            {
                "source": {"group": {"moduli": [7]}, "elements": [[2], [2]]},
                "forbidden_values": [],
            }
        )
    with pytest.raises(ValidationError, match="match the group rank"):
        ForbiddenPrefixSequencingRequest.model_validate(
            {
                "source": {"group": {"moduli": [7]}, "elements": [[1, 2]]},
                "forbidden_values": [],
            }
        )
    with pytest.raises(ValidationError, match="reduce to a source element"):
        ForbiddenPrefixSequencingRequest.model_validate(
            {
                "source": {"group": {"moduli": [7]}, "elements": [[1]]},
                "first_element": [2],
                "forbidden_values": [],
            }
        )


def test_admission_rejects_unbounded_group_source_and_work() -> None:
    with pytest.raises(ValueError, match="4,096-element"):
        _run(
            ForbiddenPrefixSequencingRequest.model_validate(
                {
                    "source": {"group": {"moduli": [4097]}, "elements": []},
                    "forbidden_values": [],
                }
            )
        )

    with pytest.raises(ValidationError, match="at most 8 items"):
        _run(
            ForbiddenPrefixSequencingRequest.model_validate(
                {
                    "source": {
                        "group": {"moduli": [11]},
                        "elements": [[value] for value in range(9)],
                    },
                    "forbidden_values": [],
                }
            )
        )

    with pytest.raises(ValidationError, match="less than or equal to 109601"):
        ForbiddenPrefixSequencingRequest.model_validate(
            {
                **_request(((1,), (2,), (4,), (5,)), (7,)).model_dump(mode="json"),
                "search_node_limit": MAX_SEQUENCING_PERMUTATION_NODES + 1,
            }
        )


def test_schema_publishes_exact_search_envelope() -> None:
    schema = ForbiddenPrefixSequencingRequest.model_json_schema()
    source_schema = schema["$defs"]["FiniteAbelianSequencingSource"]
    result_schema = ForbiddenPrefixSequencingResult.model_json_schema()

    assert (
        source_schema["properties"]["elements"]["maxItems"]
        == MAX_SEQUENCING_SOURCE_ITEMS
    )
    assert result_schema["properties"]["search_node_limit"]["maximum"] == (
        MAX_SEQUENCING_PERMUTATION_NODES
    )
    assert "EXHAUSTED" in schema["description"]
    assert "UNKNOWN" in schema["description"]
    assert "standard cyclic-sequencing convention" in schema["description"]


def test_result_rejects_incomplete_or_forged_witness_shapes() -> None:
    request = _request(
        (
            (1,),
            (2,),
        ),
        (7,),
    )
    found = _run(request)
    assert found.status == "FOUND"
    payload = found.model_dump(mode="json")

    with pytest.raises(ValidationError):
        ForbiddenPrefixSequencingResult.model_validate(
            {**payload, "ordering": payload["ordering"][:1]}
        )
    with pytest.raises(ValidationError):
        ForbiddenPrefixSequencingResult.model_validate(
            {**payload, "status": "EXHAUSTED"}
        )
