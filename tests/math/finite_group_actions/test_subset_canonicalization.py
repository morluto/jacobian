"""Contract tests for finite-action subset canonicalization."""

import math
from itertools import combinations

import pytest
from pydantic import ValidationError
from sympy.combinatorics import Permutation, PermutationGroup

from jacobian.math.finite_group_actions._models import (
    MAX_GROUP_ORDER,
    ActionBoundSubset,
    FinitePermutationAction,
    SubsetCanonicalizationRequest,
    SubsetCanonicalizationResult,
)
from jacobian.math.finite_group_actions._operations import (
    _enumerate_group,
    compute_subset_canonicalization,
)
from jacobian.math.finite_group_actions._tools import TOOLS


def _cyclic_c3(
    labels: tuple[str, str, str] = ("a", "b", "c"),
) -> FinitePermutationAction:
    return FinitePermutationAction(domain=labels, generators=((1, 2, 0),))


def _assert_error_type(error: ValidationError, expected: str) -> None:
    assert error.errors()[0]["type"] == expected


def _symmetric_s3(
    labels: tuple[str, str, str] = ("p0", "p1", "p2"),
) -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=labels,
        generators=((1, 2, 0), (1, 0, 2)),
    )


def _dihedral_d4() -> FinitePermutationAction:
    return FinitePermutationAction(
        domain=("v0", "v1", "v2", "v3"),
        generators=((1, 2, 3, 0), (1, 0, 3, 2)),
    )


def _request(
    action: FinitePermutationAction,
    positions: tuple[int, ...],
) -> SubsetCanonicalizationRequest:
    return SubsetCanonicalizationRequest(
        subset=ActionBoundSubset(action=action, positions=positions)
    )


def _oversized_symmetric_group_degree() -> int:
    degree = 1
    while math.factorial(degree) <= MAX_GROUP_ORDER:
        degree += 1
    return degree


def _mathematical_payload(result: SubsetCanonicalizationResult) -> dict:
    """The canonicalization content independent of how the action is presented."""
    return {
        "source_positions": result.source_subset.positions,
        "canonical_positions": result.canonical_subset.positions,
        "transporter": result.transporter,
        "orbit_size": result.orbit_size,
        "stabilizer_size": result.stabilizer_size,
    }


def _transport(
    permutation: tuple[int, ...], subset: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(sorted(permutation[position] for position in subset))


def _sympy_oracle(
    action: FinitePermutationAction,
    source_subset: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    group = PermutationGroup(
        [Permutation(list(generator)) for generator in action.generators]
    )
    elements = tuple(
        tuple(permutation) for permutation in group.generate_schreier_sims(af=True)
    )
    images = {_transport(permutation, source_subset) for permutation in elements}
    canonical = min(images)
    transporter = min(
        permutation
        for permutation in elements
        if _transport(permutation, source_subset) == canonical
    )
    stabilizer_size = sum(
        _transport(permutation, source_subset) == source_subset
        for permutation in elements
    )
    return canonical, transporter, len(images), stabilizer_size


def test_cyclic_action_canonicalizes_singleton_with_transporter() -> None:
    result = compute_subset_canonicalization(_request(_cyclic_c3(), (2,)))

    assert result.canonical_subset.positions == (0,)
    assert tuple(
        result.transporter[index] for index in result.source_subset.positions
    ) == (0,)
    assert result.orbit_size == 3
    assert result.stabilizer_size == 1


@pytest.mark.property
@pytest.mark.parametrize("action", [_cyclic_c3(), _symmetric_s3(), _dihedral_d4()])
def test_every_small_subset_agrees_with_independent_sympy_enumeration(
    action: FinitePermutationAction,
) -> None:
    for size in range(len(action.domain) + 1):
        for subset in combinations(range(len(action.domain)), size):
            result = compute_subset_canonicalization(_request(action, subset))
            assert (
                result.canonical_subset.positions,
                result.transporter,
                result.orbit_size,
                result.stabilizer_size,
            ) == _sympy_oracle(action, subset)
            assert _transport(result.transporter, subset) == (
                result.canonical_subset.positions
            )
            assert result.orbit_size * result.stabilizer_size == len(
                _enumerate_group(action)
            )


def test_generator_reordering_preserves_canonical_result_and_transporter() -> None:
    first = _symmetric_s3()
    second = FinitePermutationAction(
        domain=first.domain,
        generators=tuple(reversed(first.generators)),
    )

    first_result = compute_subset_canonicalization(_request(first, (2,)))
    second_result = compute_subset_canonicalization(_request(second, (2,)))

    assert _mathematical_payload(first_result) == _mathematical_payload(second_result)


def test_position_preserving_domain_relabelling_preserves_result() -> None:
    first = _symmetric_s3(("z", "a", "m"))
    relabelled = _symmetric_s3(("triangle", "edge", "witness"))

    first_result = compute_subset_canonicalization(_request(first, (1,)))
    relabelled_result = compute_subset_canonicalization(_request(relabelled, (1,)))

    assert first_result.canonical_subset.positions == (0,)
    assert first.domain[first_result.canonical_subset.positions[0]] == "z"
    assert first_result.canonical_subset.action == first
    assert _mathematical_payload(first_result) == _mathematical_payload(
        relabelled_result
    )


@pytest.mark.parametrize("subset", [(), (0, 1, 2, 3)])
def test_empty_and_full_subsets_have_the_whole_group_as_stabilizer(
    subset: tuple[int, ...],
) -> None:
    action = _dihedral_d4()
    result = compute_subset_canonicalization(_request(action, subset))

    assert result.source_subset.positions == subset
    assert result.canonical_subset.positions == subset
    assert result.transporter == (0, 1, 2, 3)
    assert result.orbit_size == 1
    assert result.stabilizer_size == 8


def test_fixed_nontrivial_subset_reports_its_setwise_stabilizer() -> None:
    result = compute_subset_canonicalization(_request(_dihedral_d4(), (0, 2)))

    assert result.canonical_subset.positions == (0, 2)
    assert result.orbit_size == 2
    assert result.stabilizer_size == 4


def test_bound_subset_normalizes_wire_order_by_domain_position() -> None:
    bound = ActionBoundSubset(action=_dihedral_d4(), positions=(3, 1))
    assert bound.positions == (1, 3)


def test_request_rejects_duplicate_and_out_of_domain_positions() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ActionBoundSubset(action=_cyclic_c3(), positions=(1, 1))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_positions_not_distinct"
    )
    with pytest.raises(ValidationError) as exc_info:
        ActionBoundSubset(action=_cyclic_c3(), positions=(3,))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_position_out_of_range"
    )
    with pytest.raises(ValidationError):
        ActionBoundSubset(action=_cyclic_c3(), positions=(-1,))

    with pytest.raises(ValidationError) as exc_info:
        _request(_cyclic_c3(), (1, 1))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_positions_not_distinct"
    )
    with pytest.raises(ValidationError) as exc_info:
        _request(_cyclic_c3(), (3,))
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_position_out_of_range"
    )


def test_request_rejects_group_immediately_above_enumeration_bound() -> None:
    degree = _oversized_symmetric_group_degree()
    symmetric_sn = FinitePermutationAction(
        domain=tuple(f"p{position}" for position in range(degree)),
        generators=(
            (*range(1, degree), 0),
            (1, 0, *range(2, degree)),
        ),
    )

    with pytest.raises(ValidationError) as exc_info:
        _request(symmetric_sn, (0,))
    _assert_error_type(exc_info.value, "finite_group_action.group_order_exceeds_bound")
    with pytest.raises(ValueError, match=rf"group order exceeds.*{MAX_GROUP_ORDER}"):
        _enumerate_group(symmetric_sn)


def test_domain_and_subset_boundary_of_fifty_positions_is_accepted() -> None:
    action = FinitePermutationAction(
        domain=tuple(f"p{position}" for position in range(50)),
        generators=(tuple(range(50)),),
    )
    result = compute_subset_canonicalization(
        _request(action, tuple(reversed(range(50))))
    )

    assert result.source_subset.positions == tuple(range(50))
    assert result.canonical_subset.positions == tuple(range(50))
    assert result.transporter == tuple(range(50))
    assert result.orbit_size == result.stabilizer_size == 1

    with pytest.raises(ValidationError):
        FinitePermutationAction(
            domain=tuple(f"p{position}" for position in range(51)),
            generators=(tuple(range(51)),),
        )


def test_generator_representation_boundary_is_explicit() -> None:
    identity = (0, 1, 2)
    accepted = FinitePermutationAction(
        domain=("a", "b", "c"),
        generators=(identity,) * 50,
    )
    assert len(accepted.generators) == 50
    result = compute_subset_canonicalization(_request(accepted, (2, 0)))
    assert result.canonical_subset.positions == (0, 2)
    assert result.orbit_size == result.stabilizer_size == 1

    with pytest.raises(ValidationError):
        FinitePermutationAction(
            domain=("a", "b", "c"),
            generators=(identity,) * 51,
        )


def _induced_pair_action_s6() -> FinitePermutationAction:
    degree = 6
    point_generators = (
        (1, 2, 3, 4, 5, 0),
        (1, 0, 2, 3, 4, 5),
    )
    induced_generators = tuple(
        tuple(
            generator[left] * degree + generator[right]
            for left in range(degree)
            for right in range(degree)
        )
        for generator in point_generators
    )
    return FinitePermutationAction(
        domain=tuple(
            f"({left},{right})" for left in range(degree) for right in range(degree)
        ),
        generators=induced_generators,
    )


def test_s6_function_graph_use_case_at_group_order_boundary() -> None:
    action = _induced_pair_action_s6()
    function = (1, 0, 0, 0, 0, 0)
    source_graph = tuple(
        source * 6 + destination for source, destination in enumerate(function)
    )

    result = compute_subset_canonicalization(_request(action, source_graph))

    canonical_pairs = tuple(
        divmod(position, 6) for position in result.canonical_subset.positions
    )
    assert len(canonical_pairs) == 6
    assert {source for source, _ in canonical_pairs} == set(range(6))
    assert all(source != destination for source, destination in canonical_pairs)
    assert _transport(result.transporter, source_graph) == (
        result.canonical_subset.positions
    )
    assert result.transporter in _enumerate_group(action)
    assert result.orbit_size * result.stabilizer_size == 720


def test_canonical_subset_serializes_unchanged_into_the_next_request() -> None:
    first = compute_subset_canonicalization(_request(_dihedral_d4(), (2, 1)))
    dumped = first.model_dump()

    # One canonical value passes through: the dumped canonical subset already
    # carries its action, so no field is manually reattached here.
    request = SubsetCanonicalizationRequest.model_validate(
        {"subset": dumped["canonical_subset"]}
    )
    second = compute_subset_canonicalization(request)

    assert request.subset == first.canonical_subset
    assert request.subset.action == first.source_subset.action
    assert request.subset.positions == first.canonical_subset.positions == (0, 1)
    assert second.source_subset == first.canonical_subset
    assert second.canonical_subset == first.canonical_subset
    assert second.transporter == tuple(range(len(first.canonical_subset.action.domain)))
    assert SubsetCanonicalizationResult.model_validate(dumped) == first


def test_bound_subset_dump_and_load_preserves_labelled_meaning() -> None:
    bound = ActionBoundSubset(action=_dihedral_d4(), positions=(3, 0))

    restored = ActionBoundSubset.model_validate(bound.model_dump())
    json_restored = ActionBoundSubset.model_validate_json(bound.model_dump_json())

    assert restored == bound
    assert json_restored == bound
    assert {bound.action.domain[position] for position in bound.positions} == {
        "v0",
        "v3",
    }


def test_same_positions_under_different_actions_are_different_values() -> None:
    under_c3 = ActionBoundSubset(action=_cyclic_c3(), positions=(0,))
    under_d4 = ActionBoundSubset(action=_dihedral_d4(), positions=(0,))
    relabelled_c3 = ActionBoundSubset(
        action=_cyclic_c3(("x", "y", "z")), positions=(0,)
    )

    assert under_c3 != under_d4
    assert under_c3 != relabelled_c3
    # The same positions denote different labels under the different domains.
    assert under_c3.action.domain[0] != under_d4.action.domain[0]
    assert under_c3.action.domain[0] != relabelled_c3.action.domain[0]


def test_positions_valid_only_under_the_original_action_rejected_under_another() -> (
    None
):
    first = compute_subset_canonicalization(_request(_dihedral_d4(), (0, 1, 2, 3)))
    positions = first.canonical_subset.positions
    assert positions == (0, 1, 2, 3)

    smaller_domain = _cyclic_c3()
    with pytest.raises(ValidationError) as exc_info:
        ActionBoundSubset(action=smaller_domain, positions=positions)
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_position_out_of_range"
    )
    with pytest.raises(ValidationError) as exc_info:
        _request(smaller_domain, positions)
    _assert_error_type(
        exc_info.value, "finite_group_action.subset_position_out_of_range"
    )


def test_result_rejects_canonical_subset_bound_to_a_different_action() -> None:
    result = compute_subset_canonicalization(_request(_cyclic_c3(), (2,)))
    forged = result.model_dump()
    other_action = _symmetric_s3()
    forged["canonical_subset"] = {
        "action": other_action.model_dump(),
        "positions": list(result.canonical_subset.positions),
    }

    with pytest.raises(ValidationError) as exc_info:
        SubsetCanonicalizationResult.model_validate(forged)
    _assert_error_type(
        exc_info.value, "finite_group_action.canonical_subset_action_mismatch"
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("source_subset", [1], "finite_group_action.transporter_mismatch"),
        ("canonical_subset", [1], "finite_group_action.canonical_subset_not_minimal"),
        ("transporter", [0, 1, 2], "tuple_type"),
        ("orbit_size", 2, "finite_group_action.orbit_size_mismatch"),
        ("stabilizer_size", 2, "finite_group_action.stabilizer_size_mismatch"),
    ],
)
def test_result_rejects_independently_forged_source_and_conclusions(
    field: str,
    value: list[int] | int,
    error_type: str,
) -> None:
    result = compute_subset_canonicalization(_request(_cyclic_c3(), (2,)))
    forged = result.model_dump()
    if isinstance(value, list):
        forged[field] = {
            "action": forged["source_subset"]["action"],
            "positions": value,
        }
    else:
        forged[field] = value

    with pytest.raises(ValidationError) as exc_info:
        SubsetCanonicalizationResult.model_validate(forged)
    _assert_error_type(exc_info.value, error_type)


def test_public_declaration_exposes_and_executes_copyable_example() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "group_action.subset.canonicalize"
    )
    assert operation.examples
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)

    assert isinstance(result, SubsetCanonicalizationResult)
    assert result.canonical_subset.positions == (0,)
    schema = operation.request_type.model_json_schema()
    bound_schema = schema["$defs"]["ActionBoundSubset"]
    assert bound_schema["properties"]["positions"]["maxItems"] == 50
    assert (
        "generated group must have order at most 10000"
        in schema["properties"]["subset"]["description"]
    )
    # The copyable example must advertise the implemented group-order bound.
    assert (
        f"generated group must have order at most {MAX_GROUP_ORDER}"
        in operation.examples[0].description
    )
    # The example is one copyable request whose subset carries its action.
    assert set(operation.examples[0].input) == {"subset"}
    assert set(operation.examples[0].input["subset"]) == {"action", "positions"}
