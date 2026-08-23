"""Contract tests for finite-action subset canonicalization."""

from itertools import combinations

import pytest
from pydantic import ValidationError
from sympy.combinatorics import Permutation, PermutationGroup

from jacobian.math.finite_group_actions._models import (
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
    action = _cyclic_c3()

    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=action, subset=(2,))
    )

    assert result.canonical_subset == (0,)
    assert tuple(result.transporter[index] for index in result.source_subset) == (0,)
    assert result.orbit_size == 3
    assert result.stabilizer_size == 1


@pytest.mark.property
@pytest.mark.parametrize("action", [_cyclic_c3(), _symmetric_s3(), _dihedral_d4()])
def test_every_small_subset_agrees_with_independent_sympy_enumeration(
    action: FinitePermutationAction,
) -> None:
    for size in range(len(action.domain) + 1):
        for subset in combinations(range(len(action.domain)), size):
            result = compute_subset_canonicalization(
                SubsetCanonicalizationRequest(action=action, subset=subset)
            )
            assert (
                result.canonical_subset,
                result.transporter,
                result.orbit_size,
                result.stabilizer_size,
            ) == _sympy_oracle(action, subset)
            assert _transport(result.transporter, subset) == result.canonical_subset
            assert result.orbit_size * result.stabilizer_size == len(
                _enumerate_group(action)
            )


def test_generator_reordering_preserves_canonical_result_and_transporter() -> None:
    first = _symmetric_s3()
    second = FinitePermutationAction(
        domain=first.domain,
        generators=tuple(reversed(first.generators)),
    )

    first_result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=first, subset=(2,))
    )
    second_result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=second, subset=(2,))
    )

    assert first_result.model_dump(exclude={"action"}) == second_result.model_dump(
        exclude={"action"}
    )


def test_position_preserving_domain_relabelling_preserves_result() -> None:
    first = _symmetric_s3(("z", "a", "m"))
    relabelled = _symmetric_s3(("triangle", "edge", "witness"))

    first_result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=first, subset=(1,))
    )
    relabelled_result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=relabelled, subset=(1,))
    )

    assert first_result.canonical_subset == (0,)
    assert first.domain[first_result.canonical_subset[0]] == "z"
    assert first_result.model_dump(exclude={"action"}) == relabelled_result.model_dump(
        exclude={"action"}
    )


@pytest.mark.parametrize("subset", [(), (0, 1, 2, 3)])
def test_empty_and_full_subsets_have_the_whole_group_as_stabilizer(
    subset: tuple[int, ...],
) -> None:
    action = _dihedral_d4()
    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=action, subset=subset)
    )

    assert result.source_subset == subset
    assert result.canonical_subset == subset
    assert result.transporter == (0, 1, 2, 3)
    assert result.orbit_size == 1
    assert result.stabilizer_size == 8


def test_fixed_nontrivial_subset_reports_its_setwise_stabilizer() -> None:
    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=_dihedral_d4(), subset=(0, 2))
    )

    assert result.canonical_subset == (0, 2)
    assert result.orbit_size == 2
    assert result.stabilizer_size == 4


def test_request_normalizes_subset_wire_order_by_domain_position() -> None:
    request = SubsetCanonicalizationRequest(action=_dihedral_d4(), subset=(3, 1))
    assert request.subset == (1, 3)


def test_request_rejects_duplicate_and_out_of_domain_positions() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        SubsetCanonicalizationRequest(action=_cyclic_c3(), subset=(1, 1))
    with pytest.raises(ValidationError, match="index the action domain"):
        SubsetCanonicalizationRequest(action=_cyclic_c3(), subset=(3,))
    with pytest.raises(ValidationError):
        SubsetCanonicalizationRequest(action=_cyclic_c3(), subset=(-1,))


def test_request_rejects_group_immediately_above_enumeration_bound() -> None:
    symmetric_s7 = FinitePermutationAction(
        domain=tuple(f"p{position}" for position in range(7)),
        generators=((1, 2, 3, 4, 5, 6, 0), (1, 0, 2, 3, 4, 5, 6)),
    )

    with pytest.raises(ValidationError, match=r"group order 5040.*maximum 720"):
        SubsetCanonicalizationRequest(action=symmetric_s7, subset=(0,))
    with pytest.raises(ValueError, match=r"group order exceeds.*720"):
        _enumerate_group(symmetric_s7)


def test_domain_and_subset_boundary_of_fifty_positions_is_accepted() -> None:
    action = FinitePermutationAction(
        domain=tuple(f"p{position}" for position in range(50)),
        generators=(tuple(range(50)),),
    )
    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(
            action=action,
            subset=tuple(reversed(range(50))),
        )
    )

    assert result.source_subset == tuple(range(50))
    assert result.canonical_subset == tuple(range(50))
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
    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=accepted, subset=(2, 0))
    )
    assert result.canonical_subset == (0, 2)
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

    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=action, subset=source_graph)
    )

    canonical_pairs = tuple(divmod(position, 6) for position in result.canonical_subset)
    assert len(canonical_pairs) == 6
    assert {source for source, _ in canonical_pairs} == set(range(6))
    assert all(source != destination for source, destination in canonical_pairs)
    assert _transport(result.transporter, source_graph) == result.canonical_subset
    assert result.transporter in _enumerate_group(action)
    assert result.orbit_size * result.stabilizer_size == 720


def test_canonical_subset_serializes_unchanged_into_the_next_request() -> None:
    first = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=_dihedral_d4(), subset=(1, 2))
    )
    dumped = first.model_dump()
    request = SubsetCanonicalizationRequest.model_validate(
        {
            "action": dumped["action"],
            "subset": dumped["canonical_subset"],
        }
    )
    second = compute_subset_canonicalization(request)

    assert request.subset == first.canonical_subset
    assert second.source_subset == first.canonical_subset
    assert second.canonical_subset == first.canonical_subset
    assert second.transporter == tuple(range(len(first.action.domain)))
    assert SubsetCanonicalizationResult.model_validate(dumped) == first


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_subset", [1], "transporter"),
        ("canonical_subset", [1], "canonical_subset"),
        ("transporter", [0, 1, 2], "transporter"),
        ("orbit_size", 2, "orbit_size"),
        ("stabilizer_size", 2, "stabilizer_size"),
    ],
)
def test_result_rejects_independently_forged_source_and_conclusions(
    field: str,
    value: list[int] | int,
    message: str,
) -> None:
    result = compute_subset_canonicalization(
        SubsetCanonicalizationRequest(action=_cyclic_c3(), subset=(2,))
    )
    forged = result.model_dump()
    forged[field] = value

    with pytest.raises(ValidationError, match=message):
        SubsetCanonicalizationResult.model_validate(forged)


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
    assert result.canonical_subset == (0,)
    schema = operation.request_type.model_json_schema()
    assert schema["properties"]["subset"]["maxItems"] == 50
    assert (
        "generated group must have order at most 720"
        in schema["properties"]["subset"]["description"]
    )
