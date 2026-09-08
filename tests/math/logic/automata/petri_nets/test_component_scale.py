"""Independent Petri components preserve minimal-family source axes."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.petri_nets._models import SiphonTrapRequest
from jacobian.math.logic.automata.petri_nets._tools import compute_siphon_trap
from jacobian.math.logic.automata.petri_nets.operations import (
    find_minimal_siphons,
    find_minimal_traps,
    verify_siphon_trap,
)
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def test_many_independent_cycles_are_admitted_with_original_axes() -> None:
    size = 64
    # Pair i with63-i, ensuring local component indexing differs from global axes.
    targets = tuple(size - 1 - i for i in range(size))
    net = PetriNet(
        place_count=size,
        transition_count=size,
        pre=tuple(tuple(int(i == j) for j in range(size)) for i in range(size)),
        post=tuple(
            tuple(int(i == targets[j]) for j in range(size)) for i in range(size)
        ),
    )
    result = compute_siphon_trap(SiphonTrapRequest(net=net))
    restored = type(result).model_validate_json(result.model_dump_json())
    expected = tuple((i, size - 1 - i) for i in range(size // 2))
    assert tuple(s.places for s in restored.siphons) == expected
    assert restored.siphons == restored.traps
    assert restored.net == net
    assert (
        find_minimal_siphons(net)
        == find_minimal_traps(net)
        == [frozenset(s) for s in expected]
    )
    assert verify_siphon_trap(restored)


def test_coupled_component_retains_exponential_refusal() -> None:
    size = 21
    net = PetriNet(
        place_count=size,
        transition_count=size,
        pre=tuple(
            tuple(int((i * 7 + j * 3) % 11 < 3) for j in range(size))
            for i in range(size)
        ),
        post=tuple(
            tuple(int((i * 5 + j * 2) % 13 < 4) for j in range(size))
            for i in range(size)
        ),
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_siphon_trap(SiphonTrapRequest(net=net))


def test_component_kernel_matches_independent_exhaustive_subset_oracle() -> None:
    from itertools import combinations
    from random import Random

    random = Random(3498)
    for places in range(1, 8):
        for _ in range(12):
            transitions = random.randrange(7)
            net = PetriNet(
                place_count=places,
                transition_count=transitions,
                pre=tuple(
                    tuple(random.randrange(3) for _ in range(transitions))
                    for _ in range(places)
                ),
                post=tuple(
                    tuple(random.randrange(3) for _ in range(transitions))
                    for _ in range(places)
                ),
            )
            subsets = [
                frozenset(c)
                for size in range(1, places + 1)
                for c in combinations(range(places), size)
            ]

            def minimal(
                reverse: bool,
                net: PetriNet = net,
                subsets: list[frozenset[int]] = subsets,
            ) -> list[frozenset[int]]:
                valid = []
                for subset in subsets:
                    if all(
                        not any(
                            (net.pre if reverse else net.post)[p][t] for p in subset
                        )
                        or any((net.post if reverse else net.pre)[p][t] for p in subset)
                        for t in range(net.transition_count)
                    ):
                        valid.append(subset)
                return [a for a in valid if not any(b < a for b in valid)]

            assert find_minimal_siphons(net) == minimal(False)
            assert find_minimal_traps(net) == minimal(True)
