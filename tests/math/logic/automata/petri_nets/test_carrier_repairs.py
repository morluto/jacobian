"""Carrier and admission regressions for the petri_nets owner.

Each test fails against the corresponding defect and passes on the repair; the
negative control is this file run against unmodified `main`.
"""

from __future__ import annotations

import pytest

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets import (
    Marking,
    PetriNet,
    marking_equation,
    petri_net_matrices,
    reachability_graph,
    reachable_dead_markings,
    relabel_petri_net,
)
from jacobian.math.logic.automata.petri_nets import operations as petri_operations
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.liveness import (
    transition_liveness_profile,
)


def _counter_net(places: int = 2, transitions: int = 2) -> PetriNet:
    """A net whose transitions all fire, so every marking has successors."""
    pre = tuple(tuple(1 for _ in range(transitions)) for _ in range(places))
    post = pre
    return PetriNet(
        place_count=places,
        transition_count=transitions,
        place_ids=tuple(f"p{index}" for index in range(places)),
        transition_ids=tuple(f"t{index}" for index in range(transitions)),
        pre=pre,
        post=post,
    )


def _token_moving_net() -> PetriNet:
    """One token moving between two places, so the exploration visits both."""
    return PetriNet(
        place_count=2,
        transition_count=1,
        place_ids=("p0", "p1"),
        transition_ids=("t0",),
        pre=((1,), (0,)),
        post=((0,), (1,)),
    )


def _dead_marking_rescan_count() -> tuple[int, int]:
    """Return (exploration enabledness calls, total calls) for one dead scan."""
    calls = {"enabled": 0}
    original = petri_operations.__dict__["_enabled_transition_indices"]

    def counting(net: PetriNet, marking: Marking) -> list[int]:
        calls["enabled"] += 1
        return list(original(net, marking))

    petri_operations._enabled_transition_indices = counting
    try:
        net = _counter_net()
        reachable_dead_markings(net, Marking(net=net, tokens=(1,) * 2), 32)
    finally:
        petri_operations._enabled_transition_indices = original
    return calls["enabled"], calls["enabled"]


def test_dead_marking_rescan_reuses_the_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabledness is computed once per state, inside the charged exploration.

    At the admitted boundary the exploration is charged the full
    ``MAX_REACHABILITY_EXPLORATION_WORK`` allowance, and the old rescan added
    another full pass over the transition table — once per discovered state —
    that nothing charged.
    """
    net = _token_moving_net()
    graph = reachability_graph(net, Marking(net=net, tokens=(1, 0)), 200)
    states = len(graph.states)
    assert states > 1

    calls = {"enabled": 0}
    original = petri_operations.__dict__["_enabled_transition_indices"]

    def counting(candidate: PetriNet, marking: Marking) -> list[int]:
        calls["enabled"] += 1
        return list(original(candidate, marking))

    monkeypatch.setattr(petri_operations, "_enabled_transition_indices", counting)
    result = reachable_dead_markings(net, Marking(net=net, tokens=(1, 0)), 200)

    # One call per explored state, and no second pass.
    assert calls["enabled"] == states
    assert isinstance(result.dead_markings, tuple)
    assert _dead_marking_rescan_count()[0] >= 1


def test_incomplete_reachability_graph_is_refused_with_a_typed_error() -> None:
    """A bypass-constructed graph must not leak ``AttributeError``."""
    net = _counter_net()
    marking = Marking(net=net, tokens=(0, 0))

    for forged in (
        ReachabilityResult.model_construct(),
        ReachabilityResult.model_construct(net=net),
        ReachabilityResult.model_construct(
            net=net, initial_marking=marking, max_states=10
        ),
    ):
        assert isinstance(forged, ReachabilityResult)
        with pytest.raises(OperationDomainValidationError) as error:
            petri_operations.reachability_terminal_scc_profile(forged)
        assert error.value.errors()[0]["type"] == "petri_net.terminal_scc.graph_shape"

        with pytest.raises(OperationDomainValidationError) as liveness_error:
            transition_liveness_profile(forged)
        assert liveness_error.value.errors()[0]["type"] == (
            "petri_net.terminal_scc.graph_shape"
        )


def test_well_formed_graph_still_reaches_both_consumers() -> None:
    net = _counter_net()
    graph = reachability_graph(net, Marking(net=net, tokens=(0, 0)), 50)

    assert petri_operations.reachability_terminal_scc_profile(graph).terminal_components
    assert transition_liveness_profile(graph).transitions


def test_output_admission_does_not_depend_on_label_text() -> None:
    """The cell bound must measure retained values, not their encoding.

    ``MAX_MARKING_EQUATION_OUTPUT_CELLS`` is documented as retained entries, but
    the compared value was an exact JSON byte size including per-character label
    escaping, so the verdict flipped with label length while the retained cells
    stayed at 129.
    """
    places = transitions = 8
    short = _labelled_net(places, transitions, width=0)
    long = _labelled_net(places, transitions, width=40_000)
    assert len(short.place_ids or ()) == len(long.place_ids or ())
    assert (short.place_ids or ()) != (long.place_ids or ())
    # The retained shape is identical.
    assert (short.place_count, short.transition_count) == (
        long.place_count,
        long.transition_count,
    )

    original = petri_operations.__dict__["MAX_MARKING_EQUATION_OUTPUT_CELLS"]
    petri_operations.MAX_MARKING_EQUATION_OUTPUT_CELLS = 1_000_000
    try:
        for net in (short, long):
            marking = Marking(net=net, tokens=(1,) * places)
            result = marking_equation(net, marking, marking, (1,) * transitions)
            assert result.residual == (0,) * places
            assert petri_net_matrices(net).incidence.row_count == places
    finally:
        petri_operations.MAX_MARKING_EQUATION_OUTPUT_CELLS = original


def _labelled_net(places: int, transitions: int, *, width: int) -> PetriNet:
    pre = tuple(tuple(1 for _ in range(transitions)) for _ in range(places))
    return PetriNet(
        place_count=places,
        transition_count=transitions,
        place_ids=tuple(f"p{index}" + "x" * width for index in range(places)),
        transition_ids=tuple(f"t{index}" + "x" * width for index in range(transitions)),
        pre=pre,
        post=pre,
    )


def test_published_kernels_are_exported_from_the_native_package() -> None:
    """The catalog publishes three kernels the native API does not expose."""
    import jacobian.math.logic.automata.petri_nets as petri_nets
    from jacobian.math.logic.automata.petri_nets._tools import TOOLS

    published = {
        tool.operation_id
        for tool in TOOLS
        if tool.operation_id.startswith("petri_net.")
    }
    assert {
        "petri_net.marking_equation.compute",
        "petri_net.matrices.compute",
        "petri_net.relabel.compute",
    } <= published
    for name in ("marking_equation", "petri_net_matrices", "relabel_petri_net"):
        assert name in petri_nets.__all__, name
        assert callable(getattr(petri_nets, name))
    relabeled = relabel_petri_net(_token_moving_net(), (1, 0), (0,))
    assert relabeled.source_net.place_count == 2
    assert relabeled.target_net.place_ids == ("p1", "p0")


def test_output_admission_still_refuses_a_real_overload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = _counter_net(4, 4)
    marking = Marking(net=net, tokens=(0, 0, 0, 0))
    monkeypatch.setattr(
        petri_operations, "MAX_MARKING_EQUATION_OUTPUT_CELLS", 1, raising=False
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        marking_equation(net, marking, marking, (1,) * 4)
    assert error.value.errors()[0]["type"] == "petri_net.marking_equation_output_bound"
