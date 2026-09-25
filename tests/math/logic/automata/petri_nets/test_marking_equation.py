"""Exact state-equation residuals with independent arithmetic checks."""

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.logic.automata.petri_nets import marking_equation
from jacobian.math.logic.automata.petri_nets._models import MarkingEquationRequest
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet


def _oracle(
    net: PetriNet,
    source: tuple[int, ...],
    target: tuple[int, ...],
    counts: tuple[int, ...],
):
    formal = tuple(
        source[p]
        + sum(
            (net.post[p][t] - net.pre[p][t]) * counts[t]
            for t in range(net.transition_count)
        )
        for p in range(net.place_count)
    )
    residual = tuple(target[p] - formal[p] for p in range(net.place_count))
    return formal, residual


def test_equation_holds_without_implying_any_transition_can_fire():
    net = PetriNet(
        place_count=2,
        transition_count=2,
        pre=((1, 0), (0, 1)),
        post=((0, 1), (1, 0)),
    )
    result = marking_equation(
        net, Marking(tokens=(0, 0)), Marking(tokens=(0, 0)), (1, 1)
    )
    assert result.formal_target == (0, 0)
    assert result.residual == (0, 0)
    assert result.satisfies_equation
    assert (
        result.model_dump_json()
        == type(result).model_validate_json(result.model_dump_json()).model_dump_json()
    )
    assert MarkingEquationRequest(
        net=net,
        source_marking=Marking(tokens=(0, 0)),
        target_marking=Marking(tokens=(0, 0)),
        transition_counts=(1, 1),
    )


def test_nonzero_residual_is_complete_and_signed():
    net = PetriNet(
        place_count=2,
        transition_count=1,
        pre=((1,), (0,)),
        post=((0,), (1,)),
    )
    result = marking_equation(net, Marking(tokens=(1, 0)), Marking(tokens=(0, 0)), (1,))
    assert result.formal_target == (0, 1)
    assert result.residual == (0, -1)
    assert not result.satisfies_equation


def test_formal_target_scalar_bound_is_sharp():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((1000,),),
    )
    result = marking_equation(
        net, Marking(tokens=(1000,)), Marking(tokens=(1000,)), (1000,)
    )
    assert result.formal_target == (1_001_000,)
    assert result.residual == (-1_000_000,)
    assert not result.satisfies_equation


def test_small_nets_match_independent_residual_oracle():
    # Exhaust 1-place, 2-transition zero/one-arc nets, source and target
    # markings in {0,1,2}, and transition multisets in {0,1}^2.
    for pre in product(range(2), repeat=2):
        for post in product(range(2), repeat=2):
            net = PetriNet(
                place_count=1,
                transition_count=2,
                pre=(pre,),
                post=(post,),
            )
            for source, target in product(range(3), repeat=2):
                for counts in product(range(2), repeat=2):
                    expected_formal, expected_residual = _oracle(
                        net, (source,), (target,), counts
                    )
                    actual = marking_equation(
                        net,
                        Marking(tokens=(source,)),
                        Marking(tokens=(target,)),
                        counts,
                    )
                    assert actual.formal_target == expected_formal
                    assert actual.residual == expected_residual
                    assert actual.satisfies_equation == (expected_residual == (0,))


def test_public_operation_invocation_preserves_nonreachability_distinction():
    result = invoke_operation(
        "petri_net.marking_equation.compute",
        {
            "net": {
                "place_count": 2,
                "transition_count": 2,
                "pre": [[1, 0], [0, 1]],
                "post": [[0, 1], [1, 0]],
            },
            "source_marking": {"tokens": [0, 0]},
            "target_marking": {"tokens": [0, 0]},
            "transition_counts": [1, 1],
        },
        Catalog.open(),
    )
    assert result.output["residual"] == [0, 0]
    assert result.output["satisfies_equation"] is True


def test_occurrence_bound_is_admitted_before_matrix_vector_expansion():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    at_limit = marking_equation(
        net, Marking(tokens=(0,)), Marking(tokens=(0,)), (1000,)
    )
    assert at_limit.satisfies_equation
    with pytest.raises(OperationResourceAdmissionError):
        marking_equation(net, Marking(tokens=(0,)), Marking(tokens=(0,)), (1001,))
    with pytest.raises(OperationDomainValidationError):
        marking_equation(net, Marking(tokens=(0,)), Marking(tokens=(0,)), None)
    with pytest.raises(OperationDomainValidationError):
        marking_equation(net, Marking(tokens=(0,)), Marking(tokens=(0,)), [0])


def test_transition_counts_reject_approximate_numeric_input():
    net = PetriNet(
        place_count=1,
        transition_count=1,
        pre=((0,),),
        post=((0,),),
    )
    with pytest.raises(ValidationError):
        MarkingEquationRequest(
            net=net,
            source_marking=Marking(tokens=(0,)),
            target_marking=Marking(tokens=(0,)),
            transition_counts=(1.0,),
        )
