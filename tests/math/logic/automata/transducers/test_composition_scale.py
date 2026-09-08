"""Reachable composition admission preserves exact transducer semantics."""

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers._models import ComposeRequest
from jacobian.math.logic.automata.transducers._tools import compute_compose
from jacobian.math.logic.automata.transducers.operations import (
    run_subsequential,
    verify_composition,
)
from jacobian.math.logic.automata.transducers.values import (
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _cycle(size: int) -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=1,
        output_alphabet_size=1,
        state_count=size,
        initial_state=0,
        transitions=tuple(
            SubseqTransition(
                source=i, input_symbol=0, target=(i + 1) % size, output=(0,)
            )
            for i in range(size)
        ),
        final_outputs=tuple(SubseqFinalOutput(state=i, output=()) for i in range(size)),
    )


@pytest.mark.parametrize("size", [9, 64])
def test_composition_admits_small_reachable_diagonal(size: int) -> None:
    machine = _cycle(size)
    result = compute_compose(ComposeRequest(first=machine, second=machine))
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.first == restored.second == machine
    assert restored.transducer.state_count == size
    for length in [0, 1, size - 1, size, size + 1]:
        word = (0,) * length
        assert run_subsequential(restored.transducer, word)[1] == word
    assert verify_composition(restored)


def test_composition_refuses_genuinely_large_reachable_product() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="reachable"):
        compute_compose(ComposeRequest(first=_cycle(8), second=_cycle(9)))
