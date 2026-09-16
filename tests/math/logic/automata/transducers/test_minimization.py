"""Tests for subsequential transducer minimization (#3762)."""

from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.logic.automata.transducers._models import (
    MinimizeRequest,
    MinimizeResult,
)
from jacobian.math.logic.automata.transducers._tools import TOOLS, compute_minimize
from jacobian.math.logic.automata.transducers.operations import (
    minimize_subsequential,
    run_subsequential,
    verify_minimization,
)
from jacobian.math.logic.automata.transducers.values import (
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _three_state_machine() -> SubsequentialTransducer:
    """Three live states with 0 and 1 behaviorally equivalent."""
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=3,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=0, input_symbol=1, target=2, output=(1,)),
            SubseqTransition(source=1, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=1, input_symbol=1, target=2, output=(1,)),
            SubseqTransition(source=2, input_symbol=0, target=2, output=(0,)),
            SubseqTransition(source=2, input_symbol=1, target=2, output=(1,)),
        ),
        final_outputs=(
            SubseqFinalOutput(state=0, output=()),
            SubseqFinalOutput(state=1, output=()),
            SubseqFinalOutput(state=2, output=(0,)),
        ),
    )


def _run_from(
    transducer: SubsequentialTransducer, state: int, word: tuple[int, ...]
) -> tuple[bool, tuple[int, ...]]:
    """The realized partial function: definedness plus output word.

    ``run_subsequential`` reports ``UNDEFINED_TRANSITION`` and
    ``NONFINAL_DOMAIN_STATE`` as distinct statuses, but both mean the word is
    outside the state's partial function.  Comparing raw statuses would treat
    two undefined runs as a separation, so collapse to definedness.
    """
    relocated = transducer.model_copy(update={"initial_state": state})
    status, output, _, _, _ = run_subsequential(relocated, word)
    return status == "OUTPUT", output


def _nonfinal_target_machine() -> SubsequentialTransducer:
    """States 1 and 2 differ only on a symbol whose target is not final.

    State 1 has a 0-transition into the non-final state 0; state 2 has no
    0-transition.  The one-symbol word ``(0,)`` is outside both partial
    functions, so it does not separate them; a word reaching a final state
    through state 0 is required.
    """
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=3,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=()),
            SubseqTransition(source=0, input_symbol=1, target=2, output=()),
            SubseqTransition(source=1, input_symbol=0, target=0, output=()),
            SubseqTransition(source=1, input_symbol=1, target=1, output=()),
            SubseqTransition(source=2, input_symbol=1, target=2, output=()),
        ),
        final_outputs=(
            SubseqFinalOutput(state=1, output=()),
            SubseqFinalOutput(state=2, output=()),
        ),
    )


class TestKnownAnswer:
    def test_three_states_minimize_to_two(self) -> None:
        result = minimize_subsequential(_three_state_machine(), 3)

        assert isinstance(result, MinimizeResult)
        assert result.minimized.state_count == 2
        assert result.partition == ((0, 1), (2,))
        assert result.old_to_new == (0, 0, 1)
        assert result.new_to_old == (0, 2)

    def test_distinguishability_table_marks_only_equivalent_pair(self) -> None:
        result = minimize_subsequential(_three_state_machine(), 3)
        table = {
            (row.first_state, row.second_state): row
            for row in result.distinguishability
        }

        assert set(table) == {(0, 1), (0, 2), (1, 2)}
        assert table[(0, 1)].equivalent
        assert table[(0, 1)].witness_word == ()
        assert not table[(0, 2)].equivalent
        assert not table[(1, 2)].equivalent

    def test_already_minimal_machine_is_unchanged(self) -> None:
        source = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
                SubseqTransition(source=0, input_symbol=1, target=0, output=(1,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        result = minimize_subsequential(source, 2)

        assert result.minimized.state_count == 1
        assert result.partition == ((0,),)
        assert result.distinguishability == ()


class TestPreservationReplay:
    def test_minimized_machine_agrees_on_an_extended_sample(self) -> None:
        source = _three_state_machine()
        result = minimize_subsequential(source, 3)

        for length in range(7):
            for raw in product((0, 1), repeat=length):
                word = tuple(raw)
                source_status, source_output, _, _, _ = run_subsequential(source, word)
                minimized_status, minimized_output, _, _, _ = run_subsequential(
                    result.minimized, word
                )
                assert (source_status, source_output) == (
                    minimized_status,
                    minimized_output,
                ), word

    def test_separating_words_distinguish_inequivalent_states(self) -> None:
        source = _three_state_machine()
        result = minimize_subsequential(source, 3)
        finals = {entry.state: entry.output for entry in source.final_outputs}

        for row in result.distinguishability:
            if row.equivalent:
                continue
            if not row.witness_word:
                assert finals.get(row.first_state) != finals.get(row.second_state)
            else:
                assert _run_from(
                    source, row.first_state, row.witness_word
                ) != _run_from(source, row.second_state, row.witness_word)

    def test_sample_size_matches_the_requested_budget(self) -> None:
        result = minimize_subsequential(_three_state_machine(), 4)

        assert result.sample_words_checked == 1 + 2 + 4 + 8 + 16
        assert result.sample_agreement


class TestWitnessSoundness:
    def test_witness_is_defined_and_separates(self) -> None:
        source = _nonfinal_target_machine()
        result = minimize_subsequential(source, 4)
        table = {
            (row.first_state, row.second_state): row
            for row in result.distinguishability
        }
        row = table[(1, 2)]

        assert not row.equivalent
        assert row.witness_word
        assert _run_from(source, row.first_state, row.witness_word) != _run_from(
            source, row.second_state, row.witness_word
        )

    def test_every_non_equivalent_witness_separates_its_pair(self) -> None:
        for source in (_three_state_machine(), _nonfinal_target_machine()):
            result = minimize_subsequential(source, 4)
            finals = {entry.state: entry.output for entry in source.final_outputs}
            for row in result.distinguishability:
                if row.equivalent:
                    continue
                assert row.witness_word or (
                    finals.get(row.first_state) != finals.get(row.second_state)
                )
                if row.witness_word:
                    assert _run_from(
                        source, row.first_state, row.witness_word
                    ) != _run_from(source, row.second_state, row.witness_word)


class TestBoundary:
    def test_oversized_sample_budget_is_rejected(self) -> None:
        wide = SubsequentialTransducer(
            input_alphabet_size=32,
            output_alphabet_size=2,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=()),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        with pytest.raises(OperationResourceAdmissionError, match="sample"):
            minimize_subsequential(wide, 12)

    def test_state_count_over_cap_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SubsequentialTransducer.model_validate(
                {
                    "input_alphabet_size": 1,
                    "output_alphabet_size": 1,
                    "state_count": 65,
                    "initial_state": 0,
                    "transitions": [],
                    "final_outputs": [],
                }
            )

    def test_empty_function_minimizes_to_the_placeholder(self) -> None:
        source = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=2,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
            ),
            final_outputs=(),
        )
        result = minimize_subsequential(source, 2)

        assert result.minimized.state_count == 1
        assert result.minimized.transitions == ()
        assert result.minimized.final_outputs == ()
        assert result.partition == ()
        assert result.distinguishability == ()
        assert result.old_to_new == (-1, -1)


class TestAdversarial:
    def test_duplicate_deterministic_transition_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SubsequentialTransducer.model_validate(
                {
                    "input_alphabet_size": 1,
                    "output_alphabet_size": 1,
                    "state_count": 1,
                    "initial_state": 0,
                    "transitions": [
                        {"source": 0, "input_symbol": 0, "target": 0, "output": []},
                        {"source": 0, "input_symbol": 0, "target": 0, "output": []},
                    ],
                    "final_outputs": [{"state": 0, "output": []}],
                }
            )

    def test_out_of_range_target_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SubsequentialTransducer.model_validate(
                {
                    "input_alphabet_size": 1,
                    "output_alphabet_size": 1,
                    "state_count": 1,
                    "initial_state": 0,
                    "transitions": [
                        {"source": 0, "input_symbol": 0, "target": 7, "output": []},
                    ],
                    "final_outputs": [{"state": 0, "output": []}],
                }
            )

    def test_dead_state_is_dropped_before_merging(self) -> None:
        source = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=3,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
                SubseqTransition(source=0, input_symbol=1, target=1, output=(1,)),
                SubseqTransition(source=1, input_symbol=0, target=1, output=(0,)),
                SubseqTransition(source=1, input_symbol=1, target=1, output=(1,)),
                SubseqTransition(source=2, input_symbol=0, target=2, output=(0,)),
                SubseqTransition(source=2, input_symbol=1, target=2, output=(1,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        result = minimize_subsequential(source, 2)

        assert result.partition == ((0,),)
        assert result.old_to_new == (0, -1, -1)
        assert result.minimized.state_count == 1


class TestNativeCatalogParity:
    def test_native_kernel_and_owner_adapter_agree(self) -> None:
        request = MinimizeRequest(
            transducer=_three_state_machine(), sample_max_length=3
        )

        assert compute_minimize(request) == minimize_subsequential(
            request.transducer, request.sample_max_length
        )

    def test_declared_example_executes_through_the_owner_adapter(self) -> None:
        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "transducer.subsequential.minimize.compute"
        )
        request = operation.request_type.model_validate_json(
            encode_strict_json(operation.examples[0].input), strict=True
        )
        result = operation.run(request)

        assert result.minimized.state_count == 2
        assert result.partition == ((0, 1), (2,))


class TestSerialization:
    def test_strict_json_round_trip_preserves_the_certificate(self) -> None:
        result = compute_minimize(
            MinimizeRequest(transducer=_three_state_machine(), sample_max_length=3)
        )
        decoded = MinimizeResult.model_validate_json(result.model_dump_json())

        assert decoded == result
        assert verify_minimization(decoded)

    def test_forged_minimization_does_not_verify(self) -> None:
        result = compute_minimize(
            MinimizeRequest(transducer=_three_state_machine(), sample_max_length=3)
        )
        forged = result.model_copy(
            update={"minimized": result.transducer},
            deep=True,
        )

        assert not verify_minimization(forged)
