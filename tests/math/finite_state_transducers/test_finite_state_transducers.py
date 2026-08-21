"""Known-answer and adversarial tests for finite-state transducers."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_state_transducers import (
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
    compose_subsequential,
    identity_transducer,
    invert_rational,
    replay_rational_path,
    run_subsequential,
    trim_subsequential,
)
from jacobian.math.finite_state_transducers._models import (
    ComposeRequest,
    RelationPathReplayRequest,
    RelationPathReplayResult,
    SubseqRunRequest,
    SubseqRunResult,
)
from jacobian.math.finite_state_transducers._operations import (
    compute_compose,
    compute_relation_path_replay,
    compute_run,
)
from jacobian.math.finite_state_transducers._tools import TOOLS


def _flip() -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=1,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=0, output=(1,)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=(0,)),
        ),
        final_outputs=(SubseqFinalOutput(state=0, output=()),),
    )


def _relation(*, initial_states: tuple[int, ...] = (0,)) -> RationalTransducer:
    return RationalTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=2,
        initial_states=initial_states,
        accepting_states=(1,),
        edges=(
            RationalEdge(source=0, target=1, input_label=(0,), output_label=(1,)),
            RationalEdge(source=1, target=1, input_label=(1,), output_label=(0,)),
        ),
    )


class TestSubsequentialRun:
    def test_successful_empty_output_is_distinct(self) -> None:
        transducer = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=()),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )

        assert run_subsequential(transducer, (0, 0)) == (
            "OUTPUT",
            (),
            0,
            None,
            (),
        )

    def test_undefined_transition_preserves_partial_trace(self) -> None:
        transducer = _flip()

        status, output, state, position, partial = run_subsequential(
            transducer.model_copy(update={"transitions": transducer.transitions[:1]}),
            (0, 1),
        )

        assert (status, output, state, position, partial) == (
            "UNDEFINED_TRANSITION",
            (),
            0,
            1,
            (1,),
        )

    def test_nonfinal_state_is_not_a_function_value(self) -> None:
        transducer = _flip().model_copy(update={"final_outputs": ()})

        assert run_subsequential(transducer, (0,))[0] == "NONFINAL_DOMAIN_STATE"

    def test_adapter_binds_transducer_and_word(self) -> None:
        request = SubseqRunRequest(transducer=_flip(), word=(0, 1))
        result = compute_run(request)

        assert result.transducer == request.transducer
        assert result.word == request.word
        assert result.output == (1, 0)

    def test_false_run_result_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exact bound"):
            SubseqRunResult(
                transducer=_flip(),
                word=(0,),
                status="OUTPUT",
                output=(0,),
                final_state=0,
                undefined_position=None,
                partial_output=(),
            )

    def test_native_run_rejects_symbol_outside_alphabet(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            run_subsequential(_flip(), (2,))


class TestComposition:
    def test_flip_after_flip_is_identity(self) -> None:
        composite = compose_subsequential(_flip(), _flip())

        assert run_subsequential(composite, (0, 1, 0))[1] == (0, 1, 0)

    def test_second_nonfinal_after_first_final_output_rejects_word(self) -> None:
        first = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=1,
            initial_state=0,
            transitions=(),
            final_outputs=(SubseqFinalOutput(state=0, output=(0,)),),
        )
        second = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=2,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=1, output=()),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )

        composite = compose_subsequential(first, second)

        assert composite.final_outputs == ()
        assert run_subsequential(composite, ())[0] == "NONFINAL_DOMAIN_STATE"

    def test_product_state_bound_is_rejected_before_composition(self) -> None:
        large = _flip().model_copy(update={"state_count": 9})

        with pytest.raises(ValidationError, match="product-state"):
            ComposeRequest(first=large, second=large)

    def test_adapter_binds_both_operands(self) -> None:
        request = ComposeRequest(first=identity_transducer(2), second=_flip())
        result = compute_compose(request)

        assert result.first == request.first
        assert result.second == request.second
        assert run_subsequential(result.transducer, (0, 1))[1] == (1, 0)


class TestNativeTransformations:
    def test_identity_is_exact(self) -> None:
        assert run_subsequential(identity_transducer(3), (0, 1, 2))[1] == (0, 1, 2)

    def test_trim_removes_unreachable_state(self) -> None:
        source = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=2,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
                SubseqTransition(source=1, input_symbol=0, target=1, output=(0,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )

        trimmed, state_map = trim_subsequential(source)

        assert trimmed.state_count == 1
        assert state_map == {0: 0}

    def test_rational_inverse_swaps_labels_and_alphabets(self) -> None:
        inverse = invert_rational(_relation())

        assert inverse.edges[0].input_label == (1,)
        assert inverse.edges[0].output_label == (0,)

    def test_only_audited_outcomes_are_public(self) -> None:
        assert {tool.operation_id for tool in TOOLS} == {
            "transducer.relation.path.replay.compute",
            "transducer.subsequential.compose.compute",
            "transducer.subsequential.run.compute",
        }


class TestRationalPathReplay:
    def test_accepting_path_binds_selected_initial_state(self) -> None:
        request = RelationPathReplayRequest(
            transducer=_relation(), initial_state=0, edge_path=(0, 1)
        )

        result = compute_relation_path_replay(request)

        assert result.status == "ACCEPTING_PAIR"
        assert result.input_word == (0, 1)
        assert result.output_word == (1, 0)
        assert result.state_trace == (0, 1, 1)

    def test_empty_path_uses_explicit_initial_state(self) -> None:
        relation = _relation(initial_states=(0, 1))

        assert replay_rational_path(relation, 0, ())[0] == "INVALID_PATH"
        assert replay_rational_path(relation, 1, ())[0] == "ACCEPTING_PAIR"

    def test_discontinuous_or_out_of_range_path_is_invalid_not_accepting(self) -> None:
        relation = _relation()

        assert replay_rational_path(relation, 0, (1,))[0] == "INVALID_PATH"
        assert replay_rational_path(relation, 0, (9,))[0] == "INVALID_PATH"

    def test_false_replay_result_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exact bound"):
            RelationPathReplayResult(
                transducer=_relation(),
                initial_state=0,
                edge_path=(0,),
                status="INVALID_PATH",
                input_word=(),
                output_word=(),
                state_trace=(),
                error="invented",
            )


class TestValueValidation:
    def test_duplicate_deterministic_transition_is_rejected(self) -> None:
        transition = SubseqTransition(source=0, input_symbol=0, target=0, output=())
        with pytest.raises(ValidationError, match="duplicate"):
            SubsequentialTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_state=0,
                transitions=(transition, transition),
                final_outputs=(),
            )

    def test_duplicate_initial_and_accepting_states_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="initial states"):
            _relation(initial_states=(0, 0))

    def test_empty_rational_edge_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both labels empty"):
            RationalTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_states=(0,),
                accepting_states=(0,),
                edges=(RationalEdge(source=0, target=0),),
            )

    def test_replay_must_select_a_declared_initial_state(self) -> None:
        with pytest.raises(ValidationError, match="select"):
            RelationPathReplayRequest(
                transducer=_relation(), initial_state=1, edge_path=()
            )
