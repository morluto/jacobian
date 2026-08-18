"""Tests for finite-state transducer operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.finite_state_transducers._models import (
    ComposeRequest,
    IdentityRequest,
    RelationInverseRequest,
    RelationPathReplayRequest,
)
from jacobian.math.finite_state_transducers._operations import (
    compute_compose,
    compute_identity,
    compute_relation_inverse,
    compute_relation_path_replay,
)
from jacobian.math.finite_state_transducers.operations import (
    compose_subsequential,
    identity_transducer,
    run_subsequential,
    trim_subsequential,
)
from jacobian.math.finite_state_transducers.values import (
    RationalEdge,
    RationalTransducer,
    SubseqFinalOutput,
    SubseqTransition,
    SubsequentialTransducer,
)


def _bit_flip_transducer() -> SubsequentialTransducer:
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


def _partial_transducer() -> SubsequentialTransducer:
    return SubsequentialTransducer(
        input_alphabet_size=2,
        output_alphabet_size=2,
        state_count=2,
        initial_state=0,
        transitions=(
            SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=1, input_symbol=0, target=1, output=(0,)),
            SubseqTransition(source=1, input_symbol=1, target=1, output=(1,)),
        ),
        final_outputs=(SubseqFinalOutput(state=1, output=()),),
    )


class TestSubseqRun:
    def test_identity_run(self):
        transducer = identity_transducer(2)
        result = run_subsequential(transducer, (0, 1, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == (0, 1, 0)
        assert result[2] == 0

    def test_bit_flip_run(self):
        transducer = _bit_flip_transducer()
        result = run_subsequential(transducer, (0, 1, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == (1, 0, 1)

    def test_undefined_transition(self):
        transducer = _partial_transducer()
        result = run_subsequential(transducer, (1,))
        assert result[0] == "UNDEFINED_TRANSITION"
        assert result[3] == 0
        assert result[4] == ()

    def test_nonfinal_domain_state(self):
        transducer = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=()),
                SubseqTransition(source=0, input_symbol=1, target=0, output=()),
            ),
            final_outputs=(),
        )
        result = run_subsequential(transducer, (0,))
        assert result[0] == "NONFINAL_DOMAIN_STATE"
        assert result[2] == 0

    def test_empty_output_distinct_from_undefined(self):
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
        result = run_subsequential(transducer, (0, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == ()

    def test_final_output_appended(self):
        transducer = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=(1,)),),
        )
        result = run_subsequential(transducer, (0, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == (0, 0, 1)


class TestIdentity:
    def test_identity_2(self):
        transducer = identity_transducer(2)
        assert transducer.state_count == 1
        assert transducer.input_alphabet_size == 2
        assert len(transducer.transitions) == 2
        result = run_subsequential(transducer, (0, 1))
        assert result[0] == "OUTPUT"
        assert result[1] == (0, 1)

    def test_identity_via_request(self):
        result = compute_identity(IdentityRequest(alphabet_size=3))
        transducer = result.transducer
        run_result = run_subsequential(transducer, (0, 1, 2))
        assert run_result[0] == "OUTPUT"
        assert run_result[1] == (0, 1, 2)


class TestComposition:
    def test_compose_identity_with_bit_flip(self):
        identity = identity_transducer(2)
        flip = _bit_flip_transducer()
        composed = compose_subsequential(identity, flip)
        result = run_subsequential(composed, (0, 1, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == (1, 0, 1)

    def test_compose_flip_with_flip_is_identity(self):
        flip = _bit_flip_transducer()
        composed = compose_subsequential(flip, flip)
        result = run_subsequential(composed, (0, 1, 0))
        assert result[0] == "OUTPUT"
        assert result[1] == (0, 1, 0)

    def test_compose_shrinks_domain(self):
        t = SubsequentialTransducer(
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
        u = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        composed = compose_subsequential(t, u)
        result0 = run_subsequential(composed, (0,))
        assert result0[0] == "OUTPUT"
        assert result0[1] == (0,)
        result1 = run_subsequential(composed, (1,))
        assert result1[0] == "UNDEFINED_TRANSITION"

    def test_compose_via_request(self):
        identity = identity_transducer(2)
        flip = _bit_flip_transducer()
        result = compute_compose(ComposeRequest(first=identity, second=flip))
        run = run_subsequential(result.transducer, (0, 1))
        assert run[0] == "OUTPUT"
        assert run[1] == (1, 0)


class TestTrim:
    def test_trim_removes_dead_states(self):
        transducer = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=3,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=1, output=(0,)),
                SubseqTransition(source=0, input_symbol=1, target=0, output=(1,)),
                SubseqTransition(source=1, input_symbol=0, target=1, output=(1,)),
                SubseqTransition(source=2, input_symbol=0, target=2, output=(0,)),
                SubseqTransition(source=2, input_symbol=1, target=0, output=(1,)),
            ),
            final_outputs=(
                SubseqFinalOutput(state=0, output=()),
                SubseqFinalOutput(state=1, output=()),
            ),
        )
        trimmed, state_map = trim_subsequential(transducer)
        assert trimmed.state_count <= 3
        assert 2 not in state_map
        result = run_subsequential(trimmed, (0,))
        assert result[0] == "OUTPUT"


class TestRationalRelation:
    def test_inverse_swaps_labels(self):
        transducer = RationalTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_states=(0,),
            accepting_states=(0,),
            edges=(
                RationalEdge(source=0, target=0, input_label=(0,), output_label=(1,)),
                RationalEdge(source=0, target=0, input_label=(1,), output_label=(0,)),
            ),
        )
        inverse = compute_relation_inverse(
            RelationInverseRequest(transducer=transducer)
        )
        inv = inverse.transducer
        edge0 = inv.edges[0]
        assert edge0.input_label == (1,)
        assert edge0.output_label == (0,)

    def test_path_replay_accepting(self):
        transducer = RationalTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=1,
            initial_states=(0,),
            accepting_states=(0,),
            edges=(
                RationalEdge(source=0, target=0, input_label=(0,), output_label=(1,)),
            ),
        )
        result = compute_relation_path_replay(
            RelationPathReplayRequest(transducer=transducer, edge_path=(0, 0))
        )
        assert result.status == "ACCEPTING_PAIR"
        assert result.input_word == (0, 0)
        assert result.output_word == (1, 1)

    def test_path_replay_invalid_edge(self):
        transducer = RationalTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=1,
            initial_states=(0,),
            accepting_states=(0,),
            edges=(
                RationalEdge(source=0, target=0, input_label=(0,), output_label=(0,)),
            ),
        )
        result = compute_relation_path_replay(
            RelationPathReplayRequest(transducer=transducer, edge_path=(5,))
        )
        assert result.status == "INVALID_PATH"


class TestValidation:
    def test_duplicate_transition_rejected(self):
        with pytest.raises(ValidationError):
            SubsequentialTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_state=0,
                transitions=(
                    SubseqTransition(source=0, input_symbol=0, target=0, output=()),
                    SubseqTransition(source=0, input_symbol=0, target=0, output=()),
                ),
                final_outputs=(),
            )

    def test_output_symbol_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SubsequentialTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_state=0,
                transitions=(
                    SubseqTransition(source=0, input_symbol=0, target=0, output=(5,)),
                ),
                final_outputs=(),
            )

    def test_both_labels_empty_edge_rejected(self):
        with pytest.raises(ValidationError):
            RationalTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_states=(0,),
                accepting_states=(0,),
                edges=(
                    RationalEdge(source=0, target=0, input_label=(), output_label=()),
                ),
            )

    def test_compose_mismatched_alphabets_rejected(self):
        with pytest.raises(ValidationError):
            ComposeRequest(
                first=identity_transducer(2),
                second=identity_transducer(3),
            )
