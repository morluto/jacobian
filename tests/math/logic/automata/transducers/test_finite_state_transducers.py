"""Known-answer and adversarial tests for finite-state transducers."""

import json

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.transducers import (
    FiniteAlphabet,
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
    verify_composition,
    verify_subsequential_run,
    word_morphism_to_subsequential,
)
from jacobian.math.logic.automata.transducers._models import (
    ComposeRequest,
    MinimizeRequest,
    MinimizeResult,
    RationalRelationInverseRequest,
    RelationPathReplayRequest,
    SubseqIdentityRequest,
    SubseqRunRequest,
    TrimRequest,
    TrimResult,
)
from jacobian.math.logic.automata.transducers._tools import (
    TOOLS,
    compute_compose,
    compute_identity,
    compute_minimize,
    compute_relation_inverse,
    compute_relation_path_replay,
    compute_run,
    compute_trim,
)
from jacobian.math.logic.automata.transducers.values import (
    MAX_FST_RESULT_WORD_LENGTH,
    MAX_FST_RUN_RESULT_BYTES,
)
from jacobian.math.logic.languages.words.operations import apply_morphism
from jacobian.math.logic.languages.words.values import FiniteWord, WordMorphism


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


def test_native_boundaries_reject_model_constructed_carriers() -> None:
    forged_subsequential = SubsequentialTransducer.model_construct()
    with pytest.raises(OperationDomainValidationError):
        run_subsequential(forged_subsequential, ())
    with pytest.raises(OperationDomainValidationError):
        compose_subsequential(forged_subsequential, _flip())

    forged_rational = RationalTransducer.model_construct()
    with pytest.raises(OperationDomainValidationError):
        replay_rational_path(forged_rational, 0, ())


class TestSubsequentialIdentityOperation:
    def test_identity_runs_match_direct_word_oracle_and_keep_parent(self) -> None:
        request = SubseqIdentityRequest(
            alphabet=FiniteAlphabet(symbols=("b", "a")), alphabet_id="two-letters"
        )
        result = compute_identity(request)
        assert result.input_alphabet == request.alphabet
        assert result.output_alphabet == request.alphabet
        assert result.input_alphabet_id == result.output_alphabet_id == "two-letters"
        assert result.transitions == (
            SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
            SubseqTransition(source=0, input_symbol=1, target=0, output=(1,)),
        )
        assert result.final_outputs == (SubseqFinalOutput(state=0, output=()),)

        words = [()]
        for length in range(1, 6):
            words.extend(
                tuple((bits >> shift) & 1 for shift in reversed(range(length)))
                for bits in range(1 << length)
            )
        for word in words:
            run = run_subsequential(result, word)
            assert run.status == "OUTPUT"
            assert run.output == word

    def test_identity_operation_manifest_and_alphabet_bound(self) -> None:
        tool = next(
            item
            for item in TOOLS
            if item.operation_id == "transducer.subsequential.identity.compute"
        )
        request = tool.request_type.model_validate(tool.examples[0].input)
        result = tool.run(request)
        assert result.input_alphabet == FiniteAlphabet(symbols=("a", "b"))

        assert len(identity_transducer(32).transitions) == 32
        with pytest.raises(OperationResourceAdmissionError) as error:
            identity_transducer(33)
        assert error.value.errors()[0]["type"] == (
            "finite_state_transducer.identity_alphabet_bound_exceeded"
        )


class TestWordMorphismTransducerConversion:
    def test_conversion_matches_independent_morphism_evaluator(self) -> None:
        morphism = WordMorphism(
            source_alphabet=("a", "b"),
            target_alphabet=("y", "x"),
            images=(("x", "y"), ()),
        )
        transducer = word_morphism_to_subsequential(morphism)
        assert transducer.input_alphabet == FiniteAlphabet(symbols=("a", "b"))
        assert transducer.output_alphabet == FiniteAlphabet(symbols=("y", "x"))
        assert tuple(edge.output for edge in transducer.transitions) == ((1, 0), ())
        assert transducer.final_outputs == (SubseqFinalOutput(state=0, output=()),)

        words = [()]
        for length in range(1, 5):
            words.extend(
                tuple((bits >> shift) & 1 for shift in reversed(range(length)))
                for bits in range(1 << length)
            )
        for indices in words:
            source_word = FiniteWord(
                alphabet=morphism.source_alphabet,
                letters=tuple(morphism.source_alphabet[index] for index in indices),
            )
            expected = apply_morphism(morphism, source_word)
            outcome = run_subsequential(transducer, indices)
            assert outcome.status == "OUTPUT"
            actual_letters = tuple(
                morphism.target_alphabet[index] for index in outcome.output
            )
            assert actual_letters == expected.letters

    def test_transition_output_limit_is_admitted_before_conversion(self) -> None:
        accepted = WordMorphism(
            source_alphabet=("a",), target_alphabet=("x",), images=(("x",) * 512,)
        )
        assert (
            len(word_morphism_to_subsequential(accepted).transitions[0].output) == 512
        )
        rejected = WordMorphism(
            source_alphabet=("a",), target_alphabet=("x",), images=(("x",) * 513,)
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            word_morphism_to_subsequential(rejected)
        assert error.value.errors()[0]["type"] == (
            "finite_state_transducer.morphism_image_bound_exceeded"
        )

    def test_alphabet_larger_than_transducer_carrier_is_rejected(self) -> None:
        symbols = tuple(f"s{index}" for index in range(33))
        morphism = WordMorphism(
            source_alphabet=symbols,
            target_alphabet=("x",),
            images=(("x",),) * 33,
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            word_morphism_to_subsequential(morphism)
        assert error.value.errors()[0]["type"] == (
            "finite_state_transducer.morphism_alphabet_bound_exceeded"
        )


class TestSubsequentialRun:
    @staticmethod
    def _observed(result: object) -> tuple[object, ...]:
        return (
            result.status,
            result.output,
            result.final_state,
            result.undefined_position,
            result.partial_output,
            result.state_trace,
            result.transition_outputs,
            result.cumulative_outputs,
            result.final_output,
            result.obstruction_position,
            result.obstruction_state,
            result.obstruction_symbol,
        )

    @staticmethod
    def _direct_run(
        transducer: SubsequentialTransducer, word: tuple[int, ...]
    ) -> tuple[object, ...]:
        """Independent small evaluator used as a semantic oracle."""
        rows = {(t.source, t.input_symbol): t for t in transducer.transitions}
        final_rows = {row.state: row.output for row in transducer.final_outputs}
        state = transducer.initial_state
        states = [state]
        step_outputs: list[tuple[int, ...]] = []
        prefixes = [()]
        emitted: tuple[int, ...] = ()
        for position, symbol in enumerate(word):
            transition = rows.get((state, symbol))
            if transition is None:
                return (
                    "UNDEFINED_TRANSITION",
                    (),
                    state,
                    position,
                    emitted,
                    tuple(states),
                    tuple(step_outputs),
                    tuple(prefixes),
                    (),
                    position,
                    state,
                    symbol,
                )
            state = transition.target
            emitted = emitted + transition.output
            states.append(state)
            step_outputs.append(transition.output)
            prefixes.append(emitted)
        if state not in final_rows:
            return (
                "NONFINAL_DOMAIN_STATE",
                (),
                state,
                None,
                emitted,
                tuple(states),
                tuple(step_outputs),
                tuple(prefixes),
                (),
                len(word),
                state,
                None,
            )
        final = final_rows[state]
        return (
            "OUTPUT",
            emitted + final,
            state,
            None,
            (),
            tuple(states),
            tuple(step_outputs),
            tuple(prefixes),
            final,
            None,
            None,
            None,
        )

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

        assert self._observed(run_subsequential(transducer, (0, 0))) == (
            self._direct_run(transducer, (0, 0))
        )
        outcome = run_subsequential(transducer, (0, 0))
        assert outcome.state_trace == (0, 0, 0)
        assert outcome.transition_outputs == ((), ())
        assert outcome.cumulative_outputs == ((), (), ())
        assert outcome.final_output == ()

    def test_undefined_transition_preserves_partial_trace(self) -> None:
        transducer = _flip()

        source = transducer.model_copy(
            update={"transitions": transducer.transitions[:1]}
        )
        outcome = run_subsequential(source, (0, 1))

        assert (
            outcome.status,
            outcome.output,
            outcome.final_state,
            outcome.undefined_position,
            outcome.partial_output,
        ) == (
            "UNDEFINED_TRANSITION",
            (),
            0,
            1,
            (1,),
        )
        assert self._observed(outcome) == self._direct_run(source, (0, 1))
        outcome = compute_run(
            SubseqRunRequest(
                transducer=transducer.model_copy(
                    update={"transitions": transducer.transitions[:1]}
                ),
                word=(0, 1),
            )
        )
        assert (outcome.obstruction_position, outcome.obstruction_state) == (1, 0)
        assert outcome.obstruction_symbol == 1
        assert outcome.transition_outputs == ((1,),)
        assert outcome.cumulative_outputs == ((), (1,))

    def test_nonfinal_state_is_not_a_function_value(self) -> None:
        transducer = _flip().model_copy(update={"final_outputs": ()})

        assert self._observed(run_subsequential(transducer, (0,))) == self._direct_run(
            transducer, (0,)
        )

    def test_trace_records_final_output_separately(self) -> None:
        transducer = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=3,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=1, output=()),
                SubseqTransition(source=1, input_symbol=1, target=2, output=(1, 0)),
            ),
            final_outputs=(SubseqFinalOutput(state=2, output=(0,)),),
        )

        result = run_subsequential(transducer, (0, 1))

        assert self._observed(result) == self._direct_run(transducer, (0, 1))
        assert result.output == (1, 0, 0)
        assert result.state_trace == (0, 1, 2)
        assert result.transition_outputs == ((), (1, 0))
        assert result.cumulative_outputs == ((), (), (1, 0))
        assert result.final_output == (0,)

    def test_run_result_byte_bound_is_checked_before_trace_expansion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import jacobian.math.logic.automata.transducers.operations as kernels

        monkeypatch.setattr(kernels, "MAX_FST_RUN_RESULT_BYTES", 1)
        monkeypatch.setattr(
            kernels,
            "_transition_map",
            lambda _transducer: pytest.fail("trace expansion began before admission"),
        )
        with pytest.raises(OperationResourceAdmissionError) as error:
            run_subsequential(_flip(), (0,))
        assert (
            error.value.errors()[0]["type"]
            == "finite_state_transducer.run_result_bytes_exceeded"
        )

    def test_maximum_admitted_output_keeps_cumulative_trace_within_bytes(self) -> None:
        transducer = SubsequentialTransducer(
            input_alphabet_size=1,
            output_alphabet_size=1,
            state_count=1,
            initial_state=0,
            transitions=(
                SubseqTransition(
                    source=0,
                    input_symbol=0,
                    target=0,
                    output=(0,) * 8,
                ),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        result = compute_run(SubseqRunRequest(transducer=transducer, word=(0,) * 512))

        assert len(result.output) == MAX_FST_RESULT_WORD_LENGTH
        assert len(result.cumulative_outputs) == 513
        assert len(result.cumulative_outputs[-1]) == MAX_FST_RESULT_WORD_LENGTH
        assert len(encode_strict_json(result.model_dump(mode="json"))) <= (
            MAX_FST_RUN_RESULT_BYTES
        )

    def test_adapter_binds_transducer_and_word(self) -> None:
        request = SubseqRunRequest(transducer=_flip(), word=(0, 1))
        result = compute_run(request)

        assert result.transducer == request.transducer
        assert result.word == request.word
        assert result.output == (1, 0)
        assert result.state_trace == (0, 0, 0)
        assert result.transition_outputs == ((1,), (0,))
        assert result.cumulative_outputs == ((), (1,), (1, 0))
        assert result.final_output == ()

        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_subsequential_run(decoded)
        assert not verify_subsequential_run(
            decoded.model_copy(update={"output": (0, 0)})
        )

    def test_native_run_rejects_symbol_outside_alphabet(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            run_subsequential(_flip(), (2,))


class TestComposition:
    def test_one_sided_intermediate_parent_is_rejected_on_both_paths(self) -> None:
        first = _flip().model_copy(update={"output_alphabet_id": "shared"})
        second = _flip()
        with pytest.raises(ValidationError):
            ComposeRequest(first=first, second=second)
        with pytest.raises(OperationDomainValidationError):
            compose_subsequential(first, second)

    def test_bound_intermediate_parent_requires_equal_identity_and_context(
        self,
    ) -> None:
        parent = FiniteAlphabet(symbols=("zero", "one"))
        first = _flip().model_copy(
            update={"output_alphabet_id": "shared", "output_alphabet": parent}
        )
        second = _flip().model_copy(
            update={"input_alphabet_id": "shared", "input_alphabet": parent}
        )
        assert ComposeRequest(first=first, second=second).first == first
        foreign = second.model_copy(
            update={"input_alphabet": FiniteAlphabet(symbols=("x", "y"))}
        )
        with pytest.raises(ValidationError):
            ComposeRequest(first=first, second=foreign)

    def test_flip_after_flip_is_identity(self) -> None:
        composite = compose_subsequential(_flip(), _flip())

        assert run_subsequential(composite, (0, 1, 0)).output == (0, 1, 0)

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
        assert run_subsequential(composite, ()).status == "NONFINAL_DOMAIN_STATE"

    def test_unreachable_cartesian_states_do_not_block_composition(self) -> None:
        large = _flip().model_copy(update={"state_count": 9})
        request = ComposeRequest(first=large, second=large)

        result = compute_compose(request)
        assert result.transducer.state_count == 1
        assert run_subsequential(result.transducer, (0, 1, 0)).output == (0, 1, 0)

    def test_adapter_binds_both_operands(self) -> None:
        request = ComposeRequest(first=identity_transducer(2), second=_flip())
        result = compute_compose(request)

        assert result.first == request.first
        assert result.second == request.second
        assert run_subsequential(result.transducer, (0, 1)).output == (1, 0)
        decoded = type(result).model_validate_json(result.model_dump_json())
        assert verify_composition(decoded)
        forged = decoded.transducer.model_copy(
            update={"transitions": decoded.transducer.transitions[:1]}
        )
        assert not verify_composition(decoded.model_copy(update={"transducer": forged}))


class TestNativeTransformations:
    def test_identity_is_exact(self) -> None:
        assert run_subsequential(identity_transducer(3), (0, 1, 2)).output == (0, 1, 2)

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

    def test_trim_drops_reachable_dead_state_and_its_transitions(self) -> None:
        # State 1 is reachable (0 -1-> 1) but dead: no final-output state is
        # reachable from it. A reachable-only trim would keep it; the exact
        # trim must drop both the state and the transition leading into it.
        source = SubsequentialTransducer(
            input_alphabet_size=2,
            output_alphabet_size=2,
            state_count=2,
            initial_state=0,
            transitions=(
                SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
                SubseqTransition(source=0, input_symbol=1, target=1, output=(1,)),
                SubseqTransition(source=1, input_symbol=0, target=1, output=(0,)),
                SubseqTransition(source=1, input_symbol=1, target=1, output=(1,)),
            ),
            final_outputs=(SubseqFinalOutput(state=0, output=()),),
        )
        result = compute_trim(TrimRequest(transducer=source))

        assert result.trimmed.state_count == 1
        assert result.old_to_new == ((0, 0),)
        assert result.new_to_old == (0,)
        assert result.trimmed.transitions == (
            SubseqTransition(source=0, input_symbol=0, target=0, output=(0,)),
        )
        assert result.trimmed.final_outputs == (SubseqFinalOutput(state=0, output=()),)

    def test_trim_of_live_machine_is_the_identity_restriction(self) -> None:
        result = compute_trim(TrimRequest(transducer=_flip()))

        assert result.trimmed == _flip()
        assert result.old_to_new == ((0, 0),)
        assert result.new_to_old == (0,)

    def test_trim_without_final_states_returns_the_canonical_placeholder(self) -> None:
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
        result = compute_trim(TrimRequest(transducer=source))

        assert result.trimmed.state_count == 1
        assert result.trimmed.initial_state == 0
        assert result.trimmed.transitions == ()
        assert result.trimmed.final_outputs == ()
        assert result.old_to_new == ()
        assert result.new_to_old == ()

    def test_trim_preserves_the_partial_function(self) -> None:
        from itertools import product

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
            final_outputs=(SubseqFinalOutput(state=0, output=(1,)),),
        )
        result = compute_trim(TrimRequest(transducer=source))

        def function_value(
            transducer: SubsequentialTransducer, word: tuple[int, ...]
        ) -> tuple[bool, tuple[int, ...]]:
            outcome = run_subsequential(transducer, word)
            return (outcome.status == "OUTPUT", outcome.output)

        for length in range(6):
            for raw in product((0, 1), repeat=length):
                word = tuple(raw)
                assert function_value(source, word) == function_value(
                    result.trimmed, word
                ), word

    def test_trimmed_states_are_exactly_the_live_states(self) -> None:
        from jacobian.math.logic.automata.transducers import (
            coaccessible_states,
            reachable_states,
        )

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
            final_outputs=(SubseqFinalOutput(state=0, output=(1,)),),
        )
        result = compute_trim(TrimRequest(transducer=source))
        live = set(range(result.trimmed.state_count))

        assert reachable_states(result.trimmed) == live
        assert coaccessible_states(result.trimmed) == live
        assert set(result.new_to_old) == reachable_states(source) & (
            coaccessible_states(source)
        )

    def test_trim_native_and_catalog_results_agree(self) -> None:
        request = TrimRequest(transducer=_flip())
        result = compute_trim(request)
        trimmed, state_map = trim_subsequential(request.transducer)

        assert result.transducer == request.transducer
        assert result.trimmed == trimmed
        assert dict(result.old_to_new) == state_map
        decoded = TrimResult.model_validate_json(result.model_dump_json())
        assert decoded == result

    def test_trim_result_rejects_disagreeing_state_maps(self) -> None:
        request = TrimRequest(transducer=_flip())
        result = compute_trim(request)
        payload = result.model_dump(mode="json")
        payload["new_to_old"] = [1]
        with pytest.raises(ValidationError, match="outside the source"):
            TrimResult.model_validate(payload)
        payload = result.model_dump(mode="json")
        payload["old_to_new"] = [[0, 1]]
        with pytest.raises(ValidationError, match="new-state range"):
            TrimResult.model_validate(payload)

    def test_trim_result_rejects_forged_alphabet_parent(self) -> None:
        source = _flip().model_copy(
            update={
                "input_alphabet_id": "input",
                "input_alphabet": FiniteAlphabet(symbols=("zero", "one")),
            }
        )
        result = compute_trim(TrimRequest(transducer=source))
        payload = result.model_dump(mode="json")
        payload["trimmed"]["input_alphabet_id"] = "foreign"
        with pytest.raises(ValidationError):
            TrimResult.model_validate_json(json.dumps(payload))

    def test_minimize_result_rejects_forged_alphabet_parent(self) -> None:
        source = _flip().model_copy(
            update={
                "output_alphabet_id": "output",
                "output_alphabet": FiniteAlphabet(symbols=("zero", "one")),
            }
        )
        result = compute_minimize(MinimizeRequest(transducer=source))
        payload = result.model_dump(mode="json")
        payload["minimized"]["output_alphabet_id"] = "foreign"
        with pytest.raises(ValidationError):
            MinimizeResult.model_validate_json(json.dumps(payload))

    def test_trim_declared_example_executes_through_the_owner_adapter(self) -> None:
        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "transducer.subsequential.trim.compute"
        )
        request = operation.request_type.model_validate_json(
            encode_strict_json(operation.examples[0].input), strict=True
        )
        result = operation.run(request)
        assert result.trimmed.state_count == 1
        assert result.new_to_old == (0,)

    def test_rational_inverse_swaps_labels_and_alphabets(self) -> None:
        relation = _relation().model_copy(update={"output_alphabet_size": 3})
        inverse = invert_rational(relation)

        assert inverse.input_alphabet_size == 3
        assert inverse.output_alphabet_size == 2
        assert inverse.edges[0].input_label == (1,)
        assert inverse.edges[0].output_label == (0,)
        assert inverse.edges[1].input_label == (0,)
        assert inverse.edges[1].output_label == (1,)

    def test_rational_inverse_is_published_and_involutive_after_json_round_trip(
        self,
    ) -> None:
        relation = _relation()
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "transducer.relation.inverse.compute"
        )
        request = RationalRelationInverseRequest(
            transducer=RationalTransducer.model_validate_json(
                relation.model_dump_json()
            )
        )

        inverse = tool.run(request)
        twice = compute_relation_inverse(
            RationalRelationInverseRequest(transducer=inverse)
        )

        assert inverse.input_alphabet == relation.output_alphabet
        assert inverse.output_alphabet == relation.input_alphabet
        assert twice == relation

    def test_only_audited_outcomes_are_public(self) -> None:
        assert {tool.operation_id for tool in TOOLS} == {
            "transducer.relation.inverse.compute",
            "transducer.relation.outputs_for_input_automaton.compute",
            "transducer.relation.path.replay.compute",
            "transducer.relation.projection.compute",
            "transducer.subsequential.compose.compute",
            "transducer.subsequential.from_word_morphism.compute",
            "transducer.subsequential.identity.compute",
            "transducer.subsequential.minimize.compute",
            "transducer.subsequential.reachable_states.compute",
            "transducer.subsequential.run.compute",
            "transducer.subsequential.trim.compute",
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


class TestValueValidation:
    def test_duplicate_deterministic_transition_is_rejected(self) -> None:
        transition = SubseqTransition(source=0, input_symbol=0, target=0, output=())
        with pytest.raises(ValidationError) as error:
            SubsequentialTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_state=0,
                transitions=(transition, transition),
                final_outputs=(),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_state_transducer.duplicate_transition"
        )

    def test_duplicate_initial_and_accepting_states_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            _relation(initial_states=(0, 0))
        assert (
            error.value.errors()[0]["type"]
            == "finite_state_transducer.initial_states_not_distinct"
        )

    def test_empty_rational_edge_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as error:
            RationalTransducer(
                input_alphabet_size=1,
                output_alphabet_size=1,
                state_count=1,
                initial_states=(0,),
                accepting_states=(0,),
                edges=(RationalEdge(source=0, target=0),),
            )
        assert (
            error.value.errors()[0]["type"]
            == "finite_state_transducer.edge_labels_empty"
        )

    def test_replay_must_select_a_declared_initial_state(self) -> None:
        request = RelationPathReplayRequest(
            transducer=_relation(), initial_state=1, edge_path=()
        )

        with pytest.raises(OperationDomainValidationError) as error:
            compute_relation_path_replay(request)
        assert (
            error.value.errors()[0]["type"]
            == "finite_state_transducer.initial_state_not_declared"
        )
