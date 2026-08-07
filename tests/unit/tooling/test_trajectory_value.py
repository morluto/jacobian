from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from jacobian.eval.trajectory_state import (
    BindingValidity,
    CandidateState,
    CheckerState,
    CleanRoomTerminalEvidence,
    ExtractedTrajectoryState,
    MilestoneKind,
    ReasoningProtocolState,
    StateBoundary,
    TerminalAcceptance,
    TrajectoryExtraction,
    TrajectoryHardState,
    TrajectorySoftState,
)
from jacobian.eval.trajectory_value import (
    EstimatorEvaluation,
    EstimatorKind,
    LabelledTrajectory,
    OfflineValueComparison,
    StateValueEstimate,
    TrajectoryValueCorpus,
    ValueSource,
    evaluate_offline_trajectories,
)

ROOT = Path(__file__).resolve().parents[3]


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _hard(
    *,
    group: str,
    candidate: CandidateState,
    checker: CheckerState,
    protocol: ReasoningProtocolState,
    transitions: tuple[MilestoneKind, ...] = (),
    binding: BindingValidity = BindingValidity.UNKNOWN,
) -> TrajectoryHardState:
    return TrajectoryHardState(
        task_family="polynomial-map-inverse",
        candidate_state=candidate,
        latest_candidate_digest=(
            None if candidate is CandidateState.ABSENT else _sha("candidate-" + group)
        ),
        checker_state=checker,
        execution_status=None if candidate is CandidateState.ABSENT else "COMPLETED",
        completeness_status=(
            None if candidate is CandidateState.ABSENT else "COMPLETE"
        ),
        completeness_assurance=(
            None if candidate is CandidateState.ABSENT else "HEURISTIC"
        ),
        assurance_level=None if candidate is CandidateState.ABSENT else "COMPUTED",
        scope_digest=None
        if candidate is CandidateState.ABSENT
        else _sha("scope-" + group),
        binding_validity=binding,
        latest_meaningful_transitions=transitions,
        reasoning_protocol_state=protocol,
    )


def _state(
    *,
    index: int,
    boundary: StateBoundary,
    hard: TrajectoryHardState,
    plan: str,
    after: str | None = None,
    transitions: tuple[MilestoneKind, ...] = (),
) -> ExtractedTrajectoryState:
    return ExtractedTrajectoryState(
        index=index,
        source_event_index=index,
        boundary=boundary,
        hard_state=hard,
        soft_state=TrajectorySoftState(
            plan_summary=plan,
            latest_after_tool_summary=after,
        ),
        hard_state_digest=_sha(f"hard-{index}-{hard.model_dump_json()}"),
        changed_fields=(),
        milestone_kinds=transitions,
        milestone_eligible=bool(transitions),
        milestone_reason=(
            "eligible typed mathematical transition"
            if transitions
            else "observation boundary only"
        ),
    )


def _trajectory(
    trajectory_id: str,
    *,
    task_group: str,
    accepted: bool,
    plan: str,
    after: str,
    checker: CheckerState,
    candidate: CandidateState,
    extra_repeated_result: bool = False,
) -> LabelledTrajectory:
    plan_hard = _hard(
        group=task_group,
        candidate=CandidateState.ABSENT,
        checker=CheckerState.NOT_CHECKED,
        protocol=ReasoningProtocolState.READY,
    )
    transitions = [MilestoneKind.CANDIDATE_PRODUCED]
    if checker is CheckerState.ACCEPTED:
        transitions.append(MilestoneKind.CHECKER_ACCEPTED)
    elif checker is CheckerState.REJECTED:
        transitions.append(MilestoneKind.CHECKER_REJECTED)
    transition_tuple = tuple(transitions)
    tool_hard = _hard(
        group=task_group,
        candidate=candidate,
        checker=checker,
        protocol=ReasoningProtocolState.AWAITING_AFTER_TOOL,
        transitions=transition_tuple,
    )
    after_hard = _hard(
        group=task_group,
        candidate=candidate,
        checker=checker,
        protocol=ReasoningProtocolState.READY,
    )
    states = [
        _state(
            index=0,
            boundary=StateBoundary.PLAN,
            hard=plan_hard,
            plan=plan,
        ),
        _state(
            index=1,
            boundary=StateBoundary.TOOL_RESULT,
            hard=tool_hard,
            plan=plan,
            transitions=transition_tuple,
        ),
    ]
    after_index = len(states)
    states.append(
        _state(
            index=after_index,
            boundary=StateBoundary.AFTER_TOOL,
            hard=after_hard,
            plan=plan,
            after=after,
        )
    )
    if extra_repeated_result:
        repeated_hard = after_hard.model_copy(
            update={
                "reasoning_protocol_state": ReasoningProtocolState.AWAITING_AFTER_TOOL
            }
        )
        states.append(
            _state(
                index=len(states),
                boundary=StateBoundary.TOOL_RESULT,
                hard=repeated_hard,
                plan=plan,
                after=after,
            )
        )
    terminal_hard = after_hard.model_copy(
        update={"binding_validity": BindingValidity.VALID}
    )
    states.append(
        _state(
            index=len(states),
            boundary=StateBoundary.TERMINAL,
            hard=terminal_hard,
            plan=plan,
            after=after,
        )
    )
    source_digest = _sha("source-" + trajectory_id)
    terminal = CleanRoomTerminalEvidence(
        verifier_digest=_sha("clean-room-polynomial-verifier-v1"),
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=(
            TerminalAcceptance.ACCEPTED if accepted else TerminalAcceptance.REJECTED
        ),
        input_binding_valid=True,
        artifact_binding_valid=True,
        source_binding_digest=source_digest,
    )
    extraction = TrajectoryExtraction(
        source_digest=source_digest,
        task_family="polynomial-map-inverse",
        states=tuple(states),
        terminal_evidence=terminal,
    )
    return LabelledTrajectory(
        trajectory_id=trajectory_id,
        task_group=task_group,
        extraction=extraction,
    )


def _with_candidate_digest(
    trajectory: LabelledTrajectory, candidate_digest: str
) -> LabelledTrajectory:
    states: list[ExtractedTrajectoryState] = []
    for state in trajectory.extraction.states:
        hard = state.hard_state
        if hard.latest_candidate_digest is not None:
            hard = hard.model_copy(update={"latest_candidate_digest": candidate_digest})
        states.append(
            state.model_copy(
                update={
                    "hard_state": hard,
                    "hard_state_digest": _sha(
                        f"variant-{state.index}-{hard.model_dump_json()}"
                    ),
                }
            )
        )
    extraction = trajectory.extraction.model_copy(update={"states": tuple(states)})
    return LabelledTrajectory(
        trajectory_id=trajectory.trajectory_id,
        task_group=trajectory.task_group,
        extraction=extraction,
    )


def _controlled_corpus(*, repeated: bool = False) -> TrajectoryValueCorpus:
    common_plan = "Check both compositions for a 2 variable polynomial map."
    common_after = "The exact checker returned a result for both compositions."
    trajectories = [
        _trajectory(
            "checker-ok-1",
            task_group="checker-scope",
            accepted=True,
            plan=common_plan,
            after=common_after,
            checker=CheckerState.ACCEPTED,
            candidate=CandidateState.CHECKED,
            extra_repeated_result=repeated,
        ),
        _trajectory(
            "checker-ok-2",
            task_group="checker-scope",
            accepted=True,
            plan=common_plan,
            after=common_after,
            checker=CheckerState.ACCEPTED,
            candidate=CandidateState.CHECKED,
        ),
        _trajectory(
            "checker-bad-1",
            task_group="checker-scope",
            accepted=False,
            plan=common_plan,
            after=common_after,
            checker=CheckerState.REJECTED,
            candidate=CandidateState.REJECTED,
        ),
        _trajectory(
            "checker-bad-2",
            task_group="checker-scope",
            accepted=False,
            plan=common_plan,
            after=common_after,
            checker=CheckerState.REJECTED,
            candidate=CandidateState.REJECTED,
        ),
        _trajectory(
            "reasoning-ok-1",
            task_group="reasoning-branch",
            accepted=True,
            plan="Compose exact maps and prove the identity in 2 directions.",
            after="Both symbolic compositions reduce exactly to the identity.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
        _trajectory(
            "reasoning-ok-2",
            task_group="reasoning-branch",
            accepted=True,
            plan="Compose exact maps and prove the identity in 2 directions.",
            after="Both symbolic compositions reduce exactly to the identity.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
        _trajectory(
            "reasoning-bad-1",
            task_group="reasoning-branch",
            accepted=False,
            plan="Guess coefficients from 2 samples and take a shortcut.",
            after="Ignore the unresolved mismatch and claim the candidate anyway.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
        _trajectory(
            "reasoning-bad-2",
            task_group="reasoning-branch",
            accepted=False,
            plan="Guess coefficients from 2 samples and take a shortcut.",
            after="Ignore the unresolved mismatch and claim the candidate anyway.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
    ]
    return TrajectoryValueCorpus(
        corpus_id="controlled-polynomial-state-aliasing-v1",
        trajectories=tuple(trajectories),
    )


def _evaluation(
    comparison: OfflineValueComparison, kind: EstimatorKind
) -> EstimatorEvaluation:
    return next(item for item in comparison.evaluations if item.estimator is kind)


def _estimate(
    comparison: OfflineValueComparison, kind: EstimatorKind, key: str
) -> StateValueEstimate:
    evaluation = _evaluation(comparison, kind)
    return next(item for item in evaluation.estimates if item.observation_id == key)


def test_compares_all_estimators_without_target_label_leakage() -> None:
    result = evaluate_offline_trajectories(_controlled_corpus())

    assert tuple(item.estimator for item in result.evaluations) == tuple(EstimatorKind)
    assert result.assurance_authority is False
    assert all(item.metrics.observation_count == 24 for item in result.evaluations)
    for evaluation in result.evaluations:
        for estimate in evaluation.estimates:
            assert estimate.trajectory_id not in estimate.supporting_trajectory_ids
            assert estimate.assurance_authority is False
            assert 0.0 <= estimate.estimated_value <= 1.0


def test_typed_compatibility_separates_textually_identical_checker_states() -> None:
    result = evaluate_offline_trajectories(_controlled_corpus())
    accepted_key = "checker-ok-1:2"
    rejected_key = "checker-bad-1:2"

    text_accepted = _estimate(result, EstimatorKind.REASONING_TEXT, accepted_key)
    text_rejected = _estimate(result, EstimatorKind.REASONING_TEXT, rejected_key)
    typed_accepted = _estimate(result, EstimatorKind.JACOBIAN_TYPED, accepted_key)
    typed_rejected = _estimate(result, EstimatorKind.JACOBIAN_TYPED, rejected_key)

    assert text_accepted.cluster_id == text_rejected.cluster_id
    assert typed_accepted.cluster_id != typed_rejected.cluster_id
    assert typed_accepted.estimated_value == 1.0
    assert typed_rejected.estimated_value == 0.0


def test_hybrid_uses_reasoning_to_split_same_typed_state() -> None:
    result = evaluate_offline_trajectories(_controlled_corpus())
    accepted_key = "reasoning-ok-1:2"
    rejected_key = "reasoning-bad-1:2"

    typed_accepted = _estimate(result, EstimatorKind.JACOBIAN_TYPED, accepted_key)
    typed_rejected = _estimate(result, EstimatorKind.JACOBIAN_TYPED, rejected_key)
    hybrid_accepted = _estimate(result, EstimatorKind.HYBRID_TYPED_TEXT, accepted_key)
    hybrid_rejected = _estimate(result, EstimatorKind.HYBRID_TYPED_TEXT, rejected_key)

    assert typed_accepted.typed_compatibility_digest == (
        typed_rejected.typed_compatibility_digest
    )
    assert typed_accepted.cluster_id == typed_rejected.cluster_id
    assert hybrid_accepted.cluster_id != hybrid_rejected.cluster_id
    assert hybrid_accepted.estimated_value == 1.0
    assert hybrid_rejected.estimated_value == 0.0


def test_exact_candidate_identity_is_a_typed_compatibility_constraint() -> None:
    def make(trajectory_id: str, *, accepted: bool) -> LabelledTrajectory:
        return _trajectory(
            trajectory_id,
            task_group="candidate-identity",
            accepted=accepted,
            plan="Inspect the same proposed inverse.",
            after="The candidate remains available.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        )

    accepted = (
        make("candidate-a-1", accepted=True),
        make("candidate-a-2", accepted=True),
    )
    rejected = tuple(
        _with_candidate_digest(
            make(trajectory_id, accepted=False),
            _sha("distinct-candidate-b"),
        )
        for trajectory_id in ("candidate-b-1", "candidate-b-2")
    )
    result = evaluate_offline_trajectories(
        TrajectoryValueCorpus(
            corpus_id="candidate-identity-v1",
            trajectories=(*accepted, *rejected),
        )
    )
    left = _estimate(result, EstimatorKind.JACOBIAN_TYPED, "candidate-a-1:1")
    right = _estimate(result, EstimatorKind.JACOBIAN_TYPED, "candidate-b-1:1")

    assert left.typed_compatibility_digest != right.typed_compatibility_digest
    assert left.cluster_id != right.cluster_id
    assert left.estimated_value == 1.0
    assert right.estimated_value == 0.0


def test_repeated_nonmilestone_tool_result_cannot_add_an_observation() -> None:
    baseline = evaluate_offline_trajectories(_controlled_corpus())
    repeated = evaluate_offline_trajectories(_controlled_corpus(repeated=True))

    for baseline_eval, repeated_eval in zip(
        baseline.evaluations, repeated.evaluations, strict=True
    ):
        assert (
            baseline_eval.metrics.observation_count
            == repeated_eval.metrics.observation_count
        )


def test_metrics_weight_each_trajectory_equally() -> None:
    result = evaluate_offline_trajectories(_controlled_corpus())
    evaluation = _evaluation(result, EstimatorKind.GROUP_ROLLOUT)
    per_trajectory: dict[str, list[float]] = {}
    for estimate in evaluation.estimates:
        error = estimate.estimated_value - estimate.eventual_terminal_reward
        per_trajectory.setdefault(estimate.trajectory_id, []).append(error * error)
    expected = sum(
        sum(errors) / len(errors) for errors in per_trajectory.values()
    ) / len(per_trajectory)

    assert evaluation.metrics.brier_score == pytest.approx(expected)


def test_singleton_clusters_fall_back_to_other_task_group_rollouts() -> None:
    trajectories = (
        _trajectory(
            "unique-alpha",
            task_group="unique-text",
            accepted=True,
            plan="Amber orchard theorem.",
            after="Cobalt lattice complete.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
        _trajectory(
            "unique-bravo",
            task_group="unique-text",
            accepted=True,
            plan="Glacier falcon lemma.",
            after="Indigo prism complete.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
        _trajectory(
            "unique-charlie",
            task_group="unique-text",
            accepted=False,
            plan="Lantern quartz argument.",
            after="Umber spiral unresolved.",
            checker=CheckerState.NOT_CHECKED,
            candidate=CandidateState.PRESENT,
        ),
    )
    result = evaluate_offline_trajectories(
        TrajectoryValueCorpus(
            corpus_id="singleton-fallback-v1", trajectories=trajectories
        )
    )
    estimate = _estimate(result, EstimatorKind.REASONING_TEXT, "unique-alpha:0")

    assert estimate.value_source is ValueSource.TASK_GROUP_PRIOR
    assert estimate.supporting_trajectory_ids == ("unique-bravo", "unique-charlie")
    assert estimate.estimated_value == 0.5


def test_binary_labels_fail_closed_on_bad_evidence_and_duplicate_rollouts() -> None:
    valid = _controlled_corpus().trajectories[0]
    evidence = valid.extraction.terminal_evidence
    assert evidence is not None
    unbound = valid.extraction.model_copy(
        update={
            "terminal_evidence": evidence.model_copy(
                update={"artifact_binding_valid": None}
            )
        }
    )
    with pytest.raises(ValidationError, match="exact input and artifact bindings"):
        LabelledTrajectory(
            trajectory_id="unbound",
            task_group="checker-scope",
            extraction=unbound,
        )

    duplicate = valid.model_copy(update={"trajectory_id": "duplicate"})
    with pytest.raises(ValidationError, match="duplicate transcript digests"):
        TrajectoryValueCorpus(
            corpus_id="duplicate-corpus",
            trajectories=(valid, duplicate),
        )


def test_terminal_evidence_bound_to_exact_trajectory_source_digest() -> None:
    valid = _controlled_corpus().trajectories[0]
    evidence = valid.extraction.terminal_evidence
    assert evidence is not None
    foreign_binding = evidence.model_copy(
        update={"source_binding_digest": _sha("foreign-source")}
    )
    foreign = valid.extraction.model_copy(update={"terminal_evidence": foreign_binding})
    with pytest.raises(
        ValidationError, match="bound to the exact extracted trajectory"
    ):
        LabelledTrajectory(
            trajectory_id="foreign-bound",
            task_group="checker-scope",
            extraction=foreign,
        )


def test_terminal_only_extraction_rejects_labelled_trajectory() -> None:
    valid = _controlled_corpus().trajectories[0]
    terminal = valid.extraction.states[-1].model_copy(update={"index": 0})
    evidence = valid.extraction.terminal_evidence
    extraction = valid.extraction.model_copy(
        update={
            "states": (terminal,),
            "terminal_evidence": evidence,
        }
    )
    with pytest.raises(ValidationError, match="PLAN observation"):
        LabelledTrajectory(
            trajectory_id="terminal-only",
            task_group="checker-scope",
            extraction=extraction,
        )


def test_single_rollout_task_groups_and_unknown_fields_are_rejected() -> None:
    trajectory = _controlled_corpus().trajectories[0]
    with pytest.raises(ValidationError, match="requires repeated rollouts"):
        TrajectoryValueCorpus(
            corpus_id="singleton-corpus",
            trajectories=(
                trajectory,
                _controlled_corpus().trajectories[4],
            ),
        )

    payload = _controlled_corpus().model_dump(mode="json")
    payload["terminal_reward"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrajectoryValueCorpus.model_validate(payload)


def test_results_are_deterministic_under_trajectory_reordering() -> None:
    corpus = _controlled_corpus()
    reversed_corpus = corpus.model_copy(
        update={"trajectories": corpus.trajectories[::-1]}
    )
    left = evaluate_offline_trajectories(corpus)
    right = evaluate_offline_trajectories(reversed_corpus)

    assert left.evaluations == right.evaluations
    assert left.corpus_digest != right.corpus_digest


def test_controlled_experiment_summary_is_immutable_and_reproducible() -> None:
    result = evaluate_offline_trajectories(_controlled_corpus())
    fixture_path = (
        ROOT
        / "tests/unit/tooling/fixtures/trajectory_value/pr2_controlled/comparison-summary.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["corpus_id"] == result.corpus_id
    assert fixture["corpus_digest"] == result.corpus_digest
    assert fixture["evaluator_id"] == result.evaluator_id
    assert fixture["evaluator_config"] == result.evaluator_config.model_dump(
        mode="json"
    )
    assert fixture["observation_count"] == 24
    for evaluation in result.evaluations:
        expected = fixture["metrics"][evaluation.estimator.value]
        actual = evaluation.metrics.model_dump(mode="json")
        assert expected == {
            key: actual[key]
            for key in (
                "cluster_count",
                "task_group_fallback_count",
                "brier_score",
                "mean_absolute_error",
            )
        }


def test_schema_files_match_closed_models_and_validate_results() -> None:
    schemas: dict[str, type[BaseModel]] = {
        "trajectory-value-corpus-v1.schema.json": TrajectoryValueCorpus,
        "trajectory-value-evaluation-v1.schema.json": OfflineValueComparison,
    }
    result = evaluate_offline_trajectories(_controlled_corpus())
    instances = {
        "trajectory-value-corpus-v1.schema.json": _controlled_corpus().model_dump(
            mode="json"
        ),
        "trajectory-value-evaluation-v1.schema.json": result.model_dump(mode="json"),
    }
    state_schema = json.loads(
        (
            ROOT / "docs/reference/evaluations/schemas/trajectory-state-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    registry = Registry().with_resource(
        state_schema["$id"], DRAFT202012.create_resource(state_schema)
    )
    for filename, model in schemas.items():
        path = ROOT / "docs/reference/evaluations/schemas" / filename
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema == model.model_json_schema()
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, registry=registry).validate(instances[filename])
