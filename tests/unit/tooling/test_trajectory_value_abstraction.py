from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from jacobian.canonical import canonicalize_json
from jacobian.eval import trajectory_value_abstraction as abstraction_module
from jacobian.eval.trajectory_state import (
    ArtifactStateRef,
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
    TypedObjectRef,
)
from jacobian.eval.trajectory_value import (
    LabelledTrajectory,
    TrajectoryValueCorpus,
    ValueSource,
)
from jacobian.eval.trajectory_value_abstraction import (
    AbstractValueStateSignature,
    EstimatorEvaluationV2,
    EstimatorKindV2,
    OfflineValueComparisonV2,
    SemanticStateValueEstimate,
    abstract_value_state,
    evaluate_semantic_trajectories,
)

ROOT = Path(__file__).resolve().parents[3]


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonicalize_json(value)).hexdigest()


def _artifact(value: str) -> str:
    return "artifact://sha256/" + hashlib.sha256(value.encode()).hexdigest()


def _hard(
    *,
    identity: str,
    candidate: CandidateState,
    checker: CheckerState,
    protocol: ReasoningProtocolState,
    transitions: tuple[MilestoneKind, ...] = (),
    binding: BindingValidity = BindingValidity.UNKNOWN,
) -> TrajectoryHardState:
    if candidate is CandidateState.ABSENT:
        return TrajectoryHardState(
            task_family="polynomial-map-inverse",
            candidate_state=candidate,
            checker_state=checker,
            binding_validity=binding,
            latest_meaningful_transitions=transitions,
            reasoning_protocol_state=protocol,
        )

    result_uri = _artifact("result-" + identity)
    obligation_uri = _artifact("obligation-" + identity)
    artifacts = (
        ArtifactStateRef(
            artifact_uri=result_uri,
            role="MATHEMATICAL_RESULT",
            source_capability_id="polynomial.map.compute",
        ),
        ArtifactStateRef(
            artifact_uri=obligation_uri,
            role="OBLIGATION",
            source_capability_id="polynomial.identity.verify",
        ),
    )
    return TrajectoryHardState(
        task_family="polynomial-map-inverse",
        typed_objects=(
            TypedObjectRef(
                object_type="polynomial.map",
                content_digest=_sha("object-" + identity),
                source_capability_id="polynomial.map.compute",
            ),
        ),
        artifacts=artifacts,
        candidate_state=candidate,
        latest_candidate_digest=_sha("candidate-" + identity),
        checker_state=checker,
        open_obligation_uris=(obligation_uri,),
        execution_status="COMPLETED",
        completeness_status="COMPLETE",
        completeness_assurance="HEURISTIC",
        assurance_level="COMPUTED",
        scope_digest=_sha("scope-" + identity),
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
) -> ExtractedTrajectoryState:
    transitions = hard.latest_meaningful_transitions
    soft = TrajectorySoftState(
        plan_summary=plan,
        latest_after_tool_summary=after,
    )
    return ExtractedTrajectoryState(
        index=index,
        source_event_index=index,
        boundary=boundary,
        hard_state=hard,
        soft_state=soft,
        soft_state_digest=_digest(soft.model_dump(mode="json")),
        hard_state_digest=_sha(f"hard-{index}-{hard.model_dump_json()}"),
        changed_fields=tuple(item.value for item in transitions),
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
    candidate: CandidateState,
    checker: CheckerState,
) -> LabelledTrajectory:
    plan_hard = _hard(
        identity=trajectory_id,
        candidate=CandidateState.ABSENT,
        checker=CheckerState.NOT_CHECKED,
        protocol=ReasoningProtocolState.READY,
    )
    transitions = [
        MilestoneKind.OBJECT_ADDED,
        MilestoneKind.ARTIFACT_ADDED,
        MilestoneKind.CANDIDATE_PRODUCED,
        MilestoneKind.OBLIGATION_OPENED,
        MilestoneKind.SCOPE_CHANGED,
        MilestoneKind.COMPLETENESS_CHANGED,
        MilestoneKind.ASSURANCE_CHANGED,
    ]
    if checker is CheckerState.ACCEPTED:
        transitions.append(MilestoneKind.CHECKER_ACCEPTED)
    elif checker is CheckerState.REJECTED:
        transitions.append(MilestoneKind.CHECKER_REJECTED)
    tool_hard = _hard(
        identity=trajectory_id,
        candidate=candidate,
        checker=checker,
        protocol=ReasoningProtocolState.AWAITING_AFTER_TOOL,
        transitions=tuple(transitions),
    )
    after_hard = _hard(
        identity=trajectory_id,
        candidate=candidate,
        checker=checker,
        protocol=ReasoningProtocolState.READY,
    )
    terminal_hard = after_hard.model_copy(
        update={"binding_validity": BindingValidity.VALID}
    )
    states = (
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
        ),
        _state(
            index=2,
            boundary=StateBoundary.AFTER_TOOL,
            hard=after_hard,
            plan=plan,
            after=after,
        ),
        _state(
            index=3,
            boundary=StateBoundary.TERMINAL,
            hard=terminal_hard,
            plan=plan,
            after=after,
        ),
    )
    source_digest = _sha("source-" + trajectory_id)
    evidence = CleanRoomTerminalEvidence(
        verifier_digest=_sha("independent-polynomial-verifier-v1"),
        source_binding_digest=source_digest,
        clean_room=True,
        verifier_execution_status="COMPLETED",
        acceptance=(
            TerminalAcceptance.ACCEPTED if accepted else TerminalAcceptance.REJECTED
        ),
        input_binding_valid=True,
        artifact_binding_valid=True,
    )
    return LabelledTrajectory(
        trajectory_id=trajectory_id,
        task_group=task_group,
        extraction=TrajectoryExtraction(
            source_digest=source_digest,
            task_family="polynomial-map-inverse",
            states=states,
            terminal_evidence=evidence,
        ),
    )


def _controlled_corpus() -> TrajectoryValueCorpus:
    checker_plan = "Check both compositions of the proposed polynomial inverse."
    checker_after = "The exact two-sided checker completed."
    reasoning_good_plan = "Prove both exact compositions reduce to the identity."
    reasoning_good_after = "Both compositions are exact identities; preserve evidence."
    reasoning_bad_plan = "Infer the inverse from samples and skip the second direction."
    reasoning_bad_after = (
        "The other composition is unresolved, but claim success anyway."
    )
    trajectories = (
        _trajectory(
            "checker-ok-1",
            task_group="checker-branch",
            accepted=True,
            plan=checker_plan,
            after=checker_after,
            candidate=CandidateState.CHECKED,
            checker=CheckerState.ACCEPTED,
        ),
        _trajectory(
            "checker-ok-2",
            task_group="checker-branch",
            accepted=True,
            plan=checker_plan,
            after=checker_after,
            candidate=CandidateState.CHECKED,
            checker=CheckerState.ACCEPTED,
        ),
        _trajectory(
            "checker-bad-1",
            task_group="checker-branch",
            accepted=False,
            plan=checker_plan,
            after=checker_after,
            candidate=CandidateState.REJECTED,
            checker=CheckerState.REJECTED,
        ),
        _trajectory(
            "checker-bad-2",
            task_group="checker-branch",
            accepted=False,
            plan=checker_plan,
            after=checker_after,
            candidate=CandidateState.REJECTED,
            checker=CheckerState.REJECTED,
        ),
        _trajectory(
            "reasoning-ok-1",
            task_group="reasoning-branch",
            accepted=True,
            plan=reasoning_good_plan,
            after=reasoning_good_after,
            candidate=CandidateState.PRESENT,
            checker=CheckerState.NOT_CHECKED,
        ),
        _trajectory(
            "reasoning-ok-2",
            task_group="reasoning-branch",
            accepted=True,
            plan=reasoning_good_plan,
            after=reasoning_good_after,
            candidate=CandidateState.PRESENT,
            checker=CheckerState.NOT_CHECKED,
        ),
        _trajectory(
            "reasoning-bad-1",
            task_group="reasoning-branch",
            accepted=False,
            plan=reasoning_bad_plan,
            after=reasoning_bad_after,
            candidate=CandidateState.PRESENT,
            checker=CheckerState.NOT_CHECKED,
        ),
        _trajectory(
            "reasoning-bad-2",
            task_group="reasoning-branch",
            accepted=False,
            plan=reasoning_bad_plan,
            after=reasoning_bad_after,
            candidate=CandidateState.PRESENT,
            checker=CheckerState.NOT_CHECKED,
        ),
    )
    return TrajectoryValueCorpus(
        corpus_id="semantic-value-state-controlled-v1",
        trajectories=trajectories,
    )


def _evaluation(
    result: OfflineValueComparisonV2, kind: EstimatorKindV2
) -> EstimatorEvaluationV2:
    return next(item for item in result.evaluations if item.estimator is kind)


def _estimate(
    result: OfflineValueComparisonV2, kind: EstimatorKindV2, observation_id: str
) -> SemanticStateValueEstimate:
    return next(
        item
        for item in _evaluation(result, kind).estimates
        if item.observation_id == observation_id
    )


def test_semantic_signature_removes_exact_identity_but_preserves_state_meaning() -> (
    None
):
    corpus = _controlled_corpus()
    left_state = corpus.trajectories[0].extraction.states[1]
    right_state = corpus.trajectories[1].extraction.states[1]
    left = abstract_value_state(left_state, corpus.trajectories[0].extraction.states[0])
    right = abstract_value_state(
        right_state, corpus.trajectories[1].extraction.states[0]
    )

    assert left == right
    assert left.object_type_counts[0].semantic_class == "polynomial.map"
    assert {item.semantic_class for item in left.artifact_role_counts} == {
        "MATHEMATICAL_RESULT",
        "OBLIGATION",
    }
    assert left.open_obligation_class_counts[0].semantic_class == "polynomial.identity"
    assert left.scope_class.value == "DECLARED"
    assert left.scope_relation.value == "INTRODUCED"
    serialized = left.model_dump_json()
    for forbidden in (
        left_state.hard_state.latest_candidate_digest,
        left_state.hard_state.scope_digest,
        left_state.hard_state.artifacts[0].artifact_uri,
    ):
        assert forbidden is not None
        assert forbidden not in serialized


def test_scope_relations_and_rejected_escalation_are_explicit() -> None:
    trajectory = _controlled_corpus().trajectories[0]
    plan, tool, after = trajectory.extraction.states[:3]

    introduced = abstract_value_state(tool, plan)
    stable = abstract_value_state(after, tool)
    changed_hard = after.hard_state.model_copy(
        update={
            "scope_digest": _sha("changed-scope"),
            "latest_meaningful_transitions": (MilestoneKind.SCOPE_CHANGED,),
        }
    )
    changed_state = _state(
        index=3,
        boundary=StateBoundary.TOOL_RESULT,
        hard=changed_hard,
        plan="Change only the bounded scope.",
    )
    removed_hard = changed_hard.model_copy(update={"scope_digest": None})
    removed_state = _state(
        index=4,
        boundary=StateBoundary.TOOL_RESULT,
        hard=removed_hard,
        plan="Remove the declared scope.",
    )
    rejected_hard = changed_hard.model_copy(
        update={
            "scope_escalation_errors": ("requested scope exceeds bound",),
            "latest_meaningful_transitions": (MilestoneKind.SCOPE_ESCALATION_REJECTED,),
        }
    )
    rejected_state = _state(
        index=5,
        boundary=StateBoundary.TOOL_RESULT,
        hard=rejected_hard,
        plan="Keep the bounded scope after rejection.",
    )

    assert introduced.scope_relation.value == "INTRODUCED"
    assert stable.scope_relation.value == "STABLE"
    assert abstract_value_state(changed_state, after).scope_relation.value == "CHANGED"
    assert (
        abstract_value_state(removed_state, changed_state).scope_relation.value
        == "REMOVED"
    )
    rejected = abstract_value_state(rejected_state, changed_state)
    assert rejected.scope_class.value == "ESCALATION_REJECTED"
    assert rejected.scope_relation.value == "ESCALATION_REJECTED"


def test_six_estimators_are_leave_one_trajectory_out_and_report_uncertainty() -> None:
    result = evaluate_semantic_trajectories(_controlled_corpus())

    assert tuple(item.estimator for item in result.evaluations) == tuple(
        EstimatorKindV2
    )
    assert result.learned_components is False
    assert result.assurance_authority is False
    assert result.source_corpus == _controlled_corpus()
    assert all(item.metrics.observation_count == 24 for item in result.evaluations)
    for evaluation in result.evaluations:
        for estimate in evaluation.estimates:
            assert estimate.trajectory_id not in estimate.supporting_trajectory_ids
            assert estimate.uncertainty.support_count == len(
                estimate.supporting_trajectory_ids
            )
            assert estimate.uncertainty.lower <= estimate.estimated_value
            assert estimate.uncertainty.upper >= estimate.estimated_value


def test_abstract_state_recovers_support_fragmented_by_exact_identities() -> None:
    result = evaluate_semantic_trajectories(_controlled_corpus())
    left_key = "checker-ok-1:1"
    right_key = "checker-ok-2:1"
    exact_left = _estimate(result, EstimatorKindV2.JACOBIAN_TYPED_EXACT, left_key)
    exact_right = _estimate(result, EstimatorKindV2.JACOBIAN_TYPED_EXACT, right_key)
    abstract_left = _estimate(result, EstimatorKindV2.ABSTRACT_VALUE_STATE, left_key)
    abstract_right = _estimate(result, EstimatorKindV2.ABSTRACT_VALUE_STATE, right_key)

    assert exact_left.exact_typed_state_digest != exact_right.exact_typed_state_digest
    assert exact_left.cluster_id != exact_right.cluster_id
    assert exact_left.value_source is ValueSource.TASK_GROUP_PRIOR
    assert abstract_left.abstract_value_state_digest == (
        abstract_right.abstract_value_state_digest
    )
    assert abstract_left.cluster_id == abstract_right.cluster_id
    assert abstract_left.value_source is ValueSource.CLUSTER
    assert abstract_left.estimated_value == 1.0
    assert (
        _evaluation(
            result, EstimatorKindV2.ABSTRACT_VALUE_STATE
        ).metrics.task_group_fallback_count
        < _evaluation(
            result, EstimatorKindV2.JACOBIAN_TYPED_EXACT
        ).metrics.task_group_fallback_count
    )


def test_text_separates_opposite_policy_branches_with_same_abstract_state() -> None:
    result = evaluate_semantic_trajectories(_controlled_corpus())
    accepted_key = "reasoning-ok-1:2"
    rejected_key = "reasoning-bad-1:2"
    abstract_accepted = _estimate(
        result, EstimatorKindV2.ABSTRACT_VALUE_STATE, accepted_key
    )
    abstract_rejected = _estimate(
        result, EstimatorKindV2.ABSTRACT_VALUE_STATE, rejected_key
    )
    hybrid_accepted = _estimate(
        result, EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT, accepted_key
    )
    hybrid_rejected = _estimate(
        result, EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT, rejected_key
    )

    assert abstract_accepted.abstract_value_state_digest == (
        abstract_rejected.abstract_value_state_digest
    )
    assert abstract_accepted.cluster_id == abstract_rejected.cluster_id
    assert abstract_accepted.reasoning_text_digest != (
        abstract_rejected.reasoning_text_digest
    )
    assert hybrid_accepted.cluster_id != hybrid_rejected.cluster_id
    assert hybrid_accepted.estimated_value == 1.0
    assert hybrid_rejected.estimated_value == 0.0
    assert (
        _evaluation(
            result, EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT
        ).metrics.brier_score
        < _evaluation(result, EstimatorKindV2.ABSTRACT_VALUE_STATE).metrics.brier_score
    )


def test_result_is_deterministic_and_source_corpus_is_digest_bound() -> None:
    corpus = _controlled_corpus()
    left = evaluate_semantic_trajectories(corpus)
    right = evaluate_semantic_trajectories(
        corpus.model_copy(update={"trajectories": corpus.trajectories[::-1]})
    )

    assert left.evaluations == right.evaluations
    assert left.corpus_digest != right.corpus_digest
    payload = left.model_dump(mode="json")
    payload["source_corpus"]["trajectories"][0]["trajectory_id"] = "substituted"
    with pytest.raises(ValueError, match="corpus digest mismatch"):
        OfflineValueComparisonV2.model_validate(payload)

    config_payload = left.model_dump(mode="json")
    config_payload["evaluator_config"]["text_similarity_threshold_millionths"] += 1
    with pytest.raises(ValueError, match="evaluator config mismatch"):
        OfflineValueComparisonV2.model_validate(config_payload)

    state_payload = left.model_dump(mode="json")
    state_payload["evaluations"][0]["estimates"][0]["exact_typed_state_digest"] = _sha(
        "substituted-exact-state"
    )
    with pytest.raises(ValueError, match="stale or source-substituted"):
        OfflineValueComparisonV2.model_validate(state_payload)


def test_validation_recomputes_cluster_members_from_bound_source_corpus() -> None:
    result = evaluate_semantic_trajectories(_controlled_corpus())
    evaluation = _evaluation(result, EstimatorKindV2.GROUP_ROLLOUT)
    original = evaluation.estimates[0]
    existing_cluster_ids = {cluster.cluster_id for cluster in evaluation.clusters}
    substituted_members, substituted_cluster_id = next(
        (
            members,
            abstraction_module._cluster_id(evaluation.estimator, members),
        )
        for members in (
            tuple(sorted((*original.cluster_member_observation_ids, candidate)))
            for candidate in (
                estimate.observation_id
                for estimate in evaluation.estimates
                if estimate.observation_id
                not in original.cluster_member_observation_ids
            )
        )
        if abstraction_module._cluster_id(evaluation.estimator, members)
        not in existing_cluster_ids
    )
    substituted_estimate = original.model_copy(
        update={
            "cluster_id": substituted_cluster_id,
            "cluster_member_observation_ids": substituted_members,
        }
    )
    estimates = (substituted_estimate, *evaluation.estimates[1:])
    referenced_cluster_ids = {estimate.cluster_id for estimate in estimates}
    clusters = (
        *(
            cluster
            for cluster in evaluation.clusters
            if cluster.cluster_id in referenced_cluster_ids
        ),
        evaluation.clusters[0].model_copy(
            update={
                "cluster_id": substituted_cluster_id,
                "member_observation_ids": substituted_members,
                "member_trajectory_ids": tuple(
                    sorted(
                        {
                            estimate.trajectory_id
                            for estimate in evaluation.estimates
                            if estimate.observation_id in substituted_members
                        }
                    )
                ),
            }
        ),
    )
    substituted_evaluation = EstimatorEvaluationV2(
        estimator=evaluation.estimator,
        clusters=clusters,
        estimates=estimates,
        metrics=abstraction_module._metrics(estimates, clusters),
    )
    payload = result.model_dump(mode="json")
    payload["evaluations"][0] = substituted_evaluation.model_dump(mode="json")

    with pytest.raises(
        ValueError, match=r"clusters are stale|cluster membership is stale"
    ):
        OfflineValueComparisonV2.model_validate(payload)


def test_controlled_comparison_summary_is_immutable() -> None:
    result = evaluate_semantic_trajectories(_controlled_corpus())
    fixture = json.loads(
        (
            ROOT
            / "tests/unit/tooling/fixtures/trajectory_value/pr6_semantic/comparison-summary.json"
        ).read_text(encoding="utf-8")
    )

    assert fixture["corpus_id"] == result.corpus_id
    assert fixture["corpus_digest"] == result.corpus_digest
    assert fixture["evaluator_id"] == result.evaluator_id
    assert fixture["evaluator_config"] == result.evaluator_config.model_dump(
        mode="json"
    )
    assert fixture["state_roles"] == result.state_roles
    assert fixture["observation_count"] == 24
    for evaluation in result.evaluations:
        assert fixture["metrics"][evaluation.estimator.value] == (
            evaluation.metrics.model_dump(mode="json")
        )


def test_schema_files_match_models_and_resolve_external_references() -> None:
    schemas: dict[str, type[BaseModel]] = {
        "trajectory-value-state-abstraction-v1.schema.json": (
            AbstractValueStateSignature
        ),
        "trajectory-value-evaluation-v2.schema.json": OfflineValueComparisonV2,
    }
    corpus = _controlled_corpus()
    result = evaluate_semantic_trajectories(corpus)
    instances = {
        "trajectory-value-state-abstraction-v1.schema.json": result.evaluations[0]
        .estimates[0]
        .abstract_value_state.model_dump(mode="json"),
        "trajectory-value-evaluation-v2.schema.json": result.model_dump(mode="json"),
    }
    schema_dir = ROOT / "docs/reference/evaluations/schemas"
    registry = Registry()
    for dependency in (
        "trajectory-state-v1.schema.json",
        "trajectory-value-corpus-v1.schema.json",
        "trajectory-value-state-abstraction-v1.schema.json",
    ):
        schema = json.loads((schema_dir / dependency).read_text(encoding="utf-8"))
        registry = registry.with_resource(
            schema["$id"], DRAFT202012.create_resource(schema)
        )
    for filename, model in schemas.items():
        schema = json.loads((schema_dir / filename).read_text(encoding="utf-8"))
        assert schema == model.model_json_schema()
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, registry=registry).validate(instances[filename])
