from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from jacobian.eval.trajectory_score import (
    CreditReason,
    ScoredTrajectoryState,
    TerminalResult,
    TrajectoryScoreError,
    TrajectoryScoreReplay,
    replay_offline_values,
)
from jacobian.eval.trajectory_state import StateBoundary
from jacobian.eval.trajectory_value import (
    ClusterSummary,
    EstimatorEvaluation,
    EstimatorKind,
    EstimatorMetrics,
    OfflineValueComparison,
    StateValueEstimate,
    ValueEstimatorConfig,
    ValueSource,
)

ROOT = Path(__file__).resolve().parents[3]


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _comparison(*, reward: Literal[0, 1] = 1) -> OfflineValueComparison:
    observation_ids = tuple(f"target:{index}" for index in range(4))
    values = (0.4, 0.7, 0.2, 0.6)
    boundaries = (
        StateBoundary.PLAN,
        StateBoundary.TOOL_RESULT,
        StateBoundary.AFTER_TOOL,
        StateBoundary.TOOL_RESULT,
    )
    milestones = (False, True, False, True)
    evaluations: list[EstimatorEvaluation] = []
    for estimator in EstimatorKind:
        cluster_id = _sha("cluster-" + estimator.value)
        cluster = ClusterSummary(
            cluster_id=cluster_id,
            estimator=estimator,
            member_observation_ids=observation_ids,
            member_trajectory_ids=("support-a", "support-b", "target"),
            feature_summary=f"explainable {estimator.value} fixture cluster",
        )
        estimates = tuple(
            StateValueEstimate(
                observation_id=observation_id,
                trajectory_id="target",
                task_group="polynomial-task",
                state_index=index,
                boundary=boundaries[index],
                milestone_eligible=milestones[index],
                estimator=estimator,
                cluster_id=cluster_id,
                estimated_value=values[index],
                eventual_terminal_reward=reward,
                value_source=ValueSource.CLUSTER,
                supporting_trajectory_ids=("support-a", "support-b"),
                cluster_member_observation_ids=observation_ids,
                typed_compatibility_digest=_sha(f"typed-{index}"),
                reasoning_text_digest=_sha(f"text-{index}"),
                numerical_milestones=("2",),
            )
            for index, observation_id in enumerate(observation_ids)
        )
        evaluations.append(
            EstimatorEvaluation(
                estimator=estimator,
                clusters=(cluster,),
                estimates=estimates,
                metrics=EstimatorMetrics(
                    observation_count=4,
                    trajectory_count=3,
                    cluster_count=1,
                    task_group_fallback_count=0,
                    brier_score=0.25,
                    mean_absolute_error=0.5,
                ),
            )
        )
    return OfflineValueComparison(
        corpus_id="trajectory-score-fixture-v1",
        corpus_digest=_sha("corpus"),
        evaluator_config=ValueEstimatorConfig(),
        evaluations=tuple(evaluations),
    )


def test_replay_exposes_state_cluster_value_delta_credit_and_terminal_result() -> None:
    replay = replay_offline_values(
        _comparison(),
        trajectory_id="target",
        estimator=EstimatorKind.HYBRID_TYPED_TEXT,
    )

    assert [state.estimated_value for state in replay.states] == [0.4, 0.7, 0.2, 0.6]
    assert [state.value_delta for state in replay.states] == [None, 0.3, -0.5, 0.4]
    assert [state.transition_credit for state in replay.states] == [0.0, 0.3, 0.0, 0.4]
    assert [state.cumulative_milestone_credit for state in replay.states] == [
        0.0,
        0.3,
        0.3,
        0.7,
    ]
    assert [state.credit_reason for state in replay.states] == [
        CreditReason.INITIAL_STATE,
        CreditReason.MILESTONE_VALUE_DELTA,
        CreditReason.NON_MILESTONE_ZERO,
        CreditReason.MILESTONE_VALUE_DELTA,
    ]
    assert replay.total_milestone_credit == 0.7
    assert replay.eventual_terminal_result is TerminalResult.ACCEPTED
    assert all(state.cluster_feature_summary for state in replay.states)


def test_nonmilestone_value_drop_is_visible_but_has_zero_credit() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    after_tool = replay.states[2]

    assert after_tool.boundary is StateBoundary.AFTER_TOOL
    assert after_tool.value_delta == -0.5
    assert after_tool.milestone_eligible is False
    assert after_tool.transition_credit == 0.0
    assert after_tool.credit_reason is CreditReason.NON_MILESTONE_ZERO


def test_replay_is_observation_only_and_cannot_change_assurance() -> None:
    comparison = _comparison()
    before = comparison.model_dump(mode="json")
    replay = replay_offline_values(comparison, trajectory_id="target")

    assert comparison.model_dump(mode="json") == before
    assert replay.observation_only is True
    assert replay.chooses_tools is False
    assert replay.changes_prompts is False
    assert replay.mutates_runtime is False
    assert replay.assurance_authority is False
    assert all(state.assurance_authority is False for state in replay.states)


def test_explicit_estimator_selects_its_bound_clusters() -> None:
    replay = replay_offline_values(
        _comparison(),
        trajectory_id="target",
        estimator=EstimatorKind.REASONING_TEXT,
    )

    assert replay.estimator is EstimatorKind.REASONING_TEXT
    assert replay.states[0].cluster_id == _sha(
        "cluster-" + EstimatorKind.REASONING_TEXT.value
    )
    assert replay.states[0].supporting_trajectory_ids == ("support-a", "support-b")


def test_rejected_terminal_result_remains_separate_from_value_and_credit() -> None:
    replay = replay_offline_values(_comparison(reward=0), trajectory_id="target")

    assert replay.eventual_terminal_result is TerminalResult.REJECTED
    assert replay.eventual_terminal_reward == 0
    assert replay.states[-1].estimated_value == 0.6
    assert replay.total_milestone_credit == 0.7


def test_missing_trajectory_and_stale_cluster_binding_fail_closed() -> None:
    comparison = _comparison()
    with pytest.raises(TrajectoryScoreError, match="absent"):
        replay_offline_values(comparison, trajectory_id="missing")

    hybrid = comparison.evaluations[-1]
    stale_hybrid = hybrid.model_copy(update={"clusters": ()})
    stale = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], stale_hybrid)}
    )
    with pytest.raises(TrajectoryScoreError, match="declared cluster"):
        replay_offline_values(stale, trajectory_id="target")


def test_replay_requires_plan_first_and_unique_state_identity() -> None:
    comparison = _comparison()
    hybrid = comparison.evaluations[-1]
    first = hybrid.estimates[0].model_copy(
        update={"boundary": StateBoundary.TOOL_RESULT}
    )
    malformed_hybrid = hybrid.model_copy(
        update={"estimates": (first, *hybrid.estimates[1:])}
    )
    malformed = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], malformed_hybrid)}
    )

    with pytest.raises(TrajectoryScoreError, match="begin with PLAN"):
        replay_offline_values(malformed, trajectory_id="target")

    substituted = hybrid.estimates[1].model_copy(update={"observation_id": "target:99"})
    substituted_hybrid = hybrid.model_copy(
        update={"estimates": (hybrid.estimates[0], substituted, *hybrid.estimates[2:])}
    )
    substituted_comparison = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], substituted_hybrid)}
    )
    with pytest.raises(TrajectoryScoreError, match="not bound to its state index"):
        replay_offline_values(substituted_comparison, trajectory_id="target")


def test_closed_state_contract_rejects_credit_for_prose_only_transition() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    payload = replay.states[2].model_dump(mode="json")
    payload["transition_credit"] = -0.5
    with pytest.raises(ValidationError, match="non-milestone transition credit"):
        ScoredTrajectoryState.model_validate(payload)

    root_payload = replay.model_dump(mode="json")
    root_payload["tool_choice"] = "math.run"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrajectoryScoreReplay.model_validate(root_payload)


def test_replay_is_deterministic_and_binds_the_complete_comparison() -> None:
    comparison = _comparison()
    left = replay_offline_values(comparison, trajectory_id="target")
    right = replay_offline_values(comparison, trajectory_id="target")
    changed = comparison.model_copy(
        update={"corpus_id": "trajectory-score-fixture-v1-changed"}
    )
    changed_replay = replay_offline_values(changed, trajectory_id="target")

    assert left == right
    assert left.comparison_digest != changed_replay.comparison_digest
    assert left.source_corpus_digest == comparison.corpus_digest


def test_controlled_replay_summary_is_immutable_and_reproducible() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    fixture_path = (
        ROOT
        / "tests/unit/tooling/fixtures/trajectory_score/pr3_controlled/replay-summary.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["scorer_id"] == replay.scorer_id
    assert fixture["comparison_digest"] == replay.comparison_digest
    assert fixture["source_corpus_digest"] == replay.source_corpus_digest
    assert fixture["estimator"] == replay.estimator.value
    assert fixture["eventual_terminal_result"] == replay.eventual_terminal_result.value
    assert fixture["eventual_terminal_reward"] == replay.eventual_terminal_reward
    assert fixture["total_milestone_credit"] == replay.total_milestone_credit
    for expected, actual in zip(fixture["states"], replay.states, strict=True):
        payload = actual.model_dump(mode="json")
        assert expected == {key: payload[key] for key in expected}


def test_replay_rejects_cluster_support_outside_cluster_trajectories() -> None:
    comparison = _comparison()
    hybrid = comparison.evaluations[-1]
    corrupted = hybrid.estimates[0].model_copy(
        update={"supporting_trajectory_ids": ("support-a", "foreign")}
    )
    hybrid_corrupted = hybrid.model_copy(
        update={"estimates": (corrupted, *hybrid.estimates[1:])}
    )
    corrupted_comparison = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], hybrid_corrupted)}
    )
    with pytest.raises(TrajectoryScoreError, match="outside its cluster members"):
        replay_offline_values(corrupted_comparison, trajectory_id="target")


def test_replay_rejects_cumulative_credit_inconsistent_with_transition_chain() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    payload = replay.states[1].model_dump(mode="json")
    payload["cumulative_milestone_credit"] = 0.31
    bad = replay.model_copy(
        update={
            "states": (
                replay.states[0],
                replay.states[1].model_copy(
                    update={"cumulative_milestone_credit": 0.31}
                ),
                *replay.states[2:],
            )
        }
    )
    with pytest.raises(ValidationError, match="running total of transition credits"):
        TrajectoryScoreReplay.model_validate(bad.model_dump(mode="json"))


def test_replay_rejects_broken_observation_chain() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    bad = replay.model_copy(
        update={
            "states": (
                replay.states[0],
                replay.states[1].model_copy(
                    update={"previous_observation_id": "target:999"}
                ),
                *replay.states[2:],
            )
        }
    )
    with pytest.raises(ValidationError, match="previous observation must chain"):
        TrajectoryScoreReplay.model_validate(bad.model_dump(mode="json"))


def test_replay_schema_matches_closed_model_and_validates_output() -> None:
    path = (
        ROOT
        / "docs/reference/evaluations/schemas/trajectory-score-replay-v1.schema.json"
    )
    schema = json.loads(path.read_text(encoding="utf-8"))
    replay = replay_offline_values(_comparison(), trajectory_id="target")

    assert schema == TrajectoryScoreReplay.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(replay.model_dump(mode="json"))
