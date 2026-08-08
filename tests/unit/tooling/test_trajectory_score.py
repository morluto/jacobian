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
    support_observation_ids = ("support-a:0", "support-b:0")
    cluster_members = tuple(sorted((*observation_ids, *support_observation_ids)))
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
            member_observation_ids=cluster_members,
            member_trajectory_ids=("support-a", "support-b", "target"),
            feature_summary=f"explainable {estimator.value} fixture cluster",
        )
        target_estimates = tuple(
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
                cluster_member_observation_ids=cluster_members,
                typed_compatibility_digest=_sha(f"typed-{index}"),
                reasoning_text_digest=_sha(f"text-{index}"),
                numerical_milestones=("2",),
            )
            for index, observation_id in enumerate(observation_ids)
        )
        support_estimates = tuple(
            StateValueEstimate(
                observation_id=observation_id,
                trajectory_id=trajectory_id,
                task_group="polynomial-task",
                state_index=0,
                boundary=StateBoundary.PLAN,
                milestone_eligible=False,
                estimator=estimator,
                cluster_id=cluster_id,
                estimated_value=0.5,
                eventual_terminal_reward=support_reward,
                value_source=ValueSource.CLUSTER,
                supporting_trajectory_ids=tuple(
                    sorted({"support-a", "support-b", "target"} - {trajectory_id})
                ),
                cluster_member_observation_ids=cluster_members,
                typed_compatibility_digest=_sha(f"typed-{trajectory_id}"),
                reasoning_text_digest=_sha(f"text-{trajectory_id}"),
                numerical_milestones=("2",),
            )
            for observation_id, trajectory_id, support_reward in (
                ("support-a:0", "support-a", 1),
                ("support-b:0", "support-b", 0),
            )
        )
        estimates = (*support_estimates, *target_estimates)
        evaluations.append(
            EstimatorEvaluation(
                estimator=estimator,
                clusters=(cluster,),
                estimates=estimates,
                metrics=EstimatorMetrics(
                    observation_count=6,
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


def test_task_group_prior_support_is_validated_before_replay() -> None:
    comparison = _comparison()
    hybrid = comparison.evaluations[-1]
    stale = hybrid.estimates[2].model_copy(
        update={
            "value_source": ValueSource.TASK_GROUP_PRIOR,
            "supporting_trajectory_ids": ("support-a",),
        }
    )
    stale_hybrid = hybrid.model_copy(
        update={"estimates": (*hybrid.estimates[:2], stale, *hybrid.estimates[3:])}
    )
    stale_comparison = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], stale_hybrid)}
    )

    with pytest.raises(TrajectoryScoreError, match="support is stale"):
        replay_offline_values(stale_comparison, trajectory_id="target")


def test_cross_estimator_terminal_reward_conflicts_are_rejected() -> None:
    comparison = _comparison()
    text = comparison.evaluations[2]
    conflicted = text.estimates[2].model_copy(update={"eventual_terminal_reward": 0})
    conflicted_text = text.model_copy(
        update={"estimates": (*text.estimates[:2], conflicted, *text.estimates[3:])}
    )
    conflicted_comparison = comparison.model_copy(
        update={
            "evaluations": (
                *comparison.evaluations[:2],
                conflicted_text,
                *comparison.evaluations[3:],
            )
        }
    )

    with pytest.raises(TrajectoryScoreError, match="conflicting terminal rewards"):
        replay_offline_values(conflicted_comparison, trajectory_id="target")


def test_replay_requires_plan_first_and_unique_state_identity() -> None:
    comparison = _comparison()
    hybrid = comparison.evaluations[-1]
    first_target = hybrid.estimates[2].model_copy(
        update={"boundary": StateBoundary.TOOL_RESULT}
    )
    malformed_hybrid = hybrid.model_copy(
        update={
            "estimates": (*hybrid.estimates[:2], first_target, *hybrid.estimates[3:])
        }
    )
    malformed = comparison.model_copy(
        update={"evaluations": (*comparison.evaluations[:-1], malformed_hybrid)}
    )

    with pytest.raises(TrajectoryScoreError, match="begin with PLAN"):
        replay_offline_values(malformed, trajectory_id="target")

    substituted = hybrid.estimates[3].model_copy(update={"observation_id": "target:99"})
    substituted_hybrid = hybrid.model_copy(
        update={
            "estimates": (*hybrid.estimates[:3], substituted, *hybrid.estimates[4:])
        }
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


def test_replay_contract_rebinds_previous_delta_and_cumulative_credit() -> None:
    replay = replay_offline_values(_comparison(), trajectory_id="target")
    wrong_previous = replay.model_dump(mode="json")
    wrong_previous["states"][2]["previous_observation_id"] = "target:0"
    with pytest.raises(ValidationError, match="previous observation ids are stale"):
        TrajectoryScoreReplay.model_validate(wrong_previous)

    wrong_delta = replay.model_dump(mode="json")
    wrong_delta["states"][1]["value_delta"] = 0.1
    wrong_delta["states"][1]["transition_credit"] = 0.1
    wrong_delta["states"][1]["cumulative_milestone_credit"] = 0.1
    with pytest.raises(ValidationError, match="value deltas are stale"):
        TrajectoryScoreReplay.model_validate(wrong_delta)

    reset_cumulative = replay.model_dump(mode="json")
    reset_cumulative["states"][2]["cumulative_milestone_credit"] = 0.0
    with pytest.raises(ValidationError, match="cumulative credit is stale"):
        TrajectoryScoreReplay.model_validate(reset_cumulative)


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
