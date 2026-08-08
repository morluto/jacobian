"""Observation-only replay scoring for offline trajectory-value estimates.

The scorer joins an immutable PR2 comparison into an inspectable trajectory.
It does not invoke a model or tool, choose an action, mutate a prompt or
runtime, or participate in mathematical assurance.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian.contracts.results import ContractModel
from jacobian.eval.trajectory_state import StateBoundary
from jacobian.eval.trajectory_value import (
    ClusterSummary,
    EstimatorEvaluation,
    EstimatorKind,
    OfflineValueComparison,
    StateValueEstimate,
    ValueSource,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"


class TrajectoryScoreError(ValueError):
    """A comparison cannot support the requested observation-only replay."""


class TerminalResult(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class CreditReason(StrEnum):
    INITIAL_STATE = "INITIAL_STATE"
    NON_MILESTONE_ZERO = "NON_MILESTONE_ZERO"
    MILESTONE_VALUE_DELTA = "MILESTONE_VALUE_DELTA"


class ScoredTrajectoryState(ContractModel):
    observation_id: str = Field(min_length=3, max_length=256)
    trajectory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    task_group: str = Field(pattern=_IDENTIFIER_PATTERN)
    state_index: int = Field(ge=0, strict=True)
    boundary: StateBoundary
    typed_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    milestone_eligible: bool
    estimator: EstimatorKind
    cluster_id: str = Field(pattern=_DIGEST_PATTERN)
    cluster_feature_summary: str = Field(min_length=1, max_length=1024)
    value_source: ValueSource
    supporting_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    estimated_value: float = Field(ge=0.0, le=1.0)
    previous_observation_id: str | None = Field(default=None, max_length=256)
    value_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    transition_credit: float = Field(ge=-1.0, le=1.0)
    cumulative_milestone_credit: float = Field(ge=-1000.0, le=1000.0)
    credit_reason: CreditReason
    eventual_terminal_result: TerminalResult
    eventual_terminal_reward: Literal[0, 1]
    observation_only: Literal[True] = True
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_local_credit_semantics(self) -> Self:
        if self.previous_observation_id is None:
            if self.value_delta is not None or self.transition_credit != 0.0:
                raise ValueError("initial state has no delta and zero credit")
            if self.credit_reason is not CreditReason.INITIAL_STATE:
                raise ValueError("initial state requires the initial credit reason")
        elif self.value_delta is None:
            raise ValueError("non-initial state requires a value delta")
        if not self.milestone_eligible and self.transition_credit != 0.0:
            raise ValueError("non-milestone transition credit must be zero")
        if (
            self.previous_observation_id is not None
            and self.milestone_eligible
            and self.transition_credit != self.value_delta
        ):
            raise ValueError("milestone credit must equal the estimated value delta")
        return self


class TrajectoryScoreReplay(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-score-replay-v1.schema.json"
        },
    )

    replay_schema_version: Literal["1"] = "1"
    scorer_id: Literal["jacobian.observation-only-trajectory-scorer.v1"] = (
        "jacobian.observation-only-trajectory-scorer.v1"
    )
    comparison_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_corpus_digest: str = Field(pattern=_DIGEST_PATTERN)
    trajectory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    task_group: str = Field(pattern=_IDENTIFIER_PATTERN)
    estimator: EstimatorKind
    eventual_terminal_result: TerminalResult
    eventual_terminal_reward: Literal[0, 1]
    states: tuple[ScoredTrajectoryState, ...] = Field(min_length=1)
    total_milestone_credit: float = Field(ge=-1000.0, le=1000.0)
    credit_semantics: Literal[
        "non-milestone=0;milestone=current-value-minus-previous-selected-value"
    ] = "non-milestone=0;milestone=current-value-minus-previous-selected-value"
    observation_only: Literal[True] = True
    chooses_tools: Literal[False] = False
    changes_prompts: Literal[False] = False
    mutates_runtime: Literal[False] = False
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_bound_ordered_replay(self) -> Self:
        _validate_replay_identity(self)
        _validate_replay_sequence(self.states)
        if self.total_milestone_credit != self.states[-1].cumulative_milestone_credit:
            raise ValueError("total credit must equal the last cumulative credit")
        return self


def _validate_replay_identity(replay: TrajectoryScoreReplay) -> None:
    states = replay.states
    if states[0].boundary is not StateBoundary.PLAN:
        raise ValueError("replay must begin at a PLAN observation")
    if any(state.trajectory_id != replay.trajectory_id for state in states):
        raise ValueError("replay state trajectory identity mismatch")
    if any(state.task_group != replay.task_group for state in states):
        raise ValueError("replay state task-group mismatch")
    if any(state.estimator is not replay.estimator for state in states):
        raise ValueError("replay state estimator mismatch")
    if tuple(state.state_index for state in states) != tuple(
        sorted(state.state_index for state in states)
    ):
        raise ValueError("replay states must be ordered by source state index")
    if len({state.state_index for state in states}) != len(states):
        raise ValueError("replay state indices must be unique")
    if any(
        state.eventual_terminal_reward != replay.eventual_terminal_reward
        or state.eventual_terminal_result is not replay.eventual_terminal_result
        for state in states
    ):
        raise ValueError("replay states must share the bound terminal result")


def _validate_replay_sequence(states: tuple[ScoredTrajectoryState, ...]) -> None:
    cumulative = 0.0
    previous: ScoredTrajectoryState | None = None
    for state in states:
        expected_previous = None if previous is None else previous.observation_id
        if state.previous_observation_id != expected_previous:
            raise ValueError("replay previous observation ids are stale")
        expected_delta = (
            None
            if previous is None
            else _round_value(state.estimated_value - previous.estimated_value)
        )
        if state.value_delta != expected_delta:
            raise ValueError("replay value deltas are stale")
        expected_credit = (
            expected_delta
            if state.milestone_eligible and expected_delta is not None
            else 0.0
        )
        if state.transition_credit != expected_credit:
            raise ValueError("replay transition credits are stale")
        cumulative = _round_value(cumulative + expected_credit)
        if state.cumulative_milestone_credit != cumulative:
            raise ValueError("replay cumulative credit is stale")
        previous = state


def _round_value(value: float) -> float:
    return round(value, 12)


def _comparison_digest(comparison: OfflineValueComparison) -> str:
    payload = json.dumps(
        comparison.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _evaluation(
    comparison: OfflineValueComparison, estimator: EstimatorKind
) -> EstimatorEvaluation:
    matches = tuple(
        evaluation
        for evaluation in comparison.evaluations
        if evaluation.estimator is estimator
    )
    if len(matches) != 1:
        raise TrajectoryScoreError(
            "comparison does not contain one requested estimator"
        )
    return matches[0]


def _terminal_result(reward: Literal[0, 1]) -> TerminalResult:
    return TerminalResult.ACCEPTED if reward == 1 else TerminalResult.REJECTED


def _terminal_rewards(
    comparison: OfflineValueComparison,
) -> dict[str, Literal[0, 1]]:
    rewards: dict[str, Literal[0, 1]] = {}
    for evaluation in comparison.evaluations:
        for estimate in evaluation.estimates:
            previous = rewards.get(estimate.trajectory_id)
            if previous is None:
                rewards[estimate.trajectory_id] = estimate.eventual_terminal_reward
            elif previous != estimate.eventual_terminal_reward:
                raise TrajectoryScoreError(
                    "trajectory has conflicting terminal rewards across estimators"
                )
    return rewards


def _validate_estimates(
    evaluation: EstimatorEvaluation,
    trajectory_id: str,
) -> tuple[
    tuple[StateValueEstimate, ...],
    dict[str, ClusterSummary],
    dict[str, StateValueEstimate],
    dict[str, tuple[str, ...]],
]:
    all_estimates, task_groups = _estimate_indexes(evaluation)
    estimates = _trajectory_estimates(evaluation, trajectory_id)
    clusters = _cluster_index(evaluation)
    return estimates, clusters, all_estimates, task_groups


def _estimate_indexes(
    evaluation: EstimatorEvaluation,
) -> tuple[dict[str, StateValueEstimate], dict[str, tuple[str, ...]]]:
    all_estimates = {
        estimate.observation_id: estimate for estimate in evaluation.estimates
    }
    if len(all_estimates) != len(evaluation.estimates):
        raise TrajectoryScoreError("estimator has duplicate observation ids")
    task_groups: dict[str, set[str]] = defaultdict(set)
    task_group_by_trajectory: dict[str, str] = {}
    for estimate in evaluation.estimates:
        previous_group = task_group_by_trajectory.get(estimate.trajectory_id)
        if previous_group is None:
            task_group_by_trajectory[estimate.trajectory_id] = estimate.task_group
        elif previous_group != estimate.task_group:
            raise TrajectoryScoreError("trajectory has inconsistent task groups")
        task_groups[estimate.task_group].add(estimate.trajectory_id)
    return all_estimates, {
        task_group: tuple(sorted(trajectory_ids))
        for task_group, trajectory_ids in task_groups.items()
    }


def _trajectory_estimates(
    evaluation: EstimatorEvaluation, trajectory_id: str
) -> tuple[StateValueEstimate, ...]:
    estimates = tuple(
        sorted(
            (
                estimate
                for estimate in evaluation.estimates
                if estimate.trajectory_id == trajectory_id
            ),
            key=lambda estimate: estimate.state_index,
        )
    )
    if not estimates:
        raise TrajectoryScoreError("trajectory is absent from the comparison")
    if len({estimate.observation_id for estimate in estimates}) != len(estimates):
        raise TrajectoryScoreError(
            "trajectory estimates have duplicate observation ids"
        )
    if len({estimate.state_index for estimate in estimates}) != len(estimates):
        raise TrajectoryScoreError("trajectory estimates have duplicate state indices")
    if any(
        estimate.observation_id != f"{trajectory_id}:{estimate.state_index}"
        for estimate in estimates
    ):
        raise TrajectoryScoreError(
            "observation identity is not bound to its state index"
        )
    if (
        estimates[0].boundary is not StateBoundary.PLAN
        or estimates[0].state_index != 0
        or estimates[0].milestone_eligible
    ):
        raise TrajectoryScoreError("trajectory replay must begin with PLAN")
    return estimates


def _cluster_index(
    evaluation: EstimatorEvaluation,
) -> dict[str, ClusterSummary]:
    clusters = {cluster.cluster_id: cluster for cluster in evaluation.clusters}
    if len(clusters) != len(evaluation.clusters):
        raise TrajectoryScoreError("estimator has duplicate cluster ids")
    if any(
        len(set(cluster.member_observation_ids)) != len(cluster.member_observation_ids)
        for cluster in evaluation.clusters
    ):
        raise TrajectoryScoreError("cluster has duplicate observation members")
    return clusters


def _expected_support(
    estimate: StateValueEstimate,
    cluster: ClusterSummary,
    estimates_by_observation: Mapping[str, StateValueEstimate],
    task_group_trajectories: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if estimate.value_source is ValueSource.TASK_GROUP_PRIOR:
        return tuple(
            trajectory_id
            for trajectory_id in task_group_trajectories.get(estimate.task_group, ())
            if trajectory_id != estimate.trajectory_id
        )
    return tuple(
        sorted(
            {
                member_estimate.trajectory_id
                for member in cluster.member_observation_ids
                if (member_estimate := estimates_by_observation.get(member)) is not None
                and member_estimate.task_group == estimate.task_group
                and member_estimate.trajectory_id != estimate.trajectory_id
            }
        )
    )


def _validate_support(
    estimate: StateValueEstimate,
    cluster: ClusterSummary,
    estimates_by_observation: Mapping[str, StateValueEstimate],
    task_group_trajectories: Mapping[str, tuple[str, ...]],
    rewards: Mapping[str, Literal[0, 1]],
) -> None:
    missing_members = [
        member
        for member in cluster.member_observation_ids
        if member not in estimates_by_observation
    ]
    if missing_members:
        raise TrajectoryScoreError("cluster references unknown observations")
    expected = _expected_support(
        estimate, cluster, estimates_by_observation, task_group_trajectories
    )
    if estimate.supporting_trajectory_ids != expected:
        raise TrajectoryScoreError("estimate support is stale or substituted")
    if not expected:
        raise TrajectoryScoreError("estimate support is empty")
    missing_rewards = [
        trajectory_id for trajectory_id in expected if trajectory_id not in rewards
    ]
    if missing_rewards:
        raise TrajectoryScoreError("estimate support lacks terminal rewards")


def replay_offline_values(
    comparison: OfflineValueComparison,
    *,
    trajectory_id: str,
    estimator: EstimatorKind = EstimatorKind.HYBRID_TYPED_TEXT,
) -> TrajectoryScoreReplay:
    """Replay frozen estimates as state, cluster, value, delta, and credit.

    This function is read-only.  It consumes estimates already computed by
    PR2 and never invokes a model, mathematical capability, or verifier.
    """

    evaluation = _evaluation(comparison, estimator)
    rewards_by_trajectory = _terminal_rewards(comparison)
    raw_estimates, raw_clusters, estimates_by_observation, task_group_trajectories = (
        _validate_estimates(evaluation, trajectory_id)
    )
    estimates = tuple(raw_estimates)
    clusters = dict(raw_clusters)
    task_groups = {estimate.task_group for estimate in estimates}
    if (
        len({estimate.eventual_terminal_reward for estimate in estimates}) != 1
        or len(task_groups) != 1
    ):
        raise TrajectoryScoreError("trajectory estimates have inconsistent bindings")
    reward = rewards_by_trajectory.get(trajectory_id)
    if reward is None:
        raise TrajectoryScoreError("trajectory lacks terminal reward")
    if any(estimate.eventual_terminal_reward != reward for estimate in estimates):
        raise TrajectoryScoreError("trajectory estimates have inconsistent bindings")
    task_group = task_groups.pop()
    terminal = _terminal_result(reward)
    states: list[ScoredTrajectoryState] = []
    cumulative = 0.0
    previous = None
    for estimate in estimates:
        cluster = clusters.get(estimate.cluster_id)
        if cluster is None or estimate.observation_id not in (
            cluster.member_observation_ids
        ):
            raise TrajectoryScoreError("estimate is not bound to its declared cluster")
        if estimate.cluster_member_observation_ids != (cluster.member_observation_ids):
            raise TrajectoryScoreError("estimate carries a stale cluster member set")
        _validate_support(
            estimate,
            cluster,
            estimates_by_observation,
            task_group_trajectories,
            rewards_by_trajectory,
        )
        delta = (
            None
            if previous is None
            else _round_value(estimate.estimated_value - previous.estimated_value)
        )
        credit = delta if estimate.milestone_eligible and delta is not None else 0.0
        cumulative = _round_value(cumulative + credit)
        reason = CreditReason.NON_MILESTONE_ZERO
        if previous is None:
            reason = CreditReason.INITIAL_STATE
        elif estimate.milestone_eligible:
            reason = CreditReason.MILESTONE_VALUE_DELTA
        states.append(
            ScoredTrajectoryState(
                observation_id=estimate.observation_id,
                trajectory_id=estimate.trajectory_id,
                task_group=estimate.task_group,
                state_index=estimate.state_index,
                boundary=estimate.boundary,
                typed_state_digest=estimate.typed_compatibility_digest,
                milestone_eligible=estimate.milestone_eligible,
                estimator=estimator,
                cluster_id=estimate.cluster_id,
                cluster_feature_summary=cluster.feature_summary,
                value_source=estimate.value_source,
                supporting_trajectory_ids=estimate.supporting_trajectory_ids,
                estimated_value=estimate.estimated_value,
                previous_observation_id=(
                    None if previous is None else previous.observation_id
                ),
                value_delta=delta,
                transition_credit=credit,
                cumulative_milestone_credit=cumulative,
                credit_reason=reason,
                eventual_terminal_result=terminal,
                eventual_terminal_reward=reward,
            )
        )
        previous = estimate
    return TrajectoryScoreReplay(
        comparison_digest=_comparison_digest(comparison),
        source_corpus_digest=comparison.corpus_digest,
        trajectory_id=trajectory_id,
        task_group=task_group,
        estimator=estimator,
        eventual_terminal_result=terminal,
        eventual_terminal_reward=reward,
        states=tuple(states),
        total_milestone_credit=cumulative,
    )


__all__ = [
    "CreditReason",
    "ScoredTrajectoryState",
    "TerminalResult",
    "TrajectoryScoreError",
    "TrajectoryScoreReplay",
    "replay_offline_values",
]
