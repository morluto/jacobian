"""Deterministic offline value estimation over observable trajectory states.

This evaluation helper compares cheap state abstractions without training a
model or entering Jacobian's mathematical assurance path.  Terminal rewards
are derived only from completed, exactly bound clean-room verifier evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations, pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, WithJsonSchema, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.results import ContractModel
from jacobian.eval.trajectory_state import (
    ExtractedTrajectoryState,
    StateBoundary,
    TerminalAcceptance,
    TrajectoryExtraction,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,127}$"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/\d+)?(?![\w.])"
)
_STATE_SCHEMA_URI = "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-state-v1.schema.json"


class TrajectoryValueError(ValueError):
    """A labelled corpus cannot support the declared offline comparison."""


class EstimatorKind(StrEnum):
    GROUP_ROLLOUT = "GROUP_ROLLOUT"
    NUMCA_NUMERICAL = "NUMCA_NUMERICAL"
    REASONING_TEXT = "REASONING_TEXT"
    JACOBIAN_TYPED = "JACOBIAN_TYPED"
    HYBRID_TYPED_TEXT = "HYBRID_TYPED_TEXT"


class ValueSource(StrEnum):
    CLUSTER = "CLUSTER"
    TASK_GROUP_PRIOR = "TASK_GROUP_PRIOR"


class ValueEstimatorConfig(ContractModel):
    config_schema_version: Literal["1"] = "1"
    text_similarity_threshold_millionths: int = Field(
        default=350_000, ge=0, le=1_000_000, strict=True
    )
    text_ngram_range: Literal["word-1-2"] = "word-1-2"
    linkage: Literal["average"] = "average"
    validation: Literal["leave-one-trajectory-out"] = "leave-one-trajectory-out"
    state_selection: Literal["plan-milestone-and-post-milestone-after-tool"] = (
        "plan-milestone-and-post-milestone-after-tool"
    )
    trajectory_metric_weighting: Literal["equal"] = "equal"


class LabelledTrajectory(ContractModel):
    trajectory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    task_group: str = Field(pattern=_IDENTIFIER_PATTERN)
    extraction: Annotated[
        TrajectoryExtraction,
        WithJsonSchema({"$ref": _STATE_SCHEMA_URI}),
    ]

    @model_validator(mode="after")
    def require_authoritative_binary_terminal_label(self) -> Self:
        evidence = self.extraction.terminal_evidence
        if evidence is None:
            raise ValueError("offline value estimation requires terminal evidence")
        if evidence.verifier_execution_status != "COMPLETED":
            raise ValueError("terminal verifier must complete")
        if evidence.acceptance not in {
            TerminalAcceptance.ACCEPTED,
            TerminalAcceptance.REJECTED,
        }:
            raise ValueError("terminal label must be accepted or rejected")
        if not (evidence.input_binding_valid and evidence.artifact_binding_valid):
            raise ValueError(
                "terminal label requires exact input and artifact bindings"
            )
        if evidence.source_binding_digest != self.extraction.source_digest:
            raise ValueError(
                "terminal evidence must be bound to the exact extracted trajectory"
            )
        states = self.extraction.states
        if not states or states[-1].boundary is not StateBoundary.TERMINAL:
            raise ValueError("labelled extraction must end at a terminal boundary")
        if states[0].boundary is not StateBoundary.PLAN or states[0].index != 0:
            raise ValueError("labelled extraction must begin at a PLAN observation")
        if not any(
            state.boundary is not StateBoundary.TERMINAL
            and state.boundary is not StateBoundary.FINAL
            for state in states
        ):
            raise ValueError(
                "labelled extraction must contain at least one selectable observation"
            )
        return self


class TrajectoryValueCorpus(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-corpus-v1.schema.json"
        },
    )

    corpus_schema_version: Literal["1"] = "1"
    corpus_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    evaluator_config: ValueEstimatorConfig = ValueEstimatorConfig()
    trajectories: tuple[LabelledTrajectory, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_independent_repeated_rollouts(self) -> Self:
        trajectory_ids = [item.trajectory_id for item in self.trajectories]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("trajectory ids must be unique")
        source_digests = [item.extraction.source_digest for item in self.trajectories]
        if len(set(source_digests)) != len(source_digests):
            raise ValueError(
                "duplicate transcript digests cannot become extra rollouts"
            )
        groups: dict[str, list[LabelledTrajectory]] = defaultdict(list)
        for trajectory in self.trajectories:
            groups[trajectory.task_group].append(trajectory)
        for group_id, members in groups.items():
            if len(members) < 2:
                raise ValueError(f"task group {group_id!r} requires repeated rollouts")
            families = {member.extraction.task_family for member in members}
            if len(families) != 1:
                raise ValueError("one task group cannot mix task families")
        return self


class ClusterSummary(ContractModel):
    cluster_id: str = Field(pattern=_DIGEST_PATTERN)
    estimator: EstimatorKind
    member_observation_ids: tuple[str, ...] = Field(min_length=1)
    member_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    feature_summary: str = Field(min_length=1, max_length=1024)


class StateValueEstimate(ContractModel):
    observation_id: str = Field(min_length=3, max_length=256)
    trajectory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    task_group: str = Field(pattern=_IDENTIFIER_PATTERN)
    state_index: int = Field(ge=0, strict=True)
    boundary: StateBoundary
    milestone_eligible: bool
    estimator: EstimatorKind
    cluster_id: str = Field(pattern=_DIGEST_PATTERN)
    estimated_value: float = Field(ge=0.0, le=1.0)
    eventual_terminal_reward: Literal[0, 1]
    value_source: ValueSource
    supporting_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    cluster_member_observation_ids: tuple[str, ...] = Field(min_length=1)
    typed_compatibility_digest: str = Field(pattern=_DIGEST_PATTERN)
    reasoning_text_digest: str = Field(pattern=_DIGEST_PATTERN)
    numerical_milestones: tuple[str, ...] = ()
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def exclude_target_trajectory_from_support(self) -> Self:
        if self.trajectory_id in self.supporting_trajectory_ids:
            raise ValueError("a trajectory cannot train its own value estimate")
        return self


class EstimatorMetrics(ContractModel):
    observation_count: int = Field(ge=1, strict=True)
    trajectory_count: int = Field(ge=2, strict=True)
    cluster_count: int = Field(ge=1, strict=True)
    task_group_fallback_count: int = Field(ge=0, strict=True)
    brier_score: float = Field(ge=0.0, le=1.0)
    mean_absolute_error: float = Field(ge=0.0, le=1.0)


class EstimatorEvaluation(ContractModel):
    estimator: EstimatorKind
    clusters: tuple[ClusterSummary, ...] = Field(min_length=1)
    estimates: tuple[StateValueEstimate, ...] = Field(min_length=1)
    metrics: EstimatorMetrics

    @model_validator(mode="after")
    def require_one_estimator_identity(self) -> Self:
        if any(cluster.estimator is not self.estimator for cluster in self.clusters):
            raise ValueError("cluster estimator identity mismatch")
        if any(estimate.estimator is not self.estimator for estimate in self.estimates):
            raise ValueError("estimate estimator identity mismatch")
        return self


class OfflineValueComparison(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-evaluation-v1.schema.json"
        },
    )

    comparison_schema_version: Literal["1"] = "1"
    evaluator_id: Literal["jacobian.offline-trajectory-value.v1"] = (
        "jacobian.offline-trajectory-value.v1"
    )
    corpus_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    corpus_digest: str = Field(pattern=_DIGEST_PATTERN)
    evaluator_config: ValueEstimatorConfig
    terminal_reward_semantics: Literal[
        "1=clean-room accepted;0=clean-room rejected;inconclusive=excluded"
    ] = "1=clean-room accepted;0=clean-room rejected;inconclusive=excluded"
    evaluations: tuple[EstimatorEvaluation, ...]
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_all_estimators_once(self) -> Self:
        kinds = tuple(item.estimator for item in self.evaluations)
        if kinds != tuple(EstimatorKind):
            raise ValueError("comparison must contain all estimators in fixed order")
        return self


@dataclass(frozen=True)
class _Observation:
    observation_id: str
    trajectory_id: str
    task_group: str
    state: ExtractedTrajectoryState
    terminal_reward: Literal[0, 1]
    numerical_milestones: tuple[str, ...]
    typed_payload: dict[str, Any]
    typed_digest: str
    reasoning_text: str
    reasoning_text_digest: str


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonicalize_json({"value": _json_compatible(value)})
        ).hexdigest()
    )


def _round_metric(value: float) -> float:
    return round(value, 12)


def _terminal_reward(extraction: TrajectoryExtraction) -> Literal[0, 1]:
    evidence = extraction.terminal_evidence
    if evidence is None:
        raise TrajectoryValueError(
            "validated labelled trajectory lost terminal evidence"
        )
    return 1 if evidence.acceptance is TerminalAcceptance.ACCEPTED else 0


def _selected_state(
    state: ExtractedTrajectoryState,
    previous: ExtractedTrajectoryState | None,
) -> bool:
    if state.boundary in {StateBoundary.FINAL, StateBoundary.TERMINAL}:
        return False
    if state.milestone_eligible or state.boundary is StateBoundary.PLAN:
        return True
    return bool(
        state.boundary is StateBoundary.AFTER_TOOL
        and previous is not None
        and previous.boundary is StateBoundary.TOOL_RESULT
        and previous.milestone_eligible
    )


def _boundary_summary(state: ExtractedTrajectoryState) -> str:
    soft = state.soft_state
    if soft is None:
        return ""
    if state.boundary is StateBoundary.PLAN:
        return soft.plan_summary or ""
    if state.boundary is StateBoundary.AFTER_TOOL:
        return soft.latest_after_tool_summary or ""
    return ""


def _reasoning_text(state: ExtractedTrajectoryState) -> str:
    soft = state.soft_state
    if soft is None:
        return ""
    fields = (
        ("plan", soft.plan_summary),
        ("after", soft.latest_after_tool_summary),
    )
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def _count_signature(
    values: Sequence[tuple[str, ...]],
) -> tuple[tuple[object, ...], ...]:
    counts = Counter(values)
    return tuple((*value, count) for value, count in sorted(counts.items()))


def _typed_payload(state: ExtractedTrajectoryState) -> dict[str, Any]:
    hard = state.hard_state
    typed_objects = [
        (item.object_type, item.content_digest, item.source_capability_id)
        for item in hard.typed_objects
    ]
    artifacts = [
        (item.artifact_uri, item.role, item.source_capability_id)
        for item in hard.artifacts
    ]
    return {
        "task_family": hard.task_family,
        "boundary": state.boundary.value,
        "typed_objects": _count_signature(typed_objects),
        "artifacts": _count_signature(artifacts),
        "candidate_state": hard.candidate_state.value,
        "latest_candidate_digest": hard.latest_candidate_digest,
        "checker_state": hard.checker_state.value,
        "open_obligation_uris": hard.open_obligation_uris,
        "discharged_obligation_uris": hard.discharged_obligation_uris,
        "execution_status": hard.execution_status,
        "completeness_status": hard.completeness_status,
        "completeness_assurance": hard.completeness_assurance,
        "assurance_level": hard.assurance_level,
        "scope_digest": hard.scope_digest,
        "scope_escalation_errors": hard.scope_escalation_errors,
        "binding_validity": hard.binding_validity.value,
        "latest_transitions": tuple(
            item.value for item in hard.latest_meaningful_transitions
        ),
        "reasoning_protocol_state": hard.reasoning_protocol_state.value,
    }


def _observations(corpus: TrajectoryValueCorpus) -> tuple[_Observation, ...]:
    observations: list[_Observation] = []
    for trajectory in sorted(corpus.trajectories, key=lambda item: item.trajectory_id):
        numbers: list[str] = []
        seen_numbers: set[str] = set()
        previous: ExtractedTrajectoryState | None = None
        for state in trajectory.extraction.states:
            for number in _NUMBER_PATTERN.findall(_boundary_summary(state)):
                if number not in seen_numbers:
                    seen_numbers.add(number)
                    numbers.append(number)
            selected = _selected_state(state, previous)
            previous = state
            if not selected:
                continue
            typed = _typed_payload(state)
            text = _reasoning_text(state)
            observations.append(
                _Observation(
                    observation_id=f"{trajectory.trajectory_id}:{state.index}",
                    trajectory_id=trajectory.trajectory_id,
                    task_group=trajectory.task_group,
                    state=state,
                    terminal_reward=_terminal_reward(trajectory.extraction),
                    numerical_milestones=tuple(numbers),
                    typed_payload=typed,
                    typed_digest=_digest(typed),
                    reasoning_text=text,
                    reasoning_text_digest=_digest(text),
                )
            )
    if not observations:
        raise TrajectoryValueError("corpus contains no eligible observation states")
    return tuple(observations)


def _ngrams(text: str) -> Counter[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    grams = Counter(tokens)
    grams.update(f"{left} {right}" for left, right in pairwise(tokens))
    return grams


def _tfidf_vectors(
    observations: tuple[_Observation, ...],
) -> dict[str, dict[str, float]]:
    term_counts = {
        item.observation_id: _ngrams(item.reasoning_text) for item in observations
    }
    document_frequency: Counter[str] = Counter()
    for counts in term_counts.values():
        document_frequency.update(counts.keys())
    document_count = len(observations)
    vectors: dict[str, dict[str, float]] = {}
    for observation_id, counts in term_counts.items():
        total = sum(counts.values())
        vectors[observation_id] = (
            {
                term: (count / total)
                * (
                    math.log((1 + document_count) / (1 + document_frequency[term]))
                    + 1.0
                )
                for term, count in counts.items()
            }
            if total
            else {}
        )
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    shared = left.keys() & right.keys()
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm)


def _average_link_similarity(
    left: tuple[str, ...],
    right: tuple[str, ...],
    vectors: dict[str, dict[str, float]],
) -> float:
    similarities = [
        _cosine(vectors[left_id], vectors[right_id])
        for left_id in left
        for right_id in right
    ]
    return sum(similarities) / len(similarities)


def _agglomerate(
    observation_ids: tuple[str, ...],
    vectors: dict[str, dict[str, float]],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    clusters: list[tuple[str, ...]] = [
        (observation_id,) for observation_id in sorted(observation_ids)
    ]
    while True:
        best: tuple[float, tuple[str, ...], int, int] | None = None
        for left_index, right_index in combinations(range(len(clusters)), 2):
            left = clusters[left_index]
            right = clusters[right_index]
            similarity = _average_link_similarity(left, right, vectors)
            members = tuple(sorted((*left, *right)))
            candidate = (similarity, members, left_index, right_index)
            if similarity < threshold:
                continue
            if (
                best is None
                or similarity > best[0]
                or (similarity == best[0] and members < best[1])
            ):
                best = candidate
        if best is None:
            break
        _, members, left_index, right_index = best
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(members)
        clusters.sort()
    return tuple(clusters)


def _exact_clusters(
    observations: tuple[_Observation, ...],
    key: Any,
) -> tuple[tuple[str, ...], ...]:
    groups: dict[object, list[str]] = defaultdict(list)
    for observation in observations:
        groups[key(observation)].append(observation.observation_id)
    return tuple(
        tuple(sorted(members))
        for _, members in sorted(groups.items(), key=lambda item: repr(item[0]))
    )


def _text_clusters(
    observations: tuple[_Observation, ...],
    vectors: dict[str, dict[str, float]],
    threshold: float,
    *,
    typed_compatible: bool,
) -> tuple[tuple[str, ...], ...]:
    partitions: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for observation in observations:
        partition: tuple[str, ...] = (observation.task_group,)
        if typed_compatible:
            partition += (observation.typed_digest,)
        partitions[partition].append(observation.observation_id)
    clusters: list[tuple[str, ...]] = []
    for partition in sorted(partitions):
        clusters.extend(_agglomerate(tuple(partitions[partition]), vectors, threshold))
    return tuple(sorted(clusters))


def _clusters_for(
    kind: EstimatorKind,
    observations: tuple[_Observation, ...],
    vectors: dict[str, dict[str, float]],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    if kind is EstimatorKind.GROUP_ROLLOUT:
        return _exact_clusters(observations, lambda item: item.task_group)
    if kind is EstimatorKind.NUMCA_NUMERICAL:
        return _exact_clusters(
            observations,
            lambda item: (item.task_group, item.numerical_milestones),
        )
    if kind is EstimatorKind.JACOBIAN_TYPED:
        return _exact_clusters(
            observations,
            lambda item: (item.task_group, item.typed_digest),
        )
    return _text_clusters(
        observations,
        vectors,
        threshold,
        typed_compatible=kind is EstimatorKind.HYBRID_TYPED_TEXT,
    )


def _cluster_id(kind: EstimatorKind, members: tuple[str, ...]) -> str:
    return _digest({"estimator": kind.value, "members": members})


def _top_terms(
    members: tuple[str, ...], vectors: dict[str, dict[str, float]]
) -> tuple[str, ...]:
    aggregate: Counter[str] = Counter()
    for member in members:
        aggregate.update(vectors[member])
    return tuple(
        term
        for term, _ in sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))[
            :8
        ]
    )


def _feature_summary(
    kind: EstimatorKind,
    members: tuple[str, ...],
    by_id: dict[str, _Observation],
    vectors: dict[str, dict[str, float]],
) -> str:
    first = by_id[members[0]]
    if kind is EstimatorKind.GROUP_ROLLOUT:
        return f"task_group={first.task_group}; no intermediate-state features"
    if kind is EstimatorKind.NUMCA_NUMERICAL:
        return _clip_summary(
            f"task_group={first.task_group}; numbers={list(first.numerical_milestones)!r}"
        )
    if kind is EstimatorKind.JACOBIAN_TYPED:
        payload = json.dumps(first.typed_payload, sort_keys=True, separators=(",", ":"))
        return _clip_summary(
            f"task_group={first.task_group}; typed_signature={first.typed_digest}; fields={payload}"
        )
    terms = _top_terms(members, vectors)
    prefix = f"task_group={first.task_group}; text_terms={list(terms)!r}"
    if kind is EstimatorKind.HYBRID_TYPED_TEXT:
        payload = json.dumps(first.typed_payload, sort_keys=True, separators=(",", ":"))
        prefix += f"; typed_signature={first.typed_digest}; fields={payload}"
    return _clip_summary(prefix)


def _clip_summary(value: str) -> str:
    if len(value) <= 1024:
        return value
    return value[:1000] + "...[truncated]"


def _cluster_summaries(
    kind: EstimatorKind,
    clusters: tuple[tuple[str, ...], ...],
    by_id: dict[str, _Observation],
    vectors: dict[str, dict[str, float]],
) -> tuple[ClusterSummary, ...]:
    return tuple(
        ClusterSummary(
            cluster_id=_cluster_id(kind, members),
            estimator=kind,
            member_observation_ids=members,
            member_trajectory_ids=tuple(
                sorted({by_id[member].trajectory_id for member in members})
            ),
            feature_summary=_feature_summary(kind, members, by_id, vectors),
        )
        for members in clusters
    )


def _training_trajectories(
    members: tuple[str, ...],
    target: _Observation,
    by_id: dict[str, _Observation],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                by_id[member].trajectory_id
                for member in members
                if by_id[member].trajectory_id != target.trajectory_id
            }
        )
    )


def _group_prior_trajectories(
    observations: tuple[_Observation, ...], target: _Observation
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.trajectory_id
                for item in observations
                if item.task_group == target.task_group
                and item.trajectory_id != target.trajectory_id
            }
        )
    )


def _estimate_value(
    trajectory_ids: tuple[str, ...], rewards: dict[str, Literal[0, 1]]
) -> float:
    if not trajectory_ids:
        raise TrajectoryValueError("leave-one-trajectory-out estimate has no support")
    return _round_metric(
        sum(rewards[trajectory_id] for trajectory_id in trajectory_ids)
        / len(trajectory_ids)
    )


def _estimates(
    kind: EstimatorKind,
    clusters: tuple[tuple[str, ...], ...],
    observations: tuple[_Observation, ...],
) -> tuple[StateValueEstimate, ...]:
    by_id = {item.observation_id: item for item in observations}
    member_lookup = {member: cluster for cluster in clusters for member in cluster}
    rewards = {item.trajectory_id: item.terminal_reward for item in observations}
    estimates: list[StateValueEstimate] = []
    for target in observations:
        members = member_lookup[target.observation_id]
        support = _training_trajectories(members, target, by_id)
        source = ValueSource.CLUSTER
        if not support:
            support = _group_prior_trajectories(observations, target)
            source = ValueSource.TASK_GROUP_PRIOR
        estimates.append(
            StateValueEstimate(
                observation_id=target.observation_id,
                trajectory_id=target.trajectory_id,
                task_group=target.task_group,
                state_index=target.state.index,
                boundary=target.state.boundary,
                milestone_eligible=target.state.milestone_eligible,
                estimator=kind,
                cluster_id=_cluster_id(kind, members),
                estimated_value=_estimate_value(support, rewards),
                eventual_terminal_reward=target.terminal_reward,
                value_source=source,
                supporting_trajectory_ids=support,
                cluster_member_observation_ids=members,
                typed_compatibility_digest=target.typed_digest,
                reasoning_text_digest=target.reasoning_text_digest,
                numerical_milestones=target.numerical_milestones,
            )
        )
    return tuple(estimates)


def _metrics(
    estimates: tuple[StateValueEstimate, ...], cluster_count: int
) -> EstimatorMetrics:
    by_trajectory: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for estimate in estimates:
        error = estimate.estimated_value - estimate.eventual_terminal_reward
        by_trajectory[estimate.trajectory_id].append((error * error, abs(error)))
    trajectory_brier = [
        sum(pair[0] for pair in errors) / len(errors)
        for errors in by_trajectory.values()
    ]
    trajectory_mae = [
        sum(pair[1] for pair in errors) / len(errors)
        for errors in by_trajectory.values()
    ]
    return EstimatorMetrics(
        observation_count=len(estimates),
        trajectory_count=len(by_trajectory),
        cluster_count=cluster_count,
        task_group_fallback_count=sum(
            estimate.value_source is ValueSource.TASK_GROUP_PRIOR
            for estimate in estimates
        ),
        brier_score=_round_metric(sum(trajectory_brier) / len(trajectory_brier)),
        mean_absolute_error=_round_metric(sum(trajectory_mae) / len(trajectory_mae)),
    )


def evaluate_offline_trajectories(
    corpus: TrajectoryValueCorpus,
) -> OfflineValueComparison:
    """Compare five deterministic estimators without target-label leakage.

    Cluster membership uses observable state only.  Every estimate excludes all
    observations from its own trajectory; singleton clusters fall back to the
    other rollouts in the same task group.  Each trajectory receives equal
    weight in reported error metrics regardless of its number of boundaries.
    """

    observations = _observations(corpus)
    by_id = {item.observation_id: item for item in observations}
    vectors = _tfidf_vectors(observations)
    evaluations: list[EstimatorEvaluation] = []
    for kind in EstimatorKind:
        clusters = _clusters_for(
            kind,
            observations,
            vectors,
            corpus.evaluator_config.text_similarity_threshold_millionths / 1_000_000,
        )
        summaries = _cluster_summaries(kind, clusters, by_id, vectors)
        estimates = _estimates(kind, clusters, observations)
        evaluations.append(
            EstimatorEvaluation(
                estimator=kind,
                clusters=summaries,
                estimates=estimates,
                metrics=_metrics(estimates, len(clusters)),
            )
        )
    return OfflineValueComparison(
        corpus_id=corpus.corpus_id,
        corpus_digest=_digest(corpus.model_dump(mode="json")),
        evaluator_config=corpus.evaluator_config,
        evaluations=tuple(evaluations),
    )


__all__ = [
    "ClusterSummary",
    "EstimatorEvaluation",
    "EstimatorKind",
    "EstimatorMetrics",
    "LabelledTrajectory",
    "OfflineValueComparison",
    "StateValueEstimate",
    "TrajectoryValueCorpus",
    "TrajectoryValueError",
    "ValueEstimatorConfig",
    "ValueSource",
    "evaluate_offline_trajectories",
]
