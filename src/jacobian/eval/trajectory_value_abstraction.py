"""Semantic value-state abstraction and deterministic six-estimator comparison.

The version 1 trajectory hard state remains the replay and integrity record.
This module derives a separate identity-free signature for clustering only.  It
does not alter mathematical assurance, terminal labels, prompts, tools, or
runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, WithJsonSchema, model_validator

from jacobian.canonical import canonicalize_json
from jacobian.contracts.results import ContractModel
from jacobian.eval import trajectory_value as _exact
from jacobian.eval.trajectory_state import (
    BindingValidity,
    CandidateState,
    CheckerState,
    ExtractedTrajectoryState,
    MilestoneKind,
    ReasoningProtocolState,
    StateBoundary,
)
from jacobian.eval.trajectory_value import (
    TrajectoryValueCorpus,
    ValueEstimatorConfig,
    ValueSource,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ABSTRACT_STATE_SCHEMA_URI = (
    "https://jacobian.invalid/docs/reference/evaluations/schemas/"
    "trajectory-value-state-abstraction-v1.schema.json"
)
_CORPUS_SCHEMA_URI = (
    "https://jacobian.invalid/docs/reference/evaluations/schemas/"
    "trajectory-value-corpus-v1.schema.json"
)
_CAPABILITY_VERBS = frozenset(
    {
        "analyze",
        "audit",
        "check",
        "compute",
        "decide",
        "enumerate",
        "find",
        "inspect",
        "normalize",
        "prove",
        "search",
        "verify",
    }
)
_WILSON_Z_95 = 1.959963984540054


class ScopeClass(StrEnum):
    ABSENT = "ABSENT"
    DECLARED = "DECLARED"
    ESCALATION_REJECTED = "ESCALATION_REJECTED"


class ScopeRelation(StrEnum):
    INITIAL = "INITIAL"
    ABSENT = "ABSENT"
    INTRODUCED = "INTRODUCED"
    STABLE = "STABLE"
    CHANGED = "CHANGED"
    REMOVED = "REMOVED"
    ESCALATION_REJECTED = "ESCALATION_REJECTED"


class SemanticClassCount(ContractModel):
    semantic_class: str = Field(min_length=1, max_length=256)
    count: int = Field(ge=1, strict=True)


class AbstractValueStateSignature(ContractModel):
    """Clustering signature that deliberately excludes exact identities."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"$id": _ABSTRACT_STATE_SCHEMA_URI},
    )

    abstraction_schema_version: Literal["1"] = "1"
    abstraction_id: Literal["jacobian.semantic-value-state.v1"] = (
        "jacobian.semantic-value-state.v1"
    )
    task_family: str = Field(min_length=1, max_length=128)
    boundary: StateBoundary
    object_type_counts: tuple[SemanticClassCount, ...] = ()
    artifact_role_counts: tuple[SemanticClassCount, ...] = ()
    candidate_state: CandidateState
    checker_state: CheckerState
    open_obligation_class_counts: tuple[SemanticClassCount, ...] = ()
    discharged_obligation_class_counts: tuple[SemanticClassCount, ...] = ()
    scope_class: ScopeClass
    scope_relation: ScopeRelation
    completeness_status: str | None = Field(default=None, max_length=64)
    completeness_assurance: str | None = Field(default=None, max_length=64)
    execution_status: str | None = Field(default=None, max_length=64)
    assurance_level: str | None = Field(default=None, max_length=64)
    binding_validity: BindingValidity
    meaningful_transitions: tuple[MilestoneKind, ...] = ()
    reasoning_protocol_state: ReasoningProtocolState
    exact_identity_fields_included: Literal[False] = False
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_canonical_class_counts(self) -> Self:
        for counts in (
            self.object_type_counts,
            self.artifact_role_counts,
            self.open_obligation_class_counts,
            self.discharged_obligation_class_counts,
        ):
            classes = tuple(item.semantic_class for item in counts)
            if classes != tuple(sorted(classes)) or len(classes) != len(set(classes)):
                raise ValueError("semantic class counts must be unique and sorted")
        return self


class EstimatorKindV2(StrEnum):
    GROUP_ROLLOUT = "GROUP_ROLLOUT"
    NUMCA_NUMERICAL = "NUMCA_NUMERICAL"
    REASONING_TEXT = "REASONING_TEXT"
    JACOBIAN_TYPED_EXACT = "JACOBIAN_TYPED_EXACT"
    ABSTRACT_VALUE_STATE = "ABSTRACT_VALUE_STATE"
    ABSTRACT_VALUE_STATE_TEXT = "ABSTRACT_VALUE_STATE_TEXT"


class EstimateUncertainty(ContractModel):
    method: Literal["wilson-score-95"] = "wilson-score-95"
    support_count: int = Field(ge=1, strict=True)
    success_count: int = Field(ge=0, strict=True)
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_consistent_interval(self) -> Self:
        if self.success_count > self.support_count:
            raise ValueError("uncertainty successes cannot exceed support")
        if self.lower > self.upper:
            raise ValueError("uncertainty interval is reversed")
        if not math.isclose(
            self.width, self.upper - self.lower, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("uncertainty width must equal upper minus lower")
        return self


class SemanticClusterSummary(ContractModel):
    cluster_id: str = Field(pattern=_DIGEST_PATTERN)
    estimator: EstimatorKindV2
    member_observation_ids: tuple[str, ...] = Field(min_length=1)
    member_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    feature_summary: str = Field(min_length=1, max_length=2048)


class SemanticStateValueEstimate(ContractModel):
    observation_id: str = Field(min_length=3, max_length=256)
    trajectory_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    task_group: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    state_index: int = Field(ge=0, strict=True)
    boundary: StateBoundary
    milestone_eligible: bool
    estimator: EstimatorKindV2
    cluster_id: str = Field(pattern=_DIGEST_PATTERN)
    estimated_value: float = Field(ge=0.0, le=1.0)
    eventual_terminal_reward: Literal[0, 1]
    value_source: ValueSource
    supporting_trajectory_ids: tuple[str, ...] = Field(min_length=1)
    cluster_member_observation_ids: tuple[str, ...] = Field(min_length=1)
    exact_typed_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    abstract_value_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    abstract_value_state: Annotated[
        AbstractValueStateSignature,
        WithJsonSchema({"$ref": _ABSTRACT_STATE_SCHEMA_URI}),
    ]
    reasoning_text_digest: str = Field(pattern=_DIGEST_PATTERN)
    numerical_milestones: tuple[str, ...] = ()
    uncertainty: EstimateUncertainty
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def bind_support_and_semantic_state(self) -> Self:
        if self.trajectory_id in self.supporting_trajectory_ids:
            raise ValueError("a trajectory cannot support its own value estimate")
        if self.supporting_trajectory_ids != tuple(
            sorted(set(self.supporting_trajectory_ids))
        ):
            raise ValueError("supporting trajectory ids must be unique and sorted")
        if self.cluster_member_observation_ids != tuple(
            sorted(set(self.cluster_member_observation_ids))
        ):
            raise ValueError("cluster member ids must be unique and sorted")
        if self.observation_id not in self.cluster_member_observation_ids:
            raise ValueError("an estimate must belong to its declared cluster")
        if self.uncertainty.support_count != len(self.supporting_trajectory_ids):
            raise ValueError("uncertainty support count must bind support ids")
        expected = self.uncertainty.success_count / self.uncertainty.support_count
        if not math.isclose(self.estimated_value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("estimated value must equal the support success rate")
        if _digest(self.abstract_value_state.model_dump(mode="json")) != (
            self.abstract_value_state_digest
        ):
            raise ValueError("abstract value-state digest mismatch")
        return self


class EstimatorMetricsV2(ContractModel):
    observation_count: int = Field(ge=1, strict=True)
    trajectory_count: int = Field(ge=2, strict=True)
    cluster_count: int = Field(ge=1, strict=True)
    cluster_observation_sizes: tuple[int, ...] = Field(min_length=1)
    cluster_trajectory_sizes: tuple[int, ...] = Field(min_length=1)
    task_group_fallback_count: int = Field(ge=0, strict=True)
    support_count_min: int = Field(ge=1, strict=True)
    support_count_max: int = Field(ge=1, strict=True)
    mean_support_count: float = Field(ge=1.0)
    mean_wilson_interval_width: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    mean_absolute_error: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def bind_cluster_counts(self) -> Self:
        if len(self.cluster_observation_sizes) != self.cluster_count:
            raise ValueError("observation cluster sizes must bind cluster count")
        if len(self.cluster_trajectory_sizes) != self.cluster_count:
            raise ValueError("trajectory cluster sizes must bind cluster count")
        if tuple(sorted(self.cluster_observation_sizes)) != (
            self.cluster_observation_sizes
        ):
            raise ValueError("observation cluster sizes must be sorted")
        if (
            tuple(sorted(self.cluster_trajectory_sizes))
            != self.cluster_trajectory_sizes
        ):
            raise ValueError("trajectory cluster sizes must be sorted")
        if self.support_count_min > self.support_count_max:
            raise ValueError("support range is reversed")
        return self


class EstimatorEvaluationV2(ContractModel):
    estimator: EstimatorKindV2
    clusters: tuple[SemanticClusterSummary, ...] = Field(min_length=1)
    estimates: tuple[SemanticStateValueEstimate, ...] = Field(min_length=1)
    metrics: EstimatorMetricsV2

    @model_validator(mode="after")
    def require_one_estimator_identity(self) -> Self:
        if any(cluster.estimator is not self.estimator for cluster in self.clusters):
            raise ValueError("cluster estimator identity mismatch")
        if any(estimate.estimator is not self.estimator for estimate in self.estimates):
            raise ValueError("estimate estimator identity mismatch")
        _validate_evaluation_bindings(self)
        return self


class OfflineValueComparisonV2(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://jacobian.invalid/docs/reference/evaluations/schemas/trajectory-value-evaluation-v2.schema.json"
        },
    )

    comparison_schema_version: Literal["2"] = "2"
    evaluator_id: Literal["jacobian.offline-trajectory-value.v2"] = (
        "jacobian.offline-trajectory-value.v2"
    )
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    corpus_digest: str = Field(pattern=_DIGEST_PATTERN)
    source_corpus: Annotated[
        TrajectoryValueCorpus,
        WithJsonSchema({"$ref": _CORPUS_SCHEMA_URI}),
    ]
    evaluator_config: ValueEstimatorConfig
    terminal_reward_semantics: Literal[
        "1=clean-room accepted;0=clean-room rejected;inconclusive=excluded"
    ] = "1=clean-room accepted;0=clean-room rejected;inconclusive=excluded"
    state_roles: Literal["exact-v1=replay-integrity;abstract-v1=clustering-only"] = (
        "exact-v1=replay-integrity;abstract-v1=clustering-only"
    )
    validation: Literal["leave-one-trajectory-out"] = "leave-one-trajectory-out"
    trajectory_metric_weighting: Literal["equal"] = "equal"
    learned_components: Literal[False] = False
    evaluations: tuple[EstimatorEvaluationV2, ...]
    assurance_authority: Literal[False] = False

    @model_validator(mode="after")
    def require_fixed_estimator_order_and_corpus_binding(self) -> Self:
        kinds = tuple(item.estimator for item in self.evaluations)
        if kinds != tuple(EstimatorKindV2):
            raise ValueError("comparison must contain all six estimators in order")
        if self.corpus_id != self.source_corpus.corpus_id:
            raise ValueError("comparison corpus identity mismatch")
        if _digest(self.source_corpus.model_dump(mode="json")) != self.corpus_digest:
            raise ValueError("comparison corpus digest mismatch")
        if self.evaluator_config != self.source_corpus.evaluator_config:
            raise ValueError("comparison evaluator config mismatch")
        _validate_comparison_source_bindings(self)
        return self


def _validate_cluster_binding(
    cluster: SemanticClusterSummary,
    estimator: EstimatorKindV2,
    estimates: Mapping[str, SemanticStateValueEstimate],
) -> None:
    members = cluster.member_observation_ids
    if members != tuple(sorted(set(members))):
        raise ValueError("cluster members must be unique and sorted")
    if _cluster_id(estimator, members) != cluster.cluster_id:
        raise ValueError("cluster digest does not bind its members")
    if not set(members) <= estimates.keys():
        raise ValueError("cluster refers to an unknown observation")
    expected_trajectories = tuple(
        sorted({estimates[member].trajectory_id for member in members})
    )
    if cluster.member_trajectory_ids != expected_trajectories:
        raise ValueError("cluster trajectory ids do not bind its observations")


def _validate_estimate_cluster_binding(
    estimate: SemanticStateValueEstimate,
    clusters: Mapping[str, SemanticClusterSummary],
) -> None:
    declared_cluster = clusters.get(estimate.cluster_id)
    if declared_cluster is None:
        raise ValueError("estimate refers to an unknown cluster")
    if estimate.cluster_member_observation_ids != (
        declared_cluster.member_observation_ids
    ):
        raise ValueError("estimate cluster members are stale or substituted")


def _validate_evaluation_bindings(evaluation: EstimatorEvaluationV2) -> None:
    clusters = {cluster.cluster_id: cluster for cluster in evaluation.clusters}
    if len(clusters) != len(evaluation.clusters):
        raise ValueError("cluster ids must be unique")
    estimates = {estimate.observation_id: estimate for estimate in evaluation.estimates}
    if len(estimates) != len(evaluation.estimates):
        raise ValueError("observation estimates must be unique")
    for cluster in evaluation.clusters:
        _validate_cluster_binding(cluster, evaluation.estimator, estimates)
    if {estimate.cluster_id for estimate in evaluation.estimates} != clusters.keys():
        raise ValueError("clusters must be exactly the estimate-referenced clusters")
    for estimate in evaluation.estimates:
        _validate_estimate_cluster_binding(estimate, clusters)
    if _metrics(evaluation.estimates, evaluation.clusters) != evaluation.metrics:
        raise ValueError("estimator metrics do not bind estimates and clusters")


def _source_bound_fields(estimate: SemanticStateValueEstimate) -> tuple[object, ...]:
    return (
        estimate.trajectory_id,
        estimate.task_group,
        estimate.state_index,
        estimate.boundary,
        estimate.milestone_eligible,
        estimate.eventual_terminal_reward,
        estimate.exact_typed_state_digest,
        estimate.reasoning_text_digest,
        estimate.numerical_milestones,
    )


def _expected_source_fields(source: Any) -> tuple[object, ...]:
    return (
        source.trajectory_id,
        source.task_group,
        source.state.index,
        source.state.boundary,
        source.state.milestone_eligible,
        source.terminal_reward,
        source.typed_digest,
        source.reasoning_text_digest,
        source.numerical_milestones,
    )


def _expected_estimate_support(
    estimate: SemanticStateValueEstimate,
    source: Any,
    observations: tuple[Any, ...],
    source_by_id: dict[str, Any],
) -> tuple[str, ...]:
    cluster_support = _exact._training_trajectories(
        estimate.cluster_member_observation_ids,
        source,
        source_by_id,
    )
    if estimate.value_source is ValueSource.CLUSTER:
        if not cluster_support:
            raise ValueError("cluster source requires cross-trajectory support")
        return cluster_support
    if cluster_support:
        raise ValueError("task-group fallback cannot replace cluster support")
    return _exact._group_prior_trajectories(observations, source)


def _expected_cluster_members(
    estimator: EstimatorKindV2,
    source: Any,
    observations: tuple[Any, ...],
    threshold: float,
    semantic_digests: Mapping[str, str],
) -> tuple[str, ...]:
    pool = _exact._clustering_observations(observations, source)
    vectors = _exact._tfidf_vectors(pool)
    clusters = _clusters_for(estimator, pool, vectors, threshold, semantic_digests)
    member_lookup = {member: cluster for cluster in clusters for member in cluster}
    return member_lookup[source.observation_id]


def _validate_source_estimate(
    estimate: SemanticStateValueEstimate,
    estimator: EstimatorKindV2,
    source: Any,
    observations: tuple[Any, ...],
    source_by_id: dict[str, Any],
    previous: Mapping[tuple[str, int], ExtractedTrajectoryState | None],
    rewards: Mapping[str, Literal[0, 1]],
    threshold: float,
    semantic_digests: Mapping[str, str],
) -> None:
    if _source_bound_fields(estimate) != _expected_source_fields(source):
        raise ValueError("estimate fields are stale or source-substituted")
    semantic = abstract_value_state(
        source.state,
        previous[(source.trajectory_id, source.state.index)],
    )
    if estimate.abstract_value_state != semantic:
        raise ValueError("abstract state does not bind the exact source state")
    expected_members = _expected_cluster_members(
        estimator, source, observations, threshold, semantic_digests
    )
    if estimate.cluster_member_observation_ids != expected_members:
        raise ValueError("estimate cluster membership is stale or substituted")
    if estimate.cluster_id != _cluster_id(estimator, expected_members):
        raise ValueError("estimate cluster id is stale or substituted")
    expected_support = _expected_estimate_support(
        estimate, source, observations, source_by_id
    )
    if estimate.supporting_trajectory_ids != expected_support:
        raise ValueError("estimate support is stale or substituted")
    expected_successes = sum(rewards[item] for item in expected_support)
    if estimate.uncertainty.success_count != expected_successes:
        raise ValueError("estimate successes do not bind terminal evidence")


def _validate_comparison_source_bindings(
    comparison: OfflineValueComparisonV2,
) -> None:
    observations = _exact._observations(comparison.source_corpus)
    source_by_id = {item.observation_id: item for item in observations}
    source_ids = tuple(item.observation_id for item in observations)
    previous = _previous_state_lookup(comparison.source_corpus)
    _, semantic_digests = _semantic_states(comparison.source_corpus, observations)
    threshold = (
        comparison.evaluator_config.text_similarity_threshold_millionths / 1_000_000
    )
    rewards = {item.trajectory_id: item.terminal_reward for item in observations}
    for evaluation in comparison.evaluations:
        if tuple(item.observation_id for item in evaluation.estimates) != source_ids:
            raise ValueError("evaluation observations do not bind the source corpus")
        expected_clusters = {
            _expected_cluster_members(
                evaluation.estimator,
                source_by_id[estimate.observation_id],
                observations,
                threshold,
                semantic_digests,
            )
            for estimate in evaluation.estimates
        }
        declared_clusters = {
            cluster.member_observation_ids for cluster in evaluation.clusters
        }
        if declared_clusters != expected_clusters:
            raise ValueError("evaluation clusters are stale or source-substituted")
        for estimate in evaluation.estimates:
            _validate_source_estimate(
                estimate,
                evaluation.estimator,
                source_by_id[estimate.observation_id],
                observations,
                source_by_id,
                previous,
                rewards,
                threshold,
                semantic_digests,
            )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def _digest(value: object) -> str:
    encoded = canonicalize_json({"value": _json_compatible(value)})
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _round_metric(value: float) -> float:
    return round(value, 12)


def _counts(values: Sequence[str]) -> tuple[SemanticClassCount, ...]:
    return tuple(
        SemanticClassCount(semantic_class=name, count=count)
        for name, count in sorted(Counter(values).items())
    )


def _capability_class(capability_id: str) -> str:
    parts = [part.lower() for part in capability_id.split(".") if part]
    semantic = [part for part in parts if part not in _CAPABILITY_VERBS]
    if not semantic:
        return "unclassified"
    return ".".join(semantic[:2])


def _obligation_classes(
    uris: Sequence[str], state: ExtractedTrajectoryState
) -> tuple[SemanticClassCount, ...]:
    artifacts = {
        artifact.artifact_uri: artifact for artifact in state.hard_state.artifacts
    }
    classes = [
        _capability_class(artifacts[uri].source_capability_id)
        if uri in artifacts
        else "unclassified"
        for uri in uris
    ]
    return _counts(classes)


def _scope_semantics(
    state: ExtractedTrajectoryState,
    previous: ExtractedTrajectoryState | None,
) -> tuple[ScopeClass, ScopeRelation]:
    hard = state.hard_state
    rejected = MilestoneKind.SCOPE_ESCALATION_REJECTED in (
        hard.latest_meaningful_transitions
    )
    scope_class = (
        ScopeClass.ESCALATION_REJECTED
        if rejected
        else ScopeClass.DECLARED
        if hard.scope_digest is not None
        else ScopeClass.ABSENT
    )
    if rejected:
        return scope_class, ScopeRelation.ESCALATION_REJECTED
    if previous is None:
        return scope_class, ScopeRelation.INITIAL
    before = previous.hard_state.scope_digest
    after = hard.scope_digest
    if before is None and after is None:
        relation = ScopeRelation.ABSENT
    elif before is None:
        relation = ScopeRelation.INTRODUCED
    elif after is None:
        relation = ScopeRelation.REMOVED
    elif before == after:
        relation = ScopeRelation.STABLE
    else:
        relation = ScopeRelation.CHANGED
    return scope_class, relation


def abstract_value_state(
    state: ExtractedTrajectoryState,
    previous: ExtractedTrajectoryState | None = None,
) -> AbstractValueStateSignature:
    """Derive an identity-free value signature from an exact v1 snapshot."""

    hard = state.hard_state
    scope_class, scope_relation = _scope_semantics(state, previous)
    return AbstractValueStateSignature(
        task_family=hard.task_family,
        boundary=state.boundary,
        object_type_counts=_counts(
            [typed_object.object_type for typed_object in hard.typed_objects]
        ),
        artifact_role_counts=_counts([artifact.role for artifact in hard.artifacts]),
        candidate_state=hard.candidate_state,
        checker_state=hard.checker_state,
        open_obligation_class_counts=_obligation_classes(
            hard.open_obligation_uris, state
        ),
        discharged_obligation_class_counts=_obligation_classes(
            hard.discharged_obligation_uris, state
        ),
        scope_class=scope_class,
        scope_relation=scope_relation,
        completeness_status=hard.completeness_status,
        completeness_assurance=hard.completeness_assurance,
        execution_status=hard.execution_status,
        assurance_level=hard.assurance_level,
        binding_validity=hard.binding_validity,
        meaningful_transitions=hard.latest_meaningful_transitions,
        reasoning_protocol_state=hard.reasoning_protocol_state,
    )


def _previous_state_lookup(
    corpus: TrajectoryValueCorpus,
) -> dict[tuple[str, int], ExtractedTrajectoryState | None]:
    lookup: dict[tuple[str, int], ExtractedTrajectoryState | None] = {}
    for trajectory in corpus.trajectories:
        previous: ExtractedTrajectoryState | None = None
        for state in trajectory.extraction.states:
            lookup[(trajectory.trajectory_id, state.index)] = previous
            previous = state
    return lookup


def _semantic_states(
    corpus: TrajectoryValueCorpus, observations: Sequence[Any]
) -> tuple[
    dict[str, AbstractValueStateSignature],
    dict[str, str],
]:
    previous = _previous_state_lookup(corpus)
    states: dict[str, AbstractValueStateSignature] = {}
    digests: dict[str, str] = {}
    for observation in observations:
        semantic = abstract_value_state(
            observation.state,
            previous[(observation.trajectory_id, observation.state.index)],
        )
        states[observation.observation_id] = semantic
        digests[observation.observation_id] = _digest(semantic.model_dump(mode="json"))
    return states, digests


def _semantic_text_clusters(
    observations: Sequence[Any],
    vectors: dict[str, dict[str, float]],
    threshold: float,
    semantic_digests: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    partitions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for observation in observations:
        partitions[
            (observation.task_group, semantic_digests[observation.observation_id])
        ].append(observation.observation_id)
    clusters: list[tuple[str, ...]] = []
    for partition in sorted(partitions):
        clusters.extend(
            _exact._agglomerate(tuple(partitions[partition]), vectors, threshold)
        )
    return tuple(sorted(clusters))


def _clusters_for(
    kind: EstimatorKindV2,
    observations: tuple[Any, ...],
    vectors: dict[str, dict[str, float]],
    threshold: float,
    semantic_digests: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    legacy = {
        EstimatorKindV2.GROUP_ROLLOUT: _exact.EstimatorKind.GROUP_ROLLOUT,
        EstimatorKindV2.NUMCA_NUMERICAL: _exact.EstimatorKind.NUMCA_NUMERICAL,
        EstimatorKindV2.REASONING_TEXT: _exact.EstimatorKind.REASONING_TEXT,
        EstimatorKindV2.JACOBIAN_TYPED_EXACT: _exact.EstimatorKind.JACOBIAN_TYPED,
    }.get(kind)
    if legacy is not None:
        return _exact._clusters_for(legacy, observations, vectors, threshold)
    if kind is EstimatorKindV2.ABSTRACT_VALUE_STATE:
        return _exact._exact_clusters(
            observations,
            lambda item: (
                item.task_group,
                semantic_digests[item.observation_id],
            ),
        )
    return _semantic_text_clusters(observations, vectors, threshold, semantic_digests)


def _cluster_id(kind: EstimatorKindV2, members: tuple[str, ...]) -> str:
    return _digest(
        {
            "evaluator": "jacobian.offline-trajectory-value.v2",
            "estimator": kind.value,
            "members": members,
        }
    )


def _clip(value: str) -> str:
    return value if len(value) <= 2048 else value[:2024] + "...[truncated]"


def _feature_summary(
    kind: EstimatorKindV2,
    members: tuple[str, ...],
    by_id: Mapping[str, Any],
    vectors: dict[str, dict[str, float]],
    semantic_states: Mapping[str, AbstractValueStateSignature],
    semantic_digests: Mapping[str, str],
) -> str:
    first = by_id[members[0]]
    prefix = f"task_group={first.task_group}"
    if kind is EstimatorKindV2.GROUP_ROLLOUT:
        return prefix + "; no intermediate-state features"
    if kind is EstimatorKindV2.NUMCA_NUMERICAL:
        return prefix + f"; numbers={list(first.numerical_milestones)!r}"
    if kind is EstimatorKindV2.JACOBIAN_TYPED_EXACT:
        payload = json.dumps(first.typed_payload, sort_keys=True, separators=(",", ":"))
        return _clip(
            prefix + f"; exact_signature={first.typed_digest}; fields={payload}"
        )
    terms = _exact._top_terms(members, vectors)
    if kind is EstimatorKindV2.REASONING_TEXT:
        return _clip(prefix + f"; text_terms={list(terms)!r}")
    semantic = semantic_states[first.observation_id]
    payload = json.dumps(
        semantic.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    summary = (
        prefix
        + f"; abstract_signature={semantic_digests[first.observation_id]}"
        + f"; fields={payload}"
    )
    if kind is EstimatorKindV2.ABSTRACT_VALUE_STATE_TEXT:
        summary += f"; text_terms={list(terms)!r}"
    return _clip(summary)


def _cluster_summaries(
    kind: EstimatorKindV2,
    clusters: tuple[tuple[str, ...], ...],
    by_id: Mapping[str, Any],
    vectors: dict[str, dict[str, float]],
    semantic_states: Mapping[str, AbstractValueStateSignature],
    semantic_digests: Mapping[str, str],
) -> tuple[SemanticClusterSummary, ...]:
    return tuple(
        SemanticClusterSummary(
            cluster_id=_cluster_id(kind, members),
            estimator=kind,
            member_observation_ids=members,
            member_trajectory_ids=tuple(
                sorted({by_id[member].trajectory_id for member in members})
            ),
            feature_summary=_feature_summary(
                kind,
                members,
                by_id,
                vectors,
                semantic_states,
                semantic_digests,
            ),
        )
        for members in clusters
    )


def _wilson(successes: int, support: int) -> EstimateUncertainty:
    proportion = successes / support
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    denominator = 1.0 + z2 / support
    center = (proportion + z2 / (2.0 * support)) / denominator
    margin = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / support + z2 / (4.0 * support * support)
        )
        / denominator
    )
    lower = _round_metric(max(0.0, center - margin))
    upper = _round_metric(min(1.0, center + margin))
    return EstimateUncertainty(
        support_count=support,
        success_count=successes,
        lower=lower,
        upper=upper,
        width=_round_metric(upper - lower),
    )


def _estimates(
    kind: EstimatorKindV2,
    observations: tuple[Any, ...],
    semantic_states: Mapping[str, AbstractValueStateSignature],
    semantic_digests: Mapping[str, str],
    threshold: float,
) -> tuple[SemanticStateValueEstimate, ...]:
    by_id = {item.observation_id: item for item in observations}
    rewards = {item.trajectory_id: item.terminal_reward for item in observations}
    estimates: list[SemanticStateValueEstimate] = []
    for target in observations:
        members = _expected_cluster_members(
            kind, target, observations, threshold, semantic_digests
        )
        support = _exact._training_trajectories(members, target, by_id)
        source = ValueSource.CLUSTER
        if not support:
            support = _exact._group_prior_trajectories(observations, target)
            source = ValueSource.TASK_GROUP_PRIOR
        successes = sum(rewards[trajectory_id] for trajectory_id in support)
        uncertainty = _wilson(successes, len(support))
        estimates.append(
            SemanticStateValueEstimate(
                observation_id=target.observation_id,
                trajectory_id=target.trajectory_id,
                task_group=target.task_group,
                state_index=target.state.index,
                boundary=target.state.boundary,
                milestone_eligible=target.state.milestone_eligible,
                estimator=kind,
                cluster_id=_cluster_id(kind, members),
                estimated_value=_round_metric(successes / len(support)),
                eventual_terminal_reward=target.terminal_reward,
                value_source=source,
                supporting_trajectory_ids=support,
                cluster_member_observation_ids=members,
                exact_typed_state_digest=target.typed_digest,
                abstract_value_state_digest=semantic_digests[target.observation_id],
                abstract_value_state=semantic_states[target.observation_id],
                reasoning_text_digest=target.reasoning_text_digest,
                numerical_milestones=target.numerical_milestones,
                uncertainty=uncertainty,
            )
        )
    return tuple(estimates)


def _metrics(
    estimates: tuple[SemanticStateValueEstimate, ...],
    clusters: tuple[SemanticClusterSummary, ...],
) -> EstimatorMetricsV2:
    errors_by_trajectory: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for estimate in estimates:
        error = estimate.estimated_value - estimate.eventual_terminal_reward
        errors_by_trajectory[estimate.trajectory_id].append((error * error, abs(error)))
    trajectory_brier = [
        sum(pair[0] for pair in errors) / len(errors)
        for errors in errors_by_trajectory.values()
    ]
    trajectory_mae = [
        sum(pair[1] for pair in errors) / len(errors)
        for errors in errors_by_trajectory.values()
    ]
    supports = [estimate.uncertainty.support_count for estimate in estimates]
    return EstimatorMetricsV2(
        observation_count=len(estimates),
        trajectory_count=len(errors_by_trajectory),
        cluster_count=len(clusters),
        cluster_observation_sizes=tuple(
            sorted(len(cluster.member_observation_ids) for cluster in clusters)
        ),
        cluster_trajectory_sizes=tuple(
            sorted(len(cluster.member_trajectory_ids) for cluster in clusters)
        ),
        task_group_fallback_count=sum(
            estimate.value_source is ValueSource.TASK_GROUP_PRIOR
            for estimate in estimates
        ),
        support_count_min=min(supports),
        support_count_max=max(supports),
        mean_support_count=_round_metric(sum(supports) / len(supports)),
        mean_wilson_interval_width=_round_metric(
            sum(estimate.uncertainty.width for estimate in estimates) / len(estimates)
        ),
        brier_score=_round_metric(sum(trajectory_brier) / len(trajectory_brier)),
        mean_absolute_error=_round_metric(sum(trajectory_mae) / len(trajectory_mae)),
    )


def evaluate_semantic_trajectories(
    corpus: TrajectoryValueCorpus,
) -> OfflineValueComparisonV2:
    """Compare six deterministic estimators without target-label leakage."""

    observations = _exact._observations(corpus)
    by_id = {item.observation_id: item for item in observations}
    vectors = _exact._tfidf_vectors(observations)
    semantic_states, semantic_digests = _semantic_states(corpus, observations)
    threshold = corpus.evaluator_config.text_similarity_threshold_millionths / 1_000_000
    evaluations: list[EstimatorEvaluationV2] = []
    for kind in EstimatorKindV2:
        estimates = _estimates(
            kind,
            observations,
            semantic_states,
            semantic_digests,
            threshold,
        )
        cluster_members = tuple(
            sorted({estimate.cluster_member_observation_ids for estimate in estimates})
        )
        clusters = _cluster_summaries(
            kind,
            cluster_members,
            by_id,
            vectors,
            semantic_states,
            semantic_digests,
        )
        evaluations.append(
            EstimatorEvaluationV2(
                estimator=kind,
                clusters=clusters,
                estimates=estimates,
                metrics=_metrics(estimates, clusters),
            )
        )
    return OfflineValueComparisonV2(
        corpus_id=corpus.corpus_id,
        corpus_digest=_digest(corpus.model_dump(mode="json")),
        source_corpus=corpus,
        evaluator_config=corpus.evaluator_config,
        evaluations=tuple(evaluations),
    )


__all__ = [
    "AbstractValueStateSignature",
    "EstimateUncertainty",
    "EstimatorEvaluationV2",
    "EstimatorKindV2",
    "EstimatorMetricsV2",
    "OfflineValueComparisonV2",
    "ScopeClass",
    "ScopeRelation",
    "SemanticClassCount",
    "SemanticClusterSummary",
    "SemanticStateValueEstimate",
    "abstract_value_state",
    "evaluate_semantic_trajectories",
]
