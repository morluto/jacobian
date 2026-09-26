"""Exact generalized arc-consistency closure for finite CSPs."""

from __future__ import annotations

from collections import deque

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures._models import FiniteCspInstance
from jacobian.math.logic.relational_structures.consistency._models import (
    CspDomainConsistency,
    CspDomainRequest,
)
from jacobian.math.logic.relational_structures.operations import _preflight_csp_instance

MAX_GAC_CANDIDATE_ROWS = 65_536
MAX_GAC_SUPPORT_WORK = 4_194_304
type SupportKey = tuple[int, int, int]
type CandidateRow = tuple[int, tuple[tuple[int, int], ...]]
type Supports = dict[SupportKey, int]
type ReverseSupports = dict[SupportKey, list[int]]


def generalized_arc_consistency(request: CspDomainRequest) -> CspDomainConsistency:
    """Prune every value lacking tuple support until the domains are stable.

    Candidate rows are occurrence-local after exact duplicate
    (relation, ordered scope) profiles are coalesced. An incremental reverse
    index removes each candidate at most once and updates supports only for
    variables in that scope.
    """

    instance = _admit_domain_request(request)
    relation_index = {
        symbol.symbol_id: i for i, symbol in enumerate(instance.template.signature)
    }
    constraint_profiles, false_nullary = _unique_constraint_profiles(
        instance, relation_index
    )
    candidate_bound = sum(
        len(instance.template.relation_tables[relation_index[symbol_id]])
        for symbol_id, _scope in constraint_profiles
    )
    scope_entries = sum(len(scope) for _symbol_id, scope in constraint_profiles)
    support_work_bound = candidate_bound * 4 + scope_entries * instance.template.carrier_size
    if candidate_bound > MAX_GAC_CANDIDATE_ROWS or support_work_bound > MAX_GAC_SUPPORT_WORK:
        raise OperationResourceAdmissionError(
            location=("instance",),
            code="relational.csp.consistency.work_bound",
            message=(
                "the generalized-arc-consistency candidate-row and support-index "
                "work exceeds the admitted envelope"
            ),
        )

    domains = [set(domain) for domain in request.domains]
    supports, reverse, rows, incident, false_nullary = _build_support_index(
        instance, domains, relation_index, constraint_profiles, false_nullary
    )
    _prune_unsupported_domains(domains, supports, reverse, rows, incident)

    final_domains = tuple(tuple(sorted(domain)) for domain in domains)
    return CspDomainConsistency(
        instance=instance,
        initial_domains=request.domains,
        domains=final_domains,
        empty_domain_variables=tuple(
            variable for variable, domain in enumerate(final_domains) if not domain
        ),
        false_nullary_constraint_ids=tuple(sorted(false_nullary)),
    )


def _admit_domain_request(request: CspDomainRequest) -> FiniteCspInstance:
    if type(request) is not CspDomainRequest:
        raise OperationDomainValidationError(
            location=("request",),
            code="relational.csp.consistency.request_type",
            message="request must be a canonical finite CSP domain request",
        )
    if not hasattr(request, "instance") or not hasattr(request, "domains"):
        raise OperationDomainValidationError(
            location=("request",),
            code="relational.csp.consistency.request_fields",
            message="request must include instance and domains",
        )
    instance = request.instance
    if type(instance) is not FiniteCspInstance:
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.consistency.instance_type",
            message="instance must be a canonical finite CSP instance",
        )
    _preflight_csp_instance(instance)
    try:
        instance = FiniteCspInstance.model_validate(instance.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("instance",),
            code="relational.csp.consistency.instance_shape",
            message="instance must have canonical variables, constraints, and template relations",
        ) from exc
    if (
        not isinstance(request.domains, tuple)
        or len(request.domains) != instance.variable_count
        or any(
            not isinstance(domain, tuple)
            or len(domain) > instance.template.carrier_size
            or any(type(value) is not int for value in domain)
            or len(set(domain)) != len(domain)
            or any(not 0 <= value < instance.template.carrier_size for value in domain)
            for domain in request.domains
        )
    ):
        raise OperationDomainValidationError(
            location=("domains",),
            code="relational.csp.consistency.domain_shape",
            message=(
                "domains must have one tuple of distinct template labels "
                "for each CSP variable"
            ),
        )
    return instance


def _unique_constraint_profiles(
    instance: FiniteCspInstance, relation_index: dict[str, int]
) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], tuple[str, ...]]:
    seen: set[tuple[str, tuple[int, ...]]] = set()
    profiles: list[tuple[str, tuple[int, ...]]] = []
    false_nullary: list[str] = []
    for constraint in instance.constraints:
        if not constraint.scope:
            if not instance.template.relation_tables[relation_index[constraint.symbol_id]]:
                false_nullary.append(constraint.constraint_id)
            continue
        profile = (constraint.symbol_id, constraint.scope)
        if profile not in seen:
            seen.add(profile)
            profiles.append(profile)
    return tuple(profiles), tuple(false_nullary)


def _build_support_index(
    instance: FiniteCspInstance,
    domains: list[set[int]],
    relation_index: dict[str, int],
    constraint_profiles: tuple[tuple[str, tuple[int, ...]], ...],
    false_nullary: tuple[str, ...],
) -> tuple[Supports, ReverseSupports, list[CandidateRow], list[list[tuple[int, int]]], tuple[str, ...]]:
    supports: Supports = {}
    reverse: ReverseSupports = {}
    rows: list[CandidateRow] = []
    incident: list[list[tuple[int, int]]] = [[] for _ in range(instance.variable_count)]
    for ci, (symbol_id, scope) in enumerate(constraint_profiles):
        request_checkpoint("during generalized arc-consistency relation expansion")
        relation = instance.template.relation_tables[relation_index[symbol_id]]
        variables = tuple(dict.fromkeys(scope))
        for variable in variables:
            incident[variable].append((ci, variable))
        for relation_row in relation:
            row_values: dict[int, int] = {}
            if not _candidate_row_is_compatible(scope, relation_row, domains, row_values):
                continue
            row_id = len(rows)
            pairs = tuple((variable, row_values[variable]) for variable in variables)
            rows.append((ci, pairs))
            for variable, value in pairs:
                key = (ci, variable, value)
                supports[key] = supports.get(key, 0) + 1
                reverse.setdefault(key, []).append(row_id)
    return supports, reverse, rows, incident, false_nullary


def _candidate_row_is_compatible(
    scope: tuple[int, ...],
    relation_row: tuple[int, ...],
    domains: list[set[int]],
    row_values: dict[int, int],
) -> bool:
    for variable, value in zip(scope, relation_row, strict=True):
        previous = row_values.setdefault(variable, value)
        if previous != value or value not in domains[variable]:
            return False
    return True


def _prune_unsupported_domains(
    domains: list[set[int]],
    supports: Supports,
    reverse: ReverseSupports,
    rows: list[CandidateRow],
    incident: list[list[tuple[int, int]]],
) -> None:
    queue: deque[SupportKey] = deque()
    queued: set[SupportKey] = set()
    for variable, domain in enumerate(domains):
        for value in domain:
            for ci, _ in incident[variable]:
                key = (ci, variable, value)
                if supports.get(key, 0) == 0:
                    queue.append(key)
                    queued.add(key)

    active = bytearray(b"\x01") * len(rows)
    while queue:
        request_checkpoint("during generalized arc-consistency pruning")
        _remove_unsupported_value(
            queue.popleft(), domains, supports, reverse, rows, incident, active, queue, queued
        )


def _remove_unsupported_value(
    key: SupportKey,
    domains: list[set[int]],
    supports: Supports,
    reverse: ReverseSupports,
    rows: list[CandidateRow],
    incident: list[list[tuple[int, int]]],
    active: bytearray,
    queue: deque[SupportKey],
    queued: set[SupportKey],
) -> None:
    _, variable, value = key
    if value not in domains[variable]:
        return
    domains[variable].remove(value)
    for incident_ci, _ in incident[variable]:
        for row_id in reverse.get((incident_ci, variable, value), ()):
            if not active[row_id]:
                continue
            active[row_id] = 0
            row_ci, pairs = rows[row_id]
            for row_variable, row_value in pairs:
                support_key = (row_ci, row_variable, row_value)
                count = supports[support_key] - 1
                supports[support_key] = count
                if (
                    count == 0
                    and row_value in domains[row_variable]
                    and support_key not in queued
                ):
                    queue.append(support_key)
                    queued.add(support_key)
