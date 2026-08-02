"""Durable agent-authored working state without mathematical promotion."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from jacobian.canonical import canonicalize_json
from jacobian.contracts.results import ContractModel
from jacobian.contracts.workspaces import (
    WorkspaceAttempt,
    WorkspaceAttemptSummary,
    WorkspaceCardId,
    WorkspaceCardState,
    WorkspaceContextView,
    WorkspaceFinding,
    WorkspaceFindingDraft,
    WorkspaceFindingKind,
    WorkspaceFocus,
    WorkspaceFocusDraft,
    WorkspaceFrontierItem,
    WorkspaceItemSummary,
    WorkspaceMark,
    WorkspaceOpenRequest,
    WorkspaceOpenResult,
    WorkspaceQueryRequest,
    WorkspaceQueryResult,
    WorkspaceQueryView,
    WorkspaceResumeView,
    WorkspaceRevision,
    WorkspaceRevisionId,
    WorkspaceRevisionOperation,
    WorkspaceScratchEntry,
    WorkspaceScratchSummary,
    WorkspaceWriteRequest,
    WorkspaceWriteResult,
)
from jacobian.schema_registry import SchemaRegistry, model_schema
from jacobian.store import ArtifactStore, StoreError
from jacobian.workspaces._helpers import (
    _attempt_summary,
    _dependency_postorder,
    _finding_summary,
    _model_from_json,
    _now,
    _opaque_id,
    _reject_new_dependency_cycles,
    _reject_supersession_cycles,
    _request_digest,
    _scratch_summary,
)
from jacobian.workspaces.errors import (
    WorkspaceConflictError,
    WorkspaceError,
    WorkspaceIdempotencyError,
    WorkspaceNotFoundError,
    WorkspaceReferenceError,
)


@dataclass(frozen=True, slots=True)
class _PreparedWrite:
    revision: WorkspaceRevision
    revision_artifact_uri: str
    id_map: dict[str, str]


@dataclass(frozen=True, slots=True)
class _WorkspaceProjection:
    findings: dict[str, WorkspaceFinding]
    finding_order: dict[str, int]
    marks: dict[str, WorkspaceMark]
    stale_due_to: dict[str, tuple[str, ...]]


_ModelT = TypeVar("_ModelT", bound=ContractModel)
_INVALIDATING_CARD_STATES = frozenset(
    {
        WorkspaceCardState.RETRACTED,
        WorkspaceCardState.SUPERSEDED,
    }
)


class WorkspaceService:
    """Persist revisioned epistemic state while keeping every assertion unverified."""

    def __init__(self, store: ArtifactStore, schemas: SchemaRegistry) -> None:
        self.store = store
        self.schemas = schemas
        self.revision_schema_uri = schemas.register(
            name="jacobian.workspace-revision",
            version="1",
            schema=model_schema(WorkspaceRevision),
        )
        self.revision_semantics_uri = store.register_descriptor(
            kind="semantics",
            name="jacobian.workspace-revision",
            version="1",
            definition={
                "description": (
                    "agent-authored epistemic workspace state; stored assertions and "
                    "retrieval views are unverified and cannot promote assurance"
                )
            },
        )

    def open(self, request: WorkspaceOpenRequest) -> WorkspaceOpenResult:
        selected = WorkspaceOpenRequest.model_validate(request)
        request_digest = _request_digest(
            WorkspaceRevisionOperation.OPEN,
            selected.model_dump(mode="json"),
        )
        with self.store.transaction(), self.store.connection() as connection:
            reused = self._reuse(
                connection,
                idempotency_key=selected.idempotency_key,
                operation=WorkspaceRevisionOperation.OPEN,
                request_digest=request_digest,
                result_type=WorkspaceOpenResult,
            )
            if reused is not None:
                return reused

            workspace_id = _opaque_id("workspace")
            branch_id = _opaque_id("branch")
            revision_id = _opaque_id("revision")
            problem_card_id = _opaque_id("card")
            now = _now()
            problem = WorkspaceFinding(
                card_id=problem_card_id,
                kind=WorkspaceFindingKind.PROBLEM,
                title=selected.name,
                body=selected.problem,
                tags=selected.tags,
                created_revision=revision_id,
                created_at=now,
            )
            focus = WorkspaceFocus(
                active_item_id=problem_card_id,
                pinned_item_ids=(problem_card_id,),
                updated_revision=revision_id,
            )
            revision = WorkspaceRevision(
                revision_id=revision_id,
                workspace_id=workspace_id,
                branch_id=branch_id,
                operation=WorkspaceRevisionOperation.OPEN,
                request_digest=request_digest,
                findings=(problem,),
                focus=focus,
                created_at=now,
            )
            stored = self.store.put(
                schema_uri=self.revision_schema_uri,
                semantics_uri=self.revision_semantics_uri,
                payload=revision.model_dump(mode="json"),
                summary=f"open epistemic workspace {selected.name}",
            )
            result = WorkspaceOpenResult(
                workspace_id=workspace_id,
                branch_id=branch_id,
                revision_id=revision_id,
                revision_artifact_uri=stored.artifact_uri,
                problem_card_id=problem_card_id,
            )
            timestamp = now.isoformat()
            connection.execute(
                """
                INSERT INTO workspaces(
                    workspace_id, name, root_branch_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (workspace_id, selected.name, branch_id, timestamp),
            )
            connection.execute(
                """
                INSERT INTO workspace_branches(
                    branch_id, workspace_id, alias, head_revision_id, created_at
                ) VALUES (?, ?, 'main', ?, ?)
                """,
                (branch_id, workspace_id, revision_id, timestamp),
            )
            self._insert_revision(connection, revision, stored.artifact_uri)
            self._insert_finding(connection, workspace_id, branch_id, problem)
            self._upsert_focus(connection, workspace_id, branch_id, focus)
            self._bind_idempotency(
                connection,
                selected.idempotency_key,
                WorkspaceRevisionOperation.OPEN,
                request_digest,
                result,
            )
        return result

    def write(self, request: WorkspaceWriteRequest) -> WorkspaceWriteResult:
        selected = WorkspaceWriteRequest.model_validate(request)
        request_digest = _request_digest(
            WorkspaceRevisionOperation.WRITE,
            selected.model_dump(mode="json"),
        )
        with self.store.transaction(), self.store.connection() as connection:
            reused = self._reuse(
                connection,
                idempotency_key=selected.idempotency_key,
                operation=WorkspaceRevisionOperation.WRITE,
                request_digest=request_digest,
                result_type=WorkspaceWriteResult,
            )
            if reused is not None:
                return reused
            prepared = self._prepare_write(connection, selected, request_digest)
            result = WorkspaceWriteResult(
                workspace_id=selected.workspace_id,
                branch_id=selected.branch_id,
                revision_id=prepared.revision.revision_id,
                revision_artifact_uri=prepared.revision_artifact_uri,
                id_map=prepared.id_map,
                scratch_written=len(prepared.revision.scratch),
                findings_written=len(prepared.revision.findings),
                attempts_written=len(prepared.revision.attempts),
                marks_written=len(prepared.revision.marks),
                focus_updated=prepared.revision.focus is not None,
                unverified_finding_ids=tuple(
                    finding.card_id for finding in prepared.revision.findings
                ),
                unresolved_dependency_ids=tuple(
                    sorted(
                        {
                            dependency_id
                            for finding in prepared.revision.findings
                            for dependency_id in (
                                *finding.dependency_ids,
                                *finding.assumption_ids,
                            )
                        }
                    )
                ),
            )
            self._insert_revision(
                connection,
                prepared.revision,
                prepared.revision_artifact_uri,
            )
            for entry in prepared.revision.scratch:
                self._insert_scratch(
                    connection,
                    selected.workspace_id,
                    selected.branch_id,
                    entry,
                )
            for finding in prepared.revision.findings:
                self._insert_finding(
                    connection,
                    selected.workspace_id,
                    selected.branch_id,
                    finding,
                )
            for attempt in prepared.revision.attempts:
                self._insert_attempt(
                    connection,
                    selected.workspace_id,
                    selected.branch_id,
                    attempt,
                )
            for mark in prepared.revision.marks:
                self._insert_mark(
                    connection,
                    selected.workspace_id,
                    selected.branch_id,
                    mark,
                )
            if prepared.revision.focus is not None:
                self._upsert_focus(
                    connection,
                    selected.workspace_id,
                    selected.branch_id,
                    prepared.revision.focus,
                )
            updated = connection.execute(
                """
                UPDATE workspace_branches
                SET head_revision_id = ?
                WHERE branch_id = ?
                  AND workspace_id = ?
                  AND head_revision_id = ?
                """,
                (
                    prepared.revision.revision_id,
                    selected.branch_id,
                    selected.workspace_id,
                    selected.base_revision,
                ),
            )
            if updated.rowcount != 1:
                raise WorkspaceConflictError(
                    "workspace branch advanced before the write could commit"
                )
            self._bind_idempotency(
                connection,
                selected.idempotency_key,
                WorkspaceRevisionOperation.WRITE,
                request_digest,
                result,
            )
        return result

    def query(self, request: WorkspaceQueryRequest) -> WorkspaceQueryResult:
        selected = WorkspaceQueryRequest.model_validate(request)
        with self.store.connection() as connection:
            connection.execute("BEGIN")
            branch = self._branch_row(
                connection,
                selected.workspace_id,
                selected.branch_id,
            )
            revision_id = branch["head_revision_id"]
            revision_artifact_uri = branch["revision_artifact_uri"]
            if selected.revision_id is not None and selected.revision_id != revision_id:
                raise WorkspaceConflictError(
                    f"query revision_id is stale; current branch head is {revision_id}"
                )
            projection = self._projection(connection, selected)
            if selected.view is WorkspaceQueryView.RESUME:
                resume = self._resume_view(connection, selected, projection)
                return WorkspaceQueryResult(
                    workspace_id=selected.workspace_id,
                    branch_id=selected.branch_id,
                    revision_id=revision_id,
                    revision_artifact_uri=revision_artifact_uri,
                    view=selected.view,
                    resume=resume,
                )
            if selected.view is WorkspaceQueryView.FRONTIER:
                frontier = self._frontier_view(connection, selected, projection)
                return WorkspaceQueryResult(
                    workspace_id=selected.workspace_id,
                    branch_id=selected.branch_id,
                    revision_id=revision_id,
                    revision_artifact_uri=revision_artifact_uri,
                    view=selected.view,
                    frontier=frontier,
                )
            if selected.view is WorkspaceQueryView.ATTEMPTS:
                attempts = self._attempts_view(connection, selected, projection)
                return WorkspaceQueryResult(
                    workspace_id=selected.workspace_id,
                    branch_id=selected.branch_id,
                    revision_id=revision_id,
                    revision_artifact_uri=revision_artifact_uri,
                    view=selected.view,
                    attempts=attempts,
                )
            if selected.view is WorkspaceQueryView.CONTEXT:
                context = self._context_view(connection, selected, projection)
                return WorkspaceQueryResult(
                    workspace_id=selected.workspace_id,
                    branch_id=selected.branch_id,
                    revision_id=revision_id,
                    revision_artifact_uri=revision_artifact_uri,
                    view=selected.view,
                    context=context,
                )
            stale_items = tuple(
                self._finding_summary(connection, item, projection)
                for item in self._ordered_findings(
                    projection,
                    lambda item: bool(projection.stale_due_to[item.card_id]),
                    selected.limit,
                )
            )
            return WorkspaceQueryResult(
                workspace_id=selected.workspace_id,
                branch_id=selected.branch_id,
                revision_id=revision_id,
                revision_artifact_uri=revision_artifact_uri,
                view=selected.view,
                stale_items=stale_items,
            )

    def _prepare_write(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceWriteRequest,
        request_digest: str,
    ) -> _PreparedWrite:
        branch = self._require_head(
            connection,
            request.workspace_id,
            request.branch_id,
            request.base_revision,
        )
        revision_id = _opaque_id("revision")
        now = _now()
        id_map: dict[str, str] = {}
        for scratch_draft in request.scratch:
            id_map[scratch_draft.client_ref] = _opaque_id("scratch")
        for finding_draft in request.findings:
            id_map[finding_draft.client_ref] = _opaque_id("card")
        for attempt_draft in request.attempts:
            id_map[attempt_draft.client_ref] = _opaque_id("attempt")
        for mark_draft in request.marks:
            id_map[mark_draft.client_ref] = _opaque_id("mark")

        existing_findings = self._referenced_findings(connection, request)
        finding_drafts = {
            finding_draft.client_ref: finding_draft
            for finding_draft in request.findings
        }

        def resolve_finding(reference: str) -> WorkspaceFinding | WorkspaceFindingDraft:
            draft = finding_drafts.get(reference)
            if draft is not None:
                return draft
            existing = existing_findings.get(reference)
            if existing is None:
                raise WorkspaceReferenceError(
                    f"workspace finding reference does not exist: {reference}"
                )
            return existing

        findings: list[WorkspaceFinding] = []
        for finding_draft in request.findings:
            if finding_draft.kind is WorkspaceFindingKind.PROBLEM:
                raise WorkspaceReferenceError(
                    "only workspace.open may create the canonical PROBLEM card"
                )
            card_id = id_map[finding_draft.client_ref]
            dependency_ids = tuple(
                self._resolved_card_id(reference, id_map, resolve_finding(reference))
                for reference in finding_draft.dependency_refs
            )
            assumption_ids: list[str] = []
            for reference in finding_draft.assumption_refs:
                assumption_target = resolve_finding(reference)
                if assumption_target.kind is not WorkspaceFindingKind.ASSUMPTION:
                    raise WorkspaceReferenceError(
                        f"assumption reference is not an ASSUMPTION: {reference}"
                    )
                assumption_ids.append(
                    self._resolved_card_id(reference, id_map, assumption_target)
                )
            if card_id in dependency_ids or card_id in assumption_ids:
                raise WorkspaceReferenceError("a finding cannot depend on itself")
            findings.append(
                WorkspaceFinding(
                    card_id=card_id,
                    kind=finding_draft.kind,
                    title=finding_draft.title,
                    body=finding_draft.body,
                    dependency_ids=dependency_ids,
                    assumption_ids=tuple(assumption_ids),
                    tags=finding_draft.tags,
                    created_revision=revision_id,
                    created_at=now,
                )
            )
        _reject_new_dependency_cycles(findings)

        scratch = tuple(
            WorkspaceScratchEntry(
                scratch_id=id_map[draft.client_ref],
                body=draft.body,
                tags=draft.tags,
                created_revision=revision_id,
                created_at=now,
            )
            for draft in request.scratch
        )
        finding_by_ref: dict[str, WorkspaceFinding | WorkspaceFindingDraft] = {
            **existing_findings,
            **finding_drafts,
        }
        attempts: list[WorkspaceAttempt] = []
        attached_artifacts: list[str] = []
        for attempt_draft in request.attempts:
            attempt_target = finding_by_ref.get(attempt_draft.target_ref)
            if attempt_target is None:
                raise WorkspaceReferenceError(
                    f"attempt target does not exist: {attempt_draft.target_ref}"
                )
            for artifact_uri in attempt_draft.artifact_uris:
                try:
                    self.store.get(artifact_uri)
                except StoreError as exc:
                    raise WorkspaceReferenceError(
                        f"attempt artifact does not exist: {artifact_uri}"
                    ) from exc
                attached_artifacts.append(artifact_uri)
            attempts.append(
                WorkspaceAttempt(
                    attempt_id=id_map[attempt_draft.client_ref],
                    target_card_id=self._resolved_card_id(
                        attempt_draft.target_ref,
                        id_map,
                        attempt_target,
                    ),
                    method=attempt_draft.method,
                    outcome=attempt_draft.outcome,
                    summary=attempt_draft.summary,
                    artifact_uris=attempt_draft.artifact_uris,
                    tags=attempt_draft.tags,
                    created_revision=revision_id,
                    created_at=now,
                )
            )

        current_marks = self._current_marks(
            connection,
            request.workspace_id,
            request.branch_id,
        )
        marks: list[WorkspaceMark] = []
        for mark_draft in request.marks:
            mark_target = finding_by_ref.get(mark_draft.target_ref)
            if mark_target is None:
                raise WorkspaceReferenceError(
                    f"mark target does not exist: {mark_draft.target_ref}"
                )
            if mark_target.kind is WorkspaceFindingKind.PROBLEM:
                raise WorkspaceReferenceError(
                    "the canonical PROBLEM card cannot be lifecycle-marked"
                )
            if (
                mark_draft.state is WorkspaceCardState.CLOSED
                and mark_target.kind
                not in (WorkspaceFindingKind.GOAL, WorkspaceFindingKind.OPEN_QUESTION)
            ):
                raise WorkspaceReferenceError(
                    "CLOSED marks apply only to GOAL or OPEN_QUESTION cards"
                )
            target_card_id = self._resolved_card_id(
                mark_draft.target_ref,
                id_map,
                mark_target,
            )
            current_mark = current_marks.get(target_card_id)
            if (
                current_mark is not None
                and current_mark.state in _INVALIDATING_CARD_STATES
                and mark_draft.state
                not in {
                    WorkspaceCardState.ACTIVE,
                    *_INVALIDATING_CARD_STATES,
                }
            ):
                raise WorkspaceReferenceError(
                    "a RETRACTED or SUPERSEDED card must be marked ACTIVE before "
                    "it can be CLOSED or ARCHIVED"
                )
            superseded_by_id: str | None = None
            if mark_draft.superseded_by_ref is not None:
                replacement = finding_by_ref.get(mark_draft.superseded_by_ref)
                if replacement is None:
                    raise WorkspaceReferenceError(
                        "superseding replacement does not exist: "
                        f"{mark_draft.superseded_by_ref}"
                    )
                superseded_by_id = self._resolved_card_id(
                    mark_draft.superseded_by_ref,
                    id_map,
                    replacement,
                )
                if superseded_by_id == target_card_id:
                    raise WorkspaceReferenceError("a card cannot supersede itself")
            marks.append(
                WorkspaceMark(
                    mark_id=id_map[mark_draft.client_ref],
                    target_card_id=target_card_id,
                    state=mark_draft.state,
                    reason=mark_draft.reason,
                    superseded_by_id=superseded_by_id,
                    created_revision=revision_id,
                    created_at=now,
                )
            )
        _reject_supersession_cycles(
            current_marks.values(),
            marks,
        )

        focus = self._resolve_focus(
            request.focus,
            id_map,
            finding_by_ref,
            revision_id,
        )
        revision = WorkspaceRevision(
            revision_id=revision_id,
            workspace_id=request.workspace_id,
            branch_id=request.branch_id,
            parent_revision=request.base_revision,
            operation=WorkspaceRevisionOperation.WRITE,
            request_digest=request_digest,
            scratch=scratch,
            findings=tuple(findings),
            attempts=tuple(attempts),
            marks=tuple(marks),
            focus=focus,
            created_at=now,
        )
        parents = tuple(
            dict.fromkeys(
                (
                    branch["revision_artifact_uri"],
                    *attached_artifacts,
                )
            )
        )
        stored = self.store.put(
            schema_uri=self.revision_schema_uri,
            semantics_uri=self.revision_semantics_uri,
            payload=revision.model_dump(mode="json"),
            parents=parents,
            summary=(
                f"workspace revision with {len(scratch)} scratch entries, "
                f"{len(findings)} findings, {len(attempts)} attempts, and "
                f"{len(marks)} marks"
            ),
        )
        return _PreparedWrite(
            revision=revision,
            revision_artifact_uri=stored.artifact_uri,
            id_map=id_map,
        )

    def _referenced_findings(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceWriteRequest,
    ) -> dict[str, WorkspaceFinding]:
        references = {
            reference
            for draft in request.findings
            for reference in (*draft.dependency_refs, *draft.assumption_refs)
            if reference.startswith("card://")
        }
        references.update(
            draft.target_ref
            for draft in request.attempts
            if draft.target_ref.startswith("card://")
        )
        references.update(
            reference
            for draft in request.marks
            for reference in (draft.target_ref, draft.superseded_by_ref)
            if reference is not None and reference.startswith("card://")
        )
        if request.focus is not None:
            if (
                request.focus.active_ref is not None
                and request.focus.active_ref.startswith("card://")
            ):
                references.add(request.focus.active_ref)
            references.update(
                reference
                for reference in request.focus.pinned_refs
                if reference.startswith("card://")
            )
        result: dict[str, WorkspaceFinding] = {}
        for reference in sorted(references):
            row = connection.execute(
                """
                SELECT payload_json
                FROM workspace_findings
                WHERE card_id = ? AND workspace_id = ? AND branch_id = ?
                """,
                (reference, request.workspace_id, request.branch_id),
            ).fetchone()
            if row is None:
                raise WorkspaceReferenceError(
                    f"workspace finding reference does not exist: {reference}"
                )
            result[reference] = _model_from_json(
                WorkspaceFinding,
                row["payload_json"],
            )
        return result

    def _resolve_focus(
        self,
        draft: WorkspaceFocusDraft | None,
        id_map: dict[str, str],
        findings: dict[str, WorkspaceFinding | WorkspaceFindingDraft],
        revision_id: WorkspaceRevisionId,
    ) -> WorkspaceFocus | None:
        if draft is None:
            return None

        def resolve(reference: str) -> str:
            target = findings.get(reference)
            if target is None:
                raise WorkspaceReferenceError(
                    f"focus finding reference does not exist: {reference}"
                )
            return self._resolved_card_id(reference, id_map, target)

        return WorkspaceFocus(
            active_item_id=(
                resolve(draft.active_ref) if draft.active_ref is not None else None
            ),
            pinned_item_ids=tuple(
                resolve(reference) for reference in draft.pinned_refs
            ),
            updated_revision=revision_id,
        )

    @staticmethod
    def _resolved_card_id(
        reference: str,
        id_map: dict[str, str],
        target: WorkspaceFinding | WorkspaceFindingDraft,
    ) -> WorkspaceCardId:
        if isinstance(target, WorkspaceFinding):
            return target.card_id
        return id_map[reference]

    def _require_head(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        base_revision: str,
    ) -> sqlite3.Row:
        branch = self._branch_row(connection, workspace_id, branch_id)
        if branch["head_revision_id"] != base_revision:
            raise WorkspaceConflictError(
                "base_revision is stale; query the workspace and retry from its head"
            )
        return branch

    @staticmethod
    def _branch_row(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
    ) -> sqlite3.Row:
        row: sqlite3.Row | None = connection.execute(
            """
            SELECT b.head_revision_id, r.revision_artifact_uri, w.name
            FROM workspace_branches AS b
            JOIN workspaces AS w ON w.workspace_id = b.workspace_id
            JOIN workspace_revisions AS r
              ON r.revision_id = b.head_revision_id
            WHERE b.workspace_id = ? AND b.branch_id = ?
            """,
            (workspace_id, branch_id),
        ).fetchone()
        if row is None:
            raise WorkspaceNotFoundError("workspace branch does not exist")
        return row

    @staticmethod
    def _current_marks(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
    ) -> dict[str, WorkspaceMark]:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM workspace_marks
            WHERE workspace_id = ? AND branch_id = ?
            ORDER BY rowid
            """,
            (workspace_id, branch_id),
        ).fetchall()
        current: dict[str, WorkspaceMark] = {}
        for row in rows:
            mark = _model_from_json(WorkspaceMark, row["payload_json"])
            current[mark.target_card_id] = mark
        return current

    def _projection(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
    ) -> _WorkspaceProjection:
        rows = connection.execute(
            """
            SELECT rowid, payload_json
            FROM workspace_findings
            WHERE workspace_id = ? AND branch_id = ?
            ORDER BY rowid
            """,
            (request.workspace_id, request.branch_id),
        ).fetchall()
        findings: dict[str, WorkspaceFinding] = {}
        finding_order: dict[str, int] = {}
        for row in rows:
            finding = _model_from_json(WorkspaceFinding, row["payload_json"])
            findings[finding.card_id] = finding
            finding_order[finding.card_id] = int(row["rowid"])
        marks = self._current_marks(
            connection,
            request.workspace_id,
            request.branch_id,
        )
        root_cache: dict[str, tuple[str, ...]] = {}
        for card_id in _dependency_postorder(findings, findings):
            finding = findings[card_id]
            mark = marks.get(card_id)
            if mark is not None and mark.state in _INVALIDATING_CARD_STATES:
                root_cache[card_id] = (card_id,)
                continue
            roots: set[str] = set()
            for reference in (*finding.dependency_ids, *finding.assumption_ids):
                roots.update(root_cache[reference])
            root_cache[card_id] = tuple(sorted(roots))

        stale_due_to: dict[str, tuple[str, ...]] = {}
        for card_id in findings:
            mark = marks.get(card_id)
            is_root = mark is not None and mark.state in _INVALIDATING_CARD_STATES
            stale_due_to[card_id] = () if is_root else root_cache[card_id]
        return _WorkspaceProjection(
            findings=findings,
            finding_order=finding_order,
            marks=marks,
            stale_due_to=stale_due_to,
        )

    @staticmethod
    def _ordered_findings(
        projection: _WorkspaceProjection,
        predicate: Callable[[WorkspaceFinding], bool],
        limit: int,
    ) -> tuple[WorkspaceFinding, ...]:
        ordered = sorted(
            projection.findings.values(),
            key=lambda item: projection.finding_order[item.card_id],
            reverse=True,
        )
        return tuple(item for item in ordered if predicate(item))[:limit]

    def _resume_view(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        projection: _WorkspaceProjection,
    ) -> WorkspaceResumeView:
        branch = self._branch_row(
            connection,
            request.workspace_id,
            request.branch_id,
        )
        problems = sorted(
            (
                finding
                for finding in projection.findings.values()
                if finding.kind is WorkspaceFindingKind.PROBLEM
            ),
            key=lambda item: projection.finding_order[item.card_id],
        )
        if not problems:
            raise WorkspaceError("workspace has no problem finding")
        problem = problems[0]

        focus_row = connection.execute(
            """
            SELECT payload_json
            FROM workspace_focus
            WHERE workspace_id = ? AND branch_id = ?
            """,
            (request.workspace_id, request.branch_id),
        ).fetchone()
        focus = (
            _model_from_json(WorkspaceFocus, focus_row["payload_json"])
            if focus_row is not None
            else None
        )
        active = (
            projection.findings.get(focus.active_item_id)
            if focus is not None and focus.active_item_id is not None
            else None
        )
        if focus is not None and focus.active_item_id is not None and active is None:
            raise WorkspaceError("workspace focus cites an unknown finding")
        pinned = (
            tuple(
                self._finding_summary(
                    connection,
                    projection.findings[card_id],
                    projection,
                )
                for card_id in focus.pinned_item_ids
            )
            if focus is not None
            else ()
        )
        goals = tuple(
            self._finding_summary(connection, item, projection)
            for item in self._ordered_findings(
                projection,
                lambda item: (
                    item.kind
                    in (WorkspaceFindingKind.GOAL, WorkspaceFindingKind.OPEN_QUESTION)
                    and (
                        item.card_id not in projection.marks
                        or projection.marks[item.card_id].state
                        is WorkspaceCardState.ACTIVE
                    )
                ),
                request.limit,
            )
        )
        stale_items = tuple(
            self._finding_summary(connection, item, projection)
            for item in self._ordered_findings(
                projection,
                lambda item: bool(projection.stale_due_to[item.card_id]),
                request.limit,
            )
        )
        recent_findings = tuple(
            self._finding_summary(connection, item, projection)
            for item in self._ordered_findings(
                projection,
                lambda item: (
                    item.kind
                    not in (
                        WorkspaceFindingKind.PROBLEM,
                        WorkspaceFindingKind.GOAL,
                        WorkspaceFindingKind.OPEN_QUESTION,
                    )
                    and (
                        item.card_id not in projection.marks
                        or projection.marks[item.card_id].state
                        is not WorkspaceCardState.ARCHIVED
                    )
                ),
                request.limit,
            )
        )
        attempts = tuple(
            self._attempt_summary(connection, item)
            for item in self._attempt_list(connection, request, None, request.limit)
        )
        scratch = tuple(
            self._scratch_summary(connection, item)
            for item in self._scratch_list(connection, request, request.limit)
        )
        return WorkspaceResumeView(
            name=branch["name"],
            problem=self._finding_summary(connection, problem, projection),
            active_item=(
                self._finding_summary(connection, active, projection)
                if active is not None
                else None
            ),
            pinned_items=pinned,
            open_goals=goals,
            stale_items=stale_items,
            recent_findings=recent_findings,
            recent_attempts=attempts,
            recent_scratch=scratch,
        )

    def _frontier_view(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        projection: _WorkspaceProjection,
    ) -> tuple[WorkspaceFrontierItem, ...]:
        goals = self._ordered_findings(
            projection,
            lambda item: (
                item.kind
                in (WorkspaceFindingKind.GOAL, WorkspaceFindingKind.OPEN_QUESTION)
                and (
                    item.card_id not in projection.marks
                    or projection.marks[item.card_id].state is WorkspaceCardState.ACTIVE
                )
            ),
            request.limit,
        )
        frontier: list[WorkspaceFrontierItem] = []
        for goal in goals:
            attempts = self._attempt_list(connection, request, goal.card_id, 1)
            count_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM workspace_attempts
                WHERE workspace_id = ? AND branch_id = ? AND target_card_id = ?
                """,
                (request.workspace_id, request.branch_id, goal.card_id),
            ).fetchone()
            frontier.append(
                WorkspaceFrontierItem(
                    goal=self._finding_summary(connection, goal, projection),
                    attempt_count=int(count_row["count"]),
                    last_attempt=(
                        self._attempt_summary(connection, attempts[0])
                        if attempts
                        else None
                    ),
                )
            )
        return tuple(frontier)

    def _attempts_view(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        projection: _WorkspaceProjection,
    ) -> tuple[WorkspaceAttemptSummary, ...]:
        if (
            request.target_card_id is not None
            and request.target_card_id not in projection.findings
        ):
            raise WorkspaceNotFoundError("workspace finding does not exist")
        return tuple(
            self._attempt_summary(connection, item)
            for item in self._attempt_list(
                connection,
                request,
                request.target_card_id,
                request.limit,
            )
        )

    def _context_view(
        self,
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        projection: _WorkspaceProjection,
    ) -> WorkspaceContextView:
        target_card_id = request.target_card_id
        if target_card_id is None:
            raise WorkspaceError("CONTEXT query has no target card")
        target = projection.findings.get(target_card_id)
        if target is None:
            raise WorkspaceNotFoundError("workspace finding does not exist")

        ordered_ids = tuple(
            card_id
            for card_id in _dependency_postorder(
                projection.findings,
                (target_card_id,),
            )
            if card_id != target_card_id
        )
        dependencies = tuple(
            self._finding_summary(
                connection,
                projection.findings[card_id],
                projection,
            )
            for card_id in ordered_ids[: request.limit]
        )
        recent_attempts = tuple(
            self._attempt_summary(connection, item)
            for item in self._attempt_list(
                connection,
                request,
                target_card_id,
                request.limit,
            )
        )
        return WorkspaceContextView(
            target=self._finding_summary(connection, target, projection),
            dependencies=dependencies,
            recent_attempts=recent_attempts,
            total_dependency_count=len(ordered_ids),
            truncated=len(ordered_ids) > request.limit,
        )

    @staticmethod
    def _attempt_list(
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        target_card_id: str | None,
        limit: int,
    ) -> tuple[WorkspaceAttempt, ...]:
        target_clause = "AND target_card_id = ?" if target_card_id is not None else ""
        parameters: list[str | int] = [
            request.workspace_id,
            request.branch_id,
        ]
        if target_card_id is not None:
            parameters.append(target_card_id)
        parameters.append(limit)
        rows = connection.execute(
            f"""
            SELECT payload_json
            FROM workspace_attempts
            WHERE workspace_id = ? AND branch_id = ? {target_clause}
            ORDER BY rowid DESC
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return tuple(
            _model_from_json(WorkspaceAttempt, row["payload_json"]) for row in rows
        )

    @staticmethod
    def _scratch_list(
        connection: sqlite3.Connection,
        request: WorkspaceQueryRequest,
        limit: int,
    ) -> tuple[WorkspaceScratchEntry, ...]:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM workspace_scratch
            WHERE workspace_id = ? AND branch_id = ?
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (request.workspace_id, request.branch_id, limit),
        ).fetchall()
        return tuple(
            _model_from_json(WorkspaceScratchEntry, row["payload_json"]) for row in rows
        )

    def _finding_summary(
        self,
        connection: sqlite3.Connection,
        finding: WorkspaceFinding,
        projection: _WorkspaceProjection,
    ) -> WorkspaceItemSummary:
        mark = projection.marks.get(finding.card_id)
        return _finding_summary(
            finding,
            self._revision_artifact_uri(connection, finding.created_revision),
            mark,
            (
                self._revision_artifact_uri(connection, mark.created_revision)
                if mark is not None
                else None
            ),
            projection.stale_due_to[finding.card_id],
        )

    def _attempt_summary(
        self,
        connection: sqlite3.Connection,
        attempt: WorkspaceAttempt,
    ) -> WorkspaceAttemptSummary:
        return _attempt_summary(
            attempt,
            self._revision_artifact_uri(connection, attempt.created_revision),
        )

    def _scratch_summary(
        self,
        connection: sqlite3.Connection,
        entry: WorkspaceScratchEntry,
    ) -> WorkspaceScratchSummary:
        return _scratch_summary(
            entry,
            self._revision_artifact_uri(connection, entry.created_revision),
        )

    @staticmethod
    def _revision_artifact_uri(
        connection: sqlite3.Connection,
        revision_id: str,
    ) -> str:
        row = connection.execute(
            """
            SELECT revision_artifact_uri
            FROM workspace_revisions
            WHERE revision_id = ?
            """,
            (revision_id,),
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace item cites an unknown revision")
        return str(row["revision_artifact_uri"])

    @staticmethod
    def _insert_revision(
        connection: sqlite3.Connection,
        revision: WorkspaceRevision,
        artifact_uri: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_revisions(
                revision_id,
                revision_artifact_uri,
                workspace_id,
                branch_id,
                parent_revision_id,
                request_digest,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                artifact_uri,
                revision.workspace_id,
                revision.branch_id,
                revision.parent_revision,
                revision.request_digest,
                revision.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_scratch(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        entry: WorkspaceScratchEntry,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_scratch(
                scratch_id,
                workspace_id,
                branch_id,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.scratch_id,
                workspace_id,
                branch_id,
                entry.created_revision,
                canonicalize_json(entry.model_dump(mode="json")),
                entry.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_finding(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        finding: WorkspaceFinding,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_findings(
                card_id,
                workspace_id,
                branch_id,
                kind,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.card_id,
                workspace_id,
                branch_id,
                finding.kind.value,
                finding.created_revision,
                canonicalize_json(finding.model_dump(mode="json")),
                finding.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_attempt(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        attempt: WorkspaceAttempt,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_attempts(
                attempt_id,
                workspace_id,
                branch_id,
                target_card_id,
                outcome,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.attempt_id,
                workspace_id,
                branch_id,
                attempt.target_card_id,
                attempt.outcome.value,
                attempt.created_revision,
                canonicalize_json(attempt.model_dump(mode="json")),
                attempt.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_mark(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        mark: WorkspaceMark,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_marks(
                mark_id,
                workspace_id,
                branch_id,
                target_card_id,
                state,
                created_revision_id,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mark.mark_id,
                workspace_id,
                branch_id,
                mark.target_card_id,
                mark.state.value,
                mark.created_revision,
                canonicalize_json(mark.model_dump(mode="json")),
                mark.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _upsert_focus(
        connection: sqlite3.Connection,
        workspace_id: str,
        branch_id: str,
        focus: WorkspaceFocus,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_focus(
                branch_id, workspace_id, updated_revision_id, payload_json
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
                updated_revision_id = excluded.updated_revision_id,
                payload_json = excluded.payload_json
            """,
            (
                branch_id,
                workspace_id,
                focus.updated_revision,
                canonicalize_json(focus.model_dump(mode="json")),
            ),
        )

    @staticmethod
    def _bind_idempotency(
        connection: sqlite3.Connection,
        idempotency_key: str,
        operation: WorkspaceRevisionOperation,
        request_digest: str,
        result: WorkspaceOpenResult | WorkspaceWriteResult,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workspace_idempotency(
                idempotency_key, operation, request_digest, response_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                idempotency_key,
                operation.value,
                request_digest,
                canonicalize_json(result.model_dump(mode="json")),
            ),
        )

    @staticmethod
    def _reuse(
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        operation: WorkspaceRevisionOperation,
        request_digest: str,
        result_type: type[_ModelT],
    ) -> _ModelT | None:
        row = connection.execute(
            """
            SELECT operation, request_digest, response_json
            FROM workspace_idempotency
            WHERE idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        if (
            row["operation"] != operation.value
            or row["request_digest"] != request_digest
        ):
            raise WorkspaceIdempotencyError(
                "idempotency key is already bound to a different workspace request"
            )
        return _model_from_json(result_type, row["response_json"])
