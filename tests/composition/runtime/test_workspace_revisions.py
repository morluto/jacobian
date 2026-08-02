from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.component.workspaces._workspace_support import _open

from jacobian.contracts.workspaces import (
    WorkspaceAttemptDraft,
    WorkspaceAttemptOutcome,
    WorkspaceFindingDraft,
    WorkspaceFindingKind,
    WorkspaceFocusDraft,
    WorkspaceOpenRequest,
    WorkspaceQueryRequest,
    WorkspaceQueryView,
    WorkspaceRevision,
    WorkspaceScratchDraft,
    WorkspaceWriteRequest,
)
from jacobian.runtime import create_runtime
from jacobian.workspaces import (
    WorkspaceConflictError,
    WorkspaceIdempotencyError,
    WorkspaceReferenceError,
)

pytestmark = pytest.mark.usefixtures("attached_complete_runtime")


def _blob_paths(blob_root: Path) -> set[Path]:
    return {
        blob
        for prefix in blob_root.iterdir()
        if prefix.is_dir()
        for blob in prefix.iterdir()
        if blob.is_file()
    }


def test_workspace_open_is_idempotent_and_restart_replays_revision(
    tmp_path: Path,
) -> None:
    runtime = create_runtime(tmp_path)
    request = WorkspaceOpenRequest(
        idempotency_key="workspace-open-replay-001",
        name="replay fixture",
        problem="Prove or refute the fixture claim.",
    )

    first = runtime.core.workspaces.open(request)
    second = runtime.core.workspaces.open(request)

    assert second == first
    revision_artifact = runtime.core.store.get(first.revision_artifact_uri)
    revision = WorkspaceRevision.model_validate(revision_artifact.payload)
    assert revision.revision_id == first.revision_id
    assert revision.parent_revision is None
    assert revision.findings[0].card_id == first.problem_card_id
    assert revision.findings[0].verification == "UNVERIFIED"
    assert revision.focus is not None
    assert revision.focus.active_item_id == first.problem_card_id

    restarted = create_runtime(tmp_path)
    resume = restarted.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=first.workspace_id,
            branch_id=first.branch_id,
            view=WorkspaceQueryView.RESUME,
        )
    )

    assert resume.revision_id == first.revision_id
    assert resume.revision_artifact_uri == first.revision_artifact_uri
    assert resume.resume is not None
    assert resume.resume.problem.verification == "UNVERIFIED"
    assert "retrieval does not promote" in resume.warning


def test_concurrent_idempotent_workspace_open_publishes_one_revision(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    request = WorkspaceOpenRequest(
        idempotency_key="workspace-open-concurrent-001",
        name="concurrent replay fixture",
        problem="Two retries must resolve to one accepted workspace.",
    )
    ready = threading.Barrier(3)
    results = []
    errors: list[BaseException] = []
    with runtime.core.store.connection() as connection:
        artifacts_before = connection.execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0]
    quota_before = runtime.core.store._blob_bytes_committed()
    blobs_before = _blob_paths(runtime.core.store.blob_root)

    def open_workspace() -> None:
        ready.wait(timeout=2)
        try:
            results.append(runtime.core.workspaces.open(request))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=open_workspace) for _ in range(2)]
    for worker in workers:
        worker.start()
    ready.wait(timeout=2)
    for worker in workers:
        worker.join(timeout=5)

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    with runtime.core.store.connection() as connection:
        artifacts_after = connection.execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0]
    assert artifacts_after == artifacts_before + 1
    quota_after = runtime.core.store._blob_bytes_committed()
    blobs_after = _blob_paths(runtime.core.store.blob_root)
    assert len(blobs_after - blobs_before) == 2
    assert quota_after > quota_before
    assert runtime.core.store._blob_bytes_committed() == quota_after


def test_workspace_write_cannot_add_a_second_problem(attached_complete_runtime) -> None:
    opened = _open(attached_complete_runtime, key="workspace-open-single-problem-001")
    problem_draft = WorkspaceFindingDraft(
        client_ref="P2",
        kind=WorkspaceFindingKind.PROBLEM,
        title="Hidden second problem",
        body="A write must not create another canonical problem.",
    )

    with pytest.raises(
        ValidationError,
        match=r"only workspace\.open may create the canonical PROBLEM",
    ):
        WorkspaceWriteRequest(
            idempotency_key="workspace-write-second-problem-contract-001",
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            base_revision=opened.revision_id,
            findings=(problem_draft,),
        )

    bypassed_contract = WorkspaceWriteRequest.model_construct(
        request_version="1",
        idempotency_key="workspace-write-second-problem-service-001",
        workspace_id=opened.workspace_id,
        branch_id=opened.branch_id,
        base_revision=opened.revision_id,
        scratch=(),
        findings=(problem_draft,),
        attempts=(),
        marks=(),
        focus=None,
    )
    with pytest.raises(
        ValidationError,
        match=r"only workspace\.open may create the canonical PROBLEM",
    ):
        attached_complete_runtime.core.workspaces.write(bypassed_contract)

    with (
        pytest.raises(
            WorkspaceReferenceError,
            match=r"only workspace\.open may create the canonical PROBLEM",
        ),
        attached_complete_runtime.core.store.connection() as connection,
    ):
        attached_complete_runtime.core.workspaces._prepare_write(
            connection,
            bypassed_contract,
            "sha256:" + ("0" * 64),
        )

    resumed = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.RESUME,
        )
    )
    assert resumed.revision_id == opened.revision_id
    assert resumed.resume is not None
    assert resumed.resume.problem.card_id == opened.problem_card_id


def test_workspace_write_builds_resume_frontier_and_attempt_views(
    attached_complete_runtime,
) -> None:
    opened = _open(attached_complete_runtime)
    request = WorkspaceWriteRequest(
        idempotency_key="workspace-write-batch-001",
        workspace_id=opened.workspace_id,
        branch_id=opened.branch_id,
        base_revision=opened.revision_id,
        scratch=(
            WorkspaceScratchDraft(
                client_ref="S1",
                body="Try induction, but keep the boundary case separate.",
                tags=("induction",),
            ),
        ),
        findings=(
            WorkspaceFindingDraft(
                client_ref="A1",
                kind=WorkspaceFindingKind.ASSUMPTION,
                title="Finite scope",
                body="The current experiment only concerns n <= 20.",
            ),
            WorkspaceFindingDraft(
                client_ref="G1",
                kind=WorkspaceFindingKind.GOAL,
                title="Close the induction step",
                body="Find a strengthened hypothesis that controls the boundary term.",
                assumption_refs=("A1",),
                tags=("frontier",),
            ),
            WorkspaceFindingDraft(
                client_ref="L1",
                kind=WorkspaceFindingKind.CLAIM,
                title="Candidate recurrence",
                body="A stronger recurrence may suffice.",
                dependency_refs=("G1",),
            ),
        ),
        attempts=(
            WorkspaceAttemptDraft(
                client_ref="T1",
                target_ref="G1",
                method="ordinary_induction",
                outcome=WorkspaceAttemptOutcome.BLOCKED,
                summary="The step requires a bound not present in the hypothesis.",
            ),
        ),
        focus=WorkspaceFocusDraft(active_ref="G1", pinned_refs=("G1", "L1")),
    )

    written = attached_complete_runtime.core.workspaces.write(request)
    reused = attached_complete_runtime.core.workspaces.write(request)

    assert reused == written
    assert written.scratch_written == 1
    assert written.findings_written == 3
    assert written.attempts_written == 1
    assert set(written.unverified_finding_ids) == {
        written.id_map["A1"],
        written.id_map["G1"],
        written.id_map["L1"],
    }
    assert set(written.unresolved_dependency_ids) == {
        written.id_map["A1"],
        written.id_map["G1"],
    }
    assert "cannot establish an exact mathematical conclusion" in (
        written.assurance_notice
    )
    revision_artifact = attached_complete_runtime.core.store.get(
        written.revision_artifact_uri
    )
    assert opened.revision_artifact_uri in revision_artifact.manifest.parents
    revision = WorkspaceRevision.model_validate(revision_artifact.payload)
    assert revision.parent_revision == opened.revision_id
    assert {item.verification for item in revision.findings} == {"UNVERIFIED"}
    assert {item.verification for item in revision.attempts} == {"UNVERIFIED"}

    resume = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.RESUME,
        )
    )
    assert resume.resume is not None
    assert resume.resume.active_item is not None
    assert resume.resume.active_item.card_id == written.id_map["G1"]
    assert (
        resume.resume.active_item.created_revision_artifact_uri
        == written.revision_artifact_uri
    )
    assert {item.card_id for item in resume.resume.pinned_items} == {
        written.id_map["G1"],
        written.id_map["L1"],
    }
    assert resume.resume.open_goals[0].assumption_ids == (written.id_map["A1"],)
    assert resume.resume.recent_attempts[0].verification == "UNVERIFIED"

    frontier = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.FRONTIER,
        )
    )
    assert len(frontier.frontier) == 1
    assert frontier.frontier[0].attempt_count == 1
    assert frontier.frontier[0].last_attempt is not None
    assert frontier.frontier[0].last_attempt.outcome is WorkspaceAttemptOutcome.BLOCKED

    attempts = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.ATTEMPTS,
            target_card_id=written.id_map["G1"],
        )
    )
    assert [item.attempt_id for item in attempts.attempts] == [written.id_map["T1"]]


def test_workspace_rejects_stale_base_without_partial_index_writes(
    attached_complete_runtime,
) -> None:
    opened = _open(attached_complete_runtime)
    accepted = attached_complete_runtime.core.workspaces.write(
        WorkspaceWriteRequest(
            idempotency_key="workspace-write-accepted-001",
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            base_revision=opened.revision_id,
            findings=(
                WorkspaceFindingDraft(
                    client_ref="G1",
                    kind=WorkspaceFindingKind.GOAL,
                    title="Accepted goal",
                    body="This entry advances the branch.",
                ),
            ),
        )
    )

    with pytest.raises(WorkspaceConflictError, match="base_revision is stale"):
        attached_complete_runtime.core.workspaces.write(
            WorkspaceWriteRequest(
                idempotency_key="workspace-write-stale-001",
                workspace_id=opened.workspace_id,
                branch_id=opened.branch_id,
                base_revision=opened.revision_id,
                findings=(
                    WorkspaceFindingDraft(
                        client_ref="G2",
                        kind=WorkspaceFindingKind.GOAL,
                        title="Must not commit",
                        body="This write uses a stale branch head.",
                    ),
                ),
            )
        )

    with sqlite3.connect(attached_complete_runtime.core.store.db_path) as connection:
        finding_count = connection.execute(
            "SELECT COUNT(*) FROM workspace_findings WHERE workspace_id = ?",
            (opened.workspace_id,),
        ).fetchone()[0]
        revision_count = connection.execute(
            "SELECT COUNT(*) FROM workspace_revisions WHERE workspace_id = ?",
            (opened.workspace_id,),
        ).fetchone()[0]
    assert finding_count == 2
    assert revision_count == 2
    current = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.RESUME,
            revision_id=accepted.revision_id,
        )
    )
    assert current.revision_id == accepted.revision_id

    with pytest.raises(
        WorkspaceConflictError,
        match="query revision_id is stale",
    ):
        attached_complete_runtime.core.workspaces.query(
            WorkspaceQueryRequest(
                workspace_id=opened.workspace_id,
                branch_id=opened.branch_id,
                view=WorkspaceQueryView.RESUME,
                revision_id=opened.revision_id,
            )
        )


def test_concurrent_workspace_writes_publish_only_the_winning_revision(
    attached_complete_runtime,
) -> None:
    runtime = attached_complete_runtime
    opened = _open(runtime, key="workspace-open-concurrent-write-001")
    ready = threading.Barrier(3)
    results = []
    errors: list[BaseException] = []
    with runtime.core.store.connection() as connection:
        artifacts_before = connection.execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0]
    quota_before = runtime.core.store._blob_bytes_committed()
    blobs_before = _blob_paths(runtime.core.store.blob_root)

    def write(client_ref: str) -> None:
        request = WorkspaceWriteRequest(
            idempotency_key=f"workspace-write-concurrent-{client_ref}",
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            base_revision=opened.revision_id,
            findings=(
                WorkspaceFindingDraft(
                    client_ref=client_ref,
                    kind=WorkspaceFindingKind.GOAL,
                    title=f"Competing goal {client_ref}",
                    body="Only one revision may advance this branch head.",
                ),
            ),
        )
        ready.wait(timeout=2)
        try:
            results.append(runtime.core.workspaces.write(request))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=write, args=(ref,)) for ref in ("G1", "G2")]
    for worker in workers:
        worker.start()
    ready.wait(timeout=2)
    for worker in workers:
        worker.join(timeout=5)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceConflictError)
    with runtime.core.store.connection() as connection:
        artifacts_after = connection.execute(
            "SELECT COUNT(*) FROM artifacts"
        ).fetchone()[0]
        revisions = connection.execute(
            "SELECT COUNT(*) FROM workspace_revisions WHERE workspace_id = ?",
            (opened.workspace_id,),
        ).fetchone()[0]
    assert artifacts_after == artifacts_before + 1
    assert revisions == 2
    quota_after = runtime.core.store._blob_bytes_committed()
    blobs_after = _blob_paths(runtime.core.store.blob_root)
    assert len(blobs_after - blobs_before) == 2
    assert quota_after > quota_before
    assert runtime.core.store._blob_bytes_committed() == quota_after


def test_workspace_rejects_idempotency_rebinding_and_invalid_references(
    attached_complete_runtime,
) -> None:
    opened = _open(attached_complete_runtime, key="workspace-open-binding-001")

    with pytest.raises(WorkspaceIdempotencyError, match="different workspace request"):
        attached_complete_runtime.core.workspaces.open(
            WorkspaceOpenRequest(
                idempotency_key="workspace-open-binding-001",
                name="different",
                problem="This must not reuse the first workspace.",
            )
        )

    with pytest.raises(WorkspaceReferenceError, match="does not exist"):
        attached_complete_runtime.core.workspaces.write(
            WorkspaceWriteRequest(
                idempotency_key="workspace-write-missing-ref-001",
                workspace_id=opened.workspace_id,
                branch_id=opened.branch_id,
                base_revision=opened.revision_id,
                findings=(
                    WorkspaceFindingDraft(
                        client_ref="L1",
                        kind=WorkspaceFindingKind.CLAIM,
                        title="Dangling dependency",
                        body="This finding cites a missing card.",
                        dependency_refs=("card://00000000000000000000000000000000",),
                    ),
                ),
            )
        )

    with pytest.raises(WorkspaceReferenceError, match="contain a cycle"):
        attached_complete_runtime.core.workspaces.write(
            WorkspaceWriteRequest(
                idempotency_key="workspace-write-cycle-001",
                workspace_id=opened.workspace_id,
                branch_id=opened.branch_id,
                base_revision=opened.revision_id,
                findings=(
                    WorkspaceFindingDraft(
                        client_ref="L1",
                        kind=WorkspaceFindingKind.CLAIM,
                        title="First",
                        body="First cyclic claim.",
                        dependency_refs=("L2",),
                    ),
                    WorkspaceFindingDraft(
                        client_ref="L2",
                        kind=WorkspaceFindingKind.CLAIM,
                        title="Second",
                        body="Second cyclic claim.",
                        dependency_refs=("L1",),
                    ),
                ),
            )
        )


def test_workspace_focus_clear_is_explicit_and_resumable(
    attached_complete_runtime,
) -> None:
    opened = _open(attached_complete_runtime, key="workspace-open-focus-clear-001")

    cleared = attached_complete_runtime.core.workspaces.write(
        WorkspaceWriteRequest(
            idempotency_key="workspace-write-focus-clear-001",
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            base_revision=opened.revision_id,
            focus=WorkspaceFocusDraft(clear=True),
        )
    )

    assert cleared.focus_updated is True
    resume = attached_complete_runtime.core.workspaces.query(
        WorkspaceQueryRequest(
            workspace_id=opened.workspace_id,
            branch_id=opened.branch_id,
            view=WorkspaceQueryView.RESUME,
        )
    )
    assert resume.resume is not None
    assert resume.resume.active_item is None
    assert resume.resume.pinned_items == ()
