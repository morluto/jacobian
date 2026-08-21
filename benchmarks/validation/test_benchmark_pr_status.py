from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence

from tools import benchmark_pr_status
from tools.benchmark_pr_status import GitHubReader, build_status, render_human
from tools.command_runner import (
    ToolCommandResult,
    ToolCommandStatus,
    operator_environment,
)


def _zip_plan(digest: str = "sha256:abc") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("plan.json", json.dumps({"planner_digest": digest}))
    return output.getvalue()


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: Sequence[str]) -> bytes:
        call = tuple(arguments)
        self.calls.append(call)
        if call[:2] == ("api", "graphql"):
            return json.dumps(
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "nodes": [
                                        {"isResolved": False},
                                        {"isResolved": True},
                                    ],
                                    "pageInfo": {
                                        "hasNextPage": False,
                                        "endCursor": None,
                                    },
                                }
                            }
                        }
                    }
                }
            ).encode()
        if call[-1].endswith("/artifacts"):
            return json.dumps(
                {
                    "artifacts": [
                        {
                            "id": 77,
                            "name": "benchmark-plan",
                            "expired": False,
                        }
                    ]
                }
            ).encode()
        if call[-1].endswith("/zip"):
            return _zip_plan()
        raise AssertionError(call)


def _payload(*, validation: str = "SUCCESS") -> dict[str, object]:
    checks = [
        {"name": name, "conclusion": "SUCCESS", "detailsUrl": ""}
        for name in (
            "Benchmark Static Quality",
            "Benchmark Contracts, Adapters, Records & Digests",
        )
    ]
    checks.append(
        {
            "name": "Benchmark Validation",
            "conclusion": validation,
            "detailsUrl": "https://github.com/o/r/actions/runs/123/job/456",
        }
    )
    return {
        "number": 42,
        "title": "benchmarks: add task",
        "url": "https://github.com/o/r/pull/42",
        "state": "OPEN",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "files": [
            {
                "path": "benchmarks/datasets/mathematical-benchmarks-v1/"
                "task-a/tests/verifier.py"
            }
        ],
        "statusCheckRollup": checks,
    }


def test_batch_status_binds_actual_plan_digest_and_reports_blocker() -> None:
    runner = FakeRunner()
    reader = GitHubReader("o/r", runner=runner)

    status = build_status(reader, 42, _payload())

    assert status.tasks == ("mathematical-benchmarks-v1/task-a",)
    assert status.unresolved_threads == 1
    assert status.preparation == "confirmed"
    assert status.plan_digest == "sha256:abc"
    assert status.ci == "passed"
    assert status.merge_ready is False
    assert status.blockers == ("unresolved-threads=1",)
    assert "#42 blocked" in render_human([status])


def test_failed_validation_is_ci_failure_even_when_plan_exists() -> None:
    runner = FakeRunner()
    status = build_status(
        GitHubReader("o/r", runner=runner), 42, _payload(validation="FAILURE")
    )

    assert status.ci == "failed"
    assert "ci=failed" in status.blockers


def test_missing_plan_artifact_is_fail_closed() -> None:
    class MissingPlan(FakeRunner):
        def __call__(self, arguments: Sequence[str]) -> bytes:
            if tuple(arguments)[-1].endswith("/artifacts"):
                return b'{"artifacts":[]}'
            return super().__call__(arguments)

    runner = MissingPlan()
    status = build_status(GitHubReader("o/r", runner=runner), 42, _payload())

    assert status.plan_digest is None
    assert "validation-plan=missing" in status.blockers


def test_skipped_preparation_check_is_not_confirmed() -> None:
    payload = _payload()
    checks = payload["statusCheckRollup"]
    assert isinstance(checks, list)
    checks[0]["conclusion"] = "SKIPPED"
    runner = FakeRunner()

    status = build_status(GitHubReader("o/r", runner=runner), 42, payload)

    assert status.preparation == "failed"
    assert "preparation=failed" in status.blockers


def test_unstable_merge_state_is_blocked() -> None:
    payload = _payload()
    payload["mergeStateStatus"] = "UNSTABLE"
    runner = FakeRunner()

    status = build_status(GitHubReader("o/r", runner=runner), 42, payload)

    assert "merge-state=UNSTABLE" in status.blockers


def test_gh_receives_only_explicit_auth_and_process_environment(
    monkeypatch,
) -> None:
    forwarded: dict[str, str] = {}

    for name, value in {
        "PATH": "/operator/bin",
        "HOME": "/operator/home",
        "XDG_CONFIG_HOME": "/operator/config",
        "GH_CONFIG_DIR": "/operator/gh",
        "GH_TOKEN": "gh-token",
        "GITHUB_TOKEN": "github-token",
        "UNRELATED_SECRET": "must-not-be-forwarded",
    }.items():
        monkeypatch.setenv(name, value)
    expected = dict(
        operator_environment(
            include=(
                "PATH",
                "HOME",
                "XDG_CONFIG_HOME",
                "GH_CONFIG_DIR",
                "GH_TOKEN",
                "GITHUB_TOKEN",
            )
        )
    )

    def fake_run_operator_command(*args, **kwargs):
        forwarded.update(kwargs["environment"])
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"authenticated",
            stderr=b"",
        )

    monkeypatch.setattr(
        benchmark_pr_status,
        "run_operator_command",
        fake_run_operator_command,
    )

    assert benchmark_pr_status._gh(("auth", "status")) == b"authenticated"
    assert forwarded == expected
    assert "UNRELATED_SECRET" not in forwarded
