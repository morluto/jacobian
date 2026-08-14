#!/usr/bin/env python3
"""Report merge readiness for one or more benchmark pull requests."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.command_runner import (  # noqa: E402
    ToolCommandStatus,
    operator_environment,
    run_operator_command,
)

_TASK_PATH = re.compile(r"^benchmarks/datasets/([^/]+)/([^/]+)/")
_RUN_ID = re.compile(r"/actions/runs/(\d+)(?:/|$)")
_PREPARATION_CHECKS = (
    "Benchmark Static Quality",
    "Benchmark Contracts, Adapters, Records & Digests",
)
_VALIDATION_CHECK = "Benchmark Validation"
_GH_ENVIRONMENT = (
    "PATH",
    "HOME",
    "XDG_CONFIG_HOME",
    "GH_CONFIG_DIR",
    "GH_TOKEN",
    "GITHUB_TOKEN",
)


class StatusError(RuntimeError):
    """Live GitHub state could not be read or validated."""


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    state: str
    link: str


@dataclass(frozen=True, slots=True)
class PullRequestStatus:
    number: int
    title: str
    url: str
    tasks: tuple[str, ...]
    unresolved_threads: int
    preparation: str
    plan_digest: str | None
    ci: str
    merge_ready: bool
    blockers: tuple[str, ...]


Runner = Callable[[Sequence[str]], bytes]


def _gh(arguments: Sequence[str]) -> bytes:
    result = run_operator_command(
        "gh",
        arguments,
        cwd=ROOT,
        timeout_seconds=120.0,
        stdout_limit_bytes=16 * 1024 * 1024,
        stderr_limit_bytes=4 * 1024 * 1024,
        environment=operator_environment(include=_GH_ENVIRONMENT),
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        detail = result.diagnostic or result.stderr.decode(errors="replace").strip()
        raise StatusError(detail or "gh command failed")
    return bytes(result.stdout)


def _json(payload: bytes, context: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise StatusError(f"invalid {context} JSON: {exc}") from exc


class GitHubReader:
    """Small read-only adapter around the authenticated gh CLI."""

    def __init__(self, repository: str, *, runner: Runner = _gh) -> None:
        if repository.count("/") != 1:
            raise StatusError(f"invalid repository slug: {repository}")
        self.repository = repository
        self.runner = runner

    def pull_request(self, number: int) -> dict[str, Any]:
        fields = (
            "number,title,url,state,isDraft,mergeable,mergeStateStatus,"
            "headRefOid,files,statusCheckRollup"
        )
        payload = _json(
            self.runner(
                ("-R", self.repository, "pr", "view", str(number), "--json", fields)
            ),
            f"PR {number}",
        )
        if not isinstance(payload, dict):
            raise StatusError(f"PR {number} response must be an object")
        return payload

    def unresolved_threads(self, number: int) -> int:
        owner, repository = self.repository.split("/", 1)
        query = (
            "query($owner:String!,$repo:String!,$number:Int!,$after:String){"
            "repository(owner:$owner,name:$repo){pullRequest(number:$number){"
            "reviewThreads(first:100,after:$after){nodes{isResolved}"
            "pageInfo{hasNextPage endCursor}}}}}"
        )
        after: str | None = None
        unresolved = 0
        while True:
            arguments = [
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-F",
                f"owner={owner}",
                "-F",
                f"repo={repository}",
                "-F",
                f"number={number}",
            ]
            if after is not None:
                arguments.extend(("-F", f"after={after}"))
            payload = _json(self.runner(arguments), f"PR {number} review threads")
            try:
                threads = payload["data"]["repository"]["pullRequest"]["reviewThreads"]
                unresolved += sum(not node["isResolved"] for node in threads["nodes"])
                page = threads["pageInfo"]
            except (KeyError, TypeError) as exc:
                raise StatusError(
                    f"PR {number} review-thread response is malformed"
                ) from exc
            if not page["hasNextPage"]:
                return unresolved
            after = page["endCursor"]
            if not isinstance(after, str) or not after:
                raise StatusError(f"PR {number} review-thread pagination is malformed")

    def plan_digest(self, run_id: int) -> str | None:
        payload = _json(
            self.runner(
                ("api", f"repos/{self.repository}/actions/runs/{run_id}/artifacts")
            ),
            f"Actions run {run_id} artifacts",
        )
        artifacts = payload.get("artifacts", []) if isinstance(payload, dict) else []
        artifact = next(
            (
                item
                for item in artifacts
                if item.get("name") == "benchmark-plan"
                and not item.get("expired", False)
            ),
            None,
        )
        if artifact is None:
            return None
        archive = self.runner(
            (
                "api",
                f"repos/{self.repository}/actions/artifacts/{artifact['id']}/zip",
            )
        )
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                members = [
                    name for name in bundle.namelist() if name.endswith("plan.json")
                ]
                if len(members) != 1:
                    raise StatusError("benchmark plan archive has an invalid shape")
                plan = json.loads(bundle.read(members[0]))
        except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise StatusError("benchmark plan artifact is invalid") from exc
        digest = plan.get("planner_digest") if isinstance(plan, dict) else None
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise StatusError("benchmark plan has no valid planner_digest")
        return digest


def _checks(payload: dict[str, Any]) -> tuple[Check, ...]:
    checks: list[Check] = []
    for item in payload.get("statusCheckRollup", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("context")
        state = item.get("conclusion") or item.get("state") or item.get("status")
        if isinstance(name, str) and isinstance(state, str):
            checks.append(
                Check(
                    name,
                    state.upper(),
                    str(item.get("detailsUrl") or item.get("targetUrl") or ""),
                )
            )
    return tuple(checks)


def _tasks(payload: dict[str, Any]) -> tuple[str, ...]:
    identities = {
        f"{match.group(1)}/{match.group(2)}"
        for item in payload.get("files", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (match := _TASK_PATH.match(item["path"])) is not None
    }
    return tuple(sorted(identities))


def _check_state(checks: Sequence[Check], names: Sequence[str]) -> str:
    if not names:
        return "missing"
    selected = [check for check in checks if check.name in names]
    if len(selected) != len(names):
        return "missing"
    if any(
        check.state in {"PENDING", "IN_PROGRESS", "QUEUED", "WAITING", "REQUESTED"}
        for check in selected
    ):
        return "pending"
    return "passed" if all(check.state == "SUCCESS" for check in selected) else "failed"


def build_status(
    reader: GitHubReader, number: int, payload: dict[str, Any]
) -> PullRequestStatus:
    checks = _checks(payload)
    tasks = _tasks(payload)
    unresolved = reader.unresolved_threads(number)
    preparation_checks = _check_state(checks, _PREPARATION_CHECKS)
    preparation = "confirmed" if preparation_checks == "passed" else preparation_checks
    validation = next(
        (check for check in checks if check.name == _VALIDATION_CHECK), None
    )
    plan_digest: str | None = None
    if validation is not None:
        match = _RUN_ID.search(validation.link)
        if match is not None:
            plan_digest = reader.plan_digest(int(match.group(1)))
    ci = _check_state(checks, tuple(check.name for check in checks))

    blockers = _merge_blockers(
        payload,
        unresolved=unresolved,
        preparation=preparation,
        ci=ci,
        plan_digest=plan_digest,
    )

    return PullRequestStatus(
        number=number,
        title=str(payload.get("title", "")),
        url=str(payload.get("url", "")),
        tasks=tasks,
        unresolved_threads=unresolved,
        preparation=preparation,
        plan_digest=plan_digest,
        ci=ci,
        merge_ready=not blockers,
        blockers=blockers,
    )


def _merge_blockers(
    payload: dict[str, Any],
    *,
    unresolved: int,
    preparation: str,
    ci: str,
    plan_digest: str | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if payload.get("state") != "OPEN":
        blockers.append(f"state={payload.get('state', 'UNKNOWN')}")
    if payload.get("isDraft"):
        blockers.append("draft")
    if payload.get("mergeable") != "MERGEABLE":
        blockers.append(f"mergeable={payload.get('mergeable', 'UNKNOWN')}")
    if payload.get("mergeStateStatus") != "CLEAN":
        blockers.append(f"merge-state={payload.get('mergeStateStatus', 'UNKNOWN')}")
    if unresolved:
        blockers.append(f"unresolved-threads={unresolved}")
    if preparation != "confirmed":
        blockers.append(f"preparation={preparation}")
    if ci != "passed":
        blockers.append(f"ci={ci}")
    if plan_digest is None:
        blockers.append("validation-plan=missing")

    return tuple(blockers)


def render_human(statuses: Sequence[PullRequestStatus]) -> str:
    lines: list[str] = []
    for status in statuses:
        readiness = "ready" if status.merge_ready else "blocked"
        lines.append(f"#{status.number} {readiness} — {status.title}")
        lines.append(f"  tasks: {', '.join(status.tasks) or 'none'}")
        lines.append(
            f"  threads={status.unresolved_threads} preparation={status.preparation} "
            f"ci={status.ci} plan={status.plan_digest or 'missing'}"
        )
        if status.blockers:
            lines.append(f"  blockers: {', '.join(status.blockers)}")
    return "\n".join(lines) + "\n"


def _repository(value: str | None, runner: Runner) -> str:
    if value is not None:
        return value
    payload = _json(runner(("repo", "view", "--json", "nameWithOwner")), "repository")
    repository = payload.get("nameWithOwner") if isinstance(payload, dict) else None
    if not isinstance(repository, str):
        raise StatusError("could not resolve repository; pass --repo owner/name")
    return repository


def main(argv: Sequence[str] | None = None, *, runner: Runner = _gh) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pull_requests", nargs="+", type=int, metavar="PR")
    parser.add_argument("--repo", help="owner/name; defaults to the current gh repo")
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    args = parser.parse_args(argv)
    try:
        reader = GitHubReader(_repository(args.repo, runner), runner=runner)
        statuses = [
            build_status(reader, number, reader.pull_request(number))
            for number in args.pull_requests
        ]
    except StatusError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps([asdict(status) for status in statuses], indent=2))
    else:
        sys.stdout.write(render_human(statuses))
    return 0 if all(status.merge_ready for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Check",
    "GitHubReader",
    "PullRequestStatus",
    "StatusError",
    "build_status",
    "main",
    "render_human",
]
