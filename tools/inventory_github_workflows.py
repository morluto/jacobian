"""Compare default-branch workflow files with GitHub-registered workflows.

Current Actions identity is the YAML under ``.github/workflows`` on the default
branch. Registrations whose files are gone (including historical
``agent-port-*`` / ``agent-rebase-*`` leftovers) are historical: disable them in
the GitHub UI and retain their run history. This tool never disables workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.command_runner import (
    ToolCommandStatus,
    operator_environment,
    run_operator_command,
)

HISTORICAL_PREFIXES = ("agent-port-", "agent-rebase-")
WORKFLOW_LIST_LIMIT = 1000


def workflow_stems(root: Path) -> set[str]:
    workflow_dir = root / ".github" / "workflows"
    return {path.stem for path in workflow_dir.glob("*.yml") if path.is_file()} | {
        path.stem for path in workflow_dir.glob("*.yaml") if path.is_file()
    }


def classify(
    registered: set[str],
    current: set[str],
) -> dict[str, tuple[str, ...]]:
    historical = tuple(
        sorted(
            name
            for name in registered - current
            if name.startswith(HISTORICAL_PREFIXES)
        )
    )
    missing_files = tuple(sorted(registered - current))
    extra_files = tuple(sorted(current - registered)) if registered else ()
    return {
        "historical_agent_leftovers": historical,
        "registered_without_files": missing_files,
        "files_without_registration": extra_files,
    }


def _workflow_list_command() -> list[str]:
    return [
        "gh",
        "workflow",
        "list",
        "--all",
        "--limit",
        str(WORKFLOW_LIST_LIMIT),
        "--json",
        "name,path,state",
    ]


def _stems_from_workflow_rows(rows: object) -> set[str]:
    if not isinstance(rows, list):
        raise SystemExit("gh workflow list must return a JSON array")
    if len(rows) >= WORKFLOW_LIST_LIMIT:
        raise SystemExit(
            "gh workflow list hit the requested limit; raise --limit so "
            "historical registrations are not dropped"
        )
    stems: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("gh workflow list entries must be objects")
        path = row.get("path")
        if isinstance(path, str) and path:
            stems.add(Path(path).stem)
            continue
        name = row.get("name")
        if isinstance(name, str):
            stems.add(name)
    return stems


def _registered_from_gh() -> set[str]:
    command = _workflow_list_command()
    completed = run_operator_command(
        command[0],
        command[1:],
        cwd=Path.cwd().resolve(),
        timeout_seconds=30.0,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
        environment=operator_environment(
            include=(
                "PATH",
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "GH_CONFIG_DIR",
                "XDG_CONFIG_HOME",
                "HOME",
            )
        ),
    )
    if completed.status is not ToolCommandStatus.EXITED or completed.exit_code != 0:
        diagnostic = completed.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(
            diagnostic or completed.diagnostic or "gh workflow list failed"
        )
    return _stems_from_workflow_rows(
        json.loads(completed.stdout.decode("utf-8", "strict"))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--registered",
        help="JSON array of registered workflow stems; omit to query gh.",
    )
    args = parser.parse_args(argv)
    current = workflow_stems(args.root)
    if args.registered is not None:
        loaded = json.loads(args.registered)
        registered = {str(name) for name in loaded}
    else:
        registered = _registered_from_gh()
    report = classify(registered, current)
    print(json.dumps({"current": sorted(current), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
