"""Compare default-branch workflow files with GitHub-registered workflows.

Current Actions identity is the YAML under ``.github/workflows`` on the default
branch. Registrations whose files are gone (including historical
``agent-port-*`` / ``agent-rebase-*`` leftovers) are historical: disable them in
the GitHub UI and retain their run history. This tool never disables workflows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HISTORICAL_PREFIXES = ("agent-port-", "agent-rebase-")


def workflow_stems(root: Path) -> set[str]:
    workflow_dir = root / ".github" / "workflows"
    return {
        path.stem
        for path in workflow_dir.glob("*.yml")
        if path.is_file()
    } | {
        path.stem
        for path in workflow_dir.glob("*.yaml")
        if path.is_file()
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


def _registered_from_gh() -> set[str]:
    completed = subprocess.run(
        ["gh", "workflow", "list", "--all", "--json", "name,path,state"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or "gh workflow list failed")
    rows = json.loads(completed.stdout)
    stems: set[str] = set()
    for row in rows:
        path = row.get("path")
        if isinstance(path, str) and path:
            stems.add(Path(path).stem)
            continue
        name = row.get("name")
        if isinstance(name, str):
            stems.add(name)
    return stems


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
