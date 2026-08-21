"""Independently elaborate one bounded declaration application with Lean."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")
_DECLARATION = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LEAN_CANDIDATES = (
    "/opt/lean/lean-4.31.0-linux/bin/lean",
    "/usr/local/bin/lean",
    "/usr/bin/lean",
)


def _load_frozen_input() -> dict[str, object]:
    try:
        frozen_path = E / "input.json"
        if frozen_path.is_symlink():
            return {}
        payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _lean_candidates() -> list[str]:
    candidates: list[str] = list(_LEAN_CANDIDATES)
    for elan_candidate in (
        str(Path.home() / ".elan" / "bin" / "lean"),
        "/home/runner/.elan/bin/lean",
        "/root/.elan/bin/lean",
        str(Path("~/.elan/bin/lean").expanduser()),
    ):
        if elan_candidate not in candidates:
            candidates.append(elan_candidate)
    which_lean = shutil.which("lean")
    if which_lean:
        candidates.append(which_lean)
    for elan_bin in ("/home/runner/.elan/bin/elan", "/root/.elan/bin/elan"):
        if Path(elan_bin).is_file() and os.access(elan_bin, os.X_OK):
            try:
                resolved = subprocess.run(
                    [elan_bin, "which", "lean"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if resolved.returncode == 0:
                    lean_path = resolved.stdout.strip().splitlines()[0].strip()
                    if lean_path and lean_path not in candidates:
                        candidates.append(lean_path)
            except (OSError, subprocess.TimeoutExpired):
                pass
    return candidates


def _elaborates(result: object, frozen: dict[str, object]) -> bool:
    if not isinstance(result, dict) or set(result) != {"theorem", "arguments"}:
        return False
    theorem = result.get("theorem")
    arguments = result.get("arguments")
    source_prefix = frozen.get("source_prefix")
    available = frozen.get("available_arguments")
    if (
        not isinstance(theorem, str)
        or len(theorem) > 128
        or _DECLARATION.fullmatch(theorem) is None
        or not isinstance(arguments, list)
        or len(arguments) > 4
        or not isinstance(source_prefix, str)
        or len(source_prefix) > 2000
        or not isinstance(available, list)
        or any(not isinstance(item, str) for item in available)
    ):
        return False
    if any(
        not isinstance(item, str)
        or len(item) > 64
        or _IDENTIFIER.fullmatch(item) is None
        or item not in available
        for item in arguments
    ):
        return False
    application = " ".join((theorem, *arguments))
    source = f"{source_prefix}\n  exact {application}\n"
    source_path = Path("/logs/verifier/Submission.lean")
    candidates = _lean_candidates()
    lean = next(
        (
            candidate
            for candidate in candidates
            if os.path.isfile(candidate)  # noqa: PTH113 -- Path is virtualized.
            and os.access(candidate, os.X_OK)
        ),
        None,
    )
    if lean is None:
        return False
    try:
        source_path.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [
                lean,
                "-T",
                "1000000",
                "-M",
                "512",
                "-j",
                "1",
                "--trust=0",
                str(source_path),
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def main() -> None:
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    frozen = _load_frozen_input()
    correctness = bool(
        submission is not None and _elaborates(submission.get("result"), frozen)
    )
    reward = float(input_binding and correctness)
    reward_path = Path("/logs/verifier/reward.json")
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(
        json.dumps(
            {
                "correctness": float(correctness),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        ),
        encoding="utf-8",
    )
    normalize_reward_file(reward_path)


if __name__ == "__main__":
    main()
