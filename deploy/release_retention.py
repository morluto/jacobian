"""Prune completed inactive releases after a deployment is accepted."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

_RELEASE_NAME = re.compile(r"[0-9a-f]{12}(?:-lean)?")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retain the active release and the newest completed rollbacks."
    )
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--current-link", required=True, type=Path)
    parser.add_argument("--retain", required=True, type=int)
    parser.add_argument("--preserve-release", action="append", default=[], type=Path)
    return parser


def prune_releases(
    release_root: Path,
    current_link: Path,
    *,
    retain: int,
    preserve_releases: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Remove only recognized, completed releases outside the retained set."""

    if retain < 1:
        raise ValueError("retain must be at least one")
    root = release_root.resolve(strict=True)
    active = current_link.resolve(strict=True)
    if active.parent != root:
        raise ValueError("the active release is not directly below the release root")

    completed = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and _RELEASE_NAME.fullmatch(path.name)
            and (path / ".git-revision").is_file()
            and (path / ".release-profile").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if active not in completed:
        raise ValueError("the active release is not a completed recognized release")

    retained = {active}
    preserved = tuple(
        dict.fromkeys(path.resolve(strict=True) for path in preserve_releases)
    )
    if any(path not in completed for path in preserved):
        raise ValueError("a preserved rollback is not a completed recognized release")
    inactive = [path for path in completed if path != active]
    preferred = [path for path in preserved if path != active]
    preferred.extend(path for path in inactive if path not in preferred)
    retained.update(preferred[: retain - 1])
    pruned = tuple(path for path in completed if path not in retained)
    for path in pruned:
        shutil.rmtree(path)
    return pruned


def main() -> int:
    args = _parser().parse_args()
    try:
        pruned = prune_releases(
            args.release_root,
            args.current_link,
            retain=args.retain,
            preserve_releases=tuple(args.preserve_release),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"release retention failed: {exc}") from None
    print(
        json.dumps(
            {"pruned_releases": [path.name for path in pruned]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
