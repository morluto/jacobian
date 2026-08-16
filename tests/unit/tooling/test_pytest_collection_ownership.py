"""Static ownership contracts for pytest's configured default roots."""

from __future__ import annotations

import tomllib
from itertools import combinations
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).parents[3]
SPECIALIST_ROOTS = (
    PurePosixPath("tests/boundary/process"),
    PurePosixPath("tests/boundary/mcp"),
)
ORDINARY_OWNERS = frozenset({"unit", "component", "domain", "composition"})


def _configured_testpaths() -> tuple[PurePosixPath, ...]:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    configured = project["tool"]["pytest"]["ini_options"]["testpaths"]
    assert isinstance(configured, list)
    assert all(isinstance(path, str) for path in configured)
    return tuple(PurePosixPath(path) for path in configured)


def _overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def test_default_testpaths_are_normalized_existing_ordinary_roots() -> None:
    for relative in _configured_testpaths():
        assert not relative.is_absolute()
        assert relative.parts[:1] == ("tests",)
        assert "." not in relative.parts and ".." not in relative.parts
        assert (ROOT / relative).is_dir()

        owner = relative.parts[1:2]
        maintained_provider = (
            relative.parts[:3]
            == (
                "tests",
                "boundary",
                "providers",
            )
            and len(relative.parts) >= 4
        )
        assert owner and (owner[0] in ORDINARY_OWNERS or maintained_provider)


def test_default_testpaths_do_not_overlap_specialist_roots() -> None:
    configured = _configured_testpaths()

    assert all(
        not _overlap(default_root, specialist_root)
        for default_root in configured
        for specialist_root in SPECIALIST_ROOTS
    )


def test_default_testpaths_do_not_overlap_each_other() -> None:
    assert all(
        not _overlap(left, right)
        for left, right in combinations(_configured_testpaths(), 2)
    )
