"""Verify that the Jacobian wheel contains exactly the installed source tree."""

from __future__ import annotations

import argparse
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
_RETIRED_PREFIXES = (
    PurePosixPath("jacobian/adapters"),
    PurePosixPath("jacobian/contracts"),
    PurePosixPath("jacobian/domains"),
    PurePosixPath("jacobian/eval"),
)
_RETIRED_MODULES = frozenset(
    {
        "jacobian/_deployment_smoke.py",
        "jacobian/bounded_process.py",
        "jacobian/builtin_operation_modules.py",
        "jacobian/math_tools.py",
        "jacobian/operation_adapters.py",
        "jacobian/operation_discovery.py",
        "jacobian/operation_dispatcher.py",
        "jacobian/serving_catalog.py",
        "jacobian/validation_diagnostics.py",
        "jacobian/worker_environment.py",
    }
)


def _expected_modules(root: Path) -> set[str]:
    source_root = root / "src"
    return {
        path.relative_to(source_root).as_posix()
        for path in (source_root / "jacobian").rglob("*.py")
    }


def wheel_content_failures(root: Path, wheel: Path) -> tuple[str, ...]:
    """Return deterministic source/wheel drift diagnostics."""

    with zipfile.ZipFile(wheel) as archive:
        members = {name for name in archive.namelist() if name.endswith(".py")}
    installed = {name for name in members if name.startswith("jacobian/")}
    expected = _expected_modules(root)
    failures: list[str] = []
    for missing in sorted(expected - installed):
        failures.append(f"wheel is missing installed module: {missing}")
    for unexpected in sorted(installed - expected):
        failures.append(f"wheel contains unexpected module: {unexpected}")
    for member in sorted(members):
        path = PurePosixPath(member)
        if member.startswith(("benchmarks/", "deploy/")):
            failures.append(f"wheel contains repository-only tooling: {member}")
        if member in _RETIRED_MODULES or any(
            path == prefix or prefix in path.parents for prefix in _RETIRED_PREFIXES
        ):
            failures.append(f"wheel contains retired runtime path: {member}")
    return tuple(failures)


def _current_wheel(root: Path) -> Path:
    with (root / "pyproject.toml").open("rb") as project_file:
        version = tomllib.load(project_file)["project"]["version"]
    matches = sorted((root / "dist").glob(f"jacobian-{version}-*.whl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one dist/jacobian-{version}-*.whl, found {len(matches)}"
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    wheel = args.wheel.resolve() if args.wheel else _current_wheel(root)
    failures = wheel_content_failures(root, wheel)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"wheel contents: OK ({wheel.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
