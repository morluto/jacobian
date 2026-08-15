from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from benchmarks.tooling.harbor_digest import (
    HarborDigestError,
    _compose_build_context,
    _compose_context_external_files,
    compose_context_supplement,
)


def _write_task(
    tmp_path: Path,
    *,
    context: str = "../../../../",
    dockerfile_copy: str = "tooling/command_runner.py /opt/command_runner.py",
    dockerignore: str = textwrap.dedent(
        """\
        **
        !tooling/
        !tooling/command_runner.py
        !datasets/
        !datasets/mathematical-benchmarks-v1/
        !datasets/mathematical-benchmarks-v1/*/
        !datasets/mathematical-benchmarks-v1/*/environment/
        !datasets/mathematical-benchmarks-v1/*/environment/**
        """
    ),
    command_runner: bytes = b"# canonical runner\n",
) -> Path:
    """Build a minimal task tree with a widened compose build context."""

    benchmarks = tmp_path / "benchmarks"
    task_dir = benchmarks / "datasets" / "mathematical-benchmarks-v1" / "test-task"
    env = task_dir / "environment"
    env.mkdir(parents=True, exist_ok=True)
    (env / "Dockerfile").write_text(
        f"FROM scratch\nWORKDIR /app\nCOPY {dockerfile_copy}\n",
        encoding="utf-8",
    )
    (env / "input.json").write_text("{}", encoding="utf-8")
    (env / "submission_schema.json").write_text("{}", encoding="utf-8")
    (env / "docker-compose.yaml").write_text(
        textwrap.dedent(
            f"""\
            services:
              main:
                build:
                  context: {context}
                  dockerfile: datasets/mathematical-benchmarks-v1/test-task/environment/Dockerfile
            """
        ),
        encoding="utf-8",
    )
    tooling = benchmarks / "tooling"
    tooling.mkdir(parents=True, exist_ok=True)
    (tooling / "command_runner.py").write_bytes(command_runner)
    (benchmarks / ".dockerignore").write_text(dockerignore, encoding="utf-8")
    return task_dir


def test_compose_build_context_resolves_outside_task_tree(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path)
    context = _compose_build_context(task_dir)
    assert context is not None
    assert context == (tmp_path / "benchmarks").resolve()


def test_compose_build_context_returns_none_without_compose(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path)
    (task_dir / "environment" / "docker-compose.yaml").unlink()
    assert _compose_build_context(task_dir) is None


def test_compose_context_external_files_includes_runner_and_dockerignore(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path)
    context_root = _compose_build_context(task_dir)
    assert context_root is not None
    external = _compose_context_external_files(task_dir, context_root)
    relative = {p.relative_to(context_root) for p in external}
    assert Path(".dockerignore") in relative
    assert Path("tooling/command_runner.py") in relative


def test_compose_context_supplement_binds_runner_content(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, command_runner=b"# version A\n")
    digest_a = compose_context_supplement(task_dir)
    assert digest_a is not None
    assert digest_a.startswith("sha256:")
    _write_task(tmp_path, command_runner=b"# version B\n")
    task_dir2 = (
        tmp_path
        / "benchmarks"
        / "datasets"
        / "mathematical-benchmarks-v1"
        / "test-task"
    )
    digest_b = compose_context_supplement(task_dir2)
    assert digest_b is not None
    assert digest_a != digest_b


def test_compose_context_supplement_binds_dockerignore_content(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path)
    digest_a = compose_context_supplement(task_dir)
    assert digest_a is not None
    benchmarks = tmp_path / "benchmarks"
    (benchmarks / ".dockerignore").write_text(
        "**\n!tooling/\n!tooling/command_runner.py\n",
        encoding="utf-8",
    )
    digest_b = compose_context_supplement(task_dir)
    assert digest_b is not None
    assert digest_a != digest_b


def test_compose_context_supplement_returns_none_for_local_context(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path, context=".")
    assert compose_context_supplement(task_dir) is None


def test_compose_context_supplement_returns_none_without_compose(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path)
    (task_dir / "environment" / "docker-compose.yaml").unlink()
    assert compose_context_supplement(task_dir) is None


def test_compose_context_supplement_detects_missing_dockerignore(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path)
    (tmp_path / "benchmarks" / ".dockerignore").unlink()
    supplement = compose_context_supplement(task_dir)
    assert supplement is not None
    external = _compose_context_external_files(
        task_dir, _compose_build_context(task_dir)
    )
    relative = {p.name for p in external}
    assert ".dockerignore" not in relative


def test_compose_context_supplement_handles_string_build_context(
    tmp_path: Path,
) -> None:
    task_dir = _write_task(tmp_path)
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        textwrap.dedent(
            """\
            services:
              main:
                build: ../../../../
            """
        ),
        encoding="utf-8",
    )
    context = _compose_build_context(task_dir)
    assert context is not None
    assert context == (tmp_path / "benchmarks").resolve()


def test_compose_build_context_rejects_repository_escape(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path, context="../../../../../..")
    with pytest.raises(HarborDigestError, match="escapes the repository"):
        _compose_build_context(task_dir)


def test_compose_copy_rejects_context_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    task_dir = _write_task(tmp_path, dockerfile_copy="../outside.py /opt/outside.py")
    context_root = _compose_build_context(task_dir)
    assert context_root is not None
    with pytest.raises(HarborDigestError, match="COPY source escapes"):
        _compose_context_external_files(task_dir, context_root)
