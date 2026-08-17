from __future__ import annotations

from pathlib import Path

import pytest
from tools import pytest_lifecycle
from tools.process_supervisor import ProcessTreeResult


def _tree_result(exit_code: int, *, timed_out: bool = False) -> ProcessTreeResult:
    return ProcessTreeResult(exit_code=exit_code, timed_out=timed_out)


def test_success_uses_unique_worktree_basetemp_and_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[object, ...]] = []

    def run(
        command: object,
        *,
        timeout: float,
        cwd: Path,
        env: object,
    ) -> ProcessTreeResult:
        del timeout, env
        observed.append((tuple(command), cwd))
        basetemp_argument = next(
            argument
            for argument in command  # type: ignore[union-attr]
            if str(argument).startswith("--basetemp=")
        )
        basetemp = Path(str(basetemp_argument).split("=", 1)[1])
        (basetemp.parent / "session-template").mkdir()
        return _tree_result(0)

    monkeypatch.setattr(pytest_lifecycle, "run_process_tree", run)

    first = pytest_lifecycle.run_pytest(
        ["tests/math/test_one.py"],
        root=tmp_path,
        name="unit",
        environment={},
    )
    second = pytest_lifecycle.run_pytest(
        ["tests/math/test_one.py"],
        root=tmp_path,
        name="unit",
        environment={},
    )

    assert first.basetemp != second.basetemp
    assert first.basetemp.is_relative_to(tmp_path / ".pytest_cache" / "basetemp")
    assert first.basetemp.name == "pytest"
    assert not first.basetemp.exists()
    assert not first.basetemp.parent.exists()
    assert not second.basetemp.exists()
    assert any(str(argument).startswith("--basetemp=") for argument in observed[0][0])


def test_failure_cleanup_is_default_and_retention_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pytest_lifecycle, "run_process_tree", lambda *_args, **_kwargs: _tree_result(1)
    )

    cleaned = pytest_lifecycle.run_pytest(
        ["test_bad.py"], root=tmp_path, name="clean", environment={}
    )
    retained = pytest_lifecycle.run_pytest(
        ["test_bad.py"],
        root=tmp_path,
        name="retain",
        environment={},
        retain_on_failure=True,
    )

    assert not cleaned.basetemp.exists()
    assert retained.retained is True
    assert retained.basetemp.is_dir()


def test_timeout_reports_timed_out_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pytest_lifecycle,
        "run_process_tree",
        lambda *_args, **_kwargs: _tree_result(1, timed_out=True),
    )

    result = pytest_lifecycle.run_pytest(
        ["test_slow.py"], root=tmp_path, name="slow", environment={}
    )

    assert result.status == "TIMED_OUT"
    assert result.exit_code == 1


def test_explicit_basetemp_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owned"):
        pytest_lifecycle.run_pytest(
            ["--basetemp=/tmp/shared"],
            root=tmp_path,
            name="unsafe",
            environment={},
        )
