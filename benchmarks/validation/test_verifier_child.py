from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    _MappedPathFactory,
    run_verifier_in_child,
)


def _write_task(task: Path) -> None:
    tests = task / "tests"
    tests.mkdir(parents=True)
    (tests / "verifier_support.py").write_text(
        "LOAD_COUNT = globals().get('LOAD_COUNT', 0) + 1\n",
        encoding="utf-8",
    )
    (tests / "verifier.py").write_text(
        """import json
from pathlib import Path
from verifier_support import LOAD_COUNT

Path('/logs/verifier/reward.json').write_text(
    json.dumps({'reward': 1.0, 'load_count': LOAD_COUNT})
)
""",
        encoding="utf-8",
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    task = tmp_path / "task"
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    _write_task(task)
    (app / "evidence").mkdir(parents=True)
    logs.mkdir()
    return task, app, logs


def test_verifier_runs_in_fresh_interpreter_without_task_bytecode(
    tmp_path: Path,
) -> None:
    task, app, logs = _workspace(tmp_path)

    first = run_verifier_in_child(task=task, app=app, logs=logs)
    second = run_verifier_in_child(task=task, app=app, logs=logs)

    assert first == {"reward": 1.0, "load_count": 1}
    assert second == first
    assert not list(task.rglob("*.pyc"))
    assert not list(app.rglob("*.pyc"))


def test_virtual_path_mapper_rejects_traversal_and_unsupported_roots(
    tmp_path: Path,
) -> None:
    task, app, logs = _workspace(tmp_path)
    mapper = _MappedPathFactory(
        {"/app": app, "/tests": task / "tests", "/logs/verifier": logs}
    )

    with pytest.raises(ValueError, match="traversal"):
        mapper("/app/../outside")
    with pytest.raises(ValueError, match="unsupported"):
        mapper("/etc/passwd")


def test_symlinked_evidence_is_rejected_before_verifier_execution(
    tmp_path: Path,
) -> None:
    task, app, logs = _workspace(tmp_path)
    marker = app / "verifier-ran"
    (task / "tests" / "verifier.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (app / "evidence" / "answer.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="symlinked workspace entry"):
        run_verifier_in_child(task=task, app=app, logs=logs)

    assert not marker.exists()


def test_reward_must_be_a_regular_json_object(tmp_path: Path) -> None:
    task, app, logs = _workspace(tmp_path)
    (task / "tests" / "verifier.py").write_text(
        "from pathlib import Path\n"
        "Path('/logs/verifier/reward.json').write_text('[]')\n",
        encoding="utf-8",
    )

    with pytest.raises(VerifierExecutionError, match="JSON object"):
        run_verifier_in_child(task=task, app=app, logs=logs)
