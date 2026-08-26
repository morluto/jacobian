from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.validation._verifier_child import (
    VerifierExecutionError,
    _MappedPathFactory,
    _read_verifier_output,
    _write_failure_output,
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
    json.dumps({'reward': 1.0})
)
Path('/logs/verifier/reward-details.json').write_text(
    json.dumps({'load_count': LOAD_COUNT})
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

    assert first.reward == 1.0
    assert first.details == {"load_count": 1}
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


def _write_output_records(logs: Path, reward: str, details: str | None) -> None:
    (logs / "reward.json").write_text(reward, encoding="utf-8")
    if details is not None:
        (logs / "reward-details.json").write_text(details, encoding="utf-8")


def test_verifier_output_schema_normalizes_a_valid_reward_record(
    tmp_path: Path,
) -> None:
    _task, _app, logs = _workspace(tmp_path)
    _write_output_records(logs, '{"reward": 1}', '{"correctness": 1.0}')

    output = _read_verifier_output(logs)

    assert output.reward == 1.0
    assert output.details == {"correctness": 1.0}


def test_verifier_output_schema_rejects_a_nonobject_reward_without_a_child(
    tmp_path: Path,
) -> None:
    _task, _app, logs = _workspace(tmp_path)
    _write_output_records(logs, "[]", "{}")

    with pytest.raises(VerifierExecutionError, match="JSON object"):
        _read_verifier_output(logs)


def test_child_execution_failure_remains_a_process_boundary(tmp_path: Path) -> None:
    task, app, logs = _workspace(tmp_path)
    (task / "tests" / "verifier.py").write_text(
        "raise RuntimeError('intentional verifier failure')\n", encoding="utf-8"
    )

    with pytest.raises(VerifierExecutionError, match="verifier child"):
        run_verifier_in_child(task=task, app=app, logs=logs)

    assert _read_verifier_output(logs).reward == 0.0


def test_failure_output_is_a_directly_checked_protocol_record(tmp_path: Path) -> None:
    _task, _app, logs = _workspace(tmp_path)

    _write_failure_output(logs)

    assert _read_verifier_output(logs).reward == 0.0


@pytest.mark.parametrize(
    ("reward", "details", "message"),
    [
        ('{"reward": 1.0, "correctness": 1.0}', "{}", "exactly reward"),
        ('{"reward": true}', "{}", "finite number"),
        ('{"reward": NaN}', "{}", "non-finite JSON"),
        ('{"reward": 1.0, "reward": 0.0}', "{}", "duplicate JSON object key"),
        ('{"reward": 1.0}', None, "reward-details.json"),
        ('{"reward": 1.0}', '{"reward": 1.0}', "must not contain reward"),
    ],
)
def test_verifier_output_schema_rejects_noncanonical_records_without_a_child(
    tmp_path: Path, reward: str, details: str | None, message: str
) -> None:
    _task, _app, logs = _workspace(tmp_path)
    _write_output_records(logs, reward, details)

    with pytest.raises(VerifierExecutionError, match=message):
        _read_verifier_output(logs)
