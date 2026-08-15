"""Fail-closed tests for Harbor Oracle result validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.tooling.validate_harbor_results import (
    _AUGMENTED_DIGEST_MANIFEST,
    _augmented_digest_manifest_path,
    _augmented_job_identity,
    _augmented_job_name,
    _validate_augmented_digest_manifest,
    _validate_payload,
)

_DIGEST = "a" * 64
_DURABLE_DIGEST = f"sha256:{_DIGEST}"
_JOB_CONFIG_DIGEST = f"sha256:{'c' * 64}"
_HARBOR_VERSION = "0.20.0"
_ATTEMPT_ID = "1" * 16
_DOCKER_BUILD_MODE = "buildkit"
_DOCKER_SERVER_VERSION = "27.5.1"
_DOCKER_COMPOSE_VERSION = "2.32.4"


def _identity(dataset: str, digests: dict[str, str]) -> str:
    return _augmented_job_identity(
        dataset=dataset,
        digests=digests,
        job_config_digest=_JOB_CONFIG_DIGEST,
        harbor_version=_HARBOR_VERSION,
        execution_args=(),
        docker_build_mode=_DOCKER_BUILD_MODE,
        docker_server_version=_DOCKER_SERVER_VERSION,
        docker_compose_version=_DOCKER_COMPOSE_VERSION,
    )


def _manifest(
    *,
    dataset: str,
    job_identity: str,
    job_name: str,
    tasks: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": "2",
        "dataset": dataset,
        "job_identity": job_identity,
        "job_name": job_name,
        "attempt_id": _ATTEMPT_ID,
        "prepared_at_ns": 0,
        "harbor_version": _HARBOR_VERSION,
        "job_config_digest": _JOB_CONFIG_DIGEST,
        "execution_args": [],
        "docker_build_mode": _DOCKER_BUILD_MODE,
        "docker_server_version": _DOCKER_SERVER_VERSION,
        "docker_compose_version": _DOCKER_COMPOSE_VERSION,
        "tasks": tasks,
    }


def _trial(*, reward: object = 1.0, **trial_overrides: object) -> dict:
    trial = {
        "task_name": "jacobian/example-task",
        # Harbor still emits this legacy field, but validation must bind to the
        # durable digest in TrialLock instead.
        "task_checksum": "sha256:legacy-dirhash",
        "verifier_result": {
            "rewards": {
                "correctness": reward,
                "witness_validity": 1.0,
                "scope_accuracy": 1.0,
                "assurance_calibration": 1.0,
                "reward": 0.9999999999999999,
                "false_certification": False,
            }
        },
        "exception_info": None,
    }
    trial.update(trial_overrides)
    return trial


def _payload() -> dict:
    return {
        "id": "job",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "n_total_trials": 1,
        "stats": {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
        },
    }


def test_complete_result_binds_task_and_digest() -> None:
    assert (
        _validate_payload(
            _payload(),
            trial_results=[_trial()],
            trial_digests=[_DURABLE_DIGEST],
            expected_tasks={"example-task"},
            expected_digests={"example-task": _DURABLE_DIGEST},
        )
        == []
    )


def test_missing_reward_is_not_a_certifying_result() -> None:
    failures = _validate_payload(
        _payload(),
        trial_results=[_trial(verifier_result={"rewards": {}})],
        trial_digests=[_DURABLE_DIGEST],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("incomplete verifier reward" in failure for failure in failures)


def test_exception_and_digest_mismatch_fail_closed() -> None:
    failures = _validate_payload(
        _payload(),
        trial_results=[
            _trial(
                exception_info={"exception_type": "TimeoutError"},
            )
        ],
        trial_digests=[f"sha256:{'b' * 64}"],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("digest mismatch" in failure for failure in failures)
    assert any("exception result" in failure for failure in failures)


def test_non_certifying_reward_fails_closed() -> None:
    failures = _validate_payload(
        _payload(),
        trial_results=[_trial(reward=0.5)],
        trial_digests=[_DURABLE_DIGEST],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("correctness must be full reward" in failure for failure in failures)


@pytest.mark.parametrize("signal", [True, 1.0])
def test_false_certification_signal_fails_closed(signal: bool | float) -> None:
    trial = _trial()
    trial["verifier_result"]["rewards"]["false_certification"] = signal

    failures = _validate_payload(
        _payload(),
        trial_results=[trial],
        trial_digests=[_DURABLE_DIGEST],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("false_certification must be zero" in failure for failure in failures)


def test_duplicate_task_trials_fail_closed() -> None:
    payload = _payload()
    payload["n_total_trials"] = 2
    payload["stats"]["n_completed_trials"] = 2
    failures = _validate_payload(
        payload,
        trial_results=[_trial(), _trial()],
        trial_digests=[_DURABLE_DIGEST, _DURABLE_DIGEST],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("exactly one trial" in failure for failure in failures)


def test_task_defined_reward_dimensions_are_validated() -> None:
    trial = _trial()
    trial["verifier_result"]["rewards"]["measurements_valid"] = True
    trial["verifier_result"]["rewards"]["task_specific"] = 0.5

    failures = _validate_payload(
        _payload(),
        trial_results=[trial],
        trial_digests=[_DURABLE_DIGEST],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("task_specific must be full reward" in failure for failure in failures)


def test_legacy_task_checksum_does_not_replace_durable_lock_digest() -> None:
    assert (
        _validate_payload(
            _payload(),
            trial_results=[_trial(task_checksum="sha256:wrong")],
            trial_digests=[_DURABLE_DIGEST],
            expected_tasks={"example-task"},
            expected_digests={"example-task": _DIGEST},
        )
        == []
    )


def test_missing_durable_lock_digest_fails_closed() -> None:
    failures = _validate_payload(
        _payload(),
        trial_results=[_trial()],
        trial_digests=[None],
        expected_tasks={"example-task"},
        expected_digests={"example-task": _DURABLE_DIGEST},
    )

    assert any("missing durable task digest" in failure for failure in failures)


def test_changed_augmented_task_digest_rejects_old_trial() -> None:
    old_augmented = f"sha256:{'a' * 64}"
    current_augmented = f"sha256:{'b' * 64}"

    failures = _validate_payload(
        _payload(),
        trial_results=[_trial()],
        trial_digests=[old_augmented],
        expected_tasks={"example-task"},
        expected_digests={"example-task": current_augmented},
    )

    assert any("task digest mismatch" in failure for failure in failures)


def test_augmented_manifest_rejects_context_change(tmp_path) -> None:
    dataset = "mathematical-benchmarks-v1"
    current_digest = f"sha256:{'b' * 64}"
    current_digests = {"example-task": current_digest}
    job_identity = _identity(dataset, current_digests)
    job_name = _augmented_job_name(job_identity=job_identity, attempt_id=_ATTEMPT_ID)
    result = tmp_path / job_name / "result.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    assert not _AUGMENTED_DIGEST_MANIFEST.startswith(".")
    manifest = _augmented_digest_manifest_path(tmp_path, job_name)
    manifest.write_text(
        json.dumps(
            _manifest(
                dataset=dataset,
                job_identity=job_identity,
                job_name=job_name,
                tasks=[{"task": "example-task", "digest": f"sha256:{'a' * 64}"}],
            )
        ),
        encoding="utf-8",
    )

    failures = _validate_augmented_digest_manifest(
        manifest_path=manifest,
        result_path=result,
        dataset=dataset,
        expected_digests=current_digests,
        expected_job_identity=job_identity,
        expected_harbor_version=_HARBOR_VERSION,
        expected_job_config_digest=_JOB_CONFIG_DIGEST,
        expected_execution_args=(),
        expected_docker_build_mode=_DOCKER_BUILD_MODE,
        expected_docker_server_version=_DOCKER_SERVER_VERSION,
        expected_docker_compose_version=_DOCKER_COMPOSE_VERSION,
    )

    assert failures == ["augmented task digest mismatch for example-task"]


def test_changed_augmented_digest_cannot_resume_stale_job(tmp_path) -> None:
    dataset = "mathematical-benchmarks-v1"
    old_digest = f"sha256:{'a' * 64}"
    current_digest = f"sha256:{'b' * 64}"
    current_digests = {"example-task": current_digest}
    current_identity = _identity(dataset, current_digests)
    current_job_name = _augmented_job_name(
        job_identity=current_identity, attempt_id=_ATTEMPT_ID
    )
    stale_identity = _identity(dataset, {"example-task": old_digest})
    stale_job_name = _augmented_job_name(
        job_identity=stale_identity, attempt_id=_ATTEMPT_ID
    )
    result = tmp_path / stale_job_name / "result.json"
    result.parent.mkdir()
    result.write_text("{}", encoding="utf-8")
    manifest = _augmented_digest_manifest_path(tmp_path, current_job_name)
    manifest.write_text(
        json.dumps(
            _manifest(
                dataset=dataset,
                job_identity=current_identity,
                job_name=current_job_name,
                tasks=[{"task": "example-task", "digest": current_digest}],
            )
        ),
        encoding="utf-8",
    )

    failures = _validate_augmented_digest_manifest(
        manifest_path=manifest,
        result_path=result,
        dataset=dataset,
        expected_digests=current_digests,
        expected_job_identity=current_identity,
        expected_harbor_version=_HARBOR_VERSION,
        expected_job_config_digest=_JOB_CONFIG_DIGEST,
        expected_execution_args=(),
        expected_docker_build_mode=_DOCKER_BUILD_MODE,
        expected_docker_server_version=_DOCKER_SERVER_VERSION,
        expected_docker_compose_version=_DOCKER_COMPOSE_VERSION,
    )

    assert failures == ["Harbor result is not bound to its augmented task attempt"]


def test_same_experiment_identity_uses_distinct_attempt_names() -> None:
    identity = _identity(
        "mathematical-benchmarks-v1", {"example-task": f"sha256:{'a' * 64}"}
    )

    first = _augmented_job_name(job_identity=identity, attempt_id="1" * 16)
    second = _augmented_job_name(job_identity=identity, attempt_id="2" * 16)

    assert first != second
    assert first.startswith(f"jacobian-oracle-{identity.removeprefix('sha256:')[:32]}-")
    jobs_dir = Path("jobs")
    assert _augmented_digest_manifest_path(
        jobs_dir, first
    ) != _augmented_digest_manifest_path(jobs_dir, second)


def test_execution_configuration_changes_experiment_identity() -> None:
    digests = {"example-task": f"sha256:{'a' * 64}"}
    baseline = _identity("mathematical-benchmarks-v1", digests)

    changed = _augmented_job_identity(
        dataset="mathematical-benchmarks-v1",
        digests=digests,
        job_config_digest=_JOB_CONFIG_DIGEST,
        harbor_version=_HARBOR_VERSION,
        execution_args=("--n-concurrent", "2"),
        docker_build_mode=_DOCKER_BUILD_MODE,
        docker_server_version=_DOCKER_SERVER_VERSION,
        docker_compose_version=_DOCKER_COMPOSE_VERSION,
    )

    assert changed != baseline
