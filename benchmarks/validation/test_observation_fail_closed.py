"""Fail-closed regressions for observation-results normalization."""

from collections import Counter


def test_trial_status_missing_status_fails_closed() -> None:
    """A trial with no ``status`` field must not default to COMPLETED."""
    from benchmarks.tooling.observation_results import _trial_status

    assert _trial_status({}, None) == "ERROR"
    assert _trial_status({"status": None}, None) == "ERROR"
    assert _trial_status({"status": 42}, None) == "ERROR"
    assert _trial_status({"status": "COMPLETED"}, None) == "COMPLETED"
    assert _trial_status({"status": "RUNNING"}, None) == "RUNNING"
    assert _trial_status({"status": "FAILED"}, None) == "FAILED"
    assert _trial_status({}, RuntimeError("boom")) == "ERROR"
    assert _trial_status({"status": "COMPLETED"}, RuntimeError("boom")) == "ERROR"
    assert _trial_status({"verifier_result": {"status": 1}}, None) == "ERROR"
    assert _trial_status({"verifier_result": {"status": "COMPLETED"}}, None) == "ERROR"
    assert (
        _trial_status(
            {"verifier_result": {"rewards": {"correctness": 1.0}}},
            None,
            job_stats={
                "n_total_trials": 1,
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
            observed_trial_count=1,
        )
        == "COMPLETED"
    )
    assert (
        _trial_status(
            {"verifier_result": {"status": "COMPLETED"}},
            None,
            job_stats={
                "n_total_trials": 1,
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
            observed_trial_count=1,
        )
        == "COMPLETED"
    )
    assert (
        _trial_status(
            {"verifier_result": {"rewards": {"correctness": 1.0}}},
            None,
            job_stats={"n_total_trials": 1},
            observed_trial_count=1,
        )
        == "ERROR"
    )


def test_trial_status_non_string_status_fails_closed() -> None:
    """Non-string status values must be treated as ERROR, not COMPLETED."""
    from benchmarks.tooling.observation_results import _trial_status

    for bad in (0, 1, True, False, [], {}):
        assert _trial_status({"status": bad}, None) == "ERROR", (
            f"{bad!r} should be ERROR"
        )


def test_observation_failures_require_authoritative_completion_counts() -> None:
    from benchmarks.tooling.observation_results import _observation_failures

    failures = _observation_failures(
        counters=Counter({"case": 1}),
        expected_tasks={"case"},
        attempts=1,
        expected_digests={},
        trials=[
            {
                "task": "case",
                "repetition": 0,
                "task_digest": None,
                "status": "COMPLETED",
            }
        ],
        payload={
            "n_total_trials": 1,
            "stats": {
                "n_errored_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
        },
    )

    assert any("completion counts" in failure for failure in failures)
