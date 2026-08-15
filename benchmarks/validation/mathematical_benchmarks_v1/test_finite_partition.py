import json

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

from ._fixtures import assert_result_witness_protocol


def test_result_witness_protocol(tmp_path):
    assert_result_witness_protocol(tmp_path, "finite-partition")


def test_rejects_string_coerced_members(tmp_path):
    task, app, logs = _fixtures._prepare_case(
        tmp_path, "finite-partition", "string-member"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["cases"][0]["members"][0] = "0"
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 0.0
