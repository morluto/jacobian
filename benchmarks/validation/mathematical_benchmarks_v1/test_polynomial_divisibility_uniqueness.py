import json
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child
from jsonschema import Draft202012Validator

from ._fixtures import _prepare_case, assert_result_witness_protocol


def test_result_protocol(tmp_path):
    assert_result_witness_protocol(tmp_path, "polynomial-divisibility-uniqueness")


def test_polynomial_size_envelope_accepts_padding_then_refuses_overflow(
    tmp_path: Path,
) -> None:
    for size, accepted in [(14, True), (15, False)]:
        task, app, logs = _prepare_case(
            tmp_path / str(size), "polynomial-divisibility-uniqueness", "padding"
        )
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        for key in ["common_gcd", "quotient", "remainder_constant", "remainder_x"]:
            coefficients = submission["result"][key]
            coefficients.extend([0] * (size - len(coefficients)))
        schema = json.loads((task / "environment/submission_schema.json").read_text())
        assert Draft202012Validator(schema).is_valid(submission) is accepted
        path.write_text(json.dumps(submission))
        assert run_verifier_in_child(task=task, app=app, logs=logs).reward == float(
            accepted
        )
