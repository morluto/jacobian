import importlib.util
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/continuous-spike-integral-separation"
)
TASK_NAME = "continuous-spike-integral-separation"


def load_verifier():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "continuous_spike_verifier", TASK / "tests/verifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        sys.modules.pop("verifier_support", None)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def candidate(alpha=Fraction(1, 4)):
    verifier = load_verifier()
    return {
        "alpha": verifier.encoded(alpha),
        "spikes": [verifier.expected_spike(n, alpha) for n in range(1, 13)],
    }


def test_accepts_alternative_width_scales():
    verifier = load_verifier()
    assert verifier.valid_result(candidate(Fraction(1, 4)))
    assert verifier.valid_result(candidate(Fraction(1, 7)))


def test_rejects_width_touching_integer_and_corrupt_area():
    verifier = load_verifier()
    assert not verifier.valid_result(candidate(Fraction(1, 2)))
    bad = candidate()
    bad["spikes"][6]["area"] = "1"
    assert not verifier.valid_result(bad)


def test_rejects_finite_or_reversed_classification():
    verifier = load_verifier()
    bad = candidate()
    bad["spikes"] = bad["spikes"][:-1]
    assert not verifier.valid_result(bad)


def test_rejects_boolean_spike_index():
    verifier = load_verifier()
    bad = candidate()
    bad["spikes"][0]["n"] = True
    assert not verifier.valid_result(bad)


def test_accepts_permuted_spike_order():
    verifier = load_verifier()
    good = candidate()
    good["spikes"] = list(reversed(good["spikes"]))
    assert verifier.valid_result(good)


def test_accepts_unreduced_rational_and_rejects_string_coercion():
    verifier = load_verifier()
    unreduced = candidate()
    unreduced["alpha"] = {"numerator": 2, "denominator": 8}
    assert verifier.valid_result(unreduced)
    bad = candidate()
    bad["alpha"] = "1/4"
    assert not verifier.valid_result(bad)


def _digest(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / TASK_NAME / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    _write_json(app / "submission.json", submission)
    return TASK, app, logs


def _rewrite(app: Path, submission: dict) -> None:
    _write_json(app / "submission.json", submission)


def test_canonical_computed_submission_passes(tmp_path: Path):
    task, app, logs = _case(tmp_path)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0
    assert result.reward == 1.0
