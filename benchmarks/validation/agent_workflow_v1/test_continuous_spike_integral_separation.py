import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT / "benchmarks/datasets/agent-workflow-v1/continuous-spike-integral-separation"
)


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "continuous_spike_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def candidate(alpha=Fraction(1, 4)):
    verifier = load_verifier()
    return {
        "alpha": str(alpha),
        "baseline_power": 2,
        "spike_height": "1",
        "spikes": [verifier.expected_spike(n, alpha) for n in range(1, 13)],
        "integral_classification": {
            "spike_area_series": "alpha*sum(1/n)",
            "status": "DIVERGENT",
        },
        "sample_series_classification": {
            "sample_series": "sum(1/n^2)",
            "status": "CONVERGENT",
        },
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
    bad = candidate()
    bad["sample_series_classification"]["status"] = "DIVERGENT"
    assert not verifier.valid_result(bad)


def test_contract_has_computed_ceiling():
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    assert schema["properties"]["claimed_assurance"] == {"const": "COMPUTED"}
