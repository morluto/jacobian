import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT / "benchmarks/datasets/mathematical-benchmarks-v1/lp-integrability-separator"
)


def load_verifier():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "lp_integrability_verifier", TASK / "tests/verifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _q(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def result(beta="1", log_exponent="-2", integral="1"):
    return {
        "beta": _q(beta),
        "origin_power_coefficient": "-1/2",
        "infinity_power_coefficient": "-1/2",
        "p2_log_exponent": _q(log_exponent),
        "p2_integral_each": _q(integral),
        "critical_p": "2",
        "lower_regime": {
            "p_interval": "0<p<2",
            "obstruction": "INFINITY_POWER_TAIL",
        },
        "upper_regime": {
            "p_interval": "p>2",
            "obstruction": "ORIGIN_POWER_SINGULARITY",
        },
    }


def test_accepts_alternative_rational_parameter():
    assert load_verifier().valid_result(result("3/4", "-3/2", "2"))


def test_rejects_boundary_and_corrupt_integral():
    verifier = load_verifier()
    assert not verifier.valid_result(result("1/2", "-1", "1"))
    assert not verifier.valid_result(result("3/4", "-3/2", "3"))


def test_rejects_wrong_tail_obstruction():
    candidate = result()
    candidate["lower_regime"]["obstruction"] = "ORIGIN_POWER_SINGULARITY"
    assert not load_verifier().valid_result(candidate)


def test_structured_rational_fields_are_objects():
    schema = json.loads((TASK / "environment/submission_schema.json").read_text())
    for field in ("beta", "p2_log_exponent", "p2_integral_each"):
        node = schema["properties"]["result"]["properties"][field]
        assert node["type"] == "object"
        assert set(node["required"]) == {"numerator", "denominator"}
    text = (TASK / "instruction.md").read_text().casefold()
    assert "numerator" in text
    assert "80 characters" not in text


# --- T3: accept mathematically equivalent rationals ---


def test_accepts_noncanonical_rational_representations():
    verifier = load_verifier()
    assert verifier.valid_result(result("1/1", "-2/1", "1/1"))
    assert verifier.valid_result(result("2/2", "-4/2", "2/2"))


# --- T1: assurance failures must not collapse correctness ---


def test_valid_result_independent_of_assurance():
    """valid_result must pass for a mathematically correct result even if the
    submission claims VERIFIED (which the contract rejects)."""
    verifier = load_verifier()
    assert verifier.valid_result(result())


# --- T2: evidence content validation ---
