import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT / "benchmarks/datasets/mathematical-benchmarks-v1/even-fixed-point-inclusion"
)


def load_verifier():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "fixed_point_verifier", TASK / "tests/verifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_two_derivations_agree():
    result = load_verifier().derive()
    assert result["signed_inclusion_terms"] == [40320, -20160, 4320, -480, 24]
    assert result["inclusion_sum"] == result["exact_even_fixed_histogram"][0] == 24024
    assert sum(result["exact_even_fixed_histogram"]) == 40320


def test_corrupt_term_or_histogram_is_rejected():
    verifier = load_verifier()
    result = verifier.derive()
    result["signed_inclusion_terms"][2] += 1
    assert not verifier.matches(result)
    result = verifier.derive()
    result["exact_even_fixed_histogram"][1] += 1
    assert not verifier.matches(result)
