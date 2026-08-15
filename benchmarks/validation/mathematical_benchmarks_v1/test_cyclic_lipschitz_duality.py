import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/cyclic-lipschitz-duality/tests/verifier.py"
)


def module():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(VERIFIER.parent))
        spec = importlib.util.spec_from_file_location("cyclic_dual", VERIFIER)
        assert spec and spec.loader
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_independent_dual_cost():
    assert module().minimum_cost() == 15


def test_noncanonical_fraction_rejected():
    assert module().fraction({"numerator": 2, "denominator": 4}) is None


def test_fraction_rejects_nonfinite_and_unbounded_values():
    loaded = module()
    assert loaded.fraction(float("inf")) is None
    assert loaded.fraction("1e1000000000") is None
    assert loaded.fraction({"numerator": True, "denominator": 1}) is None
    assert loaded.fraction({"numerator": 1, "denominator": 0}) is None
    assert loaded.fraction({"numerator": 1 << 1_024, "denominator": 1}) is None
