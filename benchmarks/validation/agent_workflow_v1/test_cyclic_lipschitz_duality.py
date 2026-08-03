import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/agent-workflow-v1/cyclic-lipschitz-duality/tests/verifier.py"
)


def module():
    sys.path.insert(0, str(VERIFIER.parent))
    spec = importlib.util.spec_from_file_location("cyclic_dual", VERIFIER)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_independent_dual_cost():
    assert module().minimum_cost() == 15


def test_noncanonical_fraction_rejected():
    assert module().fraction("2/4") is None
