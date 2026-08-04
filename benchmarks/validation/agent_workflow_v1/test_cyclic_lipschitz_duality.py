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


def test_fraction_rejects_nonfinite_and_unbounded_values():
    loaded = module()
    assert loaded.fraction(float("inf")) is None
    assert loaded.fraction("1e1000000000") is None
    assert loaded.fraction("1" * 129) is None


def test_scope_accepts_equivalent_marked_index_ordering():
    loaded = module()
    instance = loaded.load_instance()
    assert loaded.scope_is_correct(
        "A cyclic sequence on the frozen 60-cycle has marked positions "
        "60, 48, 36, 24, and 12.",
        instance,
    )


def test_scope_rejects_wrong_marked_indices():
    loaded = module()
    instance = loaded.load_instance()
    assert not loaded.scope_is_correct(
        "A cyclic sequence on the frozen 60-cycle has marked positions "
        "12, 24, 36, 48, and 59.",
        instance,
    )
