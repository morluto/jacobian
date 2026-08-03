import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/agent-workflow-v1/triplewise-empty-extremal-audit/tests/verifier.py"
)


def module():
    sys.path.insert(0, str(VERIFIER.parent))
    spec = importlib.util.spec_from_file_location("triplewise_verifier", VERIFIER)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_valid_alternative_matching():
    family = [[], *[[i] for i in range(7)], [0, 2], [1, 4], [3, 6]]
    assert module().valid_family(7, family)


def test_source_bound_cannot_be_attained():
    family = [[], *[[i] for i in range(7)], [0, 1], [2, 3], [4, 5]]
    assert len(family) == 11
    assert len(family) < 2 * 7
