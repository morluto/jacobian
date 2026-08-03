from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/agent-workflow-v1/closed-one-form-polynomial-classification/tests/verifier.py"
)


def _module():
    sys.path.insert(0, str(VERIFIER.parent))
    spec = importlib.util.spec_from_file_location("closed_form_verifier", VERIFIER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correct_constraint_rank_and_wrong_published_rank() -> None:
    v = _module()
    assert v.rank(v.TARGET) == 2
    wrong = [[0, -1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 1, 0, 0, 0, 0]]
    assert v.rank(v.TARGET + wrong) > 2


def test_derivative_replays_potential() -> None:
    v = _module()
    terms = [
        {"coefficient": "1", "x_power": 2, "y_power": 1},
        {"coefficient": "1", "x_power": 1, "y_power": 2},
    ]
    assert v.derivative(terms, 0) == {(1, 1): Fraction(2), (0, 2): Fraction(1)}
    assert v.derivative(terms, 1) == {(2, 0): Fraction(1), (1, 1): Fraction(2)}


def test_dependent_basis_is_detected() -> None:
    v = _module()
    basis = [[1 if i == j else 0 for i in range(10)] for j in range(7)]
    basis.append(basis[0])
    assert v.rank(basis) == 7
