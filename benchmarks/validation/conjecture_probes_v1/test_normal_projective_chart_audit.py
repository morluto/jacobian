from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/normal-projective-chart-audit"


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "normal_chart_verifier", TASK / "tests/verifier.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_submission_is_bounded_before_read(monkeypatch):
    module = _module()

    class UnreadablePath:
        def __init__(self, _value):
            pass

        def read_text(self):
            raise AssertionError("oversized submission must not be read")

    def reject_oversized(_path, *, max_bytes):
        assert max_bytes == module.MAX_SUBMISSION_BYTES
        return False

    monkeypatch.setattr(module, "Path", UnreadablePath)
    monkeypatch.setattr(module, "is_regular_bounded_file", reject_oversized)
    assert module._raw() is None


def test_raw_submission_rejects_duplicate_keys(monkeypatch):
    module = _module()

    class DuplicateKeyPath:
        def __init__(self, _value):
            pass

        def read_text(self):
            return '{"result": {}, "result": {}}'

    monkeypatch.setattr(module, "Path", DuplicateKeyPath)
    monkeypatch.setattr(
        module, "is_regular_bounded_file", lambda _path, *, max_bytes: True
    )
    assert module._raw() is None


def _result():
    return {
        "finite_parameters": ["-1", "0", "1"],
        "finite_points": [["0", "-1"], ["0", "1"], ["2", "0"]],
        "missing_projective_parameter": ["1", "0"],
        "missing_point": ["-2", "0"],
        "footpoint_records": [
            {"point": ["-2", "0"], "ellipse_residual": "0", "normal_residual": "0"},
            {"point": ["0", "-1"], "ellipse_residual": "0", "normal_residual": "0"},
            {"point": ["0", "1"], "ellipse_residual": "0", "normal_residual": "0"},
            {"point": ["2", "0"], "ellipse_residual": "0", "normal_residual": "0"},
        ],
    }


def test_oracle_mathematics():
    assert _module().mathematics(_result())


def test_rejects_three_point_only_boundary():
    result = _result()
    result["missing_point"] = ["2", "0"]
    assert not _module().mathematics(result)


def test_rejects_corrupted_normal_residual():
    result = _result()
    result["footpoint_records"][0]["normal_residual"] = "1"
    assert not _module().mathematics(result)


def test_rejects_parameter_order_for_coordinate_sorted_points():
    result = _result()
    result["finite_points"] = [["0", "-1"], ["2", "0"], ["0", "1"]]
    assert not _module().mathematics(result)


def test_rejects_noncanonical_rational():
    result = _result()
    result["finite_parameters"][0] = "-2/2"
    assert not _module().mathematics(result)


def test_rejects_expensive_noncanonical_rational_before_fraction(monkeypatch):
    module = _module()

    def unexpected_fraction(_value):
        raise AssertionError("noncanonical rational must not reach Fraction")

    monkeypatch.setattr(module, "Fraction", unexpected_fraction)
    try:
        module._q("1e100000000")
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical rational was accepted")
