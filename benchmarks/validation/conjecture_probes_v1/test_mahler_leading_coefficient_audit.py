from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/mahler-leading-coefficient-audit"
)


def _module():
    sys.path.insert(0, str(TASK / "tests"))
    spec = importlib.util.spec_from_file_location(
        "mahler_verifier", TASK / "tests/verifier.py"
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


def _result():
    return {
        "factors": [[1, -3, 1], [1, -1, 1], [1, 1, 1], [2, -5, 2]],
        "outside_contributions": [["3/2", "1/2"], ["1", "0"], ["1", "0"], ["2", "0"]],
        "flawed_monic_result": ["3", "1"],
        "leading_coefficient": "2",
        "corrected_mahler_measure": ["6", "2"],
    }


def test_oracle_mathematics():
    assert _module().mathematics(_result())


def test_rejects_missing_leading_coefficient():
    result = _result()
    result["corrected_mahler_measure"] = ["3", "1"]
    assert not _module().mathematics(result)


def test_rejects_corrupted_factor():
    result = _result()
    result["factors"][2] = [1, 2, 1]
    assert not _module().mathematics(result)


def test_rejects_noncanonical_fraction():
    result = _result()
    result["outside_contributions"][0] = ["6/4", "1/2"]
    assert not _module().mathematics(result)
