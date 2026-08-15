from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures

TASK = "primitive-eisenstein-norm-audit"


def test_result_witness_protocol(tmp_path: Path) -> None:
    _fixtures.assert_result_witness_protocol(tmp_path, TASK)
