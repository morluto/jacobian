from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures


def test_result_witness_protocol(tmp_path):
    _fixtures.assert_result_witness_protocol(
        tmp_path, "finite-scheme-rational-points-audit"
    )
