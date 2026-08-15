from ._fixtures import assert_result_witness_protocol


def test_result_protocol(tmp_path):
    assert_result_witness_protocol(tmp_path, "path-dependent-limit")
