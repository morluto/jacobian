from __future__ import annotations

from pathlib import Path

from benchmarks.validation._source_module import load_task_verifier

ROOT = Path(__file__).parents[3]
TASK = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/mahler-leading-coefficient-audit"
)


def _module():
    return load_task_verifier(TASK, module_name="mahler_verifier")


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


def test_accepts_equivalent_public_rational_encoding():
    result = _result()
    result["outside_contributions"][0] = ["6/4", "1/2"]
    assert _module().mathematics(result)


def test_rejects_malformed_contribution_collection_without_crashing():
    result = _result()
    result["outside_contributions"] = None
    assert not _module().mathematics(result)


def test_rejects_exponent_syntax_before_fraction_construction(monkeypatch):
    module = _module()

    def unexpected_fraction(_value):
        raise AssertionError("invalid rational syntax must not reach Fraction")

    monkeypatch.setattr(module, "Fraction", unexpected_fraction)
    try:
        module._q("1e10000000000")
    except ValueError:
        pass
    else:
        raise AssertionError("exponent syntax must be rejected")


def test_evidence_comparison_preserves_json_types():
    module = _module()
    assert not module._json_equal({"coefficient": 0}, {"coefficient": False})
    assert not module._json_equal({"coefficient": 1}, {"coefficient": 1.0})


def test_rejects_malformed_factor_shapes_without_crashing():
    result = _result()
    result["factors"] = [[1, -3, 1], {"bad": "factor"}, [1, 1, 1], "factor"]
    assert not _module().mathematics(result)


def test_accepts_schema_valid_integral_json_numbers():
    result = _result()
    result["factors"] = [
        [float(value) for value in factor] for factor in result["factors"]
    ]
    assert _module().mathematics(result)


def test_rejects_boolean_factor_coefficients():
    result = _result()
    result["factors"][0][0] = True
    assert not _module().mathematics(result)


def test_evidence_comparison_rejects_excessive_nesting():
    module = _module()
    left = right = 0
    for _ in range(129):
        left = [left]
        right = [right]
    assert not module._json_equal(left, right)
