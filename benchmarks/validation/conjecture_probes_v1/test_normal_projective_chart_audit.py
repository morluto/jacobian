from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from benchmarks.validation._source_module import load_task_verifier

ROOT = Path(__file__).parents[3]
TASK = ROOT / "benchmarks/datasets/conjecture-probes-v1/normal-projective-chart-audit"


def _q(value) -> dict[str, int]:
    parsed = Fraction(value)
    return {"numerator": parsed.numerator, "denominator": parsed.denominator}


def _point(values) -> list[dict[str, int]]:
    return [_q(item) for item in values]


def _module():
    return load_task_verifier(TASK, module_name="normal_chart_verifier")


def _result():
    return {
        "finite_parameters": _point(["-1", "0", "1"]),
        "finite_points": [_point(["0", "-1"]), _point(["0", "1"]), _point(["2", "0"])],
        "missing_projective_parameter": _point(["1", "0"]),
        "missing_point": _point(["-2", "0"]),
        "footpoint_records": [
            {
                "point": _point(["-2", "0"]),
                "ellipse_residual": _q("0"),
                "normal_residual": _q("0"),
            },
            {
                "point": _point(["0", "-1"]),
                "ellipse_residual": _q("0"),
                "normal_residual": _q("0"),
            },
            {
                "point": _point(["0", "1"]),
                "ellipse_residual": _q("0"),
                "normal_residual": _q("0"),
            },
            {
                "point": _point(["2", "0"]),
                "ellipse_residual": _q("0"),
                "normal_residual": _q("0"),
            },
        ],
    }


def test_oracle_mathematics():
    assert _module().mathematics(_result())


def test_rejects_three_point_only_boundary():
    result = _result()
    result["missing_point"] = _point(["2", "0"])
    assert not _module().mathematics(result)


def test_rejects_corrupted_normal_residual():
    result = _result()
    result["footpoint_records"][0]["normal_residual"] = _q("1")
    assert not _module().mathematics(result)


def test_accepts_permuted_points_and_records():
    result = _result()
    result["finite_parameters"] = _point(["1", "0", "-1"])
    result["finite_points"] = [
        _point(["2", "0"]),
        _point(["0", "1"]),
        _point(["0", "-1"]),
    ]
    result["footpoint_records"] = list(reversed(result["footpoint_records"]))
    assert _module().mathematics(result)


def test_accepts_unreduced_rational():
    result = _result()
    result["finite_parameters"][0] = {"numerator": -2, "denominator": 2}
    assert _module().mathematics(result)


def test_rejects_string_rational(monkeypatch):
    module = _module()

    def unexpected_fraction(_value):
        raise AssertionError("string rationals must not reach Fraction")

    monkeypatch.setattr(module, "Fraction", unexpected_fraction)
    try:
        module._q("1e100000000")
    except ValueError:
        pass
    else:
        raise AssertionError("string rational was accepted")
