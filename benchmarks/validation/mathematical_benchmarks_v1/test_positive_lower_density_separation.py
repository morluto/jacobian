from __future__ import annotations

import json
import runpy
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import _fixtures, _verifier

TASK = "positive-lower-density-separation"


def _q(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _run(tmp_path: Path, mutate=None):
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate:
        mutate(submission)
        _fixtures._write_json(app / "submission.json", submission)
    return _verifier._run_verifier(task, app, logs)


def _set_base(submission, base):
    levels = []
    for level in range(8):
        high, low = base ** (2 * level + 1), base ** (2 * level + 2)
        count = (low - 1) // (base + 1)
        levels.append(
            {
                "level": level,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": _q(Fraction(count, high)),
                "excluded_density": _q(Fraction(count, low)),
            }
        )
    submission["result"].update(
        {
            "base": base,
            "levels": levels,
            "lower_density": _q(Fraction(1, base + 1)),
            "upper_density": _q(Fraction(base, base + 1)),
        }
    )


def test_oracle_and_alternative_base_pass(tmp_path: Path) -> None:
    assert _run(tmp_path / "oracle").reward == 1.0
    assert (
        _run(
            tmp_path / "alternative", lambda submission: _set_base(submission, 7)
        ).reward
        == 1.0
    )


def test_rejects_wrong_certificate_or_legacy_field(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path / "wrong",
            lambda submission: submission["result"]["levels"][4].update(
                cumulative_count=0
            ),
        ).reward
        == 0.0
    )
    assert (
        _run(
            tmp_path / "legacy",
            lambda submission: submission.update(witness=[]),
        ).reward
        == 0.0
    )


def test_input_binding_and_type_checks_are_hard_gates(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path / "typed",
            lambda submission: submission["result"]["levels"][0].update(level=0.0),
        ).reward
        == 0.0
    )
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "input-tamper")
    (app / "input.json").write_text("{}")
    result = _verifier._run_verifier(task, app, logs)
    assert result.details["input_binding"] == 0.0
    assert result.reward == 0.0


def test_string_coerced_density_is_rejected(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission["result"].update(lower_density="1/4"),
        ).reward
        == 0.0
    )


def test_accepts_unreduced_lower_density(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            lambda submission: submission["result"].update(
                lower_density={"numerator": 2, "denominator": 8}
            ),
        ).reward
        == 1.0
    )


def test_oracle_generator_emits_typed_rationals(tmp_path: Path, monkeypatch) -> None:
    generated: dict[str, str] = {}
    original_write_text = Path.write_text

    def capture(self: Path, data: str, *args, **kwargs) -> int:
        assert self == Path("/app/submission.json")
        generated["submission"] = data
        return len(data)

    monkeypatch.setattr(Path, "write_text", capture)
    runpy.run_path(
        "benchmarks/datasets/mathematical-benchmarks-v1/"
        "positive-lower-density-separation/solution/solve.py"
    )
    monkeypatch.setattr(Path, "write_text", original_write_text)

    submission = json.loads(generated["submission"])
    assert submission["result"]["lower_density"] == _q(Fraction(1, 4))
    assert submission["result"]["levels"][0]["included_density"] == _q(Fraction(2, 3))
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "generated-oracle")
    _fixtures._write_json(app / "submission.json", submission)
    assert _verifier._run_verifier(task, app, logs).reward == 1.0


def test_equivalent_count_formula_encodings_pass(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path / "unreduced-offset",
            lambda submission: submission["result"]["count_formula"].update(
                numerator_constant=-2, denominator_offset=1
            ),
        ).reward
        == 0.0
    )
    gold = json.loads(
        (
            Path("benchmarks/datasets/mathematical-benchmarks-v1")
            / TASK
            / "solution/submission.json"
        ).read_text()
    )
    assert gold["result"]["count_formula"]["numerator_constant"] == -1
