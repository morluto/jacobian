import json
from pathlib import Path

import pytest
from tools.check_complexity import (
    ComplexityBaseline,
    ComplexityBaselineError,
    ComplexityViolation,
    compare_violations,
    load_baseline,
    write_baseline,
)


def test_complexity_ratchet_rejects_new_and_increased_violations() -> None:
    baseline = ComplexityBaseline(
        10,
        (
            ComplexityViolation("src/known.py", "known", 12),
            ComplexityViolation("src/stable.py", "stable", 11),
        ),
    )

    problems = compare_violations(
        baseline,
        (
            ComplexityViolation("src/known.py", "known", 13),
            ComplexityViolation("src/new.py", "new", 11),
            ComplexityViolation("src/stable.py", "stable", 11),
        ),
    )

    assert problems == (
        "new violation: src/new.py:new has complexity 11",
        "complexity increased: src/known.py:known 12 -> 13",
    )


def test_complexity_ratchet_requires_improvements_to_update_the_baseline() -> None:
    baseline = ComplexityBaseline(
        10,
        (
            ComplexityViolation("src/improved.py", "improved", 14),
            ComplexityViolation("src/resolved.py", "resolved", 11),
        ),
    )

    problems = compare_violations(
        baseline,
        (ComplexityViolation("src/improved.py", "improved", 12),),
    )

    assert problems == (
        "baseline is stale after improvement: src/improved.py:improved 14 -> 12",
        "remove resolved violation from baseline: src/resolved.py:resolved",
    )


def test_complexity_baseline_rejects_duplicate_path_symbol_keys(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        """
        {
          "version": 1,
          "max_complexity": 10,
          "violations": [
            {"path": "src/example.py", "symbol": "run", "complexity": 11},
            {"path": "src/example.py", "symbol": "run", "complexity": 12}
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ComplexityBaselineError, match="keys must be unique"):
        load_baseline(baseline)


def test_complexity_baseline_writer_uses_canonical_order_and_trailing_newline(
    tmp_path: Path,
) -> None:
    baseline = ComplexityBaseline(
        10,
        (
            ComplexityViolation("src/z.py", "run", 12),
            ComplexityViolation("src/a.py", "walk", 11),
            ComplexityViolation("src/a.py", "run", 13),
        ),
    )
    path = tmp_path / "baseline.json"

    write_baseline(path, baseline)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [
        (item["path"], item["symbol"], item["complexity"])
        for item in payload["violations"]
    ] == [
        ("src/a.py", "run", 13),
        ("src/a.py", "walk", 11),
        ("src/z.py", "run", 12),
    ]
    assert path.read_text(encoding="utf-8").endswith("\n")
