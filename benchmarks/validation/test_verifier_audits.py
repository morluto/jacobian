from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.verifier_audits import (
    canonical_string_rational_schema_failures,
    fraction_coprimality_failures,
)


def test_flags_fraction_coprimality_gate(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from fractions import Fraction\n"
        "def parse(value):\n"
        "    numerator = value['numerator']\n"
        "    denominator = value['denominator']\n"
        "    parsed = Fraction(numerator, denominator)\n"
        "    return parsed if parsed.numerator == numerator else None\n"
    )
    failures = fraction_coprimality_failures(verifier)
    assert failures
    assert "do not compare Fraction.numerator" in failures[0]


def test_allows_normalized_fraction_comparison(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from fractions import Fraction\n"
        "def parse(value):\n"
        "    parsed = Fraction(value['numerator'], value['denominator'])\n"
        "    return parsed == Fraction(1, 4)\n"
    )
    assert fraction_coprimality_failures(verifier) == []


def test_flags_canonical_rational_string_schema(tmp_path: Path) -> None:
    schema = tmp_path / "submission_schema.json"
    schema.write_text(
        '{"properties":{"x":{"type":"string",'
        '"pattern":"^-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?$"}}}\n'
    )
    failures = canonical_string_rational_schema_failures(schema)
    assert failures
    assert "structured objects" in failures[0]


def test_allows_structured_rational_schema(tmp_path: Path) -> None:
    schema = tmp_path / "submission_schema.json"
    schema.write_text(
        '{"properties":{"x":{"type":"object","additionalProperties":false,'
        '"required":["numerator","denominator"],'
        '"properties":{"numerator":{"type":"integer"},'
        '"denominator":{"type":"integer","minimum":1}}}}}\n'
    )
    assert canonical_string_rational_schema_failures(schema) == []
