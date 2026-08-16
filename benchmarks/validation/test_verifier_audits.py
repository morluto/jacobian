from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.verifier_audits import (
    canonical_string_rational_schema_failures,
    formula_string_schema_failures,
    fraction_coprimality_failures,
    hidden_expected_scoring_failures,
    mirror_witness_failures,
    prose_witness_failures,
    unread_hash_witness_failures,
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


def test_flags_mirror_result_witness(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "def evidence_ok(payload, result):\n"
        "    return set(payload) == {'schema_version', 'task_id', 'result'}"
        " and payload['result'] == result\n"
    )
    failures = mirror_witness_failures(verifier)
    assert failures
    assert "mirror submission.result" in failures[0]


def test_flags_mirror_result_dict_literal(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "def evidence_ok(payload, result):\n"
        "    return payload == {'schema_version': '1', 'task_id': 't', 'result': result}\n"
    )
    failures = mirror_witness_failures(verifier)
    assert failures
    assert "mirror submission.result" in failures[0]


def test_flags_unread_hash_only_witness(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from verifier_support import witness_list_is_bound\n"
        "def check(submission):\n"
        "    return witness_list_is_bound(submission.get('witness'))\n"
    )
    failures = unread_hash_witness_failures(verifier)
    assert failures
    assert "semantic evidence read" in failures[0]


def test_allows_semantically_read_witness(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "from verifier_support import read_evidence_json, witness_list_is_bound\n"
        "def check(submission):\n"
        "    return witness_list_is_bound(submission.get('witness')) and "
        "read_evidence_json(submission['witness'][0])\n"
    )
    assert unread_hash_witness_failures(verifier) == []


def test_flags_formula_string_schema(tmp_path: Path) -> None:
    schema = tmp_path / "submission_schema.json"
    schema.write_text(
        '{"properties":{"result":{"properties":{"count_formula":{"const":"(b^(2m+2)-1)/(b+1)"}}}}}\n'
    )
    failures = formula_string_schema_failures(schema)
    assert failures
    assert "structured objects" in failures[0]


def test_allows_structured_formula_schema(tmp_path: Path) -> None:
    schema = tmp_path / "submission_schema.json"
    schema.write_text(
        '{"properties":{"result":{"properties":{"count_formula":{"type":"object","required":["linear"]}}}}}\n'
    )
    assert formula_string_schema_failures(schema) == []


def test_ignores_witness_path_consts(tmp_path: Path) -> None:
    schema = tmp_path / "submission_schema.json"
    schema.write_text(
        '{"properties":{"witness":{"items":{"properties":{'
        '"path":{"const":"evidence/answer.txt"}}}}}}'
        "\n"
    )
    assert formula_string_schema_failures(schema) == []


def test_flags_hidden_expected_math(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "def _math(s, x, e):\n    return s['result'] == e['expected_rank']\n"
    )
    failures = hidden_expected_scoring_failures(verifier)
    assert failures
    assert "frozen input" in failures[0]


def test_flags_prose_result_copy_witness(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier.py"
    verifier.write_text(
        "def evidence_ok(text):\n"
        "    return 'RESULT_JSON:' in text and 'column' in text\n"
    )
    failures = prose_witness_failures(verifier)
    assert failures
    assert "RESULT_JSON" in failures[0]
