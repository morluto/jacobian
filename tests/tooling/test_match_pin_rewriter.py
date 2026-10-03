"""Recorded evidence must not lose ambiguity across repeated executions.

``tools/match_recorder.py`` writes one JSON object per pytest session, so a site
executed under two xdist workers appears in two JSONL lines. Replaying those
lines with last-write-wins let a coded observation hide an earlier code-less
one, and the rewriter then emitted a single-code assert that fails on the
code-less execution. The same overwrite would hide a second ``match`` text
asserted behind one code.

The rewriter must also refuse sites whose ``with`` statement manages sibling
context managers, because rebuilding the header from ``pytest.raises`` alone
would drop those siblings and change what the test exercises.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.match_recorder import merge_record
from tools.rewrite_match_pins import convertible_sites, load, rewrite

SITE = "tests/example/test_module.py:42:8"


def _line(info: dict[str, object]) -> str:
    return json.dumps({SITE: info})


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "record.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_coded_observation_does_not_hide_earlier_codeless_execution(
    tmp_path: Path,
) -> None:
    # Worker A ran a parametrization that raised a plain ValueError; worker B
    # ran one that raised an owner-coded error. Unioning keeps both.
    path = _write(
        tmp_path,
        _line({"code": None, "match": "boom", "codes": [None]}),
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
    )
    recorded = load(path)
    assert recorded[SITE]["codes"] == [None, "polynomial.degree"]


def test_split_parametrization_is_rejected_not_converted(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _line({"code": None, "match": "boom", "codes": [None]}),
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
    )
    usable, rejected = convertible_sites(load(path))
    assert not usable
    assert rejected["multiple_codes_across_executions"] == 1


def test_split_parametrization_to_two_codes_is_rejected(tmp_path: Path) -> None:
    # Two workers observed two different owner codes for the same line.
    path = _write(
        tmp_path,
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
        _line(
            {
                "code": "polynomial.leading",
                "match": "bang",
                "codes": ["polynomial.leading"],
            }
        ),
    )
    usable, rejected = convertible_sites(load(path))
    assert not usable
    assert rejected["multiple_codes_across_executions"] == 1


def test_agreeing_coded_observations_still_convert(tmp_path: Path) -> None:
    # The merge must not reject a site merely because two workers both saw it.
    path = _write(
        tmp_path,
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
    )
    usable, rejected = convertible_sites(load(path))
    assert usable == {("tests/example/test_module.py:42", 8): "polynomial.degree"}
    assert not rejected


def test_merge_record_preserves_codeless_member() -> None:
    merged = merge_record({"code": None, "codes": [None]}, {"code": "ops.limit"})
    assert merged["codes"] == [None, "ops.limit"]
    assert merged["code"] is None


def test_distinct_matches_across_executions_are_rejected(tmp_path: Path) -> None:
    # One code, two parameter-specific match texts. The wording is the only
    # thing distinguishing the guards, so the site cannot become one code assert.
    path = _write(
        tmp_path,
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
        _line(
            {
                "code": "polynomial.degree",
                "match": "bang",
                "codes": ["polynomial.degree"],
            }
        ),
    )
    usable, rejected = convertible_sites(load(path))
    assert not usable
    assert rejected["multiple_matches_across_executions"] == 1


def test_merge_record_collects_every_observed_match() -> None:
    merged = merge_record(
        {"code": "ops.limit", "match": "boom"},
        {"code": "ops.limit", "match": "bang"},
    )
    assert merged["matches"] == ["bang", "boom"]
    assert merged["code"] == "ops.limit"


def test_merge_record_preserves_incoming_aggregated_matches() -> None:
    merged = merge_record(
        {"code": "ops.limit", "match": "zap"},
        {
            "code": "ops.limit",
            "match": "bang",
            "matches": ["bang", "boom"],
        },
    )
    assert merged["matches"] == ["bang", "boom", "zap"]


def test_agreeing_matches_still_convert(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
        _line(
            {
                "code": "polynomial.degree",
                "match": "boom",
                "codes": ["polynomial.degree"],
            }
        ),
    )
    usable, rejected = convertible_sites(load(path))
    assert usable == {("tests/example/test_module.py:42", 8): "polynomial.degree"}
    assert not rejected


SIBLING_WITH = """\
import pytest

from contextlib import contextmanager


@contextmanager
def request_execution(deadline):
    yield deadline


def test_bounded():
    source = 1
    with request_execution(deadline=1), pytest.raises(ValueError, match="boom"):
        compute(source)
"""


def test_sibling_context_manager_is_not_dropped(tmp_path: Path) -> None:
    path = tmp_path / "test_sibling.py"
    path.write_text(SIBLING_WITH, encoding="utf-8")
    changed, multi_item_with = rewrite(path, {5: "ops.limit"})
    assert changed == 0
    assert multi_item_with == 1
    assert path.read_text(encoding="utf-8") == SIBLING_WITH


def test_single_item_with_still_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "test_solo.py"
    path.write_text(
        "import pytest\n\n\ndef test_bounded():\n"
        '    with pytest.raises(ValueError, match="boom"):\n'
        "        compute(1)\n",
        encoding="utf-8",
    )
    changed, multi_item_with = rewrite(path, {5: "ops.limit"})
    assert (changed, multi_item_with) == (1, 0)
    assert "with pytest.raises(ValueError) as exc_info:" in path.read_text(
        encoding="utf-8"
    )
