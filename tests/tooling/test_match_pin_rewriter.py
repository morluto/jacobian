"""Recorded evidence must not lose code-less executions across xdist workers.

``tools/match_recorder.py`` writes one JSON object per pytest session, so a site
executed under two xdist workers appears in two JSONL lines. Replaying those
lines with last-write-wins let a coded observation hide an earlier code-less
one, and the rewriter then emitted a single-code assert that fails on the
code-less execution.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.match_recorder import merge_record
from tools.rewrite_match_pins import convertible_sites, load

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
