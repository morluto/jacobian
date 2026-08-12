"""Tests for the affinity shard selector CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import select_affinity_shard


def test_parse_collected_nodeids_keeps_unique_nodeid_lines() -> None:
    output = "\n".join(
        (
            "tests/domain/a/test_alpha.py::test_one",
            "tests/domain/a/test_alpha.py::test_one",
            "3 tests collected in 0.01s",
            "tests/domain/b/test_beta.py::test_two[value]",
        )
    )

    assert select_affinity_shard.parse_collected_nodeids(output) == (
        "tests/domain/a/test_alpha.py::test_one",
        "tests/domain/b/test_beta.py::test_two[value]",
    )


def test_load_durations_accepts_bare_map_and_envelope(tmp_path: Path) -> None:
    bare = tmp_path / "bare.json"
    envelope = tmp_path / "envelope.json"
    bare.write_text(json.dumps({"tests/a.py::test_one": 1}), encoding="utf-8")
    envelope.write_text(
        json.dumps({"version": 1, "durations": {"tests/b.py::test_two": 2.5}}),
        encoding="utf-8",
    )

    assert select_affinity_shard.load_durations(bare) == {"tests/a.py::test_one": 1.0}
    assert select_affinity_shard.load_durations(envelope) == {
        "tests/b.py::test_two": 2.5
    }
    assert select_affinity_shard.load_durations(tmp_path / "missing.json") == {}


def test_main_writes_selected_nodeids_and_full_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collected = (
        "tests/domain/example/test_math.py::test_slow",
        "tests/domain/example/test_math.py::test_medium",
        "tests/domain/example/test_math.py::test_fast",
    )

    def fake_collect(paths: tuple[str, ...]) -> tuple[str, ...]:
        assert paths == ("tests/domain/example",)
        return collected

    monkeypatch.setattr(select_affinity_shard, "collect_nodeids", fake_collect)
    durations = tmp_path / "durations.json"
    durations.write_text(
        json.dumps(
            {
                "durations": {
                    collected[0]: 5.0,
                    collected[1]: 3.0,
                    collected[2]: 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "selected.txt"
    plan_path = tmp_path / "plan.json"

    exit_code = select_affinity_shard.main(
        [
            "--suite",
            "domain",
            "--shard",
            "1",
            "--shard-count",
            "2",
            "--paths",
            "tests/domain/example",
            "--durations",
            str(durations),
            "--output",
            str(output),
            "--json-shards",
            str(plan_path),
        ]
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output.read_text(encoding="utf-8").splitlines() == plan["shards"][0]
    assert {nodeid for shard in plan["shards"] for nodeid in shard} == set(collected)
    assert plan["suite"] == "domain"
    assert plan["shard_count"] == 2
    assert "suite=domain shard=1/2 collected=3 selected=" in capsys.readouterr().err


def test_write_nodeids_leaves_an_empty_shard_file(tmp_path: Path) -> None:
    output = tmp_path / "selected.txt"

    select_affinity_shard.write_nodeids(output, ())

    assert output.exists()
    assert output.read_text(encoding="utf-8") == ""
