from __future__ import annotations

import json
import os
import runpy
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from tests.boundary.process.tooling.ci import run_ci_script


def test_archive_download_does_not_forward_token_across_origins() -> None:
    received_authorization: list[str | None] = []

    class StorageHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"timing archive")

        def log_message(self, _format: str, *args: object) -> None:
            pass

    storage = ThreadingHTTPServer(("127.0.0.1", 0), StorageHandler)
    storage_thread = threading.Thread(target=storage.serve_forever)
    storage_thread.start()
    storage_url = f"http://127.0.0.1:{storage.server_port}/archive"

    class ApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.headers["Authorization"] == "Bearer test-token"
            self.send_response(302)
            self.send_header("Location", storage_url)
            self.end_headers()

        def log_message(self, _format: str, *args: object) -> None:
            pass

    api = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
    api_thread = threading.Thread(target=api.serve_forever)
    api_thread.start()
    try:
        namespace = runpy.run_path(
            Path(__file__).parents[4] / ".github" / "scripts" / "manage-test-timings"
        )
        download = cast(Callable[[str, str], bytes], namespace["download"])

        assert (
            download(
                f"http://127.0.0.1:{api.server_port}/artifact",
                "test-token",
            )
            == b"timing archive"
        )
        assert received_authorization == [None]
    finally:
        api.shutdown()
        storage.shutdown()
        api.server_close()
        storage.server_close()
        api_thread.join()
        storage_thread.join()


def plan_outputs() -> dict[str, str]:
    result = run_ci_script("manage-test-timings", "emit-plan-outputs")
    assert result.returncode == 0, result.stderr
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def shard_count(suite: str = "domain") -> int:
    if suite == "benchmark":
        return 4
    return int(plan_outputs()[f"{suite}-shard-count"])


def pinned_pytest_split_version() -> str:
    return plan_outputs()["pytest-split-version"]


def test_prepare_falls_back_to_equal_weighting_without_github_context(
    tmp_path: Path,
) -> None:
    output = tmp_path / "durations.json"
    env = os.environ.copy()
    env.pop("GITHUB_REPOSITORY", None)
    env.pop("GH_TOKEN", None)

    result = run_ci_script(
        "manage-test-timings",
        "prepare",
        "--output",
        output,
        "--suite",
        "domain",
        env=env,
    )

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {}
    assert "equal weighting" in result.stderr


@pytest.mark.parametrize("suite", ["domain", "composition", "benchmark"])
def test_merge_publishes_versioned_metadata_and_all_shards(
    tmp_path: Path, suite: str
) -> None:
    count = shard_count(suite)
    inputs: list[str] = []
    for shard in range(1, count + 1):
        path = tmp_path / f"shard-{shard}.json"
        prefix = "benchmarks/validation" if suite == "benchmark" else f"tests/{suite}"
        path.write_text(
            json.dumps({f"{prefix}/test_{shard}.py::test_case": shard}),
            encoding="utf-8",
        )
        inputs.extend(["--input", str(path)])
    output = tmp_path / f"{suite}-test-durations.json"

    result = run_ci_script(
        "manage-test-timings",
        "merge",
        *inputs,
        "--output",
        str(output),
        "--source-sha",
        "a" * 40,
        "--python-version",
        "3.12",
        "--suite",
        suite,
        "--pytest-split-version",
        pinned_pytest_split_version(),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["suite"] == suite
    assert payload["shard_count"] == count
    assert len(payload["durations"]) == count


def test_merge_defaults_pytest_split_version_from_pyproject(tmp_path: Path) -> None:
    count = shard_count()
    inputs: list[str] = []
    for shard in range(1, count + 1):
        path = tmp_path / f"shard-{shard}.json"
        path.write_text(
            json.dumps({f"tests/domain/test_shared.py::test_{shard}": 1.0}),
            encoding="utf-8",
        )
        inputs.extend(["--input", str(path)])

    result = run_ci_script(
        "manage-test-timings",
        "merge",
        *inputs,
        "--output",
        str(tmp_path / "output.json"),
        "--source-sha",
        "a" * 40,
        "--python-version",
        "3.12",
        "--suite",
        "domain",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert payload["pytest_split_version"] == pinned_pytest_split_version()


def test_emit_plan_outputs_exposes_ci_config_ssot() -> None:
    outputs = plan_outputs()
    assert outputs["node-version-jscpd"] == "20"
    assert outputs["node-version-npm"] == "24"
    assert outputs["pytest-randomly-shard-seed"] == "0"
    assert outputs["pytest-split-version"]
    assert int(outputs["composition-shard-count"]) == 2
    assert json.loads(outputs["composition-shards"]) == [1, 2]
    assert (
        run_ci_script("manage-test-timings", "node-version", "npm").stdout.strip()
        == outputs["node-version-npm"]
    )


def test_merge_keeps_max_duration_for_duplicate_node_ids(tmp_path: Path) -> None:
    count = shard_count()
    inputs: list[str] = []
    for shard in range(1, count + 1):
        path = tmp_path / f"shard-{shard}.json"
        path.write_text(
            json.dumps({"tests/domain/test_shared.py::test_case": shard}),
            encoding="utf-8",
        )
        inputs.extend(["--input", str(path)])

    result = run_ci_script(
        "manage-test-timings",
        "merge",
        *inputs,
        "--output",
        str(tmp_path / "output.json"),
        "--source-sha",
        "a" * 40,
        "--python-version",
        "3.12",
        "--suite",
        "domain",
        "--pytest-split-version",
        pinned_pytest_split_version(),
    )

    assert result.returncode == 0, result.stderr
    assert "duplicate" in result.stderr.lower()
    payload = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert payload["durations"]["tests/domain/test_shared.py::test_case"] == float(
        count
    )


def test_merge_handles_partial_overlap_across_shards(tmp_path: Path) -> None:
    """Duplicate entries from pytest-split overlap should keep max duration."""
    count = shard_count()
    inputs: list[str] = []
    shard_data: list[dict[str, float]] = [
        {
            "tests/domain/test_a.py::test_one": 1.0,
            "tests/domain/test_b.py::test_two": 2.0,
        },
        {
            "tests/domain/test_a.py::test_one": 3.0,
            "tests/domain/test_c.py::test_three": 4.0,
        },
    ]
    while len(shard_data) < count:
        shard_data.append({"tests/domain/test_d.py::test_d": 0.5})
    for shard, durations in enumerate(shard_data, start=1):
        path = tmp_path / f"shard-{shard}.json"
        path.write_text(json.dumps(durations), encoding="utf-8")
        inputs.extend(["--input", str(path)])

    result = run_ci_script(
        "manage-test-timings",
        "merge",
        *inputs,
        "--output",
        str(tmp_path / "output.json"),
        "--source-sha",
        "a" * 40,
        "--python-version",
        "3.12",
        "--suite",
        "domain",
        "--pytest-split-version",
        pinned_pytest_split_version(),
    )

    assert result.returncode == 0, result.stderr
    assert "duplicate" in result.stderr.lower()
    payload = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert payload["durations"]["tests/domain/test_a.py::test_one"] == 3.0
    assert payload["durations"]["tests/domain/test_b.py::test_two"] == 2.0
    assert payload["durations"]["tests/domain/test_c.py::test_three"] == 4.0
