"""Collect the benchmark planner contract corpus against its package owner.

The broad regression corpus predates extraction of ``tools.benchmark_plan`` and
also contains a SourceFileLoader smoke test for the executable adapter. Keep the
cases intact, but bind their semantic ``planner`` global to the importable
compiler before exposing them to pytest. Adapter behavior is tested only by the
explicit loader case; planner semantics and monkeypatch seams belong to the
package module.
"""

from __future__ import annotations

import hashlib

import tools.benchmark_plan.compiler as planner
from benchmarks.validation import benchmark_planner_contract_cases as _cases

_cases.planner = planner

# Preserve the complete historical case corpus and its autouse digest fixture,
# except for the old three-file digest assertion replaced below.
stable_digests = _cases.stable_digests
for _name, _value in vars(_cases).items():
    if _name.startswith("test_") and _name != (
        "test_planner_digest_binds_to_planner_and_path_policy_sources"
    ):
        globals()[_name] = _value


def test_planner_digest_binds_to_every_declared_semantic_source() -> None:
    payload = "\n".join(
        f"{path.relative_to(planner.ROOT).as_posix()}\t{path.read_bytes().hex()}"
        for path in planner.PLANNER_DIGEST_SOURCES
    ).encode()
    expected = "sha256:" + hashlib.sha256(payload).hexdigest()

    result = planner.plan(
        [
            "benchmarks/datasets/mathematical-benchmarks-v1/"
            "parameterized-sharp-bound-audit/tests/verifier.py"
        ],
        event="pull_request",
    )

    assert result["benchmark-planner-digest"] == expected


def test_extracted_path_policy_change_runs_benchmark_contracts() -> None:
    result = planner.plan(
        ["tools/benchmark_plan/paths.py"],
        event="pull_request",
    )

    assert result["run-benchmark-check"] == "true"
    assert result["run-benchmark-record-schema"] == "true"
    assert result["benchmark-plan-mode"] == "changed"


def test_benchmark_contract_tool_selects_its_owned_host_contract() -> None:
    path = "benchmarks/tooling/benchmark_contracts.py"

    result = planner.plan([path], event="pull_request")

    assert _cases._host_matrix(result) == [
        {
            "name": "control-test_benchmark_contracts",
            "selector": "benchmarks/validation/test_benchmark_contracts.py",
            "keyword": "",
            "splits": 0,
            "group": 0,
        }
    ]
    _cases._assert_plan_valid(result)


del _name, _value
