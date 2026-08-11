from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.boundary.process.tooling.ci import run_ci_script
from tools.test_plan.ci_outputs import (
    boolean_run_keys,
    matrix_lane_names,
    python_run_keys,
)

_ROOT = Path(__file__).resolve().parents[4]
_IMPACT = json.loads((_ROOT / ".github" / "ci-impact.json").read_text(encoding="utf-8"))
BOOLEAN_KEYS = boolean_run_keys(_IMPACT)
FUNCTIONAL_KEYS = tuple(
    key
    for key in BOOLEAN_KEYS
    if key not in {"run-coverage", "run-compatibility", "run-docs"}
)
PULL_REQUEST_FUNCTIONAL_KEYS = tuple(
    key for key in FUNCTIONAL_KEYS if key not in {"run-provider", "run-lean"}
)
MATRIX_LANES = matrix_lane_names(_IMPACT)
GENERIC_PYTHON_KEYS = tuple(
    key for key in python_run_keys(_IMPACT) if key not in {"run-provider"}
)
GENERIC_PYTHON_KEYS = ("run-python", *GENERIC_PYTHON_KEYS)


def _expected_plan(classification: str, *enabled: str) -> dict[str, str]:
    selected = set(enabled)
    return {
        "classification": classification,
        "product-lane-matrix": json.dumps(
            {
                "include": [
                    {"lane": lane} for lane in MATRIX_LANES if f"run-{lane}" in selected
                ]
            },
            separators=(",", ":"),
        ),
        "provider-selection": "[]",
        "run-deploy": str(classification == "exhaustive").lower(),
        **{key: str(key in selected).lower() for key in BOOLEAN_KEYS},
    }


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ((), _expected_plan("exhaustive", *BOOLEAN_KEYS)),
        (
            ("README.md", "docs/how-to/contribute.md", ".github/CODEOWNERS"),
            _expected_plan("docs", "run-docs"),
        ),
        (("AGENTS.md",), _expected_plan("docs", "run-docs")),
        ((".pre-commit-config.yaml",), _expected_plan("selective", "run-static")),
        ((".jscpd.json",), _expected_plan("selective", "run-duplicate")),
        (
            ("npm/package.json", "npm/npm-packaging.test.mjs"),
            _expected_plan("npm", "run-npm"),
        ),
        (
            ("docs/index.md", "npm/package.json"),
            _expected_plan("selective", "run-npm", "run-docs"),
        ),
        (
            ("tests/unit/test_runtime.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-unit",
                "run-static",
            ),
        ),
        (
            ("tests/composition/runtime/test_runtime_lifecycle.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-composition",
                "run-static",
            ),
        ),
        (
            ("lean/JacobianLeanRuntime.lean",),
            _expected_plan("none"),
        ),
        (
            ("tests/unit/test_runtime.py", "lean/JacobianLeanRuntime.lean"),
            _expected_plan(
                "python",
                "run-python",
                "run-unit",
                "run-static",
            ),
        ),
        (
            (
                "tests/unit/test_runtime.py",
                "tests/composition/runtime/test_runtime_lifecycle.py",
            ),
            _expected_plan(
                "python",
                "run-python",
                "run-unit",
                "run-composition",
                "run-static",
            ),
        ),
        (
            ("src/jacobian/runtime/model.py",),
            _expected_plan("selective", *PULL_REQUEST_FUNCTIONAL_KEYS),
        ),
        (
            ("tests/boundary/providers/lean/startup/test_lean.py",),
            _expected_plan(
                "selective",
                "run-static",
            ),
        ),
        (
            ("tests/component/providers/polytope/test_polytope_separation.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-component",
                "run-static",
            ),
        ),
        (
            ("src/jacobian/provider_runtime.py",),
            _expected_plan(
                "selective",
                *GENERIC_PYTHON_KEYS,
                "run-static",
                "run-build",
            ),
        ),
        (
            ("tests/support/provider_lean.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-composition",
                "run-e2e",
                "run-static",
            ),
        ),
        (
            ("tests/support/provider_external_sat.py",),
            _expected_plan("selective", "run-static"),
        ),
        (
            ("tests/composition/runtime/provider_spike_isolation.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-composition",
                "run-static",
            ),
        ),
        (
            ("tests/support/rationals.py",),
            _expected_plan(
                "python",
                "run-python",
                "run-unit",
                "run-component",
                "run-domain",
                "run-composition",
                "run-e2e",
                "run-static",
            ),
        ),
        (
            ("tests/boundary/providers/lean/test_lean_replayable_state_capability.py",),
            _expected_plan(
                "selective",
                "run-static",
            ),
        ),
        (
            ("tests/boundary/providers/lean/test_lean_statement_capabilities.py",),
            _expected_plan(
                "selective",
                "run-static",
            ),
        ),
        (
            ("src/jacobian/graph_capabilities.py",),
            _expected_plan(
                "selective", *GENERIC_PYTHON_KEYS, "run-static", "run-build"
            ),
        ),
        (
            ("src/jacobian/contracts/results.py",),
            _expected_plan(
                "selective", *GENERIC_PYTHON_KEYS, "run-static", "run-build"
            ),
        ),
        (
            ("src/jacobian/contracts/lean.py",),
            _expected_plan(
                "selective", *GENERIC_PYTHON_KEYS, "run-static", "run-build"
            ),
        ),
        (
            ("src/jacobian/contracts/plugins.py",),
            _expected_plan(
                "selective", *GENERIC_PYTHON_KEYS, "run-static", "run-build"
            ),
        ),
        (
            ("src/jacobian_checkers/graph_invariants.py",),
            _expected_plan(
                "selective",
                *GENERIC_PYTHON_KEYS,
                "run-static",
                "run-build",
            ),
        ),
        (
            ("src/jacobian/lean_proof_edit.py",),
            _expected_plan(
                "selective", *GENERIC_PYTHON_KEYS, "run-static", "run-build"
            ),
        ),
        (
            ("src/jacobian/adapters/mcp/server.py",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-component",
                "run-mcp",
                "run-e2e",
                "run-npm",
                "run-static",
                "run-build",
            ),
        ),
        (
            ("docs/index.md", "pyproject.toml"),
            _expected_plan("selective", *PULL_REQUEST_FUNCTIONAL_KEYS, "run-docs"),
        ),
        (
            (".github/workflows/ci.yml",),
            _expected_plan(
                "selective",
                *GENERIC_PYTHON_KEYS,
                "run-static",
                "run-build",
            ),
        ),
        (
            (".github/scripts/_ci_paths.py",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-process",
                "run-static",
                "run-build",
            ),
        ),
        (
            (".github/scripts/classify-ci-paths",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-process",
                "run-static",
                "run-build",
            ),
        ),
        (
            (".github/workflows/benchmarks.yml",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-process",
                "run-static",
                "run-build",
            ),
        ),
        (
            ("Makefile",),
            _expected_plan(
                "selective",
                "run-static",
                "run-build",
            ),
        ),
        (
            ("tools/test_topology.py",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-process",
                "run-static",
                "run-build",
            ),
        ),
        (
            ("tools/check_doc_commands.py",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-static",
                "run-build",
                "run-docs",
            ),
        ),
        (
            ("tools/check_benchmark_static.py",),
            _expected_plan("selective", "run-static", "run-build"),
        ),
        (
            ("tests/topology.toml",),
            _expected_plan(
                "selective",
                *GENERIC_PYTHON_KEYS,
                "run-static",
                "run-build",
            ),
        ),
        (
            ("src/jacobian/domains/number_theory/factorization.py",),
            _expected_plan(
                "selective",
                "run-python",
                "run-unit",
                "run-component",
                "run-domain",
                "run-static",
                "run-build",
            ),
        ),
        (
            ("CONTRIBUTING.md",),
            _expected_plan("docs", "run-docs"),
        ),
        (
            ("tests/support/services.py",),
            _expected_plan(
                "python",
                *GENERIC_PYTHON_KEYS,
                "run-static",
            ),
        ),
        (
            ("tests/conftest.py",),
            _expected_plan(
                "python",
                *GENERIC_PYTHON_KEYS,
                "run-static",
            ),
        ),
    ],
)
def test_ci_plan_fails_closed_outside_isolated_paths(
    paths: tuple[str, ...],
    expected: dict[str, str],
) -> None:
    completed = run_ci_script("classify-ci-paths", *paths, check=True)

    assert (
        dict(line.split("=", 1) for line in completed.stdout.splitlines()) == expected
    )


def test_full_override_expands_an_isolated_plan() -> None:
    completed = run_ci_script(
        "classify-ci-paths", "--force-full", "--", "docs/index.md", check=True
    )

    plan = dict(line.split("=", 1) for line in completed.stdout.splitlines())
    assert plan["classification"] == "exhaustive"
    assert all(value == "true" for key, value in plan.items() if key.startswith("run-"))


def test_lean_override_only_adds_lean_to_an_isolated_plan() -> None:
    completed = run_ci_script(
        "classify-ci-paths", "--force-lean", "--", "docs/index.md", check=True
    )

    plan = dict(line.split("=", 1) for line in completed.stdout.splitlines())
    assert plan == _expected_plan("selective", "run-lean", "run-docs")
