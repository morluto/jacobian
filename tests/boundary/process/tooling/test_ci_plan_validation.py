from __future__ import annotations

import pytest
from tests.boundary.process.tooling.ci import run_ci_script


def _plan_for(*paths: str) -> dict[str, str]:
    output = run_ci_script("classify-ci-paths", *paths, check=True).stdout
    return dict(line.split("=", 1) for line in output.splitlines())


def _encoded(plan: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in plan.items())


@pytest.mark.parametrize(
    "args",
    [
        ("README.md",),
        ("npm/package.json",),
        ("src/jacobian/runtime/model.py",),
        ("tests/unit/test_runtime.py",),
        ("tests/composition/runtime/test_runtime_lifecycle.py",),
        ("lean/JacobianLeanRuntime.lean",),
        ("tests/unit/test_runtime.py", "lean/JacobianLeanRuntime.lean"),
        ("--force-lean", "--", "README.md"),
        ("--force-lean", "--", "npm/package.json"),
    ],
)
def test_ci_plan_output_is_internally_consistent(args: tuple[str, ...]) -> None:
    plan = run_ci_script("classify-ci-paths", *args, check=True).stdout

    # The classifier must produce a non-empty plan with at least a
    # classification line and runnable flags.
    assert plan.strip(), "classifier produced empty plan"
    assert "classification=" in plan, "plan missing classification line"

    run_ci_script("validate-ci-plan", input_text=plan, check=True)


@pytest.mark.parametrize(
    "plan",
    [
        "",
        "classification=docs\nrun-python=flase\n",
        "classification=full\n"
        "run-python=false\n"
        "run-lean=false\n"
        "run-npm=false\n"
        "run-static=false\n"
        "run-build=false\n"
        "run-security=false\n"
        "run-duplicate=false\n"
        "run-docs=false\n",
        "classification=docs\n"
        "classification=docs\n"
        "run-python=false\n"
        "run-lean=false\n"
        "run-npm=false\n"
        "run-static=false\n"
        "run-build=false\n"
        "run-security=false\n"
        "run-duplicate=false\n"
        "run-docs=false\n",
        "classification=docs\n"
        "run-python=false\n"
        "run-unit=false\n"
        "run-domain=false\n"
        "run-coverage=false\n"
        "run-compatibility=false\n"
        "run-lean=false\n"
        "run-npm=true\n"
        "run-static=false\n"
        "run-build=false\n"
        "run-security=false\n"
        "run-duplicate=false\n"
        "run-docs=false\n",
        "classification=docs\n"
        "run-python=true\n"
        "run-unit=true\n"
        "run-domain=false\n"
        "run-coverage=false\n"
        "run-compatibility=false\n"
        "run-lean=false\n"
        "run-npm=false\n"
        "run-static=false\n"
        "run-build=false\n"
        "run-security=false\n"
        "run-duplicate=false\n"
        "run-docs=false\n",
        "classification=lean\n"
        "run-python=true\n"
        "run-unit=true\n"
        "run-domain=false\n"
        "run-coverage=false\n"
        "run-compatibility=false\n"
        "run-lean=false\n"
        "run-npm=false\n"
        "run-static=false\n"
        "run-build=false\n"
        "run-security=false\n"
        "run-duplicate=false\n"
        "run-docs=false\n",
    ],
)
def test_ci_plan_validator_rejects_malformed_or_incoherent_plans(plan: str) -> None:
    completed = run_ci_script("validate-ci-plan", input_text=plan)

    assert completed.returncode != 0


@pytest.mark.parametrize(
    "plan",
    [
        {**_plan_for("README.md"), "classification": "full"},
        {**_plan_for("tests/unit/test_runtime.py"), "classification": "docs"},
        {
            **_plan_for("tests/composition/runtime/test_runtime_lifecycle.py"),
            "classification": "docs",
        },
    ],
)
def test_ci_plan_validator_rejects_wrong_classification_shapes(
    plan: dict[str, str],
) -> None:
    completed = run_ci_script("validate-ci-plan", input_text=_encoded(plan))

    assert completed.returncode != 0
