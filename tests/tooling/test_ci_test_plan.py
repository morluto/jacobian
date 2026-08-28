"""Tests for the fail-closed mathematical CI selector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tools.ci_test_plan import TestPlan


ROOT = Path(__file__).parents[2]


def _load() -> ModuleType:
    path = ROOT / "tools" / "ci_test_plan.py"
    spec = importlib.util.spec_from_file_location("ci_test_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(paths: list[str], *, event: str = "pull_request") -> TestPlan:
    planner = _load()
    plan: TestPlan = planner.build_plan(
        event=event,
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=paths,
        repository=ROOT,
    )
    return plan


def test_model_change_selects_its_math_owner_and_public_contract_evidence() -> None:
    plan = _plan(["src/jacobian/math/combinatorics/codes/general/_models.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/combinatorics/codes/general",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True
    assert plan.run_scale is False
    assert plan.python_lanes == ()
    assert plan.boundary_lanes == ()


def test_canonical_cnf_contract_change_selects_public_contract_evidence() -> None:
    plan = _plan(["src/jacobian/math/logic/_cnf.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/logic",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_pseudomanifold_contract_change_selects_public_contract_evidence() -> None:
    plan = _plan(["src/jacobian/math/topology/_pseudomanifold.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/topology",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_structural_topology_contract_change_selects_public_contract_evidence() -> None:
    plan = _plan(["src/jacobian/math/topology/_structural.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/topology",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_prime_affine_interval_contract_change_selects_public_contract_evidence() -> (
    None
):
    plan = _plan(["src/jacobian/math/number_theory/prime_affine_forms/_interval.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/number_theory/prime_affine_forms",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


@pytest.mark.parametrize("filename", ["_sat.py", "_smt.py"])
def test_logic_solver_contract_change_selects_public_contract_evidence(
    filename: str,
) -> None:
    plan = _plan([f"src/jacobian/math/logic/{filename}"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/logic",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_public_operation_kernel_selects_catalog_examples() -> None:
    plan = _plan(["src/jacobian/math/combinatorics/codes/general/_dual_operations.py"])

    assert plan.run_math is True
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_partitioned_operation_modules_select_catalog_examples() -> None:
    for filename in (
        "_factorization_operations.py",
        "_presentation_operations.py",
        "_summary_operations.py",
        "_element_invariant_operations.py",
        "_global_invariant_operations.py",
    ):
        plan = _plan(
            [f"src/jacobian/math/number_theory/numerical_semigroups/{filename}"]
        )

        assert plan.run_math is True
        assert plan.math_tests == ("tests/math/number_theory/numerical_semigroups",)
        assert plan.run_catalog is True
        assert plan.run_catalog_examples is True


def test_nested_math_owner_selects_its_top_level_test_root() -> None:
    plan = _plan(["src/jacobian/math/matrices/canonical_forms/_models.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/matrices",)


def test_math_test_change_runs_only_the_changed_test() -> None:
    path = "tests/math/combinatorics/codes/general/test_exact_code_enumeration.py"
    plan = _plan([path])

    assert plan.run_math is True
    assert plan.math_tests == (path,)
    assert [(shard.group, shard.splits) for shard in plan.math_shards] == [(1, 1)]
    assert plan.run_catalog is False


def test_shared_runtime_path_falls_back_to_full_math_and_public_contracts() -> None:
    plan = _plan(["src/jacobian/canonical.py"])

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert [(shard.group, shard.splits) for shard in plan.math_shards] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True
    assert "shared CI or runtime path changed" in plan.reasons[0]


def test_unknown_math_owner_falls_back_to_full_math(tmp_path: Path) -> None:
    planner = _load()
    plan = planner.build_plan(
        event="pull_request",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=["src/jacobian/math/future_owner/_models.py"],
        repository=tmp_path,
    )

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert plan.run_catalog is True
    assert "no explicit test root" in plan.reasons[0]


def test_documentation_change_skips_product_evidence() -> None:
    plan = _plan(["docs/reference/testing-strategy.md"])

    assert plan.run_math is False
    assert plan.math_shards == ()
    assert plan.run_catalog is False
    assert plan.run_catalog_examples is False
    assert plan.run_scale is False
    assert plan.python_lanes == ()
    assert plan.boundary_lanes == ()
    assert plan.run_singular is False
    assert plan.run_wheel is False


@pytest.mark.parametrize(
    ("path", "lanes"),
    [
        ("tests/dispatch/test_dispatch.py", ("dispatch",)),
        ("tests/cli/test_cli.py", ("cli",)),
        ("tests/tooling/test_make_commands.py", ("tooling",)),
        ("tests/integration/algebra/test_linear.py", ("integration",)),
    ],
)
def test_boundary_owner_test_change_selects_only_its_python_lane(
    path: str, lanes: tuple[str, ...]
) -> None:
    plan = _plan([path])

    assert plan.run_math is False
    assert plan.python_lanes == lanes
    assert plan.boundary_lanes == ()


def test_mcp_runtime_change_fails_closed_to_shared_runtime_evidence() -> None:
    plan = _plan(["src/jacobian/mcp/tools.py"])

    assert plan.boundary_lanes == ("mcp", "process")
    assert plan.run_wheel is True
    assert plan.run_math is True


def test_process_polynomial_test_selects_process_and_singular() -> None:
    plan = _plan(["tests/process/polynomials/test_ideals.py"])

    assert plan.boundary_lanes == ("process",)
    assert plan.run_singular is True


def test_shared_test_support_fails_closed_to_every_ordinary_boundary() -> None:
    plan = _plan(["tests/support/math_values.py"])

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert plan.python_lanes == ("cli", "dispatch", "integration", "tooling")
    assert plan.boundary_lanes == ("mcp", "process")
    assert plan.run_singular is True
    assert plan.run_wheel is True


def test_merge_group_always_owns_full_math_and_public_contracts() -> None:
    plan = _plan(["docs/reference/testing-strategy.md"], event="merge_group")

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert [(shard.group, shard.splits) for shard in plan.math_shards] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True
    assert plan.run_scale is False
    assert plan.python_lanes == ("dispatch", "cli", "tooling", "integration")
    assert plan.boundary_lanes == ("process", "mcp")
    assert plan.run_singular is True
    assert plan.run_wheel is True


def test_merge_group_selects_scale_evidence_for_its_owning_math_domain() -> None:
    plan = _plan(
        ["src/jacobian/math/lattice_polytopes/_operations.py"], event="merge_group"
    )

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert plan.run_scale is True


def test_main_owns_full_ordinary_suite_and_coverage_without_scale() -> None:
    plan = _plan(["docs/reference/testing-strategy.md"], event="push")

    assert plan.run_math is True
    assert [(shard.group, shard.splits) for shard in plan.math_shards] == [
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert plan.run_scale is False
    assert plan.python_lanes == ("dispatch", "cli", "tooling", "integration")
    assert plan.boundary_lanes == ("process", "mcp")


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_schedule_and_manual_runs_include_scale_evidence(event: str) -> None:
    plan = _plan([], event=event)

    assert plan.run_scale is True


def test_rejects_non_normalized_paths() -> None:
    planner = _load()

    with pytest.raises(ValueError, match="normalized and repository-relative"):
        planner.build_plan(
            event="pull_request",
            base_revision="a" * 40,
            head_revision="b" * 40,
            changed_paths=["../tests/math/test_anything.py"],
            repository=ROOT,
        )
