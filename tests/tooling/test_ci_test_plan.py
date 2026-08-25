"""Tests for the fail-closed mathematical CI selector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def _load() -> ModuleType:
    path = ROOT / "tools" / "ci_test_plan.py"
    spec = importlib.util.spec_from_file_location("ci_test_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(paths: list[str], *, event: str = "pull_request") -> object:
    planner = _load()
    return planner.build_plan(
        event=event,
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=paths,
        repository=ROOT,
    )


def test_model_change_selects_its_math_owner_and_public_contract_evidence() -> None:
    plan = _plan(["src/jacobian/math/code_theory/_models.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/code_theory",)
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_public_operation_kernel_selects_catalog_examples() -> None:
    plan = _plan(["src/jacobian/math/code_theory/_dual_operations.py"])

    assert plan.run_math is True
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


def test_nested_math_owner_selects_its_top_level_test_root() -> None:
    plan = _plan(["src/jacobian/math/matrices/canonical_forms/_models.py"])

    assert plan.run_math is True
    assert plan.math_tests == ("tests/math/matrices",)


def test_math_test_change_runs_only_the_changed_test() -> None:
    path = "tests/math/code_theory/test_exact_code_enumeration.py"
    plan = _plan([path])

    assert plan.run_math is True
    assert plan.math_tests == (path,)
    assert plan.run_catalog is False


def test_shared_runtime_path_falls_back_to_full_math_and_public_contracts() -> None:
    plan = _plan(["src/jacobian/canonical.py"])

    assert plan.run_math is True
    assert plan.math_tests == ()
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
    assert plan.run_catalog is False
    assert plan.run_catalog_examples is False


def test_merge_group_always_owns_full_math_and_public_contracts() -> None:
    plan = _plan(["docs/reference/testing-strategy.md"], event="merge_group")

    assert plan.run_math is True
    assert plan.math_tests == ()
    assert plan.run_catalog is True
    assert plan.run_catalog_examples is True


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
