from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest
from tests.component.checkers.exact_domain_checker_support import (
    _CASES,
    _mutate_numeric_leaf,
)
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest

import jacobian_checkers.exact_domain_operations as checker_module
from jacobian_checkers.graph_exact_operations import check_graph_induced_tree_maximum


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_accepts_independent_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    assert checker(checker_request)["accepted"] is True


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_rejects_candidate_mutation(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(checker_request)
    assert _mutate_numeric_leaf(mutated["candidate"]["payload"])
    mutated["candidate"]["payload_digest"] = _digest(mutated["candidate"]["payload"])

    decision = checker(mutated)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_rejects_semantics_substitution(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(checker_request)
    mutated["candidate"]["semantics_uri"] = _uri("9")

    assert checker(mutated)["accepted"] is False


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_rejects_claim_binding_substitution(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(checker_request)
    mutated["claim"]["object_digest"] = "sha256:" + "9" * 64

    assert checker(mutated)["accepted"] is False


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_rejects_forged_semantics_digest(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(checker_request)
    forged = "sha256:" + "9" * 64
    mutated["expected_bindings"]["semantics_digest"] = forged
    mutated["witness"]["payload"]["bindings"]["semantics_digest"] = forged
    mutated["witness"]["payload_digest"] = _digest(mutated["witness"]["payload"])

    assert checker(mutated)["accepted"] is False


def test_exact_domain_checker_rejects_changed_flint_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker, checker_request = _CASES[0]
    monkeypatch.setattr(checker_module.flint, "__version__", "unexpected")

    decision = checker(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"
    assert "runtime is unavailable" in decision["detail"]


def test_graph_checker_does_not_require_the_unrelated_flint_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker, checker_request = next(
        case for case in _CASES if case[0] is check_graph_induced_tree_maximum
    )
    monkeypatch.setattr(checker_module.flint, "__version__", "unexpected")

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


def test_exact_domain_checker_import_boundary_excludes_producer_modules() -> None:
    script = """
import builtins
real_import = builtins.__import__
blocked = {"jacobian", "sympy"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise RuntimeError(f"forbidden checker import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import jacobian_checkers.exact_domain_operations
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_graph_checker_import_boundary_excludes_producer_and_flint_modules() -> None:
    script = """
import builtins
real_import = builtins.__import__
blocked = {"flint", "networkx", "sympy", "jacobian"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise RuntimeError(f"forbidden checker import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import jacobian_checkers.graph_exact_operations
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
