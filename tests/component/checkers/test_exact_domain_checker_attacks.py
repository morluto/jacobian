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
    _request,
)
from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest

import jacobian_checkers.exact_domain_operations as checker_module
from jacobian.contracts.exact_domain_verification import inline_exact_value_digest
from jacobian_checkers.exact_domain_operations import (
    check_matrix_determinant,
    check_matrix_rank,
)
from jacobian_checkers.graph_exact_operations import check_graph_induced_tree_maximum


@pytest.mark.parametrize(("checker", "checker_request"), _CASES)
def test_exact_domain_checker_accepts_independent_replay(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    checker_request: dict[str, Any],
) -> None:
    assert checker(checker_request)["accepted"] is True


def test_matrix_determinant_checker_accepts_supported_large_canonical_result() -> None:
    diagonal_entry = "1" + "0" * 255
    zero = {"num": "0", "den": "1"}
    source = {
        "matrix": {
            "matrix_schema_version": "1",
            "domain": "QQ",
            "entries": [
                [
                    ({"num": diagonal_entry, "den": "1"} if row == column else zero)
                    for column in range(32)
                ]
                for row in range(32)
            ],
        }
    }
    request = _request(
        "matrix.determinant.compute",
        "matrix.determinant.flint-replay",
        source,
        {
            "determinant": {"num": "1" + "0" * (255 * 32), "den": "1"},
            "method": "FRACTION_FREE_BAREISS",
        },
    )

    assert check_matrix_determinant(request)["accepted"] is True


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


def test_exact_domain_checker_rejects_rebound_inline_candidate() -> None:
    """Version-two replay binds inline values without assigning artifact URIs."""

    _checker, stored = next(case for case in _CASES if case[0] is check_matrix_rank)
    claim = stored["claim"]
    candidate = stored["candidate"]
    semantics = stored["semantics"]
    request: dict[str, Any] = {
        "request_version": "2",
        "claim": {
            "schema_uri": claim["schema_uri"],
            "semantics_uri": claim["semantics_uri"],
            "payload": copy.deepcopy(claim["payload"]),
        },
        "candidate": {
            "schema_uri": candidate["schema_uri"],
            "semantics_uri": candidate["semantics_uri"],
            "payload": copy.deepcopy(candidate["payload"]),
        },
        "semantics": copy.deepcopy(semantics),
        "scope": None,
    }
    request["expected_bindings"] = {
        "claim_digest": inline_exact_value_digest(
            schema_uri=request["claim"]["schema_uri"],
            semantics_uri=request["claim"]["semantics_uri"],
            payload=request["claim"]["payload"],
        ),
        "semantics_digest": semantics["object_digest"],
        "candidate_digest": inline_exact_value_digest(
            schema_uri=request["candidate"]["schema_uri"],
            semantics_uri=request["candidate"]["semantics_uri"],
            payload=request["candidate"]["payload"],
        ),
        "scope_digest": None,
        "encoding_digest": None,
    }

    assert check_matrix_rank(request)["accepted"] is True
    request["candidate"]["payload"]["rank"] = 1
    rejected = check_matrix_rank(request)
    assert rejected["accepted"] is False
    assert rejected["conclusion"] == "UNKNOWN"


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
