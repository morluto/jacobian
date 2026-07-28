from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest
from tests.helpers.artifacts import artifact_uri as _uri
from tests.helpers.artifacts import canonical_digest as _digest
from tests.helpers.rationals import rational_payload as _q

import jacobian_checkers.exact_domain_operations as checker_module
from jacobian_checkers.exact_domain_operations import (
    check_matrix_characteristic_polynomial,
    check_matrix_nullspace,
    check_matrix_rref,
    check_matrix_smith_normal_form,
    check_polynomial_discriminant,
    check_polynomial_gcd,
    check_polynomial_resultant,
    check_polynomial_square_free,
)
from jacobian_checkers.graph_exact_operations import (
    check_graph_induced_tree_maximum,
    check_graph_maximum_matching,
)


def _poly(*coefficients_ascending: int) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": ["x"],
        "polynomial": {
            "terms": [
                {"coefficient": _q(coefficient), "exponents": [exponent]}
                for exponent, coefficient in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if coefficient
            ]
        },
    }


def _rational_poly(
    *coefficients_ascending: tuple[int, int],
    variable: str = "x",
) -> dict[str, Any]:
    return {
        "polynomial_schema_version": "1",
        "domain": "QQ",
        "variables": [variable],
        "polynomial": {
            "terms": [
                {
                    "coefficient": _q(numerator, denominator),
                    "exponents": [exponent],
                }
                for exponent, (numerator, denominator) in reversed(
                    tuple(enumerate(coefficients_ascending))
                )
                if numerator
            ]
        },
    }


def _qq(entries: list[list[int]]) -> dict[str, Any]:
    return {"domain": "QQ", "entries": [[_q(item) for item in row] for row in entries]}


def _zz(entries: list[list[int]]) -> dict[str, Any]:
    return {"domain": "ZZ", "entries": [[str(item) for item in row] for row in entries]}


def _artifact(
    character: str,
    payload: dict[str, Any],
    *,
    semantics: str,
    parents: list[str],
) -> dict[str, Any]:
    return {
        "artifact_uri": _uri(character),
        "object_digest": "sha256:" + character * 64,
        "payload_digest": _digest(payload),
        "schema_uri": _uri(chr(ord(character) + 1)),
        "semantics_uri": semantics,
        "parents": parents,
        "payload": payload,
    }


def _request(
    operation_id: str,
    witness_format: str,
    source: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    semantics = _uri("e")
    semantics_artifact = _artifact(
        "e",
        {"kind": "semantics"},
        semantics=_uri("0"),
        parents=[],
    )
    semantics_artifact["object_digest"] = "sha256:" + "8" * 64
    claim = _artifact("1", source, semantics=semantics, parents=[])
    candidate = _artifact(
        "3", result, semantics=semantics, parents=[claim["artifact_uri"]]
    )
    bindings = {
        "claim_digest": claim["object_digest"],
        "semantics_digest": "sha256:" + "8" * 64,
        "candidate_digest": candidate["object_digest"],
        "scope_digest": None,
        "encoding_digest": None,
    }
    witness_payload = {
        "evidence_schema_version": "1",
        "witness_format": witness_format,
        "format_version": "1",
        "role": "SUPPORTS_CLAIM",
        "bindings": bindings,
        "payload": {
            "operation_id": operation_id,
            "input_uri": claim["artifact_uri"],
            "result_uri": candidate["artifact_uri"],
        },
    }
    witness = _artifact(
        "5",
        witness_payload,
        semantics=semantics,
        parents=[claim["artifact_uri"], candidate["artifact_uri"]],
    )
    return {
        "request_version": "1",
        "claim": claim,
        "candidate": candidate,
        "semantics": semantics_artifact,
        "scope": None,
        "witness": witness,
        "expected_bindings": bindings,
    }


_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_polynomial_gcd,
        _request(
            "polynomial.compute.gcd",
            "polynomial.gcd.flint-replay",
            {"left": _poly(-1, 0, 1), "right": _poly(-1, 1)},
            {
                "gcd": _poly(-1, 1),
                "bezout": {
                    "left_multiplier": _poly(),
                    "right_multiplier": _poly(1),
                },
                "normalization": "MONIC",
            },
        ),
    ),
    (
        check_polynomial_resultant,
        _request(
            "polynomial.compute.resultant",
            "polynomial.resultant.flint-replay",
            {
                "left": _poly(1, 0, 1),
                "right": _poly(-1, 1),
                "elimination_variable": "x",
            },
            {
                "elimination_variable": "x",
                "resultant": {"kind": "SCALAR", "value": _q(2)},
                "convention": "SYLVESTER_DETERMINANT",
            },
        ),
    ),
    (
        check_polynomial_discriminant,
        _request(
            "polynomial.compute.discriminant",
            "polynomial.discriminant.flint-replay",
            {"polynomial": _poly(-1, 0, 1), "variable": "x"},
            {
                "variable": "x",
                "discriminant": {"kind": "SCALAR", "value": _q(4)},
                "convention": "STANDARD_UNIVARIATE",
            },
        ),
    ),
    (
        check_polynomial_square_free,
        _request(
            "polynomial.compute.square_free_decomposition",
            "polynomial.square-free.flint-replay",
            {"polynomial": _poly(-1, 1, 1, -1)},
            {
                "coefficient": _q(-1),
                "factors": [
                    {"factor": _poly(1, 1), "multiplicity": 1},
                    {"factor": _poly(-1, 1), "multiplicity": 2},
                ],
                "reconstructed": _poly(-1, 1, 1, -1),
                "normalization": "MONIC_FACTORS",
            },
        ),
    ),
    (
        check_matrix_rref,
        _request(
            "matrix.normal_form.rref.compute",
            "matrix.rref.flint-replay",
            {"matrix": _qq([[1, 2, 3], [2, 4, 6]])},
            {
                "reduced_matrix": _qq([[1, 2, 3], [0, 0, 0]]),
                "rank": 1,
                "pivot_columns": [0],
                "free_columns": [1, 2],
                "convention": "UNIQUE_RREF_OVER_QQ",
            },
        ),
    ),
    (
        check_matrix_nullspace,
        _request(
            "matrix.nullspace.compute",
            "matrix.nullspace.flint-replay",
            {"matrix": _qq([[1, 2, 3], [2, 4, 6]])},
            {
                "ambient_dimension": 3,
                "nullity": 2,
                "basis_vectors": [[_q(-2), _q(1), _q(0)], [_q(-3), _q(0), _q(1)]],
                "free_columns": [1, 2],
                "convention": "RREF_FUNDAMENTAL_BASIS",
            },
        ),
    ),
    (
        check_matrix_characteristic_polynomial,
        _request(
            "matrix.characteristic_polynomial.compute",
            "matrix.characteristic-polynomial.flint-replay",
            {"matrix": _qq([[1, 2], [3, 4]])},
            {
                "variable": "lambda",
                "degree": 2,
                "coefficients_descending": [_q(1), _q(-5), _q(-2)],
                "monic": True,
                "convention": "DET_LAMBDA_I_MINUS_A",
            },
        ),
    ),
    (
        check_matrix_smith_normal_form,
        _request(
            "matrix.normal_form.smith.compute",
            "matrix.smith-normal-form.flint-replay",
            {"matrix": _zz([[2, 4], [6, 8]])},
            {
                "normal_form": _zz([[2, 0], [0, 4]]),
                "rank": 2,
                "invariant_factors": ["2", "4"],
                "transformation_available": False,
                "convention": "POSITIVE_DIVISIBILITY_DIAGONAL",
            },
        ),
    ),
    (
        check_graph_induced_tree_maximum,
        _request(
            "graph.induced_tree.maximum.compute",
            "graph.induced-tree.maximum.exhaustive-replay",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"], ["a", "d"]],
                }
            },
            {
                "status": "EXACT",
                "convention": "NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO",
                "order": 4,
                "optimum_value": 3,
                "incumbent_value": 3,
                "lower_bound": 3,
                "upper_bound": 3,
                "witness_vertices": ["a", "b", "c"],
            },
        ),
    ),
    (
        check_graph_maximum_matching,
        _request(
            "graph.invariant.maximum_matching.compute",
            "graph.maximum-matching.tutte-berge-v1",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["center", "x", "y", "z"],
                    "edges": [
                        ["center", "x"],
                        ["center", "y"],
                        ["center", "z"],
                    ],
                }
            },
            {
                "maximum_matching_cardinality": 1,
                "witness_edges": [["center", "x"]],
                "certificate": {
                    "certificate_schema_version": "1",
                    "kind": "TUTTE_BERGE_BARRIER",
                    "barrier_vertices": ["center"],
                    "odd_component_count": 3,
                    "upper_bound": 1,
                },
            },
        ),
    ),
)


def _mutate_numeric_leaf(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) == {"num", "den"}:
            mutated = int(value["num"]) + 1
            if mutated == 0:
                mutated = 1
            value["num"] = str(mutated)
            return True
        for key, item in value.items():
            if _mutate_numeric_leaf(item):
                return True
            if type(item) is int:
                value[key] = item + 1
                return True
        return False
    if isinstance(value, list):
        for index, item in enumerate(value):
            if _mutate_numeric_leaf(item):
                return True
            if isinstance(item, str) and item.lstrip("-").isdigit():
                mutated = int(item) + 1
                if mutated == 0:
                    mutated = 1
                value[index] = str(mutated)
                return True
            if type(item) is int:
                value[index] = item + 1
                return True
    return False


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


def test_graph_checker_reports_its_actual_exhaustive_replay_method() -> None:
    checker, checker_request = _CASES[-2]

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["detail"] == (
        "independent finite-subset exhaustive replay accepted "
        "graph.induced_tree.maximum.compute"
    )
    assert "FLINT" not in decision["detail"]


def test_graph_checker_does_not_require_the_unrelated_flint_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker, checker_request = _CASES[-2]
    monkeypatch.setattr(checker_module.flint, "__version__", "unexpected")

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


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


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(
            {
                "maximum_matching_cardinality": 0,
                "witness_edges": [],
                "certificate": {
                    **result["certificate"],
                    "upper_bound": 0,
                },
            }
        ),
        lambda result: result.update(witness_edges=[["x", "y"]]),
        lambda result: result["certificate"].update(barrier_vertices=["outside"]),
        lambda result: result["certificate"].update(odd_component_count=1),
        lambda result: result["certificate"].update(upper_bound=0),
    ),
)
def test_maximum_matching_checker_rejects_false_or_rebound_certificates(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker, checker_request = _CASES[-1]
    adversarial = copy.deepcopy(checker_request)
    mutate(adversarial["candidate"]["payload"])
    adversarial["candidate"]["payload_digest"] = _digest(
        adversarial["candidate"]["payload"]
    )

    decision = checker(adversarial)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_maximum_matching_checker_reports_tutte_berge_replay() -> None:
    checker, checker_request = _CASES[-1]

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["detail"] == (
        "independent Tutte-Berge barrier replay accepted "
        "graph.invariant.maximum_matching.compute"
    )
    assert decision["arithmetic"] == "EXACT_INTEGER"


def test_square_free_checker_normalizes_flint_factors_to_monic_contract() -> None:
    checker_request = _request(
        "polynomial.compute.square_free_decomposition",
        "polynomial.square-free.flint-replay",
        {"polynomial": _poly(2, 10, 16, 8)},
        {
            "coefficient": _q(8),
            "factors": [
                {"factor": _poly(1, 1), "multiplicity": 1},
                {
                    "factor": _rational_poly((1, 2), (1, 1)),
                    "multiplicity": 2,
                },
            ],
            "reconstructed": _poly(2, 10, 16, 8),
            "normalization": "MONIC_FACTORS",
        },
    )

    assert check_polynomial_square_free(checker_request)["accepted"] is True


def test_polynomial_checker_rejects_variable_renaming() -> None:
    checker, checker_request = _CASES[0]
    mutated = copy.deepcopy(checker_request)
    mutated["candidate"]["payload"]["gcd"]["variables"] = ["y"]
    mutated["candidate"]["payload_digest"] = _digest(mutated["candidate"]["payload"])

    decision = checker(mutated)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_polynomial_checker_accepts_consistent_nondefault_variable_name() -> None:
    checker, checker_request = _CASES[0]
    renamed = copy.deepcopy(checker_request)
    payloads = (
        renamed["claim"]["payload"]["left"],
        renamed["claim"]["payload"]["right"],
        renamed["candidate"]["payload"]["gcd"],
        renamed["candidate"]["payload"]["bezout"]["left_multiplier"],
        renamed["candidate"]["payload"]["bezout"]["right_multiplier"],
    )
    for polynomial in payloads:
        polynomial["variables"] = ["t"]
    renamed["claim"]["payload_digest"] = _digest(renamed["claim"]["payload"])
    renamed["candidate"]["payload_digest"] = _digest(renamed["candidate"]["payload"])

    assert checker(renamed)["accepted"] is True
