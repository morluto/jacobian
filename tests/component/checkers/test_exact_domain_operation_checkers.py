from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import Callable
from itertools import product
from typing import Any

import pytest
from tests.support.rationals import rational_payload as _q
from tests.unit.contracts.artifacts import artifact_uri as _uri
from tests.unit.contracts.artifacts import canonical_digest as _digest

import jacobian_checkers.exact_domain_operations as checker_module
from jacobian_checkers.exact_domain_operations import (
    check_integer_powerful_number,
    check_integer_prime_factorization,
    check_matrix_characteristic_polynomial,
    check_matrix_nullspace,
    check_matrix_rref,
    check_matrix_smith_normal_form,
    check_modular_polynomial_residue_image,
    check_polynomial_discriminant,
    check_polynomial_gcd,
    check_polynomial_resultant,
    check_polynomial_square_free,
)
from jacobian_checkers.graph_exact_operations import (
    check_graph_diameter,
    check_graph_distance_matrix,
    check_graph_induced_tree_maximum,
    check_graph_maximum_matching,
    check_graph_radius,
    check_graph_symmetry_generator_orbits,
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
                "rank": 1,
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
        check_integer_prime_factorization,
        _request(
            "integer.compute.prime_factorization",
            "integer.prime-factorization.flint-replay",
            {
                "value": "-360",
                "resource_budget": {"wall_seconds": 5},
            },
            {
                "factors": [
                    {"prime": "2", "power": 3},
                    {"prime": "3", "power": 2},
                    {"prime": "5", "power": 1},
                ]
            },
        ),
    ),
    (
        check_integer_powerful_number,
        _request(
            "integer.decide.powerful",
            "integer.powerful.flint-replay",
            {
                "value": "72",
                "resource_budget": {"wall_seconds": 5},
            },
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": True,
                "factors": [
                    {"prime": "2", "power": 3},
                    {"prime": "3", "power": 2},
                ],
                "violating_primes": [],
            },
        ),
    ),
    (
        check_modular_polynomial_residue_image,
        _request(
            "modular.polynomial_residue_image.compute",
            "modular.polynomial-residue-image.flint-replay",
            {
                "modulus": 7,
                "variables": [
                    {"name": "x", "residues": [0, 1, 2, 3, 4, 5, 6]},
                ],
                "terms": [{"coefficient": "4", "exponents": [3]}],
            },
            {
                "semantics_version": "modular-polynomial-residue-image.v1",
                "modulus": 7,
                "variable_order": ["x"],
                "domains": [[0, 1, 2, 3, 4, 5, 6]],
                "normalized_terms": [{"coefficient": 4, "exponents": [3]}],
                "enumeration_scope": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
                "total_assignments": 7,
                "image": [0, 3, 4],
                "residue_counts": [
                    {"residue": 0, "count": 1},
                    {"residue": 3, "count": 3},
                    {"residue": 4, "count": 3},
                ],
                "witnesses": [
                    {"residue": 0, "assignment": [0]},
                    {"residue": 3, "assignment": [3]},
                    {"residue": 4, "assignment": [1]},
                ],
                "table": [
                    {"assignment": [0], "residue": 0},
                    {"assignment": [1], "residue": 4},
                    {"assignment": [2], "residue": 4},
                    {"assignment": [3], "residue": 3},
                    {"assignment": [4], "residue": 4},
                    {"assignment": [5], "residue": 3},
                    {"assignment": [6], "residue": 3},
                ],
            },
        ),
    ),
    (
        check_graph_diameter,
        _request(
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                }
            },
            {
                "status": "COMPUTED",
                "diameter": 3,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
    ),
    (
        check_graph_radius,
        _request(
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                }
            },
            {
                "status": "COMPUTED",
                "radius": 2,
                "connected": True,
                "exactness": "EXACT",
                "detail": None,
            },
        ),
    ),
    (
        check_graph_distance_matrix,
        _request(
            "graph.distance_matrix.compute",
            "graph.distance-matrix.all-sources-bfs-v1",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["c", "a", "b"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            },
            {
                "semantics_version": ("unweighted-shortest-path-distance-matrix.v1"),
                "vertex_ordering": "LEXICOGRAPHIC_ASCENDING",
                "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
                "unreachable_representation": "JSON_NULL",
                "vertices": ["a", "b", "c"],
                "distances": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                "connected": True,
            },
        ),
    ),
    (
        check_graph_symmetry_generator_orbits,
        _request(
            "graph.symmetry.generator_orbits.compute",
            "graph.symmetry.generator-orbits.stdlib-replay",
            {
                "graph": {
                    "graph_schema_version": "1",
                    "vertices": ["a", "b", "c", "d"],
                    "edges": [
                        ["a", "b"],
                        ["a", "d"],
                        ["b", "c"],
                        ["c", "d"],
                    ],
                },
                "generators": [
                    {
                        "generator_id": "quarter_turn",
                        "mapping": {
                            "a": "b",
                            "b": "c",
                            "c": "d",
                            "d": "a",
                        },
                    }
                ],
                "vertex_colors": [],
                "edge_colors": [],
                "action": "DECLARED_AUTOMORPHISM_GENERATORS",
            },
            {
                "vertices": ["a", "b", "c", "d"],
                "edges": [
                    ["a", "b"],
                    ["a", "d"],
                    ["b", "c"],
                    ["c", "d"],
                ],
                "generator_ids": ["quarter_turn"],
                "generator_count": 1,
                "vertex_orbits": [
                    {
                        "orbit_index": 0,
                        "representative": "a",
                        "members": ["a", "b", "c", "d"],
                    }
                ],
                "edge_orbits": [
                    {
                        "orbit_index": 0,
                        "representative": ["a", "b"],
                        "members": [
                            ["a", "b"],
                            ["a", "d"],
                            ["b", "c"],
                            ["c", "d"],
                        ],
                    }
                ],
                "vertex_orbit_count": 1,
                "edge_orbit_count": 1,
                "vertex_color_mode": "UNCOLORED",
                "edge_color_mode": "UNCOLORED",
                "action": "DECLARED_GENERATED_SUBGROUP",
                "generator_validation": (
                    "ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"
                ),
                "orbit_completeness": "COMPLETE_FOR_DECLARED_GENERATORS",
                "automorphism_group_completeness": (
                    "FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"
                ),
                "exactness": "EXACT_COMBINATORIAL",
                "determinism": "DETERMINISTIC",
                "backend": "jacobian-stdlib",
                "backend_version": "1",
                "verification": "UNVERIFIED",
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


def test_matrix_nullspace_checker_rejects_wrong_rank() -> None:
    checker_request = copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _CASES
            if checker is check_matrix_nullspace
        )
    )
    checker_request["candidate"]["payload"]["rank"] = 2
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_matrix_nullspace(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def _modular_checker_request() -> dict[str, Any]:
    return copy.deepcopy(
        next(
            checker_request
            for checker, checker_request in _CASES
            if checker is check_modular_polynomial_residue_image
        )
    )


def test_modular_residue_checker_reports_exhaustive_integer_replay() -> None:
    decision = check_modular_polynomial_residue_image(_modular_checker_request())

    assert decision["accepted"] is True
    assert decision["arithmetic"] == "EXACT_INTEGER"
    assert decision["method"] == "EXHAUSTIVE_FINITE"
    assert decision["coverage"] == "EXHAUSTIVE"


def test_modular_residue_checker_accepts_exact_assignment_bound() -> None:
    assignments = [list(values) for values in product(range(16), repeat=3)]
    residues = [
        assignment[0] * assignment[1] * assignment[2] % 16 for assignment in assignments
    ]
    image = sorted(set(residues))
    first_assignments: dict[int, list[int]] = {}
    for assignment, residue in zip(assignments, residues, strict=True):
        first_assignments.setdefault(residue, assignment)
    checker_request = _request(
        "modular.polynomial_residue_image.compute",
        "modular.polynomial-residue-image.flint-replay",
        {
            "modulus": 16,
            "variables": [
                {"name": "x", "residues": list(range(16))},
                {"name": "y", "residues": list(range(16))},
                {"name": "z", "residues": list(range(16))},
            ],
            "terms": [{"coefficient": "1", "exponents": [1, 1, 1]}],
        },
        {
            "semantics_version": "modular-polynomial-residue-image.v1",
            "modulus": 16,
            "variable_order": ["x", "y", "z"],
            "domains": [list(range(16)), list(range(16)), list(range(16))],
            "normalized_terms": [{"coefficient": 1, "exponents": [1, 1, 1]}],
            "enumeration_scope": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
            "total_assignments": 4_096,
            "image": image,
            "residue_counts": [
                {"residue": residue, "count": residues.count(residue)}
                for residue in image
            ],
            "witnesses": [
                {
                    "residue": residue,
                    "assignment": first_assignments[residue],
                }
                for residue in image
            ],
            "table": [
                {"assignment": assignment, "residue": residue}
                for assignment, residue in zip(assignments, residues, strict=True)
            ],
        },
    )

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is True
    assert decision["method"] == "EXHAUSTIVE_FINITE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result["table"].pop(),
        lambda result: result["table"][1].update(residue=3),
        lambda result: result["residue_counts"][1].update(count=2),
        lambda result: result["witnesses"][2].update(assignment=[2]),
        lambda result: result.update(variable_order=["y"]),
        lambda result: result["normalized_terms"][0].update(coefficient=3),
    ),
    ids=(
        "partial-table",
        "wrong-evaluation",
        "wrong-multiplicity",
        "nonfirst-witness",
        "wrong-variable-order",
        "wrong-normalization",
    ),
)
def test_modular_residue_checker_rejects_one_obligation_mutation(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _modular_checker_request()
    mutate(checker_request["candidate"]["payload"])
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_modular_residue_checker_rejects_wrong_bound_scope() -> None:
    checker_request = _modular_checker_request()
    checker_request["scope"] = {
        "assignment_count": 7,
        "enumeration": "COMPLETE_DECLARED_CARTESIAN_PRODUCT",
    }

    decision = check_modular_polynomial_residue_image(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


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
    ("value", "factors"),
    (
        ("1", []),
        ("-1", []),
        ("2", [{"prime": "2", "power": 1}]),
        (
            "-360",
            [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
                {"prime": "5", "power": 1},
            ],
        ),
    ),
)
def test_prime_factorization_checker_accepts_exact_boundaries(
    value: str,
    factors: list[dict[str, object]],
) -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        {"factors": factors},
    )

    decision = check_integer_prime_factorization(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda factors: factors.__setitem__(
            slice(None),
            [{"prime": "6", "power": 1}, {"prime": "60", "power": 1}],
        ),
        lambda factors: factors.pop(),
        lambda factors: factors.append({"prime": "5", "power": 1}),
        lambda factors: factors[0].update(power=2),
        lambda factors: factors.reverse(),
        lambda factors: factors[0].update(prime="-2"),
    ),
    ids=(
        "composite-bases",
        "missing-factor",
        "duplicate-base",
        "wrong-power",
        "noncanonical-order",
        "negative-base",
    ),
)
def test_prime_factorization_checker_rejects_false_or_noncanonical_factors(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": "360", "resource_budget": {"wall_seconds": 5}},
        {
            "factors": [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
                {"prime": "5", "power": 1},
            ]
        },
    )
    mutate(checker_request["candidate"]["payload"]["factors"])
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = check_integer_prime_factorization(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


def test_prime_factorization_checker_rejects_zero_source() -> None:
    checker_request = _request(
        "integer.compute.prime_factorization",
        "integer.prime-factorization.flint-replay",
        {"value": "0", "resource_budget": {"wall_seconds": 5}},
        {"factors": []},
    )

    assert check_integer_prime_factorization(checker_request)["accepted"] is False


@pytest.mark.parametrize(
    ("value", "result"),
    (
        (
            "1",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": True,
                "factors": [],
                "violating_primes": [],
            },
        ),
        (
            "72",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": True,
                "factors": [
                    {"prime": "2", "power": 3},
                    {"prime": "3", "power": 2},
                ],
                "violating_primes": [],
            },
        ),
        (
            "12",
            {
                "semantics_version": (
                    "powerful-number.prime-exponents-at-least-two.v1"
                ),
                "is_powerful": False,
                "factors": [
                    {"prime": "2", "power": 2},
                    {"prime": "3", "power": 1},
                ],
                "violating_primes": ["3"],
            },
        ),
    ),
)
def test_powerful_number_checker_accepts_exact_decisions(
    value: str,
    result: dict[str, object],
) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        result,
    )

    decision = check_integer_powerful_number(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(is_powerful=False),
        lambda result: result["violating_primes"].append("2"),
        lambda result: result["factors"].pop(),
        lambda result: result.update(
            is_powerful=False,
            factors=[
                {"prime": "2", "power": 1},
                {"prime": "6", "power": 2},
            ],
            violating_primes=["2"],
        ),
        lambda result: result.update(semantics_version="powerful-number.v2"),
        lambda result: result["factors"].reverse(),
    ),
    ids=(
        "wrong-decision",
        "wrong-violations",
        "incomplete-factorization",
        "composite-factor-base",
        "wrong-semantics",
        "noncanonical-factor-order",
    ),
)
def test_powerful_number_checker_rejects_false_or_rebound_results(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": "72", "resource_budget": {"wall_seconds": 5}},
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [
                {"prime": "2", "power": 3},
                {"prime": "3", "power": 2},
            ],
            "violating_primes": [],
        },
    )
    result = checker_request["candidate"]["payload"]
    mutate(result)
    checker_request["candidate"]["payload_digest"] = _digest(result)

    decision = check_integer_powerful_number(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize("value", ("0", "-72"))
def test_powerful_number_checker_rejects_nonpositive_source(value: str) -> None:
    checker_request = _request(
        "integer.decide.powerful",
        "integer.powerful.flint-replay",
        {"value": value, "resource_budget": {"wall_seconds": 5}},
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [],
            "violating_primes": [],
        },
    )

    assert check_integer_powerful_number(checker_request)["accepted"] is False


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "result"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            {
                "status": "NOT_APPLICABLE",
                "diameter": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "diameter requires a nonempty connected graph",
            },
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            {
                "status": "NOT_APPLICABLE",
                "radius": None,
                "connected": False,
                "exactness": "NOT_APPLICABLE",
                "detail": "radius requires a nonempty connected graph",
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "graph",
    (
        {
            "graph_schema_version": "1",
            "vertices": [],
            "edges": [],
        },
        {
            "graph_schema_version": "1",
            "vertices": ["a", "b", "c"],
            "edges": [["a", "b"]],
        },
    ),
    ids=("empty", "disconnected"),
)
def test_graph_metric_checker_accepts_exact_inapplicable_boundary(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    result: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {"graph": graph},
        result,
    )

    decision = checker(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "field"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            "diameter",
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            "radius",
        ),
    ),
)
def test_graph_metric_checker_accepts_singleton_zero(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    field: str,
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": ["only"],
                "edges": [],
            }
        },
        {
            "status": "COMPUTED",
            field: 0,
            "connected": True,
            "exactness": "EXACT",
            "detail": None,
        },
    )

    assert checker(checker_request)["accepted"] is True


def _distance_matrix_checker_request(
    *,
    vertices: list[str],
    edges: list[list[str]],
    result_vertices: list[str],
    distances: list[list[int | None]],
    connected: bool,
) -> dict[str, Any]:
    return _request(
        "graph.distance_matrix.compute",
        "graph.distance-matrix.all-sources-bfs-v1",
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": vertices,
                "edges": edges,
            }
        },
        {
            "semantics_version": "unweighted-shortest-path-distance-matrix.v1",
            "vertex_ordering": "LEXICOGRAPHIC_ASCENDING",
            "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
            "unreachable_representation": "JSON_NULL",
            "vertices": result_vertices,
            "distances": distances,
            "connected": connected,
        },
    )


@pytest.mark.parametrize(
    "checker_request",
    (
        _distance_matrix_checker_request(
            vertices=[],
            edges=[],
            result_vertices=[],
            distances=[],
            connected=False,
        ),
        _distance_matrix_checker_request(
            vertices=["only"],
            edges=[],
            result_vertices=["only"],
            distances=[[0]],
            connected=True,
        ),
        _distance_matrix_checker_request(
            vertices=["c", "a", "b"],
            edges=[["a", "b"]],
            result_vertices=["a", "b", "c"],
            distances=[[0, 1, None], [1, 0, None], [None, None, 0]],
            connected=False,
        ),
    ),
    ids=("empty", "singleton", "disconnected"),
)
def test_distance_matrix_checker_accepts_exact_boundary_claims(
    checker_request: dict[str, Any],
) -> None:
    decision = check_graph_distance_matrix(checker_request)

    assert decision["accepted"] is True
    assert decision["conclusion"] == "TRUE"
    assert decision["method"] == "EXHAUSTIVE_FINITE"
    assert decision["coverage"] == "EXHAUSTIVE"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda result: result.update(vertices=["b", "a", "c"]),
        lambda result: result.update(
            distances=[[0, 1], [1, 0]],
        ),
        lambda result: result["distances"][0].__setitem__(0, 1),
        lambda result: result["distances"][0].__setitem__(1, 0),
        lambda result: result["distances"][0].__setitem__(2, 1),
        lambda result: result["distances"][2].__setitem__(0, 1),
        lambda result: result.update(
            distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        ),
        lambda result: result.update(connected=False),
        lambda result: result.update(
            semantics_version="unweighted-shortest-path-distance-matrix.v2"
        ),
        lambda result: result.update(extra="forged"),
        lambda result: result["distances"][0].__setitem__(1, True),
    ),
    ids=(
        "wrong-order",
        "wrong-shape",
        "diagonal",
        "off-diagonal-zero",
        "asymmetric-left",
        "asymmetric-right",
        "wrong-shortest-paths",
        "wrong-connectedness",
        "wrong-semantics",
        "extra-field",
        "boolean-distance",
    ),
)
def test_distance_matrix_checker_rejects_false_certification_paths(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    checker_request = _distance_matrix_checker_request(
        vertices=["c", "a", "b"],
        edges=[["a", "b"], ["b", "c"]],
        result_vertices=["a", "b", "c"],
        distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        connected=True,
    )
    result = checker_request["candidate"]["payload"]
    mutate(result)
    checker_request["candidate"]["payload_digest"] = _digest(result)

    decision = check_graph_distance_matrix(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


@pytest.mark.parametrize(
    ("checker", "operation_id", "witness_format", "field"),
    (
        (
            check_graph_diameter,
            "graph.invariant.diameter.compute",
            "graph.diameter.all-sources-bfs-v1",
            "diameter",
        ),
        (
            check_graph_radius,
            "graph.invariant.radius.compute",
            "graph.radius.all-sources-bfs-v1",
            "radius",
        ),
    ),
)
@pytest.mark.parametrize(
    "mutation",
    (
        lambda result, field: result.update({field: 0}),
        lambda result, field: result.update(status="NOT_APPLICABLE"),
        lambda result, field: result.update(connected=False),
        lambda result, field: result.update(exactness="NOT_APPLICABLE"),
        lambda result, field: result.update(detail="forged"),
    ),
    ids=(
        "wrong-value",
        "wrong-status",
        "wrong-connectivity",
        "wrong-exactness",
        "detail",
    ),
)
def test_graph_metric_checker_rejects_forged_connected_result(
    checker: Callable[[dict[str, Any]], dict[str, Any]],
    operation_id: str,
    witness_format: str,
    field: str,
    mutation: Callable[[dict[str, Any], str], object],
) -> None:
    checker_request = _request(
        operation_id,
        witness_format,
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": ["a", "b", "c", "d"],
                "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            }
        },
        {
            "status": "COMPUTED",
            field: 3 if field == "diameter" else 2,
            "connected": True,
            "exactness": "EXACT",
            "detail": None,
        },
    )
    mutation(checker_request["candidate"]["payload"], field)
    checker_request["candidate"]["payload_digest"] = _digest(
        checker_request["candidate"]["payload"]
    )

    decision = checker(checker_request)

    assert decision["accepted"] is False
    assert decision["conclusion"] == "UNKNOWN"


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
