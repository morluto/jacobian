"""Shared helpers and case data for exact-domain checker component tests.

Imports the checker callables and builds domain-split case tuples so that
individual domain test modules can reference only their own cases while the
generic attack module can work over the full ``_CASES`` sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.support.artifacts import artifact_uri as _uri
from tests.support.artifacts import canonical_digest as _digest
from tests.support.rationals import rational_payload as _q

from jacobian_checkers.exact_domain_operations import (
    check_integer_powerful_number,
    check_integer_prime_factorization,
    check_matrix_characteristic_polynomial,
    check_matrix_nullspace,
    check_matrix_product,
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
    check_graph_minimum_spanning_tree,
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


_POLY_CASES: tuple[
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
)

_MATRIX_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
    (
        check_matrix_product,
        _request(
            "matrix.multiply.compute",
            "matrix.product.flint-replay",
            {
                "left": _qq([[1, 2, 0], [0, 1, 1]]),
                "right": _qq([[1, 0], [0, 1], [1, 1]]),
            },
            {
                "product": _qq([[1, 2], [1, 2]]),
                "left_rows": 2,
                "inner_dimension": 3,
                "right_columns": 2,
                "convention": "STANDARD_ROW_BY_COLUMN_PRODUCT_OVER_QQ",
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
)

_NUMBER_THEORY_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
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
)

_GRAPH_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = (
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
        check_graph_minimum_spanning_tree,
        _request(
            "graph.spanning_tree.minimum.compute",
            "graph.minimum-spanning-tree.cycle-certificate-v1",
            {
                "graph": {
                    "weighted_graph_schema_version": "1",
                    "vertices": ["a", "b", "c"],
                    "edges": [
                        {
                            "endpoints": ["a", "b"],
                            "weight": _q(1),
                        },
                        {
                            "endpoints": ["b", "c"],
                            "weight": _q(2),
                        },
                        {
                            "endpoints": ["a", "c"],
                            "weight": _q(4),
                        },
                    ],
                }
            },
            {
                "result_schema_version": "1",
                "status": "EXACT",
                "vertices": ["a", "b", "c"],
                "order": 3,
                "connected": True,
                "component_count": 1,
                "components": [["a", "b", "c"]],
                "tree_edges": [
                    {
                        "endpoints": ["a", "b"],
                        "weight": _q(1),
                    },
                    {
                        "endpoints": ["b", "c"],
                        "weight": _q(2),
                    },
                ],
                "total_weight": _q(3),
                "optimality_certificate": {
                    "certificate_schema_version": "1",
                    "method": "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING",
                    "checks": [
                        {
                            "non_tree_edge": ["a", "c"],
                            "edge_weight": _q(4),
                            "tree_path_vertices": ["a", "b", "c"],
                            "maximum_tree_path_weight": _q(2),
                            "condition": ("EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT"),
                        }
                    ],
                    "required_checks": [
                        "SOURCE_CONNECTIVITY",
                        "TREE_SPANNING_ACYCLIC",
                        "TOTAL_WEIGHT_EXACT",
                        "ALL_NON_TREE_EDGES_COVERED",
                        "CYCLE_NON_IMPROVEMENT",
                    ],
                },
                "convention": (
                    "MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE"
                ),
                "completion": "COMPLETE",
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

_CASES: tuple[
    tuple[Callable[[dict[str, Any]], dict[str, Any]], dict[str, Any]], ...
] = _POLY_CASES + _MATRIX_CASES + _NUMBER_THEORY_CASES + _GRAPH_CASES
