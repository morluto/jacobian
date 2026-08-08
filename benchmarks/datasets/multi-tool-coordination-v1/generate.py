#!/usr/bin/env python3
"""Deterministically render the multi-tool-coordination-v1 PR1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tooling.harbor_suite import verifier_bundle_checksum_bytes  # noqa: E402
from benchmarks.tooling.public_contract import (  # noqa: E402
    PublicContract,
    render_instruction,
    render_submission_schema,
)

DATASET = Path(__file__).resolve().parent
VERIFIER_TEMPLATE = DATASET / "verifier_template.py"
SUPPORT_TEMPLATE = ROOT / "benchmarks/templates/task/tests/verifier_support.py"
IMAGE = "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
GENERATOR_VERSION = "multi-tool-coordination-pilot-generator@1"
CASE_VERSION = "multi-tool-coordination-v1/pr1-pilot-1"


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def rational(value: int | Fraction) -> dict[str, str]:
    parsed = Fraction(value)
    return {"num": str(parsed.numerator), "den": str(parsed.denominator)}


def cases() -> tuple[dict[str, object], ...]:
    rp2_facets = [
        [10, 11, 12],
        [10, 12, 13],
        [10, 11, 15],
        [10, 14, 15],
        [10, 13, 14],
        [11, 12, 14],
        [11, 13, 14],
        [11, 13, 15],
        [12, 13, 15],
        [12, 14, 15],
    ]
    return (
        {
            "schema_version": "1",
            "task_id": "jacobian/coordination-graph-set-distance-01",
            "case_id": "coordination-graph-set-distance-01",
            "family": "graph-set-distance",
            "primary_domain": "graph-theory",
            "case_version": CASE_VERSION,
            "conclusion": "GRAPH_SET_DISTANCE_CERTIFIED",
            "scope": "EXACT_DECLARED_FINITE_GRAPH:coordination-graph-set-distance-01",
            "limitations": ["NO_CLAIM_BEYOND_THE_DECLARED_FINITE_GRAPH"],
            "case_note": "A three-vertex maximum-degree set has a unique distance-two extremizer.",
            "source": {
                "kind": "authored-hand-auditable-fixture",
                "derived_from": "multi-tool-coordination-pr1 graph discovery observations",
            },
            "vertices": ["a", "b", "c", "d", "e", "f", "g"],
            "edges": [
                ["a", "b"],
                ["a", "c"],
                ["a", "d"],
                ["b", "c"],
                ["b", "e"],
                ["c", "f"],
                ["d", "g"],
            ],
        },
        {
            "schema_version": "1",
            "task_id": "jacobian/coordination-cycle-lattice-01",
            "case_id": "coordination-cycle-lattice-01",
            "family": "cycle-lattice",
            "primary_domain": "geometry-topology",
            "case_version": CASE_VERSION,
            "conclusion": "INTEGRAL_H1_CERTIFIED",
            "scope": "EXACT_DECLARED_SIMPLICIAL_COMPLEX:coordination-cycle-lattice-01",
            "limitations": ["NO_PROOF_ASSISTANT_OR_GEOMETRIC_HOMEOMORPHISM_CLAIM"],
            "case_note": "Relabelled six-vertex projective-plane triangulation; any valid spanning tree and ordering is accepted.",
            "source": {
                "repository": "https://github.com/sagemath/sage",
                "revision": "8ecee59e510093bf96360177c52825b8e0603e59",
                "path": "src/sage/topology/simplicial_complex_examples.py",
                "symbol": "RealProjectivePlane",
                "license": "GPL-2.0-or-later",
                "transformation": "vertex labels shifted by ten",
            },
            "vertices": [10, 11, 12, 13, 14, 15],
            "facets": rp2_facets,
            "orientation": "increasing",
        },
        {
            "schema_version": "1",
            "task_id": "jacobian/coordination-rational-slice-01",
            "case_id": "coordination-rational-slice-01",
            "family": "rational-slice-binding",
            "primary_domain": "optimization",
            "case_version": CASE_VERSION,
            "conclusion": "LOCAL_SLICE_CERTIFIED",
            "scope": "SCALARS_AND_DECLARED_3X3_RATIONAL_SLICE_ONLY:coordination-rational-slice-01",
            "limitations": ["NO_GLOBAL_CERTIFICATE_OR_THEOREM_CONCLUSION"],
            "case_note": "The named m00 scalar is y0+c00_y and is deliberately distinct from the matrix's upper-left entry.",
            "source": {
                "kind": "authored-hand-auditable-fixture",
                "derived_from": "multi-tool-coordination-pr1 repeated scalar-binding failures",
            },
            "scalar_inputs": {
                "y0": rational(Fraction(-2, 3)),
                "c00_y": rational(Fraction(1, 6)),
                "objective": rational(Fraction(5, 7)),
            },
            "matrix": [
                [rational(2), rational(1), rational(0)],
                [rational(1), rational(2), rational(1)],
                [rational(0), rational(1), rational(2)],
            ],
        },
        {
            "schema_version": "1",
            "task_id": "jacobian/coordination-directed-proportionality-01",
            "case_id": "coordination-directed-proportionality-01",
            "family": "directed-proportionality",
            "primary_domain": "geometry-topology",
            "case_version": CASE_VERSION,
            "conclusion": "DIRECTED_POLYNOMIAL_IDENTITY_CERTIFIED",
            "scope": "EXACT_DECLARED_QQ_COEFFICIENT_IDENTITY:coordination-directed-proportionality-01",
            "limitations": ["NO_CLAIM_BEYOND_THE_DECLARED_COORDINATE_IDENTITY"],
            "case_note": "The certificate explicitly defines distance_coefficients = multiplier * circle_coefficients.",
            "source": {
                "dataset": "INSAIT-Institute/OPC",
                "revision": "dcc3b4804e2d126ea34b13e3e0cd998c3302644b",
                "split": "test",
                "row": 1,
                "problem_id": "irish_2016_4",
                "license": "CC-BY-NC-SA-4.0",
                "transformation": "fixed rational parameters and explicit multiplier direction",
            },
            "k": rational(3),
            "c": rational(2),
            "coefficient_basis": ["x^2", "y^2", "x", "1"],
            "required_relation": "DISTANCE_EQUALS_MULTIPLIER_TIMES_CIRCLE",
        },
    )


def determinant(values: list[list[Fraction | int]]) -> Fraction:
    matrix = [[Fraction(item) for item in row] for row in values]
    result = Fraction(1)
    for column in range(len(matrix)):
        pivot = next(row for row in range(column, len(matrix)) if matrix[row][column])
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            result = -result
        pivot_value = matrix[column][column]
        result *= pivot_value
        for row in range(column + 1, len(matrix)):
            scale = matrix[row][column] / pivot_value
            for item in range(column, len(matrix)):
                matrix[row][item] -= scale * matrix[column][item]
    return result


def graph_solution(data: dict[str, object]) -> dict[str, object]:
    vertices = list(data["vertices"])
    adjacency = {vertex: set() for vertex in vertices}
    for left, right in data["edges"]:
        adjacency[left].add(right)
        adjacency[right].add(left)
    maximum = max(map(len, adjacency.values()))
    maximum_vertices = sorted(
        vertex for vertex in vertices if len(adjacency[vertex]) == maximum
    )
    distances = dict.fromkeys(maximum_vertices, 0)
    queue = deque(maximum_vertices)
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in distances:
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)
    maximum_distance = max(distances.values())
    return {
        "maximum_degree_vertices": maximum_vertices,
        "distance_to_set": [
            {"vertex": vertex, "distance": distances[vertex]}
            for vertex in sorted(vertices)
        ],
        "maximum_distance_to_set": maximum_distance,
        "maximizing_vertices": sorted(
            vertex for vertex in vertices if distances[vertex] == maximum_distance
        ),
    }


def cycle_solution(data: dict[str, object]) -> dict[str, object]:
    vertices = set(data["vertices"])
    facets = sorted(tuple(facet) for facet in data["facets"])
    edges = sorted({edge for facet in facets for edge in combinations(facet, 2)})
    parent = {vertex: vertex for vertex in vertices}

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    tree = []
    for left, right in edges:
        a, b = root(left), root(right)
        if a != b:
            parent[a] = b
            tree.append((left, right))
    non_tree = [edge for edge in edges if edge not in set(tree)]
    matrix = []
    for edge in non_tree:
        row = []
        for a, b, c in facets:
            row.append({(b, c): 1, (a, c): -1, (a, b): 1}.get(edge, 0))
        matrix.append(row)
    det = determinant(matrix)
    assert abs(det) == 2 and det.denominator == 1
    return {
        "spanning_tree": [list(edge) for edge in tree],
        "non_tree_edges": [list(edge) for edge in non_tree],
        "facet_order": [list(facet) for facet in facets],
        "cycle_coordinate_matrix": matrix,
        "determinant": det.numerator,
        "homology": "Z/2Z",
    }


def rational_slice_solution(data: dict[str, object]) -> dict[str, object]:
    scalars = data["scalar_inputs"]
    y0 = Fraction(int(scalars["y0"]["num"]), int(scalars["y0"]["den"]))
    c00_y = Fraction(int(scalars["c00_y"]["num"]), int(scalars["c00_y"]["den"]))
    matrix = [
        [Fraction(int(value["num"]), int(value["den"])) for value in row]
        for row in data["matrix"]
    ]
    determinants = [
        determinant([row[:size] for row in matrix[:size]])
        for size in range(1, len(matrix) + 1)
    ]
    return {
        "scalar_replay": {
            "y0": scalars["y0"],
            "c00_y": scalars["c00_y"],
            "m00": rational(y0 + c00_y),
            "objective": scalars["objective"],
        },
        "proof_mode": "SYLVESTER",
        "positive_definite_certificate": {
            "leading_principal_determinants": [
                rational(value) for value in determinants
            ]
        },
    }


def proportionality_solution(data: dict[str, object]) -> dict[str, object]:
    k = Fraction(int(data["k"]["num"]), int(data["k"]["den"]))
    c = Fraction(int(data["c"]["num"]), int(data["c"]["den"]))
    p = k * c / (k + 1)
    q = k * c / (k - 1)
    center = (p + q) / 2
    radius = abs(q - p) / 2
    circle = [Fraction(1), Fraction(1), -2 * center, center**2 - radius**2]
    multiplier = 1 - k**2
    distance = [multiplier * value for value in circle]
    return {
        "k": rational(k),
        "c": rational(c),
        "p": rational(p),
        "q": rational(q),
        "center": rational(center),
        "radius": rational(radius),
        "circle_coefficients": [rational(value) for value in circle],
        "distance_coefficients": [rational(value) for value in distance],
        "multiplier": rational(multiplier),
        "relation": data["required_relation"],
    }


def solution_result(data: dict[str, object]) -> dict[str, object]:
    family = data["family"]
    if family == "graph-set-distance":
        return graph_solution(data)
    if family == "cycle-lattice":
        return cycle_solution(data)
    if family == "rational-slice-binding":
        return rational_slice_solution(data)
    if family == "directed-proportionality":
        return proportionality_solution(data)
    raise AssertionError(f"unknown family: {family}")


RATIONAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["num", "den"],
    "properties": {
        "num": {"type": "string", "pattern": "^(?:0|-?[1-9][0-9]*)$"},
        "den": {"type": "string", "pattern": "^[1-9][0-9]*$"},
    },
}


def result_schema(data: dict[str, object]) -> dict[str, object]:
    family = data["family"]
    if family == "graph-set-distance":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "maximum_degree_vertices",
                "distance_to_set",
                "maximum_distance_to_set",
                "maximizing_vertices",
            ],
            "properties": {
                "maximum_degree_vertices": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "distance_to_set": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["vertex", "distance"],
                        "properties": {
                            "vertex": {"type": "string"},
                            "distance": {"type": "integer", "minimum": 0},
                        },
                    },
                },
                "maximum_distance_to_set": {"type": "integer", "minimum": 0},
                "maximizing_vertices": {"type": "array", "items": {"type": "string"}},
            },
        }
    if family == "cycle-lattice":
        edge = {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "integer"},
        }
        facet = {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": "integer"},
        }
        row = {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {"type": "integer"},
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "spanning_tree",
                "non_tree_edges",
                "facet_order",
                "cycle_coordinate_matrix",
                "determinant",
                "homology",
            ],
            "properties": {
                "spanning_tree": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": edge,
                },
                "non_tree_edges": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 10,
                    "items": edge,
                },
                "facet_order": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 10,
                    "items": facet,
                },
                "cycle_coordinate_matrix": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 10,
                    "items": row,
                },
                "determinant": {"type": "integer"},
                "homology": {"const": "Z/2Z"},
            },
        }
    if family == "rational-slice-binding":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "scalar_replay",
                "proof_mode",
                "positive_definite_certificate",
            ],
            "properties": {
                "scalar_replay": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["y0", "c00_y", "m00", "objective"],
                    "properties": {
                        key: {"$ref": "#/$defs/rational"}
                        for key in ("y0", "c00_y", "m00", "objective")
                    },
                },
                "proof_mode": {"const": "SYLVESTER"},
                "positive_definite_certificate": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["leading_principal_determinants"],
                    "properties": {
                        "leading_principal_determinants": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"$ref": "#/$defs/rational"},
                        }
                    },
                },
            },
        }
    if family == "directed-proportionality":
        properties = {
            key: {"$ref": "#/$defs/rational"}
            for key in ("k", "c", "p", "q", "center", "radius", "multiplier")
        }
        properties.update(
            {
                "circle_coefficients": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"$ref": "#/$defs/rational"},
                },
                "distance_coefficients": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"$ref": "#/$defs/rational"},
                },
                "relation": {"const": "DISTANCE_EQUALS_MULTIPLIER_TIMES_CIRCLE"},
            }
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }
    raise AssertionError(f"unknown family: {family}")


def public_contract(data: dict[str, object]) -> PublicContract:
    declaration = {
        "schema_version": "1",
        "task_id": data["task_id"],
        "submission_path": "/app/submission.json",
        "assurance_ceiling": "COMPUTED",
        "allowed_assurance": ["COMPUTED"],
        "allowed_completeness": ["COMPLETE"],
        "conclusion": {"const": data["conclusion"]},
        "scope": {"type": "string", "const": data["scope"]},
        "evidence": {
            "min_items": 1,
            "max_items": 1,
            "allowed_paths": ["evidence/certificate.json"],
            "digest_pattern": "^sha256:[0-9a-f]{64}$",
            "media_types": ["application/json"],
        },
        "required_artifact_filenames": ["evidence/certificate.json"],
        "public_notes": (
            "Write evidence/certificate.json as a JSON wrapper with exactly "
            "schema_version, task_id, result, scope, completeness, and limitations. "
            "The last four values must exactly match submission.json. Bind the "
            "regular file by SHA-256. The clean-room verifier independently checks "
            "the terminal mathematics, frozen input, scope, completeness, evidence, "
            "and assurance; prose phrases are not proof evidence."
        ),
        "submission_result": result_schema(data),
        "limitations": {
            "type": "array",
            "minItems": len(data["limitations"]),
            "maxItems": len(data["limitations"]),
            "prefixItems": [{"const": item} for item in data["limitations"]],
        },
        "schema_definitions": {"rational": RATIONAL_SCHEMA},
    }
    draft = PublicContract.model_validate(declaration)
    declaration["submission_schema"] = json.loads(render_submission_schema(draft))
    return PublicContract.model_validate(declaration)


def instruction_base(data: dict[str, object]) -> str:
    family = data["family"]
    specific = {
        "graph-set-distance": "Determine the complete maximum-degree vertex set, every vertex's shortest distance to that set in lexicographic vertex order, the maximum distance, and every maximizer.",
        "cycle-lattice": "Give any spanning tree, order the non-tree edges and facets, form the increasing-orientation facet-boundary matrix in the fundamental-cycle coordinates, and certify determinant of absolute value two and H_1 = Z/2Z. Either matrix orientation is accepted.",
        "rational-slice-binding": "Replay y0, c00_y, objective, and the derived scalar m00=y0+c00_y. Certify the declared matrix positive definite with all exact positive leading principal determinants. The scalar m00 is not a matrix entry.",
        "directed-proportionality": "For the supplied k and c compute P, Q, center, positive radius, and both coefficient vectors in the declared basis. The required direction is distance_coefficients = multiplier * circle_coefficients.",
    }[family]
    return f"""# Multi-tool coordination certificate

{specific}

Return the terminal mathematical object described by `submission_schema.json`.
Use exact arithmetic and any mathematical method. Choose your own decomposition,
representations, tools, verification timing, and stopping rule; no tool sequence is
prescribed. Write the mirrored JSON certificate to `evidence/certificate.json`.
Do not widen the declared finite scope or claim `VERIFIED`.
"""


def render_task(
    data: dict[str, object], verifier: bytes, support: bytes
) -> dict[Path, bytes]:
    slug = str(data["case_id"])
    task = DATASET / slug
    input_content = json_bytes(
        {key: value for key, value in data.items() if key != "case_note"}
    )
    fixture_digest = sha256_bytes(input_content)
    result = solution_result(data)
    submission = {
        "task_id": data["task_id"],
        "conclusion": data["conclusion"],
        "result": result,
        "claimed_assurance": "COMPUTED",
        "scope": data["scope"],
        "completeness": "COMPLETE",
        "evidence": [{"path": "evidence/certificate.json", "sha256": ""}],
        "limitations": data["limitations"],
    }
    evidence = {
        "schema_version": "1",
        "task_id": data["task_id"],
        "result": result,
        "scope": data["scope"],
        "completeness": "COMPLETE",
        "limitations": data["limitations"],
    }
    evidence_content = json_bytes(evidence)
    submission["evidence"][0]["sha256"] = sha256_bytes(evidence_content)
    contract = public_contract(data)
    schema_text = render_submission_schema(contract)
    contract_content = (
        json.dumps(
            contract.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    checksum = verifier_bundle_checksum_bytes(verifier, support)
    description = f"Assess one exact {data['family']} coordination certificate."
    task_toml = f'''schema_version = "1.4"
artifacts = ["/app/submission.json", "/app/evidence"]

[task]
name = "jacobian/{slug}"
version = "1.0.0"
description = "{description}"
keywords = ["mathematics", "multi-tool", "coordination", "exact-certificate"]

[metadata]
evaluation_kind = "workflow"
domain = "mathematical-sciences"
field = "mathematics"
primary_domain = "{data["primary_domain"]}"
assurance_ceiling = "COMPUTED"
answer_visibility = "hidden-at-runtime"
provenance_class = "deterministic-hand-auditable-pilot"
fixture_digest = "{fixture_digest}"
required_provider = "core"
author_name = "Jacobian contributors"
difficulty = "medium"
category = "mathematics"
tags = ["multi-tool-coordination", "offline", "clean-room-verifier"]
case_version = "{CASE_VERSION}"
generator_version = "{GENERATOR_VERSION}"
derivation = "Evidence-driven PR1 pilot from the frozen multi-tool-coordination-pr1 study."

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[environment]
network_mode = "no-network"
cpus = 1
memory_mb = 1024
storage_mb = 4096

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 1024
storage_mb = 4096
'''.encode()
    readme = f"""# jacobian/{slug}

{description}

- family: `{data["family"]}`
- case version: `{CASE_VERSION}`
- generator: `{GENERATOR_VERSION}`
- fixture digest: `{fixture_digest}`
- assurance ceiling: `COMPUTED`
- note: {data["case_note"]}

The task is offline and does not prescribe a capability or tool sequence. Its
standard-library clean-room verifier checks the mathematical object rather than
proof prose and independently binds the input, evidence, scope, completeness,
and assurance. Alternate valid lattice witnesses are accepted.
""".encode()
    instruction = render_instruction(contract, instruction_base(data)).encode()
    environment_docker = f"FROM {IMAGE}\nCOPY input.json submission_schema.json /app/\nWORKDIR /app\n".encode()
    tests_docker = f'''FROM {IMAGE}
LABEL jacobian.checksum="{checksum}"
RUN python -m pip install --no-cache-dir attrs==26.1.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.6.3 typing-extensions==4.16.0
COPY verifier.py verifier_support.py public_contract.json input.json test.sh /tests/
COPY input.json /app/input.json
RUN chmod +x /tests/test.sh
'''.encode()
    solve_sh = b"""#!/bin/sh
set -eu
mkdir -p /app/evidence
cp /solution/submission.json /app/submission.json
cp /solution/certificate.json /app/evidence/certificate.json
"""
    member = f'''schema_version = "2"
task_id = "{slug}"
task_name = "jacobian/{slug}"
evaluation_kind = "workflow"
domain = "mathematical-sciences"
field = "mathematics"
primary_domain = "{data["primary_domain"]}"
provenance_class = "deterministic-hand-auditable-pilot"
provenance_ref = "authored:multi-tool-coordination-v1/{CASE_VERSION}#{slug}"
assurance_ceiling = "COMPUTED"
required_provider = "core"
environment_profile = "core-python-minimal-verifier"
verifier_contract_version = "1"
evaluation_owner = "jacobian/multi-tool-coordination-v1"
'''.encode()
    return {
        task / "README.md": readme,
        task / "instruction.md": instruction,
        task / "task.toml": task_toml,
        task / "environment/Dockerfile": environment_docker,
        task / "environment/input.json": input_content,
        task / "environment/submission_schema.json": schema_text.encode(),
        task / "solution/certificate.json": evidence_content,
        task / "solution/submission.json": json_bytes(submission),
        task / "solution/solve.sh": solve_sh,
        task / "tests/Dockerfile": tests_docker,
        task / "tests/input.json": input_content,
        task / "tests/public_contract.json": contract_content,
        task / "tests/test.sh": b"#!/bin/sh\nset -eu\nexec python /tests/verifier.py\n",
        task / "tests/verifier.py": verifier,
        task / "tests/verifier_support.py": support,
        DATASET / "members" / f"{slug}.toml": member,
    }


def expected_files() -> tuple[dict[Path, bytes], dict[str, object]]:
    verifier = VERIFIER_TEMPLATE.read_bytes()
    support = SUPPORT_TEMPLATE.read_bytes()
    files: dict[Path, bytes] = {}
    manifest_cases = []
    for data in cases():
        rendered = render_task(data, verifier, support)
        if files.keys() & rendered.keys():
            raise AssertionError("duplicate generated path")
        files.update(rendered)
        input_content = rendered[
            DATASET / str(data["case_id"]) / "environment/input.json"
        ]
        manifest_cases.append(
            {
                "task_id": data["case_id"],
                "family": data["family"],
                "fixture_sha256": sha256_bytes(input_content),
            }
        )
    manifest = {
        "schema_version": "1",
        "dataset_id": "jacobian/multi-tool-coordination-v1",
        "case_version": CASE_VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_study": "benchmarks/config/multi-tool-coordination-pr1-adjudication.json",
        "generation": "deterministic-no-random-seed",
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    files[DATASET / "pilot-manifest.json"] = json_bytes(manifest)
    return files, manifest


def run(check: bool) -> int:
    files, manifest = expected_files()
    failures = []
    if check:
        for path, expected in sorted(files.items()):
            if not path.is_file():
                failures.append(f"missing: {path.relative_to(ROOT)}")
            elif path.read_bytes() != expected:
                failures.append(f"stale: {path.relative_to(ROOT)}")
        task_ids = {str(case["task_id"]) for case in manifest["cases"]}
        actual_tasks = {
            path.name for path in DATASET.glob("coordination-*") if path.is_dir()
        }
        actual_members = {path.stem for path in (DATASET / "members").glob("*.toml")}
        failures.extend(
            f"unexpected task: {task}" for task in sorted(actual_tasks - task_ids)
        )
        failures.extend(
            f"unexpected member: {task}" for task in sorted(actual_members - task_ids)
        )
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print(
            f"multi-tool-coordination-v1: {manifest['case_count']} generated cases are current"
        )
        return 0
    for path, content in sorted(files.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_bytes() != content:
            path.write_bytes(content)
    for case in manifest["cases"]:
        for relative in ("solution/solve.sh", "tests/test.sh"):
            (DATASET / str(case["task_id"]) / relative).chmod(0o755)
    print(f"rendered {manifest['case_count']} multi-tool-coordination-v1 cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return run(parser.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
