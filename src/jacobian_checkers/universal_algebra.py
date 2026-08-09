"""Independent exact replay for finite-magma law-evaluation certificates."""

from __future__ import annotations

from itertools import product
from typing import Any, cast

_MAX_ORDER = 8
_MAX_LAWS = 16
_MAX_VARIABLES = 4
_MAX_TERM_NODES = 31
_MAX_TERM_DEPTH = 16
_MAX_TOTAL_VALUATIONS = 1_000_000


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "CHECKED_CERTIFICATE",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def check_law_evaluation(request: dict[str, Any]) -> dict[str, Any]:
    """Replay each law without importing the producer or Pydantic contracts."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim_artifact = request["claim"]
        candidate_artifact = request["candidate"]
        scope_artifact = request["scope"]
        certificate = request["certificate"]["payload"]
        if not all(
            isinstance(value, dict)
            for value in (
                claim_artifact,
                candidate_artifact,
                scope_artifact,
                certificate,
            )
        ):
            return _reject("finite-magma replay artifacts are malformed")

        claim = claim_artifact.get("payload")
        candidate = candidate_artifact.get("payload")
        problem = scope_artifact.get("payload")
        if (
            not isinstance(claim, dict)
            or set(claim) != {"claim_schema_version", "predicate", "problem_uri"}
            or claim.get("claim_schema_version") != "1"
            or claim.get("predicate") != "EXACT_FINITE_MAGMA_LAW_EVALUATION"
        ):
            return _reject("unexpected finite-magma evaluation claim")
        if (
            certificate.get("evidence_schema_version") != "1"
            or certificate.get("certificate_type") != "universal_algebra.law_evaluation"
            or certificate.get("format_version") != "1"
            or certificate.get("bindings") != request.get("expected_bindings")
        ):
            return _reject("unexpected finite-magma certificate or bindings")
        replay = certificate.get("payload")
        if (
            not isinstance(replay, dict)
            or set(replay) != {"method", "problem_uri", "evaluation_uri"}
            or replay.get("method") != "EXHAUSTIVE_LEXICOGRAPHIC_REPLAY"
        ):
            return _reject("finite-magma replay payload is malformed")

        problem_uri = scope_artifact.get("artifact_uri")
        evaluation_uri = candidate_artifact.get("artifact_uri")
        if (
            claim.get("problem_uri") != problem_uri
            or replay.get("problem_uri") != problem_uri
            or replay.get("evaluation_uri") != evaluation_uri
        ):
            return _reject("finite-magma artifact identities do not match")

        order, table, laws = _parse_problem(problem)
        _replay_candidate(
            candidate,
            problem_uri=problem_uri,
            order=order,
            table=table,
            laws=laws,
        )
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "CHECKED_CERTIFICATE",
            "coverage": "NOT_APPLICABLE",
            "detail": (
                "all finite-magma laws replayed exactly over canonical "
                "lexicographic valuations"
            ),
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed finite-magma law-evaluation request")


type Term = tuple[str, str | None, Term | None, Term | None]
type Law = tuple[str, tuple[str, ...], Term, Term]


def _parse_problem(
    value: object,
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[Law, ...]]:
    if not isinstance(value, dict) or set(value) != {
        "problem_schema_version",
        "structure",
        "laws",
    }:
        raise ValueError("malformed finite-magma problem")
    structure = value["structure"]
    laws_value = value["laws"]
    if (
        value["problem_schema_version"] != "1"
        or not isinstance(structure, dict)
        or set(structure) != {"structure_schema_version", "operation", "order", "table"}
        or structure["structure_schema_version"] != "1"
        or structure["operation"] != "binary"
        or not isinstance(structure["order"], int)
        or isinstance(structure["order"], bool)
        or not 1 <= structure["order"] <= _MAX_ORDER
        or not isinstance(laws_value, list)
        or not 1 <= len(laws_value) <= _MAX_LAWS
    ):
        raise ValueError("invalid finite-magma problem metadata")
    order = structure["order"]
    table_value = structure["table"]
    if (
        not isinstance(table_value, list)
        or len(table_value) != order
        or any(not isinstance(row, list) or len(row) != order for row in table_value)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < order
            for row in table_value
            for item in row
        )
    ):
        raise ValueError("invalid finite-magma operation table")
    table = tuple(tuple(row) for row in table_value)
    laws = tuple(_parse_law(law) for law in laws_value)
    law_ids = tuple(law[0] for law in laws)
    if len(set(law_ids)) != len(law_ids):
        raise ValueError("duplicate law identifiers")
    if sum(order ** len(law[1]) for law in laws) > _MAX_TOTAL_VALUATIONS:
        raise ValueError("valuation budget exceeded")
    return order, table, laws


def _parse_law(value: object) -> Law:
    if not isinstance(value, dict) or set(value) != {
        "law_id",
        "variables",
        "left",
        "right",
    }:
        raise ValueError("malformed magma law")
    law_id = value["law_id"]
    variables_value = value["variables"]
    if (
        not isinstance(law_id, str)
        or not law_id
        or not isinstance(variables_value, list)
        or not 1 <= len(variables_value) <= _MAX_VARIABLES
        or any(
            not isinstance(variable, str) or not variable
            for variable in variables_value
        )
        or variables_value != sorted(set(variables_value))
    ):
        raise ValueError("invalid magma law identifiers")
    left = _parse_term(value["left"])
    right = _parse_term(value["right"])
    variables = tuple(variables_value)
    if _term_variables(left) | _term_variables(right) != frozenset(variables):
        raise ValueError("declared variables do not match the law terms")
    return law_id, variables, left, right


def _parse_term(value: object, *, depth: int = 1) -> Term:
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "variable", "left", "right"}
        or depth > _MAX_TERM_DEPTH
    ):
        raise ValueError("malformed magma term")
    kind = value["kind"]
    if kind == "VARIABLE":
        variable = value["variable"]
        if (
            not isinstance(variable, str)
            or not variable
            or value["left"] is not None
            or value["right"] is not None
        ):
            raise ValueError("invalid variable term")
        term: Term = ("VARIABLE", variable, None, None)
    elif kind == "PRODUCT":
        if value["variable"] is not None:
            raise ValueError("invalid product term")
        term = (
            "PRODUCT",
            None,
            _parse_term(value["left"], depth=depth + 1),
            _parse_term(value["right"], depth=depth + 1),
        )
    else:
        raise ValueError("unknown magma term kind")
    if _term_node_count(term) > _MAX_TERM_NODES:
        raise ValueError("magma term exceeds node budget")
    return term


def _term_tuple(term: object) -> Term:
    if type(term) is not tuple or len(term) != 4:
        raise ValueError("parsed magma term is not an exact four-tuple")
    return cast(Term, term)


def _variable_name(term: object) -> str:
    term = _term_tuple(term)
    variable = term[1]
    if (
        term[0] != "VARIABLE"
        or not isinstance(variable, str)
        or not variable
        or term[2] is not None
        or term[3] is not None
    ):
        raise ValueError("parsed variable term is inconsistent")
    return variable


def _product_children(term: object) -> tuple[Term, Term]:
    term = _term_tuple(term)
    left = term[2]
    right = term[3]
    if term[0] != "PRODUCT" or term[1] is not None or left is None or right is None:
        raise ValueError("parsed product term is inconsistent")
    return left, right


def _term_node_count(term: object) -> int:
    term = _term_tuple(term)
    if term[0] == "VARIABLE":
        _variable_name(term)
        return 1
    left, right = _product_children(term)
    return 1 + _term_node_count(left) + _term_node_count(right)


def _term_variables(term: object) -> frozenset[str]:
    term = _term_tuple(term)
    if term[0] == "VARIABLE":
        return frozenset((_variable_name(term),))
    left, right = _product_children(term)
    return _term_variables(left) | _term_variables(right)


def _evaluate_term(
    term: object,
    table: tuple[tuple[int, ...], ...],
    assignment: dict[str, int],
) -> int:
    term = _term_tuple(term)
    if term[0] == "VARIABLE":
        return assignment[_variable_name(term)]
    left, right = _product_children(term)
    return table[_evaluate_term(left, table, assignment)][
        _evaluate_term(right, table, assignment)
    ]


def _expected_record(
    law: Law,
    order: int,
    table: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    law_id, variables, left, right = law
    checked = 0
    for values in product(range(order), repeat=len(variables)):
        checked += 1
        assignment = dict(zip(variables, values, strict=True))
        left_value = _evaluate_term(left, table, assignment)
        right_value = _evaluate_term(right, table, assignment)
        if left_value != right_value:
            return {
                "law_id": law_id,
                "holds": False,
                "coverage": "COUNTEREXAMPLE_FOUND",
                "checked_valuations": checked,
                "counterexample": {
                    "assignment": [
                        {"variable": variable, "value": value}
                        for variable, value in zip(variables, values, strict=True)
                    ],
                    "left_value": left_value,
                    "right_value": right_value,
                },
            }
    return {
        "law_id": law_id,
        "holds": True,
        "coverage": "EXHAUSTIVE",
        "checked_valuations": checked,
        "counterexample": None,
    }


def _replay_candidate(
    value: object,
    *,
    problem_uri: object,
    order: int,
    table: tuple[tuple[int, ...], ...],
    laws: tuple[Law, ...],
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "evaluation_schema_version",
        "problem_uri",
        "records",
        "arithmetic",
        "determinism",
    }:
        raise ValueError("malformed finite-magma evaluation candidate")
    records = value["records"]
    if (
        value["evaluation_schema_version"] != "1"
        or value["problem_uri"] != problem_uri
        or value["arithmetic"] != "EXACT_FINITE"
        or value["determinism"] != "DETERMINISTIC"
        or not isinstance(records, list)
        or len(records) != len(laws)
    ):
        raise ValueError("finite-magma evaluation metadata does not match")
    expected = [_expected_record(law, order, table) for law in laws]
    if records != expected:
        raise ValueError("finite-magma law evaluation does not replay")
