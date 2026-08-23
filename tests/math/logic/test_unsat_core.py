from __future__ import annotations

from pathlib import Path

import pytest
import z3
from pydantic import ValidationError

from jacobian.math.logic import _unsat_core as unsat_core
from jacobian.math.logic._unsat_core import (
    SmtUnsatCoreRequest,
    SmtUnsatCoreResult,
    compute_smt_unsat_core,
)

CONTRADICTORY_LIA = """\
(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 1))
(assert (<= x 0))
(assert (<= x 10))
(check-sat)
"""


def _request(
    smtlib: str = CONTRADICTORY_LIA,
    *,
    rlimit: int = 100_000,
) -> SmtUnsatCoreRequest:
    return SmtUnsatCoreRequest(
        logic="QF_LIA",
        smtlib=smtlib,
        timeout_ms=1_000,
        rlimit=rlimit,
        max_memory_mb=128,
    )


def test_unsat_core_is_an_indexed_replayable_source_subset() -> None:
    result = compute_smt_unsat_core(_request())

    assert result.outcome == "UNSAT"
    assert result.core_indices == tuple(sorted(result.core_indices))
    assert result.core_indices == (0, 1)
    assertions = tuple(z3.parse_smt2_string(result.source.smtlib))
    replay = z3.SolverFor(result.source.logic.value)
    replay.add(*(assertions[index] for index in result.core_indices))
    assert replay.check() == z3.unsat


def test_unsat_core_result_rejects_a_core_detached_from_its_source() -> None:
    result = compute_smt_unsat_core(_request())
    payload = result.model_dump(mode="json")
    payload["source"]["smtlib"] = CONTRADICTORY_LIA.replace("(>= x 1)", "(>= x -1)")

    with pytest.raises(ValidationError, match="selected source assertions"):
        SmtUnsatCoreResult.model_validate(payload)


def test_unsat_core_result_rejects_a_forged_proper_subset() -> None:
    result = compute_smt_unsat_core(_request())
    payload = result.model_dump(mode="json")
    payload["core_indices"] = [0]

    with pytest.raises(ValidationError, match="selected source assertions"):
        SmtUnsatCoreResult.model_validate(payload)


def test_unsat_core_result_rejects_a_forged_sat_conclusion() -> None:
    result = compute_smt_unsat_core(_request())
    payload = result.model_dump(mode="json")
    payload["outcome"] = "SAT"
    payload["core_indices"] = []

    with pytest.raises(ValidationError, match="complete source assertions"):
        SmtUnsatCoreResult.model_validate(payload)


def test_empty_assertion_collection_is_satisfiable() -> None:
    result = compute_smt_unsat_core(_request("(set-logic QF_LIA)\n(check-sat)\n"))

    assert result.outcome == "SAT"
    assert result.core_indices == ()
    assert result.detail is None


def test_resource_exhaustion_is_unknown_not_unsat() -> None:
    result = compute_smt_unsat_core(_request(rlimit=1))

    assert result.outcome == "UNKNOWN"
    assert result.core_indices == ()
    assert result.detail


def test_replay_resource_exhaustion_is_a_typed_unknown(monkeypatch) -> None:
    monkeypatch.setattr(
        unsat_core,
        "_extract_source_core",
        lambda _source: ("UNSAT", (0, 1), None),
    )
    monkeypatch.setattr(
        unsat_core,
        "_replay_source",
        lambda _source, _indices: ("UNKNOWN", "canceled"),
    )

    result = compute_smt_unsat_core(_request())

    assert result.outcome == "UNKNOWN"
    assert result.core_indices == ()
    assert result.detail is not None
    assert "result-validation replay" in result.detail


@pytest.mark.parametrize(
    ("logic", "smtlib"),
    (
        (
            "QF_UF",
            "(set-logic QF_UF)\n"
            "(declare-const p Bool)\n"
            "(assert p)\n"
            "(assert (not p))\n"
            "(check-sat)\n",
        ),
        (
            "QF_LIA",
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (> x 0))\n"
            "(assert (< x 0))\n"
            "(check-sat)\n",
        ),
        (
            "QF_LRA",
            "(set-logic QF_LRA)\n"
            "(declare-const x Real)\n"
            "(assert (> x 0))\n"
            "(assert (< x 0))\n"
            "(check-sat)\n",
        ),
    ),
)
def test_unsat_core_supports_each_admitted_smt_fragment(
    logic: str, smtlib: str
) -> None:
    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic=logic, smtlib=smtlib))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_comments_cannot_impersonate_indexed_assertions_or_execute_text(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    source = f"""\
(set-logic QF_LIA)
(declare-const x Int)
; (assert false) __import__('pathlib').Path({str(marker)!r}).write_text('x')
(assert (> x 0))
(assert (< x 0))
(check-sat)
"""

    result = compute_smt_unsat_core(_request(source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)
    assert not marker.exists()


def test_tracking_symbols_do_not_alias_caller_boolean_constants() -> None:
    source = """\
(set-logic QF_LIA)
(declare-const jacobian_unsat_core_0 Bool)
(declare-const x Int)
(assert (not jacobian_unsat_core_0))
(assert (> x 0))
(assert (< x 0))
(check-sat)
"""

    result = compute_smt_unsat_core(_request(source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (1, 2)


def test_request_validates_smtlib_syntax_before_execution() -> None:
    with pytest.raises(ValidationError, match="could not be parsed"):
        _request(
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (+ 1 2))\n(check-sat)\n"
        )


def test_request_bounds_the_number_of_indexed_assertions() -> None:
    accepted = "\n".join(
        ["(set-logic QF_LIA)", *("(assert true)" for _ in range(512)), "(check-sat)"]
    )
    rejected = accepted.replace("(check-sat)", "(assert true)\n(check-sat)")

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=accepted).assertion_count == 512
    with pytest.raises(ValidationError, match="at most 512 source assertions"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=rejected)


def test_request_bounds_numeric_coefficient_digits() -> None:
    accepted_number = "1" * 256
    accepted = (
        "(set-logic QF_LIA)\n"
        f"(assert (= {accepted_number} {accepted_number}))\n"
        "(check-sat)\n"
    )
    rejected = accepted.replace(accepted_number, "1" * 257, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=accepted)
    with pytest.raises(ValidationError, match="numeral may contain at most 256 digits"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=rejected)


def test_request_bounds_parsed_ast_nodes() -> None:
    def source(term_count: int) -> str:
        equalities = " ".join(f"(= x {value})" for value in range(term_count))
        return (
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            f"(assert (and {equalities}))\n"
            "(check-sat)\n"
        )

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(2_047))
    with pytest.raises(ValidationError, match="at most 4096 distinct AST nodes"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(2_048))


def test_unsat_result_requires_a_nonempty_canonical_core() -> None:
    result = compute_smt_unsat_core(_request())
    payload = result.model_dump(mode="json")
    payload["core_indices"] = []

    with pytest.raises(ValidationError, match="nonempty core"):
        SmtUnsatCoreResult.model_validate(payload)

    payload["core_indices"] = [1, 0]
    with pytest.raises(ValidationError, match="strictly increasing"):
        SmtUnsatCoreResult.model_validate(payload)


def test_request_schema_explains_validator_owned_indexing() -> None:
    schema = SmtUnsatCoreRequest.model_json_schema()

    assert "zero-based index" in schema["description"]
    assert schema["examples"]
