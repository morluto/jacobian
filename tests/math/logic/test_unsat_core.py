from __future__ import annotations

from pathlib import Path
from weakref import ReferenceType, ref

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


def test_repeated_calls_have_a_stable_exact_outcome() -> None:
    results = tuple(compute_smt_unsat_core(_request()) for _ in range(16))

    assert tuple(result.outcome for result in results) == ("UNSAT",) * 16
    assert tuple(result.core_indices for result in results) == ((0, 1),) * 16


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


def test_core_extraction_failure_is_a_typed_unknown(monkeypatch) -> None:
    def fail_core_extraction(_solver: z3.Solver) -> None:
        raise z3.Z3Exception("core extraction failed")

    monkeypatch.setattr(z3.Solver, "unsat_core", fail_core_extraction)

    result = compute_smt_unsat_core(_request())

    assert result.outcome == "UNKNOWN"
    assert result.core_indices == ()
    assert result.detail == "Z3 could not complete the bounded source check."


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


def test_qf_uf_accepts_boolean_uninterpreted_functions() -> None:
    source = """\
(set-logic QF_UF)
(declare-fun f (Bool) Bool)
(declare-const p Bool)
(assert (f p))
(assert (not (f p)))
(check-sat)
"""

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_UF", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


@pytest.mark.parametrize(
    ("logic", "source"),
    (
        (
            "QF_LIA",
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (>= (* (- 2) x) 2))\n"
            "(assert (>= x 0))\n"
            "(check-sat)\n",
        ),
        (
            "QF_LRA",
            "(set-logic QF_LRA)\n"
            "(declare-const x Real)\n"
            "(assert (= (* (/ 1 3) x) 2.0))\n"
            "(assert (= x 0.0))\n"
            "(check-sat)\n",
        ),
        (
            "QF_LIA",
            "(set-logic QF_LIA)\n"
            "(declare-const x Int)\n"
            "(assert (= (* 2 3 x) 6))\n"
            "(assert (= x 0))\n"
            "(check-sat)\n",
        ),
        (
            "QF_LRA",
            "(set-logic QF_LRA)\n"
            "(declare-const x Real)\n"
            "(assert (= (* (/ 1 2) (/ 2 3) x) 1.0))\n"
            "(assert (= x 0.0))\n"
            "(check-sat)\n",
        ),
    ),
)
def test_linear_closed_coefficients_are_admitted(logic: str, source: str) -> None:
    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic=logic, smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


@pytest.mark.parametrize(
    ("logic", "declarations", "assertion"),
    (
        ("QF_LIA", "(declare-const x Int)", "(assert (= (* x x) 2))"),
        ("QF_LIA", "(declare-const x Int)", "(assert (= (* 0 x x) 0))"),
        ("QF_LIA", "(declare-const x Real)", "(assert (> x 0.0))"),
        (
            "QF_LIA",
            "",
            "(assert (forall ((x Int)) (= x x)))",
        ),
        (
            "QF_LRA",
            "(declare-const x (_ BitVec 8))",
            "(assert (= x (_ bv0 8)))",
        ),
        ("QF_UF", "(declare-const x Int)", "(assert (= x x))"),
    ),
)
def test_request_rejects_terms_outside_the_declared_fragment(
    logic: str,
    declarations: str,
    assertion: str,
) -> None:
    source = "\n".join((f"(set-logic {logic})", declarations, assertion, "(check-sat)"))

    with pytest.raises(ValidationError, match=f"declared {logic} fragment"):
        SmtUnsatCoreRequest(logic=logic, smtlib=source)


def test_retained_validation_error_does_not_retain_a_z3_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_refs: list[ReferenceType[z3.Context]] = []
    parse_assertions = unsat_core._parse_assertions

    def track_context(
        smtlib: str,
        *,
        context: z3.Context | None = None,
    ) -> tuple[object, ...]:
        owned_context = z3.Context() if context is None else context
        context_refs.append(ref(owned_context))
        return parse_assertions(smtlib, context=owned_context)

    monkeypatch.setattr(unsat_core, "_parse_assertions", track_context)
    with pytest.raises(ValidationError) as retained_error:
        SmtUnsatCoreRequest(
            logic="QF_LIA",
            smtlib=(
                "(set-logic QF_LIA)\n"
                "(declare-const x Int)\n"
                "(assert (= (* x x) 2))\n"
                "(check-sat)\n"
            ),
        )

    assert retained_error.value
    assert context_refs
    assert all(context_ref() is None for context_ref in context_refs)


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


def test_request_bounds_source_tokens() -> None:
    def source(term_count: int) -> str:
        terms = " ".join("x" for _ in range(term_count))
        return (
            "(set-logic QF_LIA)\n"
            "(declare-const x Bool)\n"
            f"(assert (and {terms}))\n"
            "(check-sat)\n"
        )

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(32_750))
    with pytest.raises(ValidationError, match="at most 32768 tokens"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(32_751))


def test_request_bounds_source_nesting() -> None:
    def source(negation_count: int) -> str:
        term = "(not " * negation_count + "true" + ")" * negation_count
        return f"(set-logic QF_LIA)\n(assert {term})\n(check-sat)\n"

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(255))
    with pytest.raises(ValidationError, match="nesting may not exceed 256"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source(256))


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


def test_request_bounds_negative_numeric_atom_digits() -> None:
    rejected_number = "-" + "1" * 257
    source = f"(set-logic QF_LIA)\n(assert (= {rejected_number} 0))\n(check-sat)\n"

    with pytest.raises(ValidationError, match="numeral may contain at most 256 digits"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


def test_request_bounds_normalized_closed_coefficient_digits() -> None:
    factor = "9" * 256
    source = f"(set-logic QF_LIA)\n(assert (= (* {factor} {factor}) 0))\n(check-sat)\n"

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


def test_request_bounds_nested_product_coefficient_digits() -> None:
    factor = "9" * 128
    boundary = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"(assert (= (* {factor} (* {factor} x)) 0))\n"
        "(check-sat)\n"
    )
    over = boundary.replace(factor, "9" * 129, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=boundary)
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=over)


@pytest.mark.parametrize("nesting", (2, 3, 4))
def test_request_bounds_deeply_nested_coefficient_products(nesting: int) -> None:
    factor = "1" * 256
    term = f"(* {factor} " * nesting + "x" + ")" * nesting
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"(assert (= {term} 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


def test_request_bounds_outer_factor_against_folded_inner_coefficient() -> None:
    inner = "9" * 200
    outer = "9" * 57
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"(assert (= (* {outer} (* {inner} x)) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


def test_request_bounds_negated_nested_product_coefficient_digits() -> None:
    factor = "9" * 128
    boundary = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"(assert (= (* {factor} (- (* {factor} x))) 0))\n"
        "(check-sat)\n"
    )
    over = boundary.replace(factor, "9" * 129, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=boundary)
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=over)


@pytest.mark.parametrize(
    ("logic", "source_template"),
    (
        (
            "QF_LIA",
            "(assert (= (* {outer} (- (- (* {inner} x)))) 0))",
        ),
        (
            "QF_LIA",
            "(assert (= (* {outer} (+ (* {inner} x))) 0))",
        ),
        (
            "QF_LRA",
            "(declare-const x Real)\n(assert (= (* {outer}.0 (/ (* {inner} x) 7.0)) 0.0))",
        ),
    ),
)
def test_request_bounds_nested_coefficients_through_preserving_wrappers(
    logic: str,
    source_template: str,
) -> None:
    outer = "9" * 200
    inner = "9" * 200
    declaration = "(declare-const x Int)" if logic == "QF_LIA" else ""
    source = (
        f"(set-logic {logic})\n"
        f"{declaration}\n"
        f"{source_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic=logic, smtlib=source)


def test_request_bounds_nested_real_coefficient_digits() -> None:
    numerator = "9" * 200
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* (/ {numerator} 3) (* (/ {numerator} 3) x)) 0.0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


@pytest.mark.parametrize(
    ("logic", "declaration", "assertion_template"),
    (
        (
            "QF_LIA",
            "(declare-const x Int)",
            "(assert (= (* {outer} (+ (* {inner} x) 1)) 0))",
        ),
        (
            "QF_LIA",
            "(declare-const x Int)",
            "(assert (= (* {outer} (- (* {inner} x) 1)) 0))",
        ),
        (
            "QF_LIA",
            "(declare-const x Int)",
            "(assert (= (* {outer} (+ 1 (- (* {inner} x)) 2)) 0))",
        ),
        (
            "QF_LRA",
            "(declare-const y Real)",
            "(assert (= (* {outer}.0 (+ (* {inner}.0 y) (/ 1.0 3.0))) 0.0))",
        ),
    ),
)
def test_request_bounds_translated_product_coefficient_digits(
    logic: str,
    declaration: str,
    assertion_template: str,
) -> None:
    outer = "9" * 200
    inner = "9" * 200
    source = (
        f"(set-logic {logic})\n"
        f"{declaration}\n"
        f"{assertion_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic=logic, smtlib=source)


def test_request_bounds_translated_constant_digits() -> None:
    boundary_constant = "9" * 255
    over_constant = "9" * 256
    template = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (= (* 2 (+ (* 2 x) {constant})) 0))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(
        logic="QF_LIA", smtlib=template.format(constant=boundary_constant)
    )
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(
            logic="QF_LIA", smtlib=template.format(constant=over_constant)
        )


@pytest.mark.parametrize(
    ("logic", "declarations", "assertion_template"),
    (
        (
            "QF_LIA",
            "(declare-const x Int)\n(declare-const y Int)",
            "(assert (= (* {outer} (+ (* {inner} x) (* {inner} y))) 0))",
        ),
        (
            "QF_LIA",
            "(declare-const x Int)\n(declare-const y Int)\n(declare-const z Int)",
            "(assert (= (* {outer} (+ (* {inner} x) (- (* {inner} y)) (* {inner} z))) 0))",
        ),
        (
            "QF_LRA",
            "(declare-const x Real)\n(declare-const y Real)",
            "(assert (= (* {outer}.0 (+ (* {inner}.0 x) (* {inner}.0 y))) 0.0))",
        ),
    ),
)
def test_request_bounds_multi_term_affine_sum_coefficient_digits(
    logic: str,
    declarations: str,
    assertion_template: str,
) -> None:
    outer = "9" * 200
    inner = "9" * 200
    source = (
        f"(set-logic {logic})\n"
        f"{declarations}\n"
        f"{assertion_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic=logic, smtlib=source)


def test_request_bounds_multi_term_boundary_coefficients() -> None:
    factor = "9" * 128
    boundary = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(declare-const y Int)\n"
        f"(assert (= (* {factor} (+ (* {factor} x) (* {factor} y))) 0))\n"
        "(check-sat)\n"
    )
    over = boundary.replace(factor, "9" * 129, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=boundary)
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=over)


def test_request_bounds_merged_duplicate_summand_coefficients() -> None:
    summand = "9" * 128
    outer = "9" * 127
    boundary = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"(assert (= (* {outer} (+ (* {summand} x) (* {summand} x))) 0))\n"
        "(check-sat)\n"
    )
    over = boundary.replace(outer, "9" * 128, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=boundary)
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=over)


@pytest.mark.parametrize(
    "assertion_template",
    (
        "(assert (= (* {outer} (div (* {inner} x) 1)) 0))",
        "(assert (= (* {outer} (div (+ (* {inner} x) 7) 1)) 0))",
        "(assert (= (* {outer} (div (* {inner} x) (- 1))) 0))",
    ),
)
def test_request_bounds_exact_integer_division_coefficient_digits(
    assertion_template: str,
) -> None:
    outer = "9" * 200
    inner = "9" * 200
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        f"{assertion_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


def test_request_bounds_integer_division_translated_constant_digits() -> None:
    constant = "9" * 255
    template = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (= (* 4 (div (+ (* 2 x) {constant}) 1)) 0))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(
        logic="QF_LIA", smtlib=template.format(constant=constant)
    )
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=template.format(constant="9" * 256))


def test_request_bounds_arithmetic_ite_branch_coefficient_digits() -> None:
    factor = "9" * 128
    boundary = (
        "(set-logic QF_LIA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Int)\n"
        f"(assert (= (* {factor} (ite p (* {factor} x) 0)) 0))\n"
        "(check-sat)\n"
    )
    over = boundary.replace(factor, "9" * 129, 1)

    assert SmtUnsatCoreRequest(logic="QF_LIA", smtlib=boundary)
    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=over)


@pytest.mark.parametrize(
    "assertion_template",
    (
        "(assert (= (* {outer} (ite p (* {inner} x) 0)) 0))",
        "(assert (= (* {outer} (ite p x (* {inner} x))) 0))",
        "(assert (= (* {outer} (- (ite p (* {inner} x) 0))) 0))",
        "(assert (= (* {outer} (- (- (ite p (* {inner} x) 0)))) 0))",
        "(assert (= (* {outer} (+ (ite p (* {inner} x) 0) 1)) 0))",
    ),
)
def test_request_bounds_scaled_ite_branch_coefficient_digits(
    assertion_template: str,
) -> None:
    outer = "9" * 200
    inner = "9" * 200
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Int)\n"
        f"{assertion_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


@pytest.mark.parametrize("nesting", (2, 3))
def test_request_bounds_deeply_nested_ite_branch_coefficients(nesting: int) -> None:
    factor = "9" * 128
    term = f"(* {factor} " * nesting + f"(ite p (* {factor} x) 0)" + ")" * nesting
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Int)\n"
        f"(assert (= {term} 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source)


@pytest.mark.parametrize(
    "assertion_template",
    (
        "(assert (= (* {outer} (ite p (* {inner} x) x)) 0))",
        "(assert (= (* {outer} (ite p x (* {inner} x))) 0))",
        "(assert (= (* {outer} (ite p (* {inner} x) (* {inner} x))) 0))",
    ),
)
def test_request_bounds_reciprocal_ite_branch_denominator_digits(
    assertion_template: str,
) -> None:
    outer = "0." + "9" * 150
    inner = "0." + "9" * 150
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"{assertion_template.format(outer=outer, inner=inner)}\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_ite_branch_digits_when_scalars_cancel_the_height() -> None:
    larger = "9" * 150
    smaller = "0." + "9" * 150
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* {smaller} (ite p (* {larger} x) (* {smaller} x))) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_division_of_formless_ite_branch_denominator_digits() -> None:
    digits = "9" * 200
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (/ (ite p (/ x {digits}) x) {digits}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_formless_ite_sum_denominator_digits() -> None:
    odd = "9" * 200
    even = "1" + "0" * 200
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (+ (ite p (/ x {odd}) x) (ite q (/ x {even}) x)) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_mixed_denominator_formless_ite_sum_scaling() -> None:
    power = "3" * 170
    scalar = "7" * 100
    ite_template = "(ite {p} (* 4.5 x) (* (/ 1 {power}) x))"
    sum_term = (
        f"(+ {ite_template.format(p='p', power=power)} "
        f"{ite_template.format(p='q', power=power)})"
    )
    scaled_source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* {scalar} {sum_term}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=scaled_source)

    unscaled_source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= {sum_term} 0))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=unscaled_source)


def test_request_bounds_opposite_sign_comparison_numerator_digits() -> None:
    divisor = "8" + "0" * 254 + "9"
    numerator = "9" * 256
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (ite p (* (- (/ "
        f"{numerator} {divisor}))) x) "
        "(ite q (* (/ "
        f"{numerator} {divisor})) x)))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_reciprocal_cancelling_formless_ite_still_admitted() -> None:
    digits = "9" * 256
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* (/ 1 {digits}) (ite p (* {digits} x) x)) 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_request_bounds_four_shared_denominator_formless_ite_terms() -> None:
    digits = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const r Bool)\n"
        "(declare-const s Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (+ "
        f"(ite p (/ x {digits}) x) "
        f"(ite q (/ x {digits}) x) "
        f"(ite r (/ x {digits}) x) "
        f"(ite s (/ x {digits}) x)"
        ") 4))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_request_bounds_comparison_of_formless_ite_branch_denominator_digits() -> None:
    odd = "9" * 200
    even = "1" + "0" * 200
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (ite p (/ x {odd}) x) (ite q (/ x {even}) x)))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_comparison_of_formless_ite_branches_still_admitted() -> None:
    digits = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (ite p (/ x {digits}) x) (ite q (/ x {digits}) x)))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_request_bounds_shared_denominator_comparison_numerator_digits() -> None:
    odd = "9" * 255
    small = "9" * 50
    power = "3" * 170
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* (/ {odd} 2) x) (* (/ {small} {power}) x)))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_scaled_formless_comparison_lifted_numerators() -> None:
    scalar = "1" + "0" * 99
    odd = "9" * 200
    ite_template = "(ite {p} (/ (* 9 x) 2) (/ x {odd}))"
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* {scalar} {ite_template.format(p='p', odd=odd)}) "
        f"(* {scalar} {ite_template.format(p='q', odd=odd)})))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_scaled_formless_sum_retains_unmatched_coefficients() -> None:
    digits = "9" * 255
    scalar = "7" * 10
    left = f"(/ (+ (* (+ {digits} 1) {{variable}}) (+ {digits} 1)) (* 2 {digits}))"
    right = f"(/ (+ (* (- {digits} 1) {{variable}}) (- {digits} 1)) (* 2 {digits}))"
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(declare-const y Real)\n"
        "(declare-const u Real)\n"
        "(declare-const v Real)\n"
        f"(assert (= (/ (- (ite p {left.format(variable='x')} "
        f"{left.format(variable='y')}) (ite q {right.format(variable='u')} "
        f"{right.format(variable='v')})) (/ 1 {scalar})) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_scaled_formless_comparison_still_admitted() -> None:
    scalar = "1" + "0" * 79
    odd = "9" * 160
    ite_template = "(ite {p} (/ (* 9 x) 2) (/ x {odd}))"
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* {scalar} {ite_template.format(p='p', odd=odd)}) "
        f"(* {scalar} {ite_template.format(p='q', odd=odd)})))\n"
        "(assert (>= x 0))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_request_bounds_nested_division_of_formless_ite_denominator_digits() -> None:
    outer = "9" * 200
    inner = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (/ (/ (ite p (* {outer} x) x) {outer}) {inner}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_nested_division_of_formless_ite_still_admitted() -> None:
    outer = "9" * 60
    inner = "3" * 60
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (/ (/ (ite p (* {outer} x) x) {outer}) {inner}) 0))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_request_bounds_compared_pairless_chain_denominator_digits() -> None:
    constants = [str(10**199 + offset) for offset in range(5)]

    def chain(prefix: str) -> str:
        term = "x"
        for index in range(5):
            term = f"(ite {prefix}{index} (* (/ 1 {constants[index]}) x) {term})"
        return term

    declarations = "".join(
        f"(declare-const left{index} Bool)\n(declare-const right{index} Bool)\n"
        for index in range(5)
    )
    source = (
        "(set-logic QF_LRA)\n"
        f"{declarations}"
        "(declare-const x Real)\n"
        f"(assert (= {chain('left')} {chain('right')}))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_bounds_scaled_formless_ite_nested_division_digits() -> None:
    scalar = "9" * 150
    outer = "9" * 150
    inner = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ (/ "
        f"(ite p (* {scalar} {outer} x) (* {outer} x)) "
        f"{outer}) {scalar}) {inner}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_scaled_formless_ite_nested_division_still_admitted() -> None:
    scalar = "9" * 60
    outer = "7" * 60
    inner = "3" * 60
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ (/ "
        f"(ite p (* {scalar} {outer} x) (* {outer} x)) "
        f"{outer}) {scalar}) {inner}) 0))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_request_bounds_sum_division_of_formless_ite_denominator_digits() -> None:
    inner = "9" * 200
    outer = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ "
        f"(+ (ite p (* {inner} x) x) (ite q (* {inner} x) x)) "
        f"(* 2 {inner})) {outer}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_sum_division_of_formless_ite_still_admitted() -> None:
    inner = "3" * 60
    outer = "9" * 60
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ "
        f"(+ (ite p (* {inner} x) x) (ite q (* {inner} x) x)) "
        f"(* 2 {inner})) {outer}) 0))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_request_keeps_oversized_sum_cross_sums_typed_and_admitted() -> None:
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (+ (ite p x (ite q (* 2 x) (ite p (* 4 x) (ite q (* 8 x) "
        "(* 16 x)))))) (ite p (* 16 x) (ite q (* 8 x) (ite p (* 4 x) "
        "(ite q (* 2 x) x))))))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_request_bounds_shared_denominator_sum_division_digits() -> None:
    inner = "9" * 130
    outer = "9" * 130
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ "
        f"(+ (ite p (/ x {inner}) x) (ite q (/ x {inner}) x)) "
        f"(* 2 {inner})) {outer}) 0))\n"
        "(check-sat)\n"
    )

    with pytest.raises(ValidationError, match="normalized SMT coefficient"):
        SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_bounded_shared_denominator_sum_division_still_admitted() -> None:
    inner = "9" * 60
    outer = "3" * 60
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (/ (/ "
        f"(+ (ite p (/ x {inner}) x) (ite q (/ x {inner}) x)) "
        f"(* 2 {inner})) {outer}) 0))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_request_keeps_shared_denominator_formless_ite_sums_admitted() -> None:
    digits = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const r Bool)\n"
        "(declare-const x Real)\n"
        "(assert (= (+ "
        f"(ite p (/ x {digits}) x) "
        f"(ite q (/ x {digits}) x) "
        f"(ite r (/ x {digits}) x)"
        ") 3))\n"
        "(assert (>= x 1))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_bounded_formless_ite_sums_flatten_and_still_solve() -> None:
    digits = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const q Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (+ (ite p (/ x {digits}) x) (ite q (/ x {digits}) x)) 0))\n"
        "(assert (>= x 1))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_bounded_division_of_formless_ite_branches_still_admitted() -> None:
    digits = "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (/ (ite p (/ x {digits}) x) {digits}) 0))\n"
        f"(assert (= (/ x {digits}) 1))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_request_keeps_exact_division_by_boundary_digit_divisor() -> None:
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const x Real)\n"
        f"(assert (= (/ x {'9' * 256}) 0))\n"
        "(check-sat)\n"
    )

    assert SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source)


def test_translated_nested_coefficients_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (>= (* 2 (+ (* 2 x) 1)) 10))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_nested_closed_coefficients_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (>= (* 2 (* 2 x)) 8))\n"
        "(assert (<= x 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_negated_nested_coefficients_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (>= (- (* 2 (- (* 2 x)))) 8))\n"
        "(assert (<= x 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_wrapped_lra_coefficients_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const y Real)\n"
        "(assert (>= (* 2.0 (/ (* 2.0 y) 4.0)) 3.0))\n"
        "(assert (<= y 1.0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_multi_term_affine_sums_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(declare-const y Int)\n"
        "(assert (>= (* 2 (+ (* 2 x) (* 3 y))) 13))\n"
        "(assert (<= x 1))\n"
        "(assert (<= y 1))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1, 2)


def test_cancelled_affine_sums_normalize_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (>= (* 2 (+ (* 3 x) (- (* 3 x)))) 0))\n"
        "(assert (> x 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_exact_integer_division_flattens_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (= (div (+ (* 6 x) 7) 2) 10))\n"
        "(assert (= x 2))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_non_exact_integer_division_remains_admitted() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const x Int)\n"
        "(assert (>= (div (* 3 x) 2) 2))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_bounded_ite_branch_coefficients_flatten_and_still_solve() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Int)\n"
        "(assert (>= (* 2 (ite p (* 2 x) 0)) 8))\n"
        "(assert (<= x 1))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_small_ite_branch_scalars_remain_admitted_and_satisfiable() -> None:
    source = (
        "(set-logic QF_LIA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Int)\n"
        "(assert (>= (* 2 (ite p x 0)) 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LIA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


def test_bounded_reciprocal_ite_branches_flatten_and_still_solve() -> None:
    larger = "9" * 100
    smaller = "0." + "9" * 100
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        f"(assert (= (* {smaller} (ite p (* {larger} x) (* {smaller} x))) 0))\n"
        "(assert (distinct x 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "UNSAT"
    assert result.core_indices == (0, 1)


def test_small_reciprocal_ite_scalars_remain_admitted_and_satisfiable() -> None:
    source = (
        "(set-logic QF_LRA)\n"
        "(declare-const p Bool)\n"
        "(declare-const x Real)\n"
        "(assert (>= (* 0.5 (ite p (* 0.25 x) x)) 0))\n"
        "(check-sat)\n"
    )

    result = compute_smt_unsat_core(SmtUnsatCoreRequest(logic="QF_LRA", smtlib=source))

    assert result.outcome == "SAT"
    assert result.core_indices == ()


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
    assert "does not claim a hard request-local byte limit" in schema["description"]
    assert "Boolean-sorted" in schema["properties"]["logic"]["description"]
    assert schema["examples"]
