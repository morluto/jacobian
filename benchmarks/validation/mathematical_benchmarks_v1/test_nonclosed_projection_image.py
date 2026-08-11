from __future__ import annotations

import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support
from jsonschema import Draft202012Validator

TASK_NAME = "nonclosed-projection-image"
TASK = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "datasets"
    / "mathematical-benchmarks-v1"
    / TASK_NAME
)
PREFIX_LENGTH = 12
TERMS = 100
SCOPE = (
    "a closed Hilbert subspace and orthogonal projection with nonclosed projected image"
)
LIMITATION = (
    "The verifier checks exact sequence identities and analytic bounds but does "
    "not formalize Hilbert-space topology in a proof assistant."
)


def _witness(*, weight_reciprocal_square: bool) -> dict:
    """Build a valid diagonal-operator graph witness.

    ``weight_reciprocal_square=False`` is the Oracle reference (weights 1/n,
    limit 1/n, forced preimage 1). ``True`` is the review's alternative
    (weights 1/n^2, limit 1/n, forced preimage n): a bounded operator, a closed
    graph, the same convergent limit, and a non-square-summable preimage.
    """
    if weight_reciprocal_square:
        weights = [Fraction(1, n * n) for n in range(1, TERMS + 1)]
        preimage = [Fraction(n) for n in range(1, PREFIX_LENGTH + 1)]
        operator = "diagonal T with weights 1/n^2 on ell2"
        limit_preimage = (
            "the forced preimage x_n=n is not in ell2 because sum n^2 diverges"
        )
    else:
        weights = [Fraction(1, n) for n in range(1, TERMS + 1)]
        preimage = [Fraction(1)] * PREFIX_LENGTH
        operator = "diagonal T with weights 1/n on ell2"
        limit_preimage = (
            "the forced preimage x_n=1 is not in ell2 because sum 1 diverges"
        )
    limit_coords = [Fraction(1, n) for n in range(1, TERMS + 1)]
    prefixes: list[dict] = []
    limit_partial = Fraction(0)
    preimage_partial = Fraction(0)
    for n in range(1, PREFIX_LENGTH + 1):
        weight = weights[n - 1]
        coordinate = preimage[n - 1]
        limit = limit_coords[n - 1]
        assert limit == weight * coordinate
        limit_partial += limit * limit
        preimage_partial += coordinate * coordinate
        prefixes.append(
            {
                "n": n,
                "weight": str(weight),
                "preimage_coordinate": str(coordinate),
                "limit_norm_sq_partial": str(limit_partial),
                "preimage_norm_sq_partial": str(preimage_partial),
            }
        )
    return {
        "space": "ell2 direct sum ell2",
        "operator": operator,
        "subspace": "closed graph of bounded T",
        "projection": "P(u,v)=(0,v) onto the second summand",
        "operator_bound": "1",
        "prefixes": prefixes,
        "limit_coordinates": [str(y) for y in limit_coords],
        "tail_bound": {
            "bound_coefficient": "1",
            "bound_exponent": 1,
            "verification_terms": TERMS,
        },
        "limit_preimage": limit_preimage,
    }


def _proof(*, weight_reciprocal_square: bool) -> dict:
    absent = (
        "The forced preimage has coordinates x_n=n, which is not in ell2 "
        "since sum n^2 diverges, so y has no square-summable preimage."
        if weight_reciprocal_square
        else "The forced preimage of y has coordinates x_n=1, which is not in "
        "ell2 since sum x_n^2=sum 1 diverges, so y has no square-summable "
        "preimage."
    )
    return {
        "boundedness": (
            "The diagonal weights satisfy 0<w_n<=1 for every n, so the operator "
            "T is bounded on ell2 with operator norm at most 1."
        ),
        "closedness": (
            "A bounded operator on a Hilbert space has a closed graph, so the "
            "graph of T is a closed linear subspace of ell2 direct sum ell2."
        ),
        "range_identification": (
            "Projecting the graph of T onto the second summand gives P(graph T) "
            "equal to the range of T, identifying the projection image with the "
            "operator range."
        ),
        "convergence": (
            "The truncations of y_n=1/n lie in the range and converge to y in "
            "ell2 because the tail sum_{n>m}1/n^2 is bounded by 1/m, which "
            "tends to zero."
        ),
        "absent_preimage": absent,
    }


def _write_case(
    tmp_path: Path,
    *,
    result: dict,
    proof: dict | None,
    evidence_text: str | None = None,
    label: str = "case",
) -> tuple[Path, Path, Path]:
    root = tmp_path / label
    app = root / "app"
    logs = root / "logs"
    shutil.rmtree(root, ignore_errors=True)
    (app / "evidence").mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    if evidence_text is None:
        assert proof is not None
        evidence_text = "Witness certificate.\n\n" + "\n\n".join(proof.values()) + "\n"
    (app / "evidence" / "answer.txt").write_text(evidence_text)
    digest = "sha256:" + hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    submission = {
        "task_id": "jacobian/nonclosed-projection-image",
        "conclusion": "NONCLOSED_IMAGE_CERTIFIED",
        "result": {**result, "proof_obligations": proof or {}},
        "claimed_assurance": "COMPUTED",
        "scope": SCOPE,
        "completeness": "COMPLETE",
        "evidence": [{"path": "evidence/answer.txt", "sha256": digest}],
        "limitations": [LIMITATION],
    }
    support._write_json(app / "submission.json", submission)
    return TASK, app, logs


def test_nonclosed_projection_instruction_is_strategy_free() -> None:
    instruction = (TASK / "instruction.md").read_text().lower()
    # The prompt must not prescribe the diagonal-operator graph decomposition.
    assert "diagonal-operator graph" not in instruction
    assert "use an" not in instruction
    # It must still state the counterexample requirements and evidence links.
    for term in ("closed", "projection", "convergence", "preimage", "ell2"):
        assert term in instruction


def test_nonclosed_projection_accepts_alternate_diagonal_weights(
    tmp_path: Path,
) -> None:
    # Review thread: weights 1/n^2, limit y_n=1/n, forced preimage x_n=n is a
    # bounded operator with a closed graph, the same convergent limit, and a
    # non-square-summable preimage. It must not be rejected as a false negative.
    result = support._run_verifier(
        *_write_case(
            tmp_path,
            result=_witness(weight_reciprocal_square=True),
            proof=_proof(weight_reciprocal_square=True),
            label="alternate",
        )
    )
    assert result.details["correctness"] == 1.0
    assert result.details["evidence_validity"] == 1.0
    assert result.reward == pytest.approx(1.0)


def test_nonclosed_projection_rejects_token_only_evidence(tmp_path: Path) -> None:
    # Review thread: an evidence file containing only the legacy tokens must no
    # longer receive full evidence validity.
    result = support._run_verifier(
        *_write_case(
            tmp_path,
            result=_witness(weight_reciprocal_square=False),
            proof=None,
            evidence_text="ell2 graph projection all-ones computed\n",
            label="token_only",
        )
    )
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_nonclosed_projection_rejects_missing_proof_obligation(
    tmp_path: Path,
) -> None:
    proof = _proof(weight_reciprocal_square=False)
    del proof["absent_preimage"]
    result = support._run_verifier(
        *_write_case(
            tmp_path,
            result=_witness(weight_reciprocal_square=False),
            proof=proof,
            label="missing_obligation",
        )
    )
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_nonclosed_projection_rejects_keyword_only_proof_argument(
    tmp_path: Path,
) -> None:
    proof = _proof(weight_reciprocal_square=False)
    proof["boundedness"] = "bound"
    result = support._run_verifier(
        *_write_case(
            tmp_path,
            result=_witness(weight_reciprocal_square=False),
            proof=proof,
            label="keyword_only",
        )
    )
    assert result.details["correctness"] == 0.0
    assert result.details["evidence_validity"] == 0.0
    assert result.reward == 0.0


def test_nonclosed_projection_ignores_stale_hidden_result_marker(
    tmp_path: Path,
) -> None:
    # The structured submission is authoritative. Legacy marker text in the
    # human-readable artifact must not become an unadvertised second contract.
    result = _witness(weight_reciprocal_square=False)
    proof = _proof(weight_reciprocal_square=False)
    stale = dict(result)
    stale["operator_bound"] = "2"
    marker = "RESULT_JSON: " + json.dumps(stale, sort_keys=True, separators=(",", ":"))
    proof_marker = "PROOF_JSON: " + json.dumps(
        proof, sort_keys=True, separators=(",", ":")
    )
    evidence_text = "Witness.\n\n" + marker + "\n" + proof_marker + "\n"
    out = support._run_verifier(
        *_write_case(
            tmp_path,
            result=result,
            proof=proof,
            evidence_text=evidence_text,
            label="stale_marker",
        )
    )
    assert out.details["correctness"] == 1.0
    assert out.details["evidence_validity"] == 1.0
    assert out.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda r: r["prefixes"][0].__setitem__("preimage_coordinate", "2"), "range"),
        (lambda r: r.__setitem__("operator_bound", "1/2"), "boundedness"),
        (lambda r: r["tail_bound"].__setitem__("bound_coefficient", "1/1000"), "tail"),
        (lambda r: r["prefixes"][0].__setitem__("weight", "2"), "weight"),
    ],
)
def test_nonclosed_projection_rejects_corrupted_witnesses(
    tmp_path: Path,
    mutate,
    field: str,
) -> None:
    result = _witness(weight_reciprocal_square=False)
    mutate(result)
    out = support._run_verifier(
        *_write_case(
            tmp_path,
            result=result,
            proof=_proof(weight_reciprocal_square=False),
            label=f"corrupt_{field}",
        )
    )
    assert out.details["correctness"] == 0.0
    assert out.reward == 0.0


def test_nonclosed_projection_solution_satisfies_agent_schema() -> None:
    schema = json.loads((TASK / "environment" / "submission_schema.json").read_text())
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    Draft202012Validator(schema).validate(submission)
