"""Public-behavior tests for the integer-matrix search plugin."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from jacobian.plugins.matrices import (
    evaluate_capability,
    find_witness_capability,
    materialize,
    reductions_capability,
)


def _kernel_candidate() -> dict:
    return {
        "rows": 2,
        "cols": 2,
        "entries": [["2", "4"], ["1", "2"]],
    }


def _maxdet_scope() -> dict:
    return {
        "rows": 3,
        "cols": 3,
        "entries": [-1, 1],
    }


def _maxdet_maximizer() -> dict:
    return {
        "rows": 3,
        "cols": 3,
        "entries": [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
        ],
    }


def test_determinant_predicate_rejects_rectangular_matrix() -> None:
    candidate = {
        "rows": 2,
        "cols": 3,
        "entries": [[1, 0, 0], [0, 1, 0]],
    }

    with pytest.raises(ValueError, match="matrix evaluation request does not match"):
        evaluate_capability(
            {"claim": {"predicate": "is_nonsingular"}, "candidate": candidate}
        )


def test_maxdet_rejects_rectangular_scope() -> None:
    claim = {
        "predicate": "maximize_absolute_determinant",
        "scope": {"rows": 2, "cols": 3, "entries": [-1, 1]},
    }

    with pytest.raises(ValueError, match="matrix evaluation request does not match"):
        evaluate_capability({"claim": claim, "candidate": _kernel_candidate()})


def test_evaluate_kernel_matrix_is_singular() -> None:
    claim = {"predicate": "is_nonsingular"}
    response = evaluate_capability({"claim": claim, "candidate": _kernel_candidate()})
    assert response["conclusion"] == "FALSE"
    assert response["objectives"] == {"determinant": {"num": "0", "den": "1"}}
    assert response["failure_classifications"] == ["nontrivial_kernel"]


def test_evaluate_matrix_accepts_canonical_entries_beyond_python_digit_limit() -> None:
    value = "1" + ("0" * 5_000)

    response = evaluate_capability(
        {
            "claim": {"predicate": "is_nonsingular"},
            "candidate": {"rows": 1, "cols": 1, "entries": [[value]]},
        }
    )

    assert response["conclusion"] == "TRUE"


def test_find_witness_kernel_vector() -> None:
    claim = {"predicate": "is_nonsingular"}
    resp = find_witness_capability(
        {
            "claim": claim,
            "candidate": _kernel_candidate(),
            "witness_role": "DEFEATS_CANDIDATE",
        }
    )
    assert resp["status"] == "FOUND"
    assert resp["witness_format"] == "matrix.kernel_vector"
    vec = resp["witness"]["vector"]
    assert _matrix_times_vector(_kernel_candidate()["entries"], vec) == [
        Fraction(0),
        Fraction(0),
    ]
    assert any(v != {"num": "0", "den": "1"} for v in vec)


def test_find_witness_rejects_unsupported_role() -> None:
    with pytest.raises(ValueError, match="supports only DEFEATS_CANDIDATE"):
        find_witness_capability(
            {
                "claim": {"predicate": "is_nonsingular"},
                "candidate": _kernel_candidate(),
                "witness_role": "SUPPORTS_CLAIM",
            }
        )


def test_evaluate_maxdet_maximizer() -> None:
    claim = {"predicate": "maximize_absolute_determinant", "scope": _maxdet_scope()}
    response = evaluate_capability({"claim": claim, "candidate": _maxdet_maximizer()})
    assert response["conclusion"] == "UNKNOWN"
    assert response["objectives"] == {"abs_determinant": {"num": "4", "den": "1"}}


def test_evaluate_maxdet_rejects_candidate_outside_scope() -> None:
    claim = {"predicate": "maximize_absolute_determinant", "scope": _maxdet_scope()}
    candidate = _maxdet_maximizer()
    candidate["entries"][0][0] = 100

    with pytest.raises(ValueError, match="matrix evaluation request does not match"):
        evaluate_capability({"claim": claim, "candidate": candidate})


def test_find_witness_maxdet_returns_maximizer() -> None:
    claim = {"predicate": "maximize_absolute_determinant", "scope": _maxdet_scope()}
    resp = find_witness_capability({"claim": claim, "witness_role": "SUPPORTS_CLAIM"})
    assert resp["status"] == "FOUND"
    assert resp["witness_format"] == "matrix.maximizer"
    mat = resp["witness"]["matrix"]
    assert all(entry in (-1, 1) for row in mat["entries"] for entry in row)
    det = _det_3x3(mat["entries"])
    assert abs(det) == 4


def test_find_witness_maxdet_requires_supporting_role() -> None:
    claim = {"predicate": "maximize_absolute_determinant", "scope": _maxdet_scope()}

    with pytest.raises(ValueError, match="supports only SUPPORTS_CLAIM"):
        find_witness_capability(
            {
                "claim": claim,
                "witness_role": "DEFEATS_CANDIDATE",
            }
        )


def test_find_witness_maxdet_rejects_over_budget_before_search() -> None:
    claim = {
        "predicate": "maximize_absolute_determinant",
        "scope": {"rows": 5, "cols": 5, "entries": [-1, 1]},
    }

    with pytest.raises(ValueError, match="scope exceeds witness search limit"):
        find_witness_capability({"claim": claim, "witness_role": "SUPPORTS_CLAIM"})


def test_materialize_maxdet_scope_count() -> None:
    claim = {"predicate": "maximize_absolute_determinant", "scope": _maxdet_scope()}
    resp = materialize({"claim": claim})
    assert len(resp["family"]) == 512
    # The scenario maximizer must be present.
    entries = [m["candidate"]["entries"] for m in resp["family"]]
    assert _maxdet_maximizer()["entries"] in entries


def test_reductions_kernel_fixture_is_minimal() -> None:
    claim = {"predicate": "is_nonsingular"}
    resp = reductions_capability(
        {"target_kind": "candidate", "target": _kernel_candidate(), "claim": claim}
    )
    assert resp["reductions"] == []


def test_reductions_finds_singular_principal_submatrix() -> None:
    # 3x3 singular matrix with a singular 2x2 principal submatrix.
    cand = {
        "rows": 3,
        "cols": 3,
        "entries": [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
    }
    claim = {"predicate": "is_nonsingular"}
    resp = reductions_capability(
        {"target_kind": "candidate", "target": cand, "claim": claim}
    )
    kinds = {r["reducer"] for r in resp["reductions"]}
    assert "delete_row_column" in kinds


def test_evaluation_does_not_grant_verification_authority() -> None:
    response = evaluate_capability(
        {"claim": {"predicate": "is_nonsingular"}, "candidate": _kernel_candidate()}
    )
    assert "verified" not in response


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _matrix_times_vector(
    entries: list[list[Any]], vector: list[dict[str, str]]
) -> list[Fraction]:
    mat = [[Fraction(int(x)) for x in row] for row in entries]
    vec = [Fraction(int(v["num"]), int(v["den"])) for v in vector]
    result: list[Fraction] = []
    for row in mat:
        total = Fraction(0)
        for a, b in zip(row, vec, strict=True):
            total += a * b
        result.append(total)
    return result


def _det_3x3(m: list[list[int]]) -> int:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
