"""Tests for SAT UNSAT proof verification."""

from __future__ import annotations

import pytest

from jacobian.domains.logic.operations import (
    SatProofVerifyRequest,
    SatProofVerifyResult,
    verify_sat_proof,
)


def test_unsat_formula_is_verified():
    """{(a v b) & (~a v b) & (~b)} is UNSAT and should be verified."""
    request = SatProofVerifyRequest.model_validate({
        "cnf": {
            "variables": ["a", "b"],
            "clauses": [[-1, 2], [1, 2], [-2]],
        },
        "proof_lines": ["0"],
        "timeout_ms": 5000,
    })
    result = verify_sat_proof(request)
    assert result.status == "VERIFIED"
    assert result.cnf_digest.startswith("sha256:")


def test_sat_formula_is_rejected():
    """{(a v b) & (~a v b)} is SAT, so UNSAT claim should be rejected."""
    request = SatProofVerifyRequest.model_validate({
        "cnf": {
            "variables": ["a", "b"],
            "clauses": [[-1, 2], [1, 2]],
        },
        "proof_lines": ["0"],
        "timeout_ms": 5000,
    })
    result = verify_sat_proof(request)
    assert result.status == "REJECTED"


def test_empty_proof_is_invalid():
    """An empty proof artifact should return INVALID."""
    request = SatProofVerifyRequest.model_validate({
        "cnf": {
            "variables": ["a", "b"],
            "clauses": [[-1, 2], [1, 2], [-2]],
        },
        "proof_lines": [],
        "timeout_ms": 5000,
    })
    result = verify_sat_proof(request)
    assert result.status == "INVALID"


def test_one_clause_corruption_is_rejected():
    """If we corrupt one clause, the formula becomes satisfiable."""
    # {(a) & (~b)} = SAT (a=True, b=False)
    request = SatProofVerifyRequest.model_validate({
        "cnf": {
            "variables": ["a", "b"],
            "clauses": [[1], [-2]],
        },
        "proof_lines": ["0"],
        "timeout_ms": 5000,
    })
    result = verify_sat_proof(request)
    assert result.status == "REJECTED"


def test_lrat_format_detected():
    """LRAT-style proof lines should be detected correctly."""
    request = SatProofVerifyRequest.model_validate({
        "cnf": {
            "variables": ["a", "b"],
            "clauses": [[-1, 2], [1, 2], [-2]],
        },
        "proof_lines": ["0 1 2 0"],
        "timeout_ms": 5000,
    })
    result = verify_sat_proof(request)
    assert result.status == "VERIFIED"
    assert result.proof_format == "LRAT"


def test_operation_discoverable():
    """The operation should be in LOGIC_OPERATIONS."""
    from jacobian.domains.logic import logic_operations

    ops = logic_operations()
    assert any(op.operation_id == "sat.verify.proof" for op in ops)
