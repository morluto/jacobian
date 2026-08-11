"""Regression coverage for malformed internal contract states."""

from __future__ import annotations

import pytest

from jacobian.contracts.jacobian_syzygy import (
    GradedJacobianSyzygyRequest,
    _compute_homogeneous_source_degree,
)
from jacobian.contracts.plugin_matrices import MatrixCandidate, MatrixClaim
from jacobian.contracts.universal_algebra import (
    CountermodelSearchStatus,
    FiniteMagmaCountermodelArtifact,
    MagmaTerm,
)


def test_magma_term_methods_reject_missing_fields_in_constructed_states() -> None:
    product = MagmaTerm.model_construct(kind="PRODUCT")
    variable = MagmaTerm.model_construct(kind="VARIABLE")

    with pytest.raises(ValueError, match="exactly two child terms"):
        product.node_count()
    with pytest.raises(ValueError, match="exactly two child terms"):
        product.depth()
    with pytest.raises(ValueError, match="exactly two child terms"):
        product.variable_names()
    with pytest.raises(ValueError, match="only a variable name"):
        variable.variable_names()


def test_contract_guards_reject_missing_required_nullability_state() -> None:
    with pytest.raises(ValueError, match="labelled linear factors are required"):
        _compute_homogeneous_source_degree(None, None)

    syzygy_request = GradedJacobianSyzygyRequest.model_construct(
        polynomial=None,
        linear_factors=None,
        linear_factor_variables=None,
    )
    with pytest.raises(ValueError, match="supply exactly one"):
        syzygy_request.require_bounded_homogeneous_three_variable_input()

    artifact = FiniteMagmaCountermodelArtifact.model_construct(
        status=CountermodelSearchStatus.WITNESS_FOUND,
        structure=None,
        source_records=None,
        target_record=None,
    )
    with pytest.raises(ValueError, match="complete witness evidence"):
        artifact.require_status_evidence_shape()

    candidate = MatrixCandidate.model_construct(rows=1, cols=1, entries=((1,),))
    claim = MatrixClaim.model_construct(
        predicate="maximize_absolute_determinant",
        scope=None,
    )
    with pytest.raises(ValueError, match="requires a scope"):
        candidate.validate_for_claim(claim)
