from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.simplicial_sets import truncate as truncate_module
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.standard import standard_simplex
from jacobian.math.topology.simplicial_sets.truncate import truncate_simplicial_set
from jacobian.math.topology.simplicial_sets.truncate_models import (
    SimplicialSetTruncateRequest,
)


def test_truncation_retains_exact_tables_and_rechecks_visible_identities() -> None:
    source = standard_simplex(2, 3)
    result = truncate_simplicial_set(
        SimplicialSetTruncateRequest(simplicial_set=source, max_degree=2)
    )
    prefix = result

    assert prefix.max_degree == 2
    assert prefix.sets == source.sets[:3]
    assert prefix.face_maps == source.face_maps[:2]
    assert prefix.degeneracy_maps == source.degeneracy_maps[:2]
    assert prefix.checked_identities < source.checked_identities

    repeated = truncate_simplicial_set(
        SimplicialSetTruncateRequest(simplicial_set=prefix, max_degree=2)
    )
    assert repeated == prefix


def test_degree_zero_truncation_keeps_vertices_and_no_maps() -> None:
    source = standard_simplex(1, 2)
    prefix = truncate_simplicial_set(
        SimplicialSetTruncateRequest(simplicial_set=source, max_degree=0)
    )
    assert prefix.sets == source.sets[:1]
    assert prefix.face_maps == prefix.degeneracy_maps == ()
    assert prefix.checked_identities == 0


def test_truncation_rejects_degree_above_source() -> None:
    source = standard_simplex(1, 1)
    with pytest.raises(OperationDomainValidationError, match="must not exceed"):
        truncate_simplicial_set(
            SimplicialSetTruncateRequest.model_construct(
                simplicial_set=source, max_degree=2
            )
        )


def test_truncation_requires_strict_integer_degree() -> None:
    source = standard_simplex(1, 1)
    with pytest.raises(ValidationError):
        SimplicialSetTruncateRequest(simplicial_set=source, max_degree=True)


def test_truncation_rechecks_retained_caller_tables() -> None:
    source = standard_simplex(1, 2)
    # Keep structural axes valid but corrupt a face identity. The carrier's
    # ordinary validator checks table shape, not the mathematical identities.
    faces = ((source.face_maps[0][0], (1, 1, 1)), *source.face_maps[1:])
    forged = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=source.sets,
        face_maps=faces,
        degeneracy_maps=source.degeneracy_maps,
        total_simplices=source.total_simplices,
        checked_identities=source.checked_identities,
    )
    with pytest.raises(OperationDomainValidationError, match="do not satisfy"):
        truncate_simplicial_set(
            SimplicialSetTruncateRequest.model_construct(
                simplicial_set=forged, max_degree=1
            )
        )


def test_output_admission_accepts_exact_estimate_and_rejects_one_byte_less(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = standard_simplex(1, 1)
    estimate = truncate_module._estimate_output_bytes(source.sets, 1, 8)
    request = SimplicialSetTruncateRequest(simplicial_set=source, max_degree=1)

    monkeypatch.setattr(truncate_module, "MAX_TRUNCATE_OUTPUT_BYTES", estimate)
    assert truncate_simplicial_set(request).max_degree == 1
    monkeypatch.setattr(truncate_module, "MAX_TRUNCATE_OUTPUT_BYTES", estimate - 1)
    with pytest.raises(
        OperationResourceAdmissionError, match="estimated truncation output"
    ):
        truncate_simplicial_set(request)
