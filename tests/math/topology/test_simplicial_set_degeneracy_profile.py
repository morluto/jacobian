"""Exact immediate degeneracy profiles on finite simplicial sets."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.simplicial_sets._models import FiniteTruncatedSimplicialSet
from jacobian.math.topology.simplicial_sets.degeneracy import degeneracy_profile
from jacobian.math.topology.simplicial_sets.standard import standard_simplex


def test_delta_one_profile_retains_axes_and_reports_first_witness() -> None:
    source = standard_simplex(1, 2)
    result = degeneracy_profile(source)

    assert result.simplicial_set == source
    assert tuple(p.nondegenerate_indices for p in result.degrees) == (
        (0, 1),
        (1,),
        (),
    )
    # In degree one, the degenerate loops are s_0 of the two vertices.
    witnesses = result.degrees[1].immediate_witnesses
    assert witnesses[0] is not None
    assert (witnesses[0].degeneracy_index, witnesses[0].source_simplex_index) == (
        0,
        0,
    )
    assert witnesses[1] is None
    assert witnesses[2] is not None
    assert (witnesses[2].degeneracy_index, witnesses[2].source_simplex_index) == (
        0,
        1,
    )
    # Each degree-two simplex has at least one degeneracy presentation. The
    # operation's documented lexicographic rule picks the first such map.
    for profile in result.degrees[2:]:
        assert all(witness is not None for witness in profile.immediate_witnesses)


def test_profile_rechecks_caller_supplied_simplicial_identities() -> None:
    source = standard_simplex(1, 2)
    faces = ((source.face_maps[0][0], (1, 1, 1)), *source.face_maps[1:])
    forged = FiniteTruncatedSimplicialSet._from_kernel(
        max_degree=source.max_degree,
        sets=source.sets,
        face_maps=faces,
        degeneracy_maps=source.degeneracy_maps,
        total_simplices=source.total_simplices,
        checked_identities=source.checked_identities,
    )

    with pytest.raises(OperationDomainValidationError, match="fail a visible"):
        degeneracy_profile(forged)


def test_degree_zero_profile_and_empty_degree_axis() -> None:
    source = standard_simplex(1, 0)
    result = degeneracy_profile(source)
    assert result.degrees[0].nondegenerate_indices == (0, 1)
    assert result.degrees[0].immediate_witnesses == (None, None)
