"""Homology's shared Smith certificates compose into explicit bounded replay."""

import pytest

from jacobian.math.matrices.certified_snf.operations import (
    verify_smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.values import SmithNormalFormCertificate
from jacobian.math.topology.chain_complexes.operations import homology_groups
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
    HomologyResult,
    IntegralHomologyGroupValue,
)


@pytest.mark.parametrize("size", (16, 17, 32))
@pytest.mark.parametrize("coefficient", (None, 0, 1, 6))
def test_serialized_homology_certificates_replay_at_full_owner_boundary(
    size: int, coefficient: int | None
) -> None:
    isolated = coefficient is None
    source = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        degree_min=0,
        degree_max=0 if isolated else 1,
        basis_sizes=(size,) if isolated else (size, size),
        differential_matrices=()
        if isolated
        else (
            tuple(
                tuple((coefficient or 0) if i == j else 0 for j in range(size))
                for i in range(size)
            ),
        ),
    )
    result = HomologyResult.model_validate_json(
        homology_groups(source).model_dump_json()
    )
    first = result.homology_groups[0]
    assert isinstance(first, IntegralHomologyGroupValue)
    assert first.free_rank == (size if coefficient in (None, 0) else 0)
    assert first.torsion_invariant_factors == ((6,) * size if coefficient == 6 else ())
    if not isolated:
        assert isinstance(result.homology_groups[1], IntegralHomologyGroupValue)
        assert result.homology_groups[1].free_rank == (size if coefficient == 0 else 0)
    for group in result.homology_groups:
        assert isinstance(group, IntegralHomologyGroupValue)
        for certificate in (
            group.outgoing_smith_certificate,
            group.incoming_smith_certificate,
        ):
            assert certificate is not None
            # In this diagonal family all transformations are identity matrices;
            # check the defining identity independently, including empty axes.
            assert certificate.source == certificate.diagonal
            for transformation in (
                certificate.left_transformation,
                certificate.right_transformation,
            ):
                assert transformation.entries == tuple(
                    tuple(int(i == j) for j in range(transformation.column_count))
                    for i in range(transformation.row_count)
                )
            assert certificate.left_determinant == certificate.right_determinant == 1
            assert verify_smith_normal_form_certificate(certificate)
            forged = SmithNormalFormCertificate.model_validate_json(
                certificate.model_copy(
                    update={"left_determinant": -1}
                ).model_dump_json()
            )
            assert not verify_smith_normal_form_certificate(forged)
