from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.affine_semigroups import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    normalization,
)
from jacobian.math.affine_semigroups.semigroup_models import (
    AffineSemigroupNormalizationRequest,
)
from jacobian.math.affine_semigroups.semigroup_tools import TOOLS


def _semigroup(vectors: tuple[tuple[int, int], ...]) -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=tuple(f"g{index}" for index in range(len(vectors))),
        entries=tuple(tuple(vector[row] for vector in vectors) for row in range(2)),
    )
    return PositiveAffineSemigroup(
        configuration=configuration,
        grading=(
            {"num": 1, "den": 1},
            {"num": 1, "den": 1},
        ),
    )


def test_normalization_matches_small_exact_lattice_oracle() -> None:
    # In the first case gp(S)=2Z^2, so the normalization is not the Hilbert
    # basis in the ambient integer grid. In the second, gp(S)=Z^2 and the
    # missing lattice point (1,1) is an explicit normalization generator.
    cases = (
        (
            ((2, 0), (0, 2), (2, 2)),
            ((0, 2), (2, 0)),
        ),
        (
            ((1, 0), (1, 2), (2, 1)),
            ((1, 0), (1, 1), (1, 2)),
        ),
    )
    for vectors, expected in cases:
        result = normalization(_semigroup(vectors))
        assert result.generators == expected
        assert result.semigroup.configuration.columns_vectors == vectors


def test_normalization_catalog_contract_and_example() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "affine_semigroup.normalization.compute"
    )
    request = AffineSemigroupNormalizationRequest.model_validate_json(
        json.dumps(tool.examples[0].input)
    )
    result = tool.run(request)
    assert result.generators == ((0, 2), (2, 0))


def test_normalization_requires_full_rank_and_bounded_hilbert_search() -> None:
    with pytest.raises(ValueError, match="full-rank"):
        normalization(_semigroup(((1, 0), (2, 0))))

    # The group lattice is Z^2 due to the interior vector, while the cone rays
    # have determinant 1,001. The request is rejected before Hilbert enumeration.
    with pytest.raises(OperationResourceAdmissionError, match="determinant"):
        normalization(_semigroup(((1, 0), (1, 1001), (2, 1))))
