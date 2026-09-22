from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.affine_semigroups.semigroup import (
    AffineConfiguration,
    PositiveAffineSemigroup,
    construct,
    fiber,
    positive_grading,
)
from jacobian.math.combinatorics.algebraic.biword import BiwordRSKPair
from jacobian.math.combinatorics.algebraic.biword_ops import inverse_biword
from jacobian.math.combinatorics.matroids.delta.extra import BinarySymmetricMatrix
from jacobian.math.combinatorics.matroids.delta.extra_ops import dual, minor
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)


def _quadrant() -> PositiveAffineSemigroup:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=((1, 0), (0, 1)),
    )
    return construct(
        configuration,
        (CanonicalRational(num=1, den=1), CanonicalRational(num=1, den=1)),
    )


def test_positive_grading_is_complete_beyond_old_search_box() -> None:
    configuration = AffineConfiguration(
        row_labels=("x", "y"),
        generator_labels=("a", "b"),
        entries=((10, -9), (-1, 1)),
    )
    result = positive_grading(configuration)
    assert result.positive is True
    assert tuple(value.as_fraction() for value in result.grading) == (
        Fraction(2),
        Fraction(19),
    )


def test_affine_fiber_rejects_target_before_recursion() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="candidate bound"):
        fiber(_quadrant(), (10**100, 0))


def test_delta_minor_compacts_axes_and_uses_deletion_semantics() -> None:
    feasible = tuple(tuple(i for i in range(3) if mask >> i & 1) for mask in range(8))
    delta = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=tuple(sorted(feasible)))
    result = minor(delta, delete=(1,))
    assert result.ground == ("a", "c")
    assert result.feasible == ((), (0,), (0, 1), (1,))


def test_delta_extra_operations_reject_forged_non_delta_sources() -> None:
    forged = FiniteDeltaMatroid(ground=("a", "b", "c"), feasible=((), (0, 1, 2)))
    with pytest.raises(OperationDomainValidationError, match="not a delta-matroid"):
        dual(forged)


def test_binary_principal_minor_envelope_is_visible_and_preflighted() -> None:
    with pytest.raises(ValidationError, match="limited to 12"):
        BinarySymmetricMatrix(
            ground=tuple(str(i) for i in range(13)),
            entries=tuple(tuple(0 for _ in range(13)) for _ in range(13)),
        )
    forged = BinarySymmetricMatrix.model_construct(
        ground=tuple(str(i) for i in range(13)),
        entries=tuple(tuple(0 for _ in range(13)) for _ in range(13)),
    )
    from jacobian.math.combinatorics.matroids.delta.extra_ops import binary

    with pytest.raises(OperationDomainValidationError, match="limited to 12"):
        binary(forged)


def test_inverse_biword_rejects_forged_alphabet_pair() -> None:
    pair = BiwordRSKPair.model_construct(
        top_alphabet=(),
        bottom_alphabet=(),
        insertion_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        recording_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        shape=IntegerPartition.model_construct(parts=(1,)),
        source_kind="BIWORD",
        convention="ROW_INSERTION_RSK_V1",
    )
    with pytest.raises(OperationDomainValidationError, match="insertion entries"):
        inverse_biword(pair)
