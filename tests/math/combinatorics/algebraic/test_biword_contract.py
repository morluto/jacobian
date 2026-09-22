from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.biword import (
    Biword,
    BiwordNormalizeRequest,
    BiwordRSKPair,
)
from jacobian.math.combinatorics.algebraic.biword_ops import (
    greene,
    inverse_biword,
    normalize_biword,
    rsk_biword,
)
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)
from jacobian.math.logic.languages.words.values import FiniteWord


def test_normalization_rejects_oversized_source_before_sorting() -> None:
    request = BiwordNormalizeRequest.model_construct(
        top_alphabet=("a",),
        bottom_alphabet=("a",),
        top=("a",) * 501,
        bottom=("a",) * 501,
    )
    with pytest.raises(OperationDomainValidationError):
        normalize_biword(request)


def test_native_algebraic_boundaries_reject_raw_invalid_values() -> None:
    with pytest.raises(OperationDomainValidationError):
        rsk_biword(None)  # type: ignore[arg-type]
    with pytest.raises(OperationDomainValidationError):
        greene(None)
    forged = Biword.model_construct(
        top_alphabet=("a",), bottom_alphabet=("a",), top=("a",), bottom=("a",)
    )
    assert rsk_biword(forged).shape.parts == (1,)
    with pytest.raises(OperationDomainValidationError):
        greene(FiniteWord.model_construct(alphabet=("a",), letters=("b",)))


def test_greene_decreasing_invariants_are_cumulative_column_totals() -> None:
    result = greene(
        FiniteWord(
            alphabet=("1", "2", "3"),
            letters=("2", "1", "3"),
        ),
        requested_k=2,
    )

    assert result.shape.parts == (2, 1)
    assert result.increasing_totals == (2, 3)
    assert result.decreasing_totals == (2, 3)


def test_inverse_rejects_forged_pair_content_at_native_boundary() -> None:
    pair = BiwordRSKPair.model_construct(
        top_alphabet=(),
        bottom_alphabet=(),
        insertion_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        recording_tableau=SemistandardYoungTableau.model_construct(rows=((1,),)),
        shape=IntegerPartition.model_construct(parts=(1,)),
        source_kind="BIWORD",
    )
    with pytest.raises(OperationDomainValidationError):
        inverse_biword(pair)
