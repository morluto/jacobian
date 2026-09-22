from __future__ import annotations

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.algebraic.biword import BiwordRSKPair
from jacobian.math.combinatorics.algebraic.biword_ops import inverse_biword
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)


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
