"""Matching and unification return the substitution consumer's own carrier."""

import pytest
from pydantic import ValidationError

from jacobian.math.logic.term_rewriting.operations import (
    matching_result,
    substitution_result,
    unification_result,
)
from jacobian.math.logic.term_rewriting.values import RankedSignature, Term


@pytest.mark.parametrize("unify", [False, True])
def test_serialized_substitution_composes(unify: bool) -> None:
    signature = RankedSignature(arities=(0,))
    variable = Term(is_variable=True, symbol=7, children=())
    constant = Term(is_variable=False, symbol=0, children=())
    result = (unification_result if unify else matching_result)(
        signature, variable, constant
    )
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert (
        substitution_result(signature, variable, decoded.substitution).result
        == constant
    )
    payload = result.model_dump()
    payload["substitution"]["mapping"] = {-1: constant.model_dump()}
    with pytest.raises(ValidationError):
        type(result).model_validate(payload)
